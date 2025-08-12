# python/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import requests, traceback, pickle, numpy as np
import time, os

load_dotenv()

OLLAMA_GENERATE_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3:8b")

K_TOP = int(os.getenv("K_TOP", "8"))
MAX_CTX_TOTAL = int(os.getenv("MAX_CTX_TOTAL", "2000"))   # символы
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "800"))

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "300"))

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")
FALLBACK_EMPTY = os.getenv("FALLBACK_EMPTY")
NO_CONTEXT = os.getenv("NO_CONTEXT")
SERVICE_BUSY = os.getenv("SERVICE_BUSY")

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

BASE_DIR = Path(__file__).parent
STORE_DIR = BASE_DIR / "rag_store"

# Загрузка индекса и метаданных
try:
    faiss_index = faiss.read_index(str(STORE_DIR / "faiss.index"))
    with open(STORE_DIR / "meta.pkl", "rb") as f:
        DOCS = pickle.load(f)
    emb_model_name_path = STORE_DIR / "model_name.txt"
    if emb_model_name_path.exists():
        with open(emb_model_name_path, "r", encoding="utf-8") as f:
            EMB_MODEL_NAME = f.read().strip()
    else:
        EMB_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
except Exception as e:
    raise RuntimeError(f"Не удалось загрузить индекс из {STORE_DIR}: {e}")

emb_model = SentenceTransformer(EMB_MODEL_NAME)

def _truncate(s: str, limit: int) -> str:
    s = (s or "").strip()
    if len(s) <= limit:
        return s
    cut = s[:limit]
    dot = cut.rfind(".")
    if dot >= int(limit * 0.5):
        cut = cut[:dot+1]
    return cut + " …"

def retrieve(query: str, k: int = K_TOP):
    qv = emb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    distances, indices = faiss_index.search(qv, k)
    idxs = indices[0]
    scores = distances[0]
    results = []
    for i, idx in enumerate(idxs):
        if idx < 0 or idx >= len(DOCS):
            continue
        d = DOCS[int(idx)]
        results.append({
            "law": d.get("law",""),
            "chapter": d.get("chapter",""),
            "article": d.get("article",""),
            "text": d.get("text",""),
            "score": float(scores[i]),
        })
    return results

def build_context(found):
    parts, cites = [], []
    used = 0
    for r in found:
        head = r['article'] or r.get('chapter','')
        body = _truncate(r["text"], MAX_CHUNK_CHARS)
        piece = f"[{r['law']} — {head}] {body}".strip()
        if not piece:
            continue
        if used + len(piece) > MAX_CTX_TOTAL:
            break
        parts.append(piece)
        cites.append(f"[{r['law']}, {head}]")
        used += len(piece)
    ctx = "\n---\n".join(parts)
    cite_str = " ".join(cites) if cites else "[]"
    return ctx, cite_str

def make_prompt(system_prompt: str, ctx: str, question: str, cites: str) -> str:
    return (
        f"{system_prompt}\n\n"
        f"КОНТЕКСТ:\n{ctx}\n\n"
        f"ВОПРОС: {question}\n\n"
        f"Сформулируй ответ строго по контексту. Ответь как можно меньше но не потеряй смысл. "
        f"Закончи ответ полностью. Не обрывай предложение. "
        f"Если в контексте нет ответа — откажись фразой из инструкции."
    )

def call_ollama_generate(prompt: str, retry: int = 1) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "5m",
        "options": {
            "num_ctx": 2048,
            "num_predict": NUM_PREDICT,
            "num_thread": 4,
            "temperature": 0.2,
            "repeat_penalty": 1.1,
        },
    }
    last_err = None
    for attempt in range(retry + 1):
        try:
            resp = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return (data.get("response") or "").strip()
        except requests.ReadTimeout as e:
            last_err = e
            time.sleep(2)
        except Exception as e:
            last_err = e
            break
    raise HTTPException(status_code=503, detail=f"Ollama недоступен: {last_err}")

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    question = (request.question or "").strip()
    if not question:
        return {"answer": FALLBACK_EMPTY}

    found = retrieve(question, k=K_TOP)
    if not found:
        return {"answer": NO_CONTEXT}

    ctx, cites = build_context(found)
    if not ctx.strip():
        return {"answer": NO_CONTEXT}

    prompt = make_prompt(SYSTEM_PROMPT, ctx, question, cites)

    try:
        answer = call_ollama_generate(prompt, retry=1)
        if not answer:
            return {"answer": NO_CONTEXT}
        return {"answer": answer}
    except HTTPException as e:
        if e.status_code == 503:
            return {"answer": SERVICE_BUSY}
        raise
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервиса")
