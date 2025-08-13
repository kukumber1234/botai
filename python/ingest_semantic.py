import os, glob, pickle, re
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

KB_DIR = Path(__file__).parent / "kb"
STORE_DIR = Path(__file__).parent / "rag_store"
STORE_DIR.mkdir(parents=True, exist_ok=True)

EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def _norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _chunk(text: str, max_len=850, overlap=120):
    text = _norm(text)
    if len(text) <= max_len:
        return [text]
    chunks, i = [], 0
    while i < len(text):
        piece = text[i:i+max_len]
        cut = piece.rfind(". ")
        if cut >= int(max_len * 0.5):
            piece = piece[:cut+1]
        chunks.append(piece.strip())
        i += max_len - overlap
    return [c for c in chunks if c]

def _head(d):
    parts = [d.get("law",""), d.get("chapter",""), d.get("article","")]
    return _norm(" | ".join([p for p in parts if p]))

def _with_head_boost(d, boost=2):
    head = _head(d)
    if head:
        return ((head + " ") * boost) + d["text"]
    return d["text"]

# скачать
def load_docs_from_txt():
    docs = []
    for path in glob.glob(str(KB_DIR / "*.txt")):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        law_name = os.path.basename(path)
        parts = re.split(r"(?=Статья\s+\d+)", text, flags=re.IGNORECASE) or [text]
        for i, part in enumerate(parts):
            part = _norm(part)
            if not part:
                continue
            m = re.match(r"(Статья\s+\d+)", part, flags=re.IGNORECASE)
            article = m.group(1) if m else f"Фрагмент {i}"
            for j, ch in enumerate(_chunk(part)):
                docs.append({
                    "law": law_name,
                    "chapter": "",
                    "article": f"{article} / ч.{j+1}",
                    "text": ch
                })
    return docs

def load_docs_from_xlsx(xlsx_path: Path):
    df = pd.read_excel(xlsx_path)
    cols = {str(c).strip().lower(): c for c in df.columns}
    def get(name): return cols.get(name)

    chapter_col = get("chapter")
    article_col = get("article")
    text_col = get("text")

    docs = []
    for i in range(len(df)):
        t = "" if text_col is None else str(df[text_col].iloc[i] or "")
        t = _norm(t)
        if not t or t.lower() == "nan":
            continue
        chapter = "" if chapter_col is None else _norm(str(df[chapter_col].iloc[i] or ""))
        article = "" if article_col is None else _norm(str(df[article_col].iloc[i] or ""))
        for j, ch in enumerate(_chunk(t)):
            docs.append({
                "law": xlsx_path.name,
                "chapter": chapter,
                "article": f"{article} / ч.{j+1}" if article else f"Фрагмент {j+1}",
                "text": ch
            })
    return docs

def load_docs():
    docs = []
    for x in glob.glob(str(KB_DIR / "*.xlsx")):
        docs.extend(load_docs_from_xlsx(Path(x)))
    docs.extend(load_docs_from_txt())
    # удалить дубли (по тексту)
    seen, uniq = set(), []
    for d in docs:
        k = d["text"]
        if k in seen: 
            continue
        seen.add(k)
        uniq.append(d)
    return uniq

if __name__ == "__main__":
    docs = load_docs()
    if not docs:
        print("Нет данных: положи .xlsx/.txt в python/kb/")
        raise SystemExit(1)

    model = SentenceTransformer(EMB_MODEL_NAME)
    corpus = [_with_head_boost(d, boost=2) for d in docs]

    emb = model.encode(
        corpus,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    faiss.write_index(index, str(STORE_DIR / "faiss.index"))
    with open(STORE_DIR / "meta.pkl", "wb") as f:
        pickle.dump(docs, f)
    with open(STORE_DIR / "model_name.txt", "w", encoding="utf-8") as f:
        f.write(EMB_MODEL_NAME)

    print(f"Индекс построен: {len(docs)} фрагментов, dim={dim}")
