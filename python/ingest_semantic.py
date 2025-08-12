import os, glob, pickle, re
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

KB_DIR = Path(__file__).parent / "kb"
STORE_DIR = Path(__file__).parent / "rag_store"
STORE_DIR.mkdir(parents=True, exist_ok=True)

# Лёгкая многоязычная модель (рус/каз/англ)
EMB_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def _norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def load_docs_from_txt():
    docs = []
    for path in glob.glob(str(KB_DIR / "*.txt")):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        law_name = os.path.basename(path)
        parts = re.split(r"(?=Статья\s+\d+)", text, flags=re.IGNORECASE)
        for i, chunk in enumerate(parts):
            chunk = _norm(chunk)
            if not chunk:
                continue
            m = re.match(r"(Статья\s+\d+)", chunk, flags=re.IGNORECASE)
            article = m.group(1) if m else f"Фрагмент {i}"
            docs.append({
                "law": law_name,
                "chapter": "",
                "article": article,
                "text": chunk
            })
    return docs

def load_docs_from_xlsx(xlsx_path: Path):
    df = pd.read_excel(xlsx_path)
    cols = {str(c).strip().lower(): c for c in df.columns}

    def get_col(name):
        col = cols.get(name)
        return df[col] if col in df else None

    chapter = get_col("chapter")
    article = get_col("article")
    text = get_col("text")

    docs = []
    for i in range(len(df)):
        t = "" if text is None else str(text.iloc[i])
        t = _norm(t)
        if not t or t.lower() == "nan":
            continue
        docs.append({
            "law": xlsx_path.name,
            "chapter": "" if chapter is None else _norm(str(chapter.iloc[i] or "")),
            "article": "" if article is None else _norm(str(article.iloc[i] or "")),
            "text": t
        })
    return docs

def load_docs():
    docs = []
    for xlsx in glob.glob(str(KB_DIR / "*.xlsx")):
        docs.extend(load_docs_from_xlsx(Path(xlsx)))
    docs.extend(load_docs_from_txt())
    return docs

if __name__ == "__main__":
    docs = load_docs()
    if not docs:
        print("Нет данных: добавь .xlsx/.txt в python/kb/")
        raise SystemExit(1)

    corpus = [d["text"] for d in docs]

    model = SentenceTransformer(EMB_MODEL_NAME)
    embeddings = model.encode(
        corpus,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(STORE_DIR / "faiss.index"))
    with open(STORE_DIR / "meta.pkl", "wb") as f:
        pickle.dump(docs, f)
    with open(STORE_DIR / "model_name.txt", "w", encoding="utf-8") as f:
        f.write(EMB_MODEL_NAME)

    print(f"Индекс построен: {len(docs)} фрагментов. dim={dim}")
