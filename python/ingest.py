import os, glob, pickle, re
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

KB_DIR = Path(__file__).parent / "kb"
STORE_DIR = Path(__file__).parent / "rag_store"
STORE_DIR.mkdir(parents=True, exist_ok=True)

def load_docs_from_txt():
    docs = []
    for path in glob.glob(str(KB_DIR / "*.txt")):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        law_name = os.path.basename(path)
        parts = re.split(r"(?=Статья\s+\d+)", text, flags=re.IGNORECASE)
        for i, chunk in enumerate(parts):
            chunk = (chunk or "").strip()
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
    cols = {c.lower(): c for c in df.columns}
    get = lambda name: df[cols[name]] if name in cols else None

    section = get("section")
    chapter = get("chapter")
    article = get("article")
    text = get("text")

    docs = []
    for i in range(len(df)):
        t = str(text.iloc[i]).strip() if text is not None else ""
        if not t or t.lower() == "nan":
            continue
        docs.append({
            "law": xlsx_path.name,
            "chapter": ("" if chapter is None else str(chapter.iloc[i] or "").strip()),
            "article": ("" if article is None else str(article.iloc[i] or "").strip()),
            "text": t
        })
    return docs

def load_docs():
    docs = []
    for xlsx in glob.glob(str(KB_DIR / "*.xlsx")):
        docs.extend(load_docs_from_xlsx(Path(xlsx)))
    docs.extend(load_docs_from_txt())
    return docs

def build_index(docs):
    corpus = [d["text"] for d in docs]
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1,2),
        max_df=0.9,
        min_df=1,
        norm="l2",
    )
    X = vec.fit_transform(corpus)
    return vec, X

if __name__ == "__main__":
    docs = load_docs()
    if not docs:
        print("Нет данных: добавь .xlsx в python/kb/ (или .txt).")
        raise SystemExit(1)
    vec, X = build_index(docs)
    with open(STORE_DIR / "index.pkl", "wb") as f:
        pickle.dump({"vec": vec, "X": X}, f)
    with open(STORE_DIR / "meta.pkl", "wb") as f:
        pickle.dump(docs, f)
    print(f"Индекс построен: {len(docs)} фрагментов.")
