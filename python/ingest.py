import os, glob, pickle, re
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

KB_DIR = Path(__file__).parent / "kb"
STORE_DIR = Path(__file__).parent / "rag_store"
STORE_DIR.mkdir(parents=True, exist_ok=True)

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

    section = get_col("section")
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

def _with_head_boost(d, boost_times=3):
    head = " ".join([x for x in [d.get("law",""), d.get("chapter",""), d.get("article","")] if x])
    head = _norm(head)
    if head:
        return ((head + " ") * boost_times) + d["text"]
    return d["text"]

def build_index(docs):
    corpus = [_with_head_boost(d, boost_times=3) for d in docs]

    vec_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=1,
        lowercase=True,
        token_pattern=r"(?u)\b[\w-]{2,}\b",
        norm="l2",
    )
    Xw = vec_word.fit_transform(corpus)

    vec_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_df=0.95,
        min_df=1,
        lowercase=True,
        norm="l2",
    )
    Xc = vec_char.fit_transform(corpus)

    X = hstack([Xw, Xc], format="csr")
    return vec_word, vec_char, X

if __name__ == "__main__":
    docs = load_docs()
    if not docs:
        print("Нет данных: добавь .xlsx в python/kb/ (или .txt).")
        raise SystemExit(1)

    vec_word, vec_char, X = build_index(docs)

    with open(STORE_DIR / "index.pkl", "wb") as f:
        pickle.dump({"vec_word": vec_word, "vec_char": vec_char, "X": X}, f)

    with open(STORE_DIR / "meta.pkl", "wb") as f:
        pickle.dump(docs, f)

    print(f"Индекс построен: {len(docs)} фрагментов.")
