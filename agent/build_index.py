#!/usr/bin/env python3
# build_index.py
# Построение TF-IDF индекса для папки agent/data
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import json

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_FILE = ROOT / "tfidf_index.pkl"
HARD_RULES = DATA_DIR / "hard_rules.json"

print("DATA_DIR =", DATA_DIR)


def load_txt_files():
    docs = []
    filenames = []
    for p in sorted(DATA_DIR.glob("*.txt")):
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            text = p.read_text(encoding="cp1251", errors="ignore")
        docs.append(text)
        filenames.append(p.name)
    return docs, filenames


def load_faq_pairs(fpath):
    # faq files with '||' separator on each entry: keys separated by ';' before ||
    items = []
    if not fpath.exists():
        return items
    raw = fpath.read_text(encoding="utf-8", errors="ignore")
    for block in raw.split("\n\n"):
        if "||" in block:
            left, right = block.split("||", 1)
            keys = [k.strip().lower() for k in left.split(";") if k.strip()]
            answer = right.strip()
            if keys and answer:
                items.append((keys, answer))
    return items


def main():
    docs, filenames = load_txt_files()
    if len(docs) == 0:
        raise SystemExit(
            "В папке agent/data нет .txt файлов — создайте их и запустите снова"
        )

    vectorizer = TfidfVectorizer(lowercase=True, max_features=50000, ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(docs)

    # save index
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "tfidf": tfidf,
            "docs": docs,
            "filenames": filenames,
        },
        OUT_FILE,
    )

    # load and save faq pairs (if exist)
    faq_pairs = load_faq_pairs(DATA_DIR / "faq_modules.txt") + load_faq_pairs(
        DATA_DIR / "faq_theory.txt"
    )
    joblib.dump(faq_pairs, ROOT / "faq_pairs.pkl")

    # copy hard_rules (just validate)
    if HARD_RULES.exists():
        try:
            hr = json.loads(HARD_RULES.read_text(encoding="utf-8"))
            print(f"Loaded hard_rules: {len(hr)} keys")
        except Exception:
            print("hard_rules.json exists but is invalid JSON")

    print("TF-IDF индекс сохранён в:", OUT_FILE)
    print("FAQ пар сохранено в: faq_pairs.pkl")


if __name__ == "__main__":
    main()
