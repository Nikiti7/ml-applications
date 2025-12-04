#!/usr/bin/env python3
# web_app.py — UI для RAG агента (совместим с tfidf_index.pkl)

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import joblib
import json
import re
import torch
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ======= Paths =======
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
INDEX_FILE = ROOT / "tfidf_index.pkl"
FAQ_FILE = ROOT / "faq_pairs.pkl"
HARD_RULES_FILE = DATA_DIR / "hard_rules.json"

MODEL_NAME = "cointegrated/rut5-base-multitask"

# ========= Load index =========
if not INDEX_FILE.exists():
    raise SystemExit("Индекс не найден. Сначала выполните python build_index.py")

idx = joblib.load(INDEX_FILE)
vectorizer = idx["vectorizer"]
tfidf = idx["tfidf"]
docs = idx["docs"]
filenames = idx["filenames"]

faq_pairs = joblib.load(FAQ_FILE) if FAQ_FILE.exists() else []
hard_rules = json.loads(HARD_RULES_FILE.read_text(encoding="utf-8")) if HARD_RULES_FILE.exists() else {}

# ========= Load model =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading model:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)


# ========= Helpers =========
def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\wа-яё\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def try_hard_rules(question: str):
    q = normalize_text(question)
    for key, target in hard_rules.items():
        if key in q:
            return target
    return None


def try_faq(question: str, threshold=0.45):
    if not faq_pairs:
        return None
    q_vec = vectorizer.transform([question])
    text_keys = [" ".join(keys) for keys, _ in faq_pairs]
    faq_vecs = vectorizer.transform(text_keys)
    sims = cosine_similarity(q_vec, faq_vecs).flatten()
    best = sims.argmax()
    if sims[best] >= threshold:
        return faq_pairs[best][1]
    return None


def retrieve_tf_idf(question: str, top_k=3):
    qv = vectorizer.transform([question])
    sims = cosine_similarity(qv, tfidf).flatten()
    idxs = sims.argsort()[::-1][:top_k]
    res = []
    for i in idxs:
        res.append({"filename": filenames[i], "text": docs[i], "score": float(sims[i])})
    return res


def generate_answer(question, context):
    prompt = (
        "Ты — эксперт по проекту. Используй ТОЛЬКО информацию из Контекста.\n\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос: {question}\n\n"
        "Ответ:"
    )
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    out = model.generate(**inp, max_new_tokens=200, num_beams=4, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def answer_question(question: str):
    # 1 — Hard Rules
    hr = try_hard_rules(question)
    if hr:
        return hr

    # 2 — FAQ
    faq = try_faq(question)
    if faq:
        return faq

    # 3 — TF-IDF
    ret = retrieve_tf_idf(question)
    if not ret or ret[0]["score"] < 0.03:
        return "Недостаточно информации в документации. Уточните вопрос."

    ctx = "\n\n---\n\n".join(f"[{x['filename']}]\n{x['text'][:2000]}" for x in ret)
    ans = generate_answer(question, ctx)
    ans += "\n\nИсточники:\n" + "\n".join(f"- {x['filename']}" for x in ret)
    return ans


# =========== FASTAPI APP ===========
app = FastAPI(title="DocAgent RAG UI")

templates = Jinja2Templates(directory=str(ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "answer": None})


@app.post("/", response_class=HTMLResponse)
def chat(request: Request, question: str = Form(...)):
    ans = answer_question(question)
    return templates.TemplateResponse("index.html", {"request": request, "answer": ans, "question": question})


# ======= RUN SERVER =======
if __name__ == "__main__":
    import uvicorn
    print("\nStarting UI at http://127.0.0.1:8000\n")
    uvicorn.run("web_app:app", host="127.0.0.1", port=8000, reload=True)
