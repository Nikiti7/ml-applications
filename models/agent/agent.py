#!/usr/bin/env python3
# agent.py
# CLI RAG agent: hard rules -> FAQ -> TF-IDF -> ruT5 generation

import joblib
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
INDEX_FILE = ROOT / "tfidf_index.pkl"
FAQ_FILE = ROOT / "faq_pairs.pkl"
HARD_RULES_FILE = DATA_DIR / "hard_rules.json"

MODEL_NAME = "cointegrated/rut5-base-multitask"  # ruT5, хорош для русскоязычных ответов

# ---------------- load index ----------------
if not INDEX_FILE.exists():
    raise SystemExit("Индекс не найден. Запустите python agent/build_index.py")

idx = joblib.load(INDEX_FILE)
vectorizer = idx["vectorizer"]
tfidf = idx["tfidf"]
docs = idx["docs"]
filenames = idx["filenames"]

faq_pairs = []
if FAQ_FILE.exists():
    faq_pairs = joblib.load(FAQ_FILE)

hard_rules = {}
if HARD_RULES_FILE.exists():
    try:
        hard_rules = json.loads(HARD_RULES_FILE.read_text(encoding="utf-8"))
    except Exception:
        hard_rules = {}

# ---------------- load model ----------------
print("Загрузка модели:", MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to(device)


# ---------------- helpers ----------------
def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\wа-яёё\s]", " ", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def try_hard_rules(question: str):
    q = normalize_text(question)
    for key, target in hard_rules.items():
        if key in q:
            # target like "MODULE:text" or "TOPIC:latency"
            if target.startswith("MODULE:"):
                module_tag = target.split(":", 1)[1]
                # find in knowledge_modules.txt block
                p = DATA_DIR / "knowledge_modules.txt"
                if p.exists():
                    txt = p.read_text(encoding="utf-8", errors="ignore")
                    pattern = r"\[{}]\n(.*?)\n(?=\[|$)".format(
                        re.escape(
                            module_tag.split(":")[-1]
                            if ":" in module_tag
                            else module_tag
                        )
                    )
                    # simpler: search [MODULE:xxx] blocks
                    m = re.search(
                        rf"\[MODULE:{re.escape(module_tag.split(':')[-1])}\]\n(.+?)(?=\n\[MODULE:|\Z)",
                        txt,
                        flags=re.S | re.I,
                    )
                    if m:
                        return m.group(1).strip()
            elif target.startswith("TOPIC:"):
                topic = target.split(":", 1)[1]
                # try topic file
                topic_file = DATA_DIR / f"theory_{topic}.txt"
                if topic_file.exists():
                    return topic_file.read_text(encoding="utf-8", errors="ignore")
    return None


def try_faq(question: str, threshold=0.45):
    if not faq_pairs:
        return None
    q_vec = vectorizer.transform([question])
    # build small FAQ vector matrix
    faq_questions = []
    for keys, _ in faq_pairs:
        # compound key text for vectorization
        faq_questions.append(" ".join(keys))
    if not faq_questions:
        return None
    faq_vecs = vectorizer.transform(faq_questions)
    sims = cosine_similarity(q_vec, faq_vecs).flatten()
    best = sims.argmax()
    if sims[best] >= threshold:
        return faq_pairs[best][1]
    return None


def retrieve_tf_idf(question: str, top_k=4):
    qv = vectorizer.transform([question])
    sims = cosine_similarity(qv, tfidf).flatten()
    idxs = sims.argsort()[::-1][:top_k]
    results = []
    for i in idxs:
        results.append(
            {"filename": filenames[i], "text": docs[i], "score": float(sims[i])}
        )
    return results


def generate_answer(question: str, context: str, max_new_tokens: int = 180):
    prompt = (
        "Ты — эксперт по проекту и по теории ML/LLM. "
        "Используй ТОЛЬКО информацию из Контекста и отвечай на русском языке.\n\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос: {question}\n\n"
        "Формат ответа:\n1) Краткое резюме (1-2 предложения).\n2) Развернутый ответ (2-6 предложений).\n3) Источники (файлы).\n\nОтвет:\n"
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,
            early_stopping=True,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # strip prompt prefix if present
    if text.startswith(prompt):
        text = text[len(prompt) :].strip()
    return text


# ---------------- main interface ----------------
def answer_question(question: str):
    # 1. hard rules
    hr = try_hard_rules(question)
    if hr:
        return f"{hr} + \n\n(Источник: hard_rules / knowledge_modules.txt)"

    # 2. FAQ exact/fuzzy
    faq = try_faq(question, threshold=0.45)
    if faq:
        return faq + "\n\n(Источник: FAQ)"

    # 3. TF-IDF retrieval
    retrieved = retrieve_tf_idf(question, top_k=5)
    if not retrieved:
        return "Я не нашёл информацию в документации проекта."

    # confidence check: if top score too low -> say not found
    if retrieved[0]["score"] < 0.03:
        return "Я не нашёл достаточно релевантной информации в документах проекта. Уточните вопрос."

    # build context from top-3
    ctx_parts = []
    srcs = []
    for item in retrieved[:3]:
        ctx = item["text"]
        # shorten large docs
        if len(ctx) > 2000:
            ctx = ctx[:2000] + "..."
        ctx_parts.append(f"[{item['filename']}]\n{ctx}")
        srcs.append(item["filename"])
    context = "\n\n---\n\n".join(ctx_parts)

    # 4. generate
    ans = generate_answer(question, context)
    # append sources if not present
    ans = ans.strip()
    ans += "\n\nИсточники:\n" + "\n".join(f"- {s}" for s in srcs)
    return ans


# CLI
if __name__ == "__main__":
    print("\nDocAgent (TF-IDF + ruT5) готов. Введите вопрос (exit для выхода).\n")
    while True:
        q = input("> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        print("\n--- Answer ---\n")
        print(answer_question(q))
        print("\n")
