from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz

# === Пути ===
ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "index"
TFIDF_PATH = INDEX_DIR / "tfidf.pkl"
MATRIX_PATH = INDEX_DIR / "matrix.npz"
METAS_PATH = INDEX_DIR / "metas.pkl"

# === Настройки ===
MODEL_NAME = "cointegrated/rut5-base-multitask"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

print("Loading TF-IDF index...")
with open(TFIDF_PATH, "rb") as f:
    vectorizer = pickle.load(f)

matrix = load_npz(MATRIX_PATH)  # sparse matrix
with open(METAS_PATH, "rb") as f:
    metas = pickle.load(f)

# FastAPI
app = FastAPI(title="DocChat – Offline RAG (TFIDF)")
templates = Jinja2Templates(directory=str(ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


# === Retrieval через TF-IDF ===
def retrieve(query: str, k: int = 3):
    q_vec = vectorizer.transform([query])
    sim = cosine_similarity(q_vec, matrix)[0]

    top_i = sim.argsort()[::-1][:k]

    context_parts = []
    sources = []

    for i in top_i:
        doc = metas[i]["text"]
        src = metas[i]["source"]

        short = doc[:1500] + "..." if len(doc) > 1500 else doc
        context_parts.append(f"[{src}]\n{short}\n")
        sources.append(src)

    return "\n---\n".join(context_parts), sources


# === Генерация ответа ===
def generate_answer(question: str, context: str):
    prompt = f"""
Ты — DocChat, ассистент по проекту.
Используй только информацию из контекста. 
Если данных недостаточно — скажи об этом.

Контекст:
{context}

Вопрос: {question}
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(
        DEVICE
    )

    outputs = model.generate(
        **inputs, max_new_tokens=256, temperature=0.4, top_p=0.9, do_sample=False
    )

    ans = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return ans.strip()


# === Маршруты ===
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "answer": None}
    )


@app.post("/", response_class=HTMLResponse)
async def chat(request: Request, question: str = Form(...)):
    context, sources = retrieve(question)
    answer = generate_answer(question, context)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "question": question,
            "answer": answer,
            "sources": sources,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
