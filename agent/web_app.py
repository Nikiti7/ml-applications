from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import chromadb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# === Настройки ===
MODEL_NAME = "cointegrated/rut5-base-multitask"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(__file__).resolve().parent
PERSIST_DIR = ROOT / "chroma_db"

# === Загрузка модели и БД ===
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
client = chromadb.PersistentClient(path=str(PERSIST_DIR))
collection = client.get_collection("docchat")

# === FastAPI setup ===
app = FastAPI(title="DocChat – RAG Agent WebUI")
templates = Jinja2Templates(directory=str(ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


def retrieve(query: str, k: int = 3):
    results = collection.query(query_texts=[query], n_results=k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    context_parts, sources = [], []
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        snippet = doc.strip()
        if len(snippet) > 1500:
            snippet = snippet[:1500] + "..."
        context_parts.append(f"[{meta.get('source', 'unknown')}]\n{snippet}\n")
        sources.append(meta.get("source", "unknown"))
    return "\n---\n".join(context_parts), sources


def generate_answer(query: str, context: str):
    prompt = f"""
Ты — DocChat, ассистент по проектной документации.
Отвечай подробно и структурированно по информации из контекста ниже.
Если данных нет — честно скажи, что не знаешь.

Контекст:
{context}

Вопрос: {query}
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(
        DEVICE
    )
    outputs = model.generate(
        **inputs, max_new_tokens=256, temperature=0.4, top_p=0.9, do_sample=False
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


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
            "answer": answer,
            "question": question,
            "sources": sources,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
