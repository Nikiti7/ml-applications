from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import pipeline

# Инициализация FastAPI
app = FastAPI(title="Sentiment Analysis API", version="1.1")

# Загружаем готовую модель Hugging Face
classifier = pipeline("sentiment-analysis")

# Модель запроса для одного текста
class TextRequest(BaseModel):
    text: str

# Модель запроса для списка текстов
class BatchRequest(BaseModel):
    texts: List[str]

# Эндпоинт для одного текста
@app.post("/analyze")
def analyze(req: TextRequest):
    result = classifier(req.text)[0]
    return {
        "text": req.text,
        "label": result["label"],
        "score": round(result["score"], 3)
    }

# Эндпоинт для списка текстов
@app.post("/batch-analyze")
def batch_analyze(req: BatchRequest):
    results = classifier(req.texts)
    response = []
    for text, res in zip(req.texts, results):
        response.append({
            "text": text,
            "label": res["label"],
            "score": round(res["score"], 3)
        })
    return response

