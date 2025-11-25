from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="ML Model API (Mocked for CI)")


class TextRequest(BaseModel):
    text: str


class BatchRequest(BaseModel):
    texts: List[str]


# --- Simple rule-based classifier for CI tests ---
NEG_WORDS = [
    "грустно", "плохо", "ужас", "ужасный", "отвратительно",
    "ненавижу", "страшно", "печально", "депрессия", "плохой"
]

POS_WORDS = [
    "отличный", "классный", "прекрасный", "люблю", "супер",
    "радостный", "замечательный", "хороший", "великолепный"
]


def simple_classifier(text: str):
    t = text.lower()

    if any(w in t for w in POS_WORDS):
        return "Positive", 0.95

    if any(w in t for w in NEG_WORDS):
        return "NEGATIVE", 0.95

    return "NEUTRAL", 0.5


@app.post("/analyze")
def analyze(req: TextRequest):
    label, score = simple_classifier(req.text)
    return {"text": req.text, "label": label, "score": score}


@app.post("/batch-analyze")
def batch_analyze(req: BatchRequest):
    results = []
    for t in req.texts:
        label, score = simple_classifier(t)
        results.append({"text": t, "label": label, "score": score})
    return results
