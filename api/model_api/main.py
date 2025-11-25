import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="ML Model API (Mocked for CI)")


class TextRequest(BaseModel):
    text: str


class BatchRequest(BaseModel):
    texts: List[str]


def simple_classifier(text: str) -> str:
    t = text.lower()
    if any(x in t for x in ["good", "excellent", "love", "happy", "great"]):
        return "POSITIVE"
    if any(x in t for x in ["bad", "terrible", "hate", "awful"]):
        return "NEGATIVE"
    return "NEUTRAL"


@app.post("/analyze")
def analyze(req: TextRequest):
    label = simple_classifier(req.text)
    return {"text": req.text, "label": label}


@app.post("/batch-analyze")
def batch_analyze(req: BatchRequest):
    return [{"text": t, "label": simple_classifier(t)} for t in req.texts]
