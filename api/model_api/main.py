import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"  # чтобы не грузил видео/image_utils


MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

app = FastAPI(title="ML Model API")

# Load tokenizer and model (PyTorch backend only)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()


class TextRequest(BaseModel):
    text: str


class BatchRequest(BaseModel):
    texts: List[str]


def classify(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1)[0]

    label = "POSITIVE" if scores[1] > scores[0] else "NEGATIVE"
    score = float(scores.max().item())

    return {"text": text, "label": label, "score": round(score, 3)}


@app.post("/analyze")
def analyze(req: TextRequest):
    return classify(req.text)


@app.post("/batch-analyze")
def batch_analyze(req: BatchRequest):
    return [classify(t) for t in req.texts]
