from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI(title="GPT-2 Text Generation API", version="1.0")

# Загружаем GPT-2
model_name = "gpt2"

print("Loading GPT-2...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Модель запроса
class PromptRequest(BaseModel):
    prompt: str
    max_length: int = 80


@app.post("/generate")
def generate_text(req: PromptRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt")

    output = model.generate(
        **inputs,
        max_length=req.max_length,
        do_sample=True,
        top_p=0.95,
        top_k=50
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {
        "prompt": req.prompt,
        "generated_text": text
    }


@app.get("/")
def root():
    return {"message": "GPT-2 text generation API is running!"}
