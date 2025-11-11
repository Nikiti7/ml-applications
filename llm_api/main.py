from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# ==============================
#  Настройки и инициализация
# ==============================

MODEL_NAME = "IlyaGusev/saiga_mistral_7b"

app = FastAPI(
    title="LLM Text Generation API",
    version="1.0",
    description="API для локальной LLM (генерация текста на основе prompt)",
)

# Загружаем токенизатор и модель
print(f"Загрузка модели: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)


# ==============================
#  Модели запросов и ответов
# ==============================


class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 100


class GenerationResponse(BaseModel):
    prompt: str
    generated_text: str


# ==============================
#  Эндпоинты
# ==============================


@app.get("/")
def root():
    return {"message": "LLM API is running. Use /docs for Swagger UI."}


@app.post("/generate", response_model=GenerationResponse)
def generate_text(req: PromptRequest):
    """Генерация текста на основе промпта"""
    outputs = generator(
        req.prompt,
        max_new_tokens=req.max_tokens,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
    )
    return {"prompt": req.prompt, "generated_text": outputs[0]["generated_text"]}
