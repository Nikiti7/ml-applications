import os
from pathlib import Path
import chromadb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# === Настройки ===
MODEL_NAME = os.environ.get("DOCCHAT_LLM", "cointegrated/rut5-base-multitask")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(__file__).resolve().parent
PERSIST_DIR = ROOT / "chroma_db"

print(f"Загрузка модели: {MODEL_NAME} (device={DEVICE})")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

# === Подключение к Chroma ===
client = chromadb.PersistentClient(path=str(PERSIST_DIR))
collection = client.get_collection("docchat")


def retrieve(query: str, k: int = 3):
    """Извлекаем k наиболее релевантных документов из Chroma"""
    results = collection.query(query_texts=[query], n_results=k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    # формируем короткий контекст (обрезаем длинные документы)
    context_parts = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        # возьмём первые 800 символов каждого документа, чтобы не переполнить
        snippet = doc.strip().replace("\n", " ")
        if len(snippet) > 800:
            snippet = snippet[:800] + "..."
        context_parts.append(f"[{meta.get('source', 'doc')}] {snippet}")
    context = "\n\n".join(context_parts)
    return context


def generate_answer(query: str, context: str) -> str:
    """Формируем чёткий инструкционный промпт и генерируем 1-3 предложения"""
    prompt = (
        "Ты — помощник DocChat. Используй только информацию из контекста ниже.\n"
        "Дай короткий (1–3 предложения), конкретный и понятный ответ на поставленный вопрос.\n"
        "Если ответа в контексте нет — честно скажи, что не знаешь.\n\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос: {query}\n\n"
        "Короткий ответ (1–3 предложения):"
    )

    # Токенизация с обрезкой
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(
        DEVICE
    )

    # Параметры генерации: даём модели свободу, но короткий ответ
    out = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id,
    )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # Убираем возможную повтораную часть промпта (иногда модель копирует промпт)
    # Если модель возвращает весь prompt+answer, берём хвост после "Короткий ответ"
    marker = "Короткий ответ (1–3 предложения):"
    if marker in text:
        answer = text.split(marker, 1)[-1].strip()
    else:
        # иначе считаем, что модель вернула только ответ
        answer = text.strip()

    # Удаляем лишние пробелы/новые строки
    answer = " ".join(answer.split())
    # Ограничим до первых 3 предложений
    sentences = [
        s.strip()
        for s in answer.replace("?", "?.").replace("!", "!.").split(".")
        if s.strip()
    ]
    if len(sentences) > 3:
        answer = ". ".join(sentences[:3]) + "."
    else:
        # восстановим точки если нужно
        if (
            not answer.endswith(".")
            and not answer.endswith("?")
            and not answer.endswith("!")
        ):
            answer = answer + "."
    return answer


def cli_loop(k=3):
    print("DocChat is ready. Type your question (or 'exit'):\n")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n Bye!")
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        context = retrieve(q, k)
        answer = generate_answer(q, context)

        print("\n--- Answer ---\n")
        print(answer)
        print("\n")  # пустая строка после ответа


if __name__ == "__main__":
    cli_loop()
