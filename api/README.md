# API — Sentiment Analysis (FastAPI)

## Задача

Создать REST API для анализа тональности текста.
API поддерживает:

- `/analyze` → анализ **одного текста**
- `/batch-analyze` → анализ **списка текстов**

---

## Используемые технологии

- **FastAPI** — современный фреймворк для REST API.
- **Uvicorn** — ASGI-сервер для запуска.
- **Hugging Face Transformers** — готовая модель анализа тональности.

---

## Запуск проекта

### 1. Установите зависимости

```bash
cd api
pip install -r ../requirements.txt
uvicorn main:app --reload
```

## Примеры использования

1. Анализ одного текста

```bash
curl -X POST http://127.0.0.1:8000/analyze \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"I love machine learning!\"}"
```

Ответ:

```json
{
  "text": "I love machine learning!",
  "label": "POSITIVE",
  "score": 0.999
}
```

2. Анализ списка текстов

```bash
curl -X POST http://127.0.0.1:8000/batch-analyze \
    -H "Content-Type: application/json" \
    -d "{\"texts\": [\"I love machine learning!\", \"Сегодня ужасная погода.\", \"The movie was fantastic.\", \"The service was terrible.\"]}"

```

Ответ:

```json
[
  {
    "text": "I love machine learning!",
    "label": "POSITIVE",
    "score": 0.999
  },
  {
    "text": "Сегодня ужасная погода.",
    "label": "NEGATIVE",
    "score": 0.998
  },
  {
    "text": "The movie was fantastic.",
    "label": "POSITIVE",
    "score": 0.995
  },
  {
    "text": "The service was terrible.",
    "label": "NEGATIVE",
    "score": 0.997
  }
]

```

3. Swagger UI

Откройте в браузере:
http://127.0.0.1:8000/docs

## Метрики качества

Accuracy, F1-score — для оценки модели на тестовом наборе.

Для API можно добавить логирование: количество запросов, распределение позитив/негатив
