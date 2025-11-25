# API для локальной LLM

Этот модуль реализует REST API для работы с локальной языковой моделью (LLM).

## Запуск

```bash
uvicorn llm_api.main:app --reload --port 8003
```

Эндпоинты
Метод URL Описание
POST /generate Генерирует текст по prompt

## Пример запроса

```bash
curl -X POST "http://127.0.0.1:8003/generate" \
     -H "Content-Type: application/json" \
     -d "{\"prompt\": \"In the future, artificial intelligence will\"}"
```

## Пример ответа

```bash
{
  "prompt": "In the future, artificial intelligence will",
  "generated_text": "In the future, artificial intelligence will have to perform much more sophisticated cognitive functions..."
}
```

1. Swagger UI

Откройте в браузере:
<http://127.0.0.1:8003/docs>
