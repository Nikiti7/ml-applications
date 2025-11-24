# Compare - Сравнение языковых моделей (LLM Benchmark Suite)

Директория compare/ содержит модуль для оценки и сравнения языковых моделей по трём основным метрикам:

1. Perplexity (PPL) - насколько хорошо модель предсказывает текст.

2. Factual QA Accuracy - точность ответов на фактологические вопросы.

3. Continuation Quality - качество продолжения заданного текста.

Модуль был создан для того, чтобы на защите проекта можно было продемонстрировать объективные различия между моделями (например GPT-2, ruGPT, Saiga, T5 и др.) на стандартизированных заданиях.

## Структура

```powershell
compare/
│ compare_llms.py        # основной скрипт сравнения LLM
│ requirements-compare.txt
│ README.md              # этот файл
│
└── data/
    ├── continuations.jsonl   # тексты для продолжения
    ├── factual_qa.jsonl      # фактологические Q&A
    └── perp_texts.txt        # тексты для измерения perplexity
```

## Функциональность

1. Perplexity Benchmark

Скрипт вычисляет perplexity (PPL) — стандартную метрику качества языковых моделей.

Чем ниже PPL, тем лучше модель предсказывает текст.

Используются данные из:

```bash
data/perp_texts.txt
```

2. Factual QA Benchmark

Оценка точности ответов на фактологические вопросы.

Формат:

```bash
{"question": "...", "answer": "..."}
```

Если модель даёт ответ, содержащий правильный факт → вопрос считается решённым.

Файл:

```bash
data/factual_qa.jsonl
```

3. Continuation Quality Test

Проверка способности модели продолжать текст.

Данные:

```bash
data/continuations.jsonl
```

Модель генерирует продолжение, а скрипт проверяет:

- длину,

- связность,

- отсутствие повторов,

- разнообразие лексики.

Используется простая метрика, не требующая сложных оценщиков (так как оценка нужна для учебного проекта).

## Настройка и запуск

1. Установить зависимости:

```bash
pip install -r compare/requirements-compare.txt
```

2. Запустить бенчмарк:

```bash
python compare/compare_llms.py
```

## Пример результата

```yaml
=== Model: gpt2 ===
Perplexity: 54.21
Factual QA accuracy: 0.37
Continuation quality: 0.52

=== Model: cointegrated/rut5-base-multitask ===
Perplexity: 11.02
Factual QA accuracy: 0.79
Continuation quality: 0.83
```

Выводы печатаются в консоль и сохраняются в файл results.txt.
