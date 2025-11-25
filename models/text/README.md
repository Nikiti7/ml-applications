# Text ML — Sentiment Analysis (Hugging Face)

## Задача

Анализ тональности текста (определение **положительной** или **отрицательной** окраски).
Примеры применения: отзывы пользователей, комментарии в соцсетях, модерация контента.

---

## Используемые технологии

- **Hugging Face Transformers** — готовая модель для анализа текста.
- **PyTorch** — фреймворк, на котором работает модель.

---

## Запуск проекта

### 1. Установите зависимости

```bash
pip install -r ../requirements.txt
python main.py
```

## Пример вывода

```yaml
Text: I love machine learning!
Label: POSITIVE, Score: 0.999

Text: Сегодня ужасная погода и всё идёт плохо.
Label: NEGATIVE, Score: 0.997

```

## Метрики качества

- **Accuracy** — доля верных предсказаний.

- **F1-score** — баланс точности и полноты (важно при несбалансированных классах).

## Полезные ссылки

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

- [Sentiment Analysis Pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.SentimentAnalysisPipeline)

- [Русская модель RuBERT Sentiment](https://huggingface.co/blanchefort/rubert-base-cased-sentiment)
