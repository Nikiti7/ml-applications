## Аудио — Speech-to-Text (STT)

### Задача

задача преобразования аудиозаписи в текст.

### Модель

Hugging Face `distilbert-base-uncased-finetuned-sst-2-english`.

### Запуск

```bash
pip install transformers torch
python main.py
```

## Пример вывода

```yaml
Text: I love machine learning!
Label: POSITIVE, Score: 0.999
```

## Метрики

- Accuracy

- F1-score
