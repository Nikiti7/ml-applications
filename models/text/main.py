from transformers import pipeline

# Загружаем пайплайн анализа тональности
classifier = pipeline("sentiment-analysis")

# Примеры для анализа
texts = [
    "I love machine learning!",
    "Сегодня ужасная погода и всё идёт плохо.",
    "Этот фильм был просто великолепен!",
    "Мне не понравилось обслуживание в ресторане.",
]

# Прогоняем через модель
for t in texts:
    result = classifier(t)[0]
    print(f"Text: {t}")
    print(f"Label: {result['label']}, Score: {result['score']:.3f}\n")
