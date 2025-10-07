# ML Applications

## Описание

Проект демонстрирует применение готовых библиотек **Machine Learning / AI** для решения прикладных задач.

Реализованы решения для 4 типов данных + локальная LLM + API:

1. **Text** — анализ тональности текста (Hugging Face).
2. **Audio** — классификация аудио (TensorFlow YAMNet).
3. **Image** — классификация изображений (PyTorch).
4. **Video** — детекция объектов/жестов (PyTorch/Hugging Face).
5. **LLM** — запуск локальной языковой модели GPT-2.
6. **API** — FastAPI-сервис для анализа тональности текста.

---

## Используемые технологии

- **Hugging Face Transformers** (Text, LLM, API)
- **TensorFlow Hub** (Audio)
- **PyTorch** (Image, Video)
- **FastAPI + Uvicorn** (API)

---

## Структура репозитория

```
ml-applications/
│
├── text/ # Анализ тональности текста
│ ├── main.py
│ └── README.md
│
├── audio/ # Классификация аудио (YAMNet)
│ ├── main.py
│ ├── speech.wav
│ ├── music.wav
│ ├── noise.wav
│ └── README.md
│
├── image/ # Классификация изображений (PyTorch)
│ └── ...
│
├── video/ # Обработка видео (PyTorch/HF)
│ └── ...
│
├── llm/ # Локальная LLM (GPT-2)
│ ├── main.py
│ └── README.md
│
├── api/ # FastAPI сервис для анализа текста
│ ├── main.py
│ └── README.md
│
├── docs/ # Отчёт и документация
│ └── ...
│
├── requirements.txt
└── README.md # Этот файл
```

---

## Установка и запуск

### 1. Клонировать репозиторий

```bash
git clone https://github.com/Nikiti7/ml-applications.git
cd ml-applications
```

### 2. Создать виртуальное окружение

```bash
python -m venv env
source env/bin/activate   # Linux/Mac
.\env\Scripts\activate    # Windows
```

### 3. Установить зависимости

```bash
pip install -r requirements.txt
```

## Запуск отдельных модулей

### Text (Sentiment Analysis)

```bash
cd text
python main.py
```

### Audio (YAMNet)

```bash
cd audio
python main.py
```

### LLM (GPT-2)

```bash
cd llm
python main.py
```

### API (FastAPI)

```bash
cd api
uvicorn main:app --reload
```

- Swagger UI: <http://127.0.0.1:8000/docs#/>

## Метрики

**Text:** Accuracy, F1-score.

**Audio:** Accuracy, F1-score, WER (для STT).

**Image:** Accuracy, Top-5 Accuracy.

**Video:** mAP (mean Average Precision).

**LLM:** Perplexity, BLEU/ROUGE, субъективная оценка качества текста.

**API:** работоспособность, скорость ответа

## Полезные ссылки

- [Hugging Face](https://huggingface.co/)

- [TensorFlow Hub](https://www.tensorflow.org/hub?hl=ru)

- [PyTorch Hub](https://pytorch.org/hub/)

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
