# LLM — Локальная языковая модель (GPT-2)

## Задача

Показать работу **Large Language Model (LLM)** на локальной машине.
Модель должна уметь **генерировать связный текст** по заданному началу (prompt).

Для демонстрации используется **GPT-2** — одна из первых публичных больших языковых моделей от OpenAI.
Она компактна (~500 МБ) и легко запускается даже без GPU.

---

## Используемые технологии

- **Hugging Face Transformers** — библиотека для работы с LLM.
- **PyTorch** — фреймворк для запуска модели.

---

## Запуск проекта

### 1. Установите зависимости

```bash
cd llm
pip install -r ../requirements.txt
python main.py
```

## Пример вывода

```yaml
---
Generated text ---

In the future, artificial intelligence will be a force for good for our societies too. It will make us smarter. And it will make us a better person.

But it doesn't make us better. You see, I'm not an expert in AI or neuroscience. I'm a professor at MIT. My first job was as a graduate student at Yale University. And while I didn't work
```

## Метрики

- **Perplexity (PPL)** — насколько хорошо модель предсказывает следующий токен.

- **BLEU / ROUGE** — сравнение с эталонными текстами (используется в задачах перевода и суммаризации).

- **Человеческая оценка** — связность, осмысленность и грамматическая правильность текста.

## Полезные ссылки

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

- [GPT-2 Model Card](https://huggingface.co/openai-community/gpt2)

- [Sberbank RuGPT (русская версия GPT-2)](https://huggingface.co/ai-forever/rugpt3small_based_on_gpt2)
