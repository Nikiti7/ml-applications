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
--- Generated text ---

In the future, artificial intelligence will be a force for good for our societies too. It will make us smarter. And it will make us a better person.        

But it doesn't make us better. You see, I'm not an expert in AI or neuroscience. I'm a professor at MIT. My first job was as a graduate student at Yale University. And while I didn't work
```

## Метрики

- Perplexity (PPL)

- BLEU / ROUGE

- Человеческая оценка
