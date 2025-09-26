# Audio ML — Классификация аудио (TensorFlow + YAMNet)

## Задача

Определение содержания в аудиозаписи (речь, музыка, шумы и т.д.) с помощью готовой модели **YAMNet** от Google.

YAMNet обучена на датасете **AudioSet** и способна классифицировать аудио на **521 категорию**: речь, аплодисменты, музыка, лай собаки и многое другое.

---

## Используемые технологии

- **TensorFlow** — для запуска модели.
- **TensorFlow Hub** — загрузка готовой модели YAMNet.
- **Librosa** — обработка и преобразование аудиофайлов.
- **NumPy** — работа с массивами данных.

---

### Запуск

```bash
cd audio
pip install -r ../requirements.txt
python main.py
```

## Пример вывода

```yaml
Analyzing file: speech.wav
Top-3 predictions:
- display_name (0.943)
- Crunch (0.028)
- Conversation (0.021)

Analyzing file: music.wav
Top-3 predictions:
- Whale vocalization (0.971)
- Music for children (0.176)
- Happy music (0.085)

Analyzing file: noise.wav
Top-3 predictions:
- Cheering (0.649)
- display_name (0.110)
- Crunch (0.063)
```

## Метрики

- Accuracy

- F1-score

- WER
