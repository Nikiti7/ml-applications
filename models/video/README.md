# Video ML — Object Detection (YOLOv8)

## Задача

Обнаружение объектов в видеопотоке с помощью модели YOLOv8.  
Программа принимает видео и создаёт новый файл с визуализированными детекциями.

---

## Используемые технологии

- **Ultralytics YOLOv8**
- **PyTorch**
- **OpenCV**

---

## Запуск

1. Установить зависимости:

```bash
pip install -r ../requirements.txt
```

2. Подготовить видеофайл в папку samples/:

```bash
video/samples/street.mp4
```

3. Запустить:

```bash
cd video
python main.py
```

4. Результат появится в:

```bash
video/outputs/result.mp4
```

## Параметры

- Изменить путь к видео → в начале main.py (VIDEO_PATH).

- Нажмите **Q**, чтобы остановить обработку.

## Метрики и результаты

- Скорость обработки (FPS)

- Количество найденных объектов каждого класса

- Визуальный результат с bounding boxes

## Полезные ссылки

- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)

- [OpenCV Python Docs](https://docs.opencv.org/)

## Папка 'video/' после реализации

```bash
video/
├── main.py
├── samples/
│ └── street.mp4
│ └── people.mp4
├── outputs/
│ └── result.mp4
└── README.md
```
