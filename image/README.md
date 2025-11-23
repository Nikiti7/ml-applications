# Image ML — Классификация изображений (PyTorch)

## Задача

Классификация изображений с использованием предобученной модели (transfer learning).

## Технологии

- PyTorch, torchvision  
- Pillow, matplotlib  
- Предобученная модель: ResNet-18 (можно заменить на MobileNet для CPU)

## Структура данных

Используется формат `ImageFolder`:

```bash
data/
├── train/
│ ├── class1/
│ └── class2/
└── val/
├── class1/
└── class2/
```

## Как запустить

1. Установить зависимости:

```bash
cd image
pip install -r ../requirements.txt
```

2. Подготовить данные (data/train, data/val), либо использовать CIFAR-10 для теста.

3. Обучение:

```bash
python train.py
```

Лучшая модель будет сохранена в best_resnet18.pth.

4. Инференс (один/несколько файлов):

```bash
python infer.py --model best_resnet18.pth --images data\val\cat\cat.2000.jpg data\val\dog\dog.2000.jpg
```

## Метрики

- Accuracy
- Top-5 Accuracy
- Confusion Matrix
- Precision/Recall/F1.

## Советы

- Размер входа: 224x224 (стандарт для ResNet).

- Для небольших задач и CPU выбирайте MobileNet/VGG Mobile варианты.

- Не храните тяжёлые модели в Git (.gitignore уже настроен`).

## Источники

- [PyTorch docs](https://pytorch.org/)

- [torchvision models](https://pytorch.org/vision/stable/models.html)