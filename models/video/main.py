import os
import cv2
from ultralytics import YOLO

# ====== Настройки ======
VIDEO_PATH = "samples/people.mp4"  # входное видео
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "result.mp4")

# ====== Модель ======
model = YOLO("yolov8n.pt")  # маленькая модель YOLOv8

# ====== Чтение видео ======
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Не удалось открыть видео: {VIDEO_PATH}")

# Получаем параметры видео
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Настраиваем writer для сохранения результата
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print(f" Обработка видео: {VIDEO_PATH}")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Детекция объектов
    results = model(frame)

    # Рисуем предсказания прямо на кадре
    annotated = results[0].plot()
    out.write(annotated)

    # показывать окно
    cv2.imshow("Result", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f" Результат сохранён в: {OUTPUT_PATH}")
