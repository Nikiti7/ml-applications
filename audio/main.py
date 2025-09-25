import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa

# Загружаем модель YAMNet (Google, обучена на AudioSet)
print("Loading model...")
model = hub.load("https://tfhub.dev/google/yamnet/1")

# Загружаем список классов (категории звуков)
class_map_path = model.class_map_path().numpy()
class_names = list(tf.io.gfile.GFile(class_map_path))

# Файлы для теста
files = [
    r"C:\Users\Nikol\Desktop\ml-applications\audio\speech.wav",
    r"C:\Users\Nikol\Desktop\ml-applications\audio\music.wav",
    r"C:\Users\Nikol\Desktop\ml-applications\audio\noise.wav"
]


for fname in files:
    print(f"\nAnalyzing file: {fname}")
    
    # Загружаем аудио в формате 16kHz mono
    waveform, sr = librosa.load(fname, sr=16000)
    
    # Прогоняем через модель
    scores, embeddings, spectrogram = model(waveform)
    mean_scores = np.mean(scores.numpy(), axis=0)
    
    # Топ-3 предсказания
    top3 = np.argsort(mean_scores)[::-1][:3]
    
    print("Top-3 predictions:")
    for i in top3:
        print(f"- {class_names[i].strip()} ({mean_scores[i]:.3f})")
