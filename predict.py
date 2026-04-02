"""
predict.py - Применение обученной модели к новым видео
Запускается локально или на сервере
"""

import os
import torch
import cv2
import numpy as np
import pickle
from model import get_model

# -----------------------------------------------------------
# НАСТРОЙКИ (измените под себя)
# -----------------------------------------------------------
MODEL_PATH = "model.pth"           # путь к обученной модели
ENCODER_PATH = "label_encoder.pkl" # путь к энкодеру
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Параметры обработки видео
NUM_FRAMES = 16
TARGET_SIZE = (224, 224)

# -----------------------------------------------------------
# ЗАГРУЗКА МОДЕЛИ
# -----------------------------------------------------------
def load_model():
    with open(ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
    
    model = get_model('handreader', num_classes=len(le.classes_))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    return model, le

# -----------------------------------------------------------
# ОБРАБОТКА ВИДЕО
# -----------------------------------------------------------
def process_video(video_path, model, le):
    """Применяет модель к одному видео"""
    
    # Загрузка кадров
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames > NUM_FRAMES:
        indices = np.linspace(0, total_frames-1, NUM_FRAMES, dtype=int)
    else:
        indices = range(total_frames)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, TARGET_SIZE)
            frame = frame / 255.0
            frames.append(frame)
        else:
            frames.append(np.zeros((*TARGET_SIZE, 3)))
    
    cap.release()
    
    while len(frames) < NUM_FRAMES:
        frames.append(np.zeros((*TARGET_SIZE, 3)))
    
    # Преобразование в тензор
    frames = np.array(frames, dtype=np.float32)
    frames = torch.FloatTensor(frames).permute(0, 3, 1, 2).unsqueeze(0).to(DEVICE)
    
    # Предсказание
    with torch.no_grad():
        outputs = model(frames)
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, 5)
    
    # Преобразование в слова
    top_words = le.inverse_transform(top_indices[0].cpu().numpy())
    top_probs = top_probs[0].cpu().numpy()
    
    return list(zip(top_words, top_probs))

# -----------------------------------------------------------
# ОБРАБОТКА ПАПКИ
# -----------------------------------------------------------
def process_folder(folder_path, model, le):
    """Обрабатывает все видео в папке"""
    results = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.mp4', '.webm', '.avi')):
            video_path = os.path.join(folder_path, filename)
            predictions = process_video(video_path, model, le)
            results.append({
                'video': filename,
                'predictions': predictions
            })
            print(f"\n🎬 {filename}")
            for word, prob in predictions:
                print(f"   {word}: {prob*100:.2f}%")
    
    return results

# -----------------------------------------------------------
# ЗАПУСК
# -----------------------------------------------------------
if __name__ == "__main__":
    import sys
    
    print("🚀 Загрузка модели...")
    model, le = load_model()
    
    if len(sys.argv) > 1:
        # Если передан путь к файлу или папке
        path = sys.argv[1]
        if os.path.isdir(path):
            process_folder(path, model, le)
        elif os.path.isfile(path):
            predictions = process_video(path, model, le)
            print(f"\n🎬 {os.path.basename(path)}")
            for word, prob in predictions:
                print(f"   {word}: {prob*100:.2f}%")
        else:
            print(f"❌ Путь не найден: {path}")
    else:
        # Интерактивный режим
        print("\n📹 Введите путь к видео или папке:")
        path = input().strip()
        if os.path.isdir(path):
            process_folder(path, model, le)
        else:
            predictions = process_video(path, model, le)
            for word, prob in predictions:
                print(f"{word}: {prob*100:.2f}%")
