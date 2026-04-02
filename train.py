"""
train.py - Обучение модели
Запускается из Colab, DataSphere или локально
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Импортируем модель
from model import get_model

# -----------------------------------------------------------
# ПАРАМЕТРЫ (меняйте здесь под свою платформу)
# -----------------------------------------------------------

# ГДЕ ЛЕЖАТ ДАННЫЕ? (измените под себя)
# Для Colab: /content/drive/MyDrive/DactDetect
# Для DataSphere: /datasphere/datasets/DactDetect
# Для Shadow: /mnt/data/DactDetect
# Для локального: ./data/DactDetect

DATA_ROOT = "/content/drive/MyDrive/DactDetect"  # ← ИЗМЕНИТЕ ЗДЕСЬ
EXCEL_FILE = "Расшифровки видео.xlsx"  # имя файла с таблицей

# Параметры обучения
MAX_VIDEOS = 100        # сколько видео взять (None = все)
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 0.001
NUM_FRAMES = 16
TARGET_SIZE = (224, 224)

# -----------------------------------------------------------
# ДАТАСЕТ
# -----------------------------------------------------------

class DactDataset(Dataset):
    def __init__(self, video_list, word_labels):
        self.video_list = video_list
        self.labels = torch.LongTensor(word_labels)
        
    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, idx):
        video = self.video_list[idx]
        frames = self._load_video(video['path'])
        return frames, self.labels[idx]
    
    def _load_video(self, video_path):
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
        
        frames = np.array(frames, dtype=np.float32)
        frames = torch.FloatTensor(frames).permute(0, 3, 1, 2)
        
        return frames

# -----------------------------------------------------------
# ЗАГРУЗКА ДАННЫХ
# -----------------------------------------------------------

def load_data():
    """Загружает видео и метки"""
    print(f"📂 Загрузка данных из {DATA_ROOT}")
    
    # Загружаем таблицу
    excel_path = os.path.join(DATA_ROOT, EXCEL_FILE)
    df = pd.read_excel(excel_path, sheet_name='Корпус')
    print(f"✅ Таблица: {len(df)} записей")
    
    # Находим видео
    videos = []
    for _, row in df.iterrows():
        filename = str(row['файл'])
        if pd.isna(filename):
            continue
        
        video_path = os.path.join(DATA_ROOT, filename)
        if os.path.exists(video_path):
            videos.append({
                'filename': filename,
                'path': video_path,
                'word': str(row['расшифровка'])
            })
    
    print(f"✅ Найдено видео: {len(videos)}")
    
    if MAX_VIDEOS and MAX_VIDEOS < len(videos):
        videos = videos[:MAX_VIDEOS]
        print(f"⚠️ Используем {MAX_VIDEOS} видео")
    
    # Кодируем слова
    words = [v['word'] for v in videos]
    le = LabelEncoder()
    labels = le.fit_transform(words)
    
    print(f"✅ Уникальных слов: {len(le.classes_)}")
    
    return videos, labels, le

# -----------------------------------------------------------
# ОБУЧЕНИЕ
# -----------------------------------------------------------

def train():
    print("=" * 60)
    print("🚀 ЗАПУСК ОБУЧЕНИЯ")
    print("=" * 60)
    
    # Загрузка данных
    videos, labels, le = load_data()
    
    # Датасет
    dataset = DactDataset(videos, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Модель
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model('simple', num_classes=len(le.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"🔥 Устройство: {device}")
    print(f"📊 Модель: {sum(p.numel() for p in model.parameters()):,} параметров")
    print(f"📊 Train: {train_size}, Val: {val_size}")
    print("=" * 60)
    
    # Обучение
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for frames, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
            frames, labels = frames.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for frames, labels in val_loader:
                frames, labels = frames.to(device), labels.to(device)
                outputs = model(frames)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
    
    # Сохранение
    model_path = os.path.join(os.path.dirname(DATA_ROOT), 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"✅ Модель сохранена: {model_path}")

if __name__ == "__main__":
    train()
