"""
Основной скрипт обучения
Запускается на любой платформе

# Локально
python train.py --platform local --max_videos 100

# Google Colab
python train.py --platform colab --max_videos 500

# Yandex DataSphere
python train.py --platform datasphere --batch_size 16

# Shadow сервер
python train.py --platform shadow --epochs 50
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import config
from data_loader import DataLoader as DactDataLoader

# -----------------------------------------------------------
# Парсинг аргументов командной строки
# -----------------------------------------------------------
parser = argparse.ArgumentParser(description='Обучение модели дактилем')
parser.add_argument('--platform', type=str, default='local',
                    choices=['colab', 'datasphere', 'shadow', 'local'],
                    help='Платформа для запуска')
parser.add_argument('--max_videos', type=int, default=None,
                    help='Максимум видео для обучения')
parser.add_argument('--epochs', type=int, default=None,
                    help='Количество эпох')
parser.add_argument('--batch_size', type=int, default=None,
                    help='Размер батча')
args = parser.parse_args()

# -----------------------------------------------------------
# Настройка платформы
# -----------------------------------------------------------
config.set_platform(args.platform)
paths = config.get_paths()
train_cfg = config.TRAIN_CONFIG.copy()

if args.max_videos:
    train_cfg['max_videos'] = args.max_videos
if args.epochs:
    train_cfg['epochs'] = args.epochs
if args.batch_size:
    train_cfg['batch_size'] = args.batch_size

print("=" * 60)
print("🚀 ЗАПУСК ОБУЧЕНИЯ")
print("=" * 60)
print(f"📁 Платформа: {config.PLATFORM}")
print(f"📁 Путь к данным: {paths['data_root']}")
print(f"⚙️ Параметры: {train_cfg}")
print("=" * 60)

# -----------------------------------------------------------
# Загрузка данных
# -----------------------------------------------------------
print("\n📂 Загрузка данных...")
loader = DactDataLoader()
df = loader.load_metadata()
videos = loader.get_video_list(max_videos=train_cfg['max_videos'])

print(f"✅ Загружено {len(videos)} видео")

# -----------------------------------------------------------
# Здесь продолжение вашего кода обучения
# -----------------------------------------------------------

if __name__ == "__main__":
    # Точка входа
    print("\n✅ Обучение завершено!")
