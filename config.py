"""
Конфигурация проекта
Легко менять под любую платформу
"""

import os
from pathlib import Path

# Корень проекта
PROJECT_ROOT = Path(__file__).parent

# -----------------------------------------------------------
# НАСТРОЙКИ ПУТЕЙ (меняйте здесь под свою платформу)
# -----------------------------------------------------------

# Тип платформы: 'colab', 'datasphere', 'local', 'shadow'
PLATFORM = 'colab'  # ← ИЗМЕНИТЕ ПРИ ЗАПУСКЕ

# Пути к данным (в зависимости от платформы)
PATHS = {
    'colab': {
        'data_root': '/content/drive/MyDrive/DactDetect',
        'excel_file': 'Расшифровки видео.xlsx',
        'model_save': '/content/models',
    },
    'datasphere': {
        'data_root': '/datasphere/datasets/DactDetect',
        'excel_file': 'Расшифровки видео.xlsx',
        'model_save': '/datasphere/output/models',
    },
    'shadow': {
        'data_root': '/mnt/data/DactDetect',
        'excel_file': 'Расшифровки видео.xlsx',
        'model_save': '/home/user/models',
    },
    'local': {
        'data_root': './data/DactDetect',
        'excel_file': 'Расшифровки видео.xlsx',
        'model_save': './models',
    }
}

def get_paths():
    """Возвращает пути для текущей платформы"""
    return PATHS.get(PLATFORM, PATHS['local'])

def set_platform(platform_name):
    """Установить платформу перед запуском"""
    global PLATFORM
    if platform_name in PATHS:
        PLATFORM = platform_name
        print(f"✅ Платформа: {PLATFORM}")
    else:
        print(f"❌ Неизвестная платформа: {platform_name}")

# -----------------------------------------------------------
# НАСТРОЙКИ ОБУЧЕНИЯ
# -----------------------------------------------------------

TRAIN_CONFIG = {
    'batch_size': 8,
    'epochs': 30,
    'learning_rate': 0.001,
    'num_frames': 16,
    'target_size': (224, 224),
    'max_videos': None,  # None = все видео
    'test_size': 0.2,
    'random_seed': 42,
}

# -----------------------------------------------------------
# НАСТРОЙКИ МОДЕЛИ
# -----------------------------------------------------------

MODEL_CONFIG = {
    'name': 'HandReader',
    'backbone': 'mobilenet_v3_small',
    'hidden_dim': 256,
    'dropout': 0.3,
}

# -----------------------------------------------------------
# НАСТРОЙКИ ЛОГИРОВАНИЯ
# -----------------------------------------------------------

LOG_CONFIG = {
    'level': 'INFO',
    'save_dir': 'logs',
    'log_to_file': True,
}
