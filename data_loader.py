"""
Загрузка данных с разных источников
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import cv2
from tqdm import tqdm
import config

class DataLoader:
    """Класс для загрузки данных с любой платформы"""
    
    def __init__(self):
        self.paths = config.get_paths()
        self.data_root = Path(self.paths['data_root'])
        self.df = None
        self.video_files = []
        
    def load_metadata(self):
        """Загружает таблицу с расшифровками"""
        excel_path = self.data_root / self.paths['excel_file']
        
        if not excel_path.exists():
            raise FileNotFoundError(f"Таблица не найдена: {excel_path}")
        
        # Пробуем разные форматы
        try:
            self.df = pd.read_excel(excel_path, sheet_name='Корпус')
        except:
            try:
                self.df = pd.read_csv(excel_path)
            except:
                raise ValueError(f"Не удалось загрузить {excel_path}")
        
        print(f"✅ Загружено {len(self.df)} записей")
        return self.df
    
    def find_videos(self):
        """Находит все видео из таблицы"""
        if self.df is None:
            self.load_metadata()
        
        self.video_files = []
        file_column = 'файл' if 'файл' in self.df.columns else self.df.columns[0]
        word_column = 'расшифровка' if 'расшифровка' in self.df.columns else self.df.columns[1]
        
        for _, row in self.df.iterrows():
            filename = str(row[file_column])
            if pd.isna(filename):
                continue
            
            video_path = self.data_root / filename
            if video_path.exists():
                self.video_files.append({
                    'filename': filename,
                    'path': str(video_path),
                    'word': str(row[word_column]),
                })
        
        print(f"✅ Найдено {len(self.video_files)} видео")
        return self.video_files
    
    def get_video_list(self, max_videos=None):
        """Возвращает список видео"""
        if not self.video_files:
            self.find_videos()
        
        videos = self.video_files
        if max_videos and max_videos < len(videos):
            videos = videos[:max_videos]
        
        return videos
