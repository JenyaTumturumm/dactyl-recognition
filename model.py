"""
model.py - Архитектура нейросети для распознавания дактилем
Этот файл будут менять участники команды
"""

import torch
import torch.nn as nn
import torchvision.models as models

class HandReader(nn.Module):
    """
    Модель для распознавания дактилем
    
    Архитектура:
    - Backbone: MobileNetV3 (предобученный)
    - Временной модуль: GRU или LSTM
    - Классификатор: полносвязные слои
    """
    
    def __init__(self, num_classes=534, backbone='mobilenet_v3_small', hidden_dim=256):
        super().__init__()
        
        # Выбор backbone
        if backbone == 'mobilenet_v3_small':
            self.backbone = models.mobilenet_v3_small(pretrained=True)
            self.feature_dim = 576
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            self.feature_dim = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Убираем классификатор backbone
        if backbone == 'mobilenet_v3_small':
            self.features = self.backbone.features
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        else:  # resnet
            self.features = nn.Sequential(*list(self.backbone.children())[:-2])
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Временной модуль (усреднение или RNN)
        self.temporal_pool = 'mean'  # 'mean', 'gru', 'lstm'
        
        if self.temporal_pool == 'gru':
            self.rnn = nn.GRU(
                input_size=self.feature_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=0.3
            )
            classifier_input = hidden_dim * 2
        elif self.temporal_pool == 'lstm':
            self.rnn = nn.LSTM(
                input_size=self.feature_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=0.3
            )
            classifier_input = hidden_dim * 2
        else:  # mean
            classifier_input = self.feature_dim
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x: [batch, time, channels, height, width]
        batch_size, time_steps, c, h, w = x.shape
        
        # Обрабатываем каждый кадр
        x = x.view(batch_size * time_steps, c, h, w)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # [batch*time, feature_dim]
        
        # Восстанавливаем временное измерение
        x = x.view(batch_size, time_steps, -1)  # [batch, time, feature_dim]
        
        # Временная обработка
        if self.temporal_pool == 'mean':
            x = x.mean(dim=1)  # усреднение по времени
        else:  # gru или lstm
            x, _ = self.rnn(x)
            x = x[:, -1, :]  # последний выход
        
        # Классификация
        output = self.classifier(x)
        
        return output


class SimpleHandReader(nn.Module):
    """
    Упрощенная версия (для быстрых экспериментов)
    """
    
    def __init__(self, num_classes=534):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        batch_size, time_steps, c, h, w = x.shape
        x = x.view(batch_size * time_steps, c, h, w)
        x = self.cnn(x)
        x = x.view(batch_size, time_steps, -1)
        x = x.mean(dim=1)
        return self.classifier(x)


# Словарь доступных моделей
MODELS = {
    'handreader': HandReader,
    'simple': SimpleHandReader,
}


def get_model(model_name='handreader', num_classes=534, **kwargs):
    """Фабрика моделей"""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    return MODELS[model_name](num_classes=num_classes, **kwargs)
