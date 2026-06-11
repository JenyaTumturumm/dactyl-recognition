# Dactyl Recognition - распознавание дактилем

## Описание проекта
Модель для распознавания продактилированных слов
(последовательностей дактилем) по видео. Обучается
на датасете русских жестов.

## Участники
- [@JenyaTumturumm](https://github.com/JenyaTumturumm) &mdash; Предобработка данных, реализация _SupCon Loss_, аренда серверов, обучение модели
- [@AjnoEO](https://github.com/AjnoEO) &mdash; Работа с литературой, сбор данных, адаптация кода _HandReader_, предобработка видео, приложение _StreamLit_

## Опробовать модель

Сайт на Streamlit будет здесь

Веса модели есть на [HuggingFace](https://huggingface.co/Ajno/DactDetect)

## Быстрый старт

### 1. Клонировать репозиторий
```bash
git clone https://github.com/JenyaTumturumm/dactyl-recognition.git
cd dactyl-recognition
```

### 2. Создать виртуальную среду
```bash
python3.10 -m .venv venv
src venv/bin/activate
```

### 3. Установить зависимости
```bash
pip install -r requirements.txt
```

### 4. Запустить приложение Streamlit
```bash
streamlit run app.py
```

## Источники

Korotaev et al. 2025 &mdash; [HandReader: Advanced Techniques for Efficient Fingerspelling Recognition](https://arxiv.org/abs/2505.10267)

Pannattee et al. 2023 &mdash; [American Sign language fingerspelling recognition in the wild with spatio temporal feature extraction and multi-task learning](https://www.sciencedirect.com/science/article/pii/S0957417423034036)
