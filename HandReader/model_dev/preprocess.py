import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import List, Tuple, Union
from enum import Enum
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import face_mesh as mp_face

# ============================================================
# НОРМАЛИЗАЦИЯ (ImageNet)
# ============================================================
NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Handedness(Enum):
    LEFT = 0
    RIGHT = 1
    MOUTH = 2

# ============================================================
# ДЕТЕКЦИЯ РУКИ
# ============================================================
MAX_NUM_HANDS = 2
base_hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def crop_hand_region(frame_rgb: np.ndarray, padding=0.7, hands_detector: mp_hands.Hands = base_hands_detector):
    """
    Обнаруживает руку и вырезает КВАДРАТ вокруг неё (чтобы не искажать пропорции).
    """
    h, w, _ = frame_rgb.shape
    results = hands_detector.process(frame_rgb)

    crops = [(frame_rgb, None), (frame_rgb, None)]
    certainty = [0.0, 0.0]

    if not results.multi_hand_landmarks:
        return crops
    
    labels = [handedness.ListFields()[0][1][0].ListFields() for handedness in results.multi_handedness]
    # Если обе руки левые или обе руки правые, то лучше вообще не анализировать этот кадр
    if len(labels) == 2 and labels[0] == labels[1]:
        return crops
    
    # print("Cropping")
    for handedness, landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
        # Магия тупейшего интерфейса гугловских классов
        ind, score, label = [
            item[1] for item in handedness.ListFields()[0][1][0].ListFields()
        ]
        # print(ind, score, label)
        # print(handedness.ListFields()[0][1][0].ListFields()[0][1], handedness.ListFields()[0][1][0].ListFields()[2][1])
        # ind = Handedness[handedness.ListFields()[0][1][0].ListFields()[2][1].upper()].value
        if ind is None:
            continue
        if certainty[ind] > score:
            continue
        # print(f"Assigning {Handedness(ind)}")
        certainty[ind] = score
        # print(f"{len(landmarks)} landmarks")
        x_coords = [lm.x * w for lm in landmarks.landmark]
        y_coords = [lm.y * h for lm in landmarks.landmark]

        # Центр руки
        center_x = (min(x_coords) + max(x_coords)) / 2
        center_y = (min(y_coords) + max(y_coords)) / 2

        # Размер квадрата = максимальный размер руки + отступ
        hand_width = max(x_coords) - min(x_coords)
        hand_height = max(y_coords) - min(y_coords)
        half_size = max(hand_width, hand_height) / 2
        if isinstance(padding, float):
            padding = int(half_size * padding)
        half_size += padding

        x_min = max(0, int(center_x - half_size))
        y_min = max(0, int(center_y - half_size))
        x_max = min(w, int(center_x + half_size))
        y_max = min(h, int(center_y + half_size))

        # Если вышли за границы — корректируем, сохраняя квадрат
        if x_max - x_min != y_max - y_min:
            size = min(x_max - x_min, y_max - y_min)
            x_max = x_min + size
            y_max = y_min + size

        crops[ind] = (frame_rgb[y_min:y_max, x_min:x_max], [x_min, y_min, x_max, y_max])
    return crops

base_face_detector = mp_face.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
LIPS_KP = sorted(np.unique(list(mp_face.FACEMESH_LIPS)).flatten())

def crop_mouth_region(frame_rgb: np.ndarray, padding=0.7, face_detector: mp_face.FaceMesh = base_face_detector):
    """
    Обнаруживает рот и вырезает КВАДРАТ вокруг него (чтобы не искажать пропорции).
    """
    h, w, _ = frame_rgb.shape
    results = face_detector.process(frame_rgb)

    crops = (frame_rgb, None)

    if not results.multi_face_landmarks:
        return crops

    landmark: List = results.multi_face_landmarks[0].landmark

    # Костыльненько но собрать ключевые точки губ
    for ind in LIPS_KP:
        landmark.append(landmark[ind])
    while len(landmark) > len(LIPS_KP):
        landmark.pop(0)
    
    x_coords = [lm.x * w for lm in landmark]
    y_coords = [lm.y * h for lm in landmark]

    # Центр руки
    center_x = (min(x_coords) + max(x_coords)) / 2
    center_y = (min(y_coords) + max(y_coords)) / 2

    # Размер квадрата = максимальный размер руки + отступ
    mouth_width = max(x_coords) - min(x_coords)
    mouth_height = max(y_coords) - min(y_coords)
    half_size = max(mouth_width, mouth_height) / 2
    if isinstance(padding, float):
        padding = int(half_size * padding)
    half_size += padding

    x_min = max(0, int(center_x - half_size))
    y_min = max(0, int(center_y - half_size))
    x_max = min(w, int(center_x + half_size))
    y_max = min(h, int(center_y + half_size))

    # Если вышли за границы — корректируем, сохраняя квадрат
    if x_max - x_min != y_max - y_min:
        size = min(x_max - x_min, y_max - y_min)
        x_max = x_min + size
        y_max = y_min + size

    crops = (frame_rgb[y_min:y_max, x_min:x_max], [x_min, y_min, x_max, y_max])

    return crops

# ============================================================
# ПРЕДОБРАБОТКА КАДРОВ (С КРОПОМ)
# ============================================================
def preprocess_frames(video, img_size=224, crop=True, dynamic_crop=False, handedness=Handedness.RIGHT, convert_colors=True):
    """
    Возвращает предобработанный список кадров видео

    Args:
        video: путь к видео или список фреймов
        img_size: размер для ресайза
        crop: детектировать и кропать область тела (True/False)
        dynamic_crop: учитывать динамику при кропе
    """
    frames: List[np.ndarray] = []

    if isinstance(video, List):
        frames = video
    else:
        if not os.path.exists(video):
            raise FileExistsError(f"Файл {video} не найден")
        cap = cv2.VideoCapture(video)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

    crop_bboxes: List[Union[None, Tuple[int, int, int, int]]] = []

    if crop and dynamic_crop:
        if handedness == Handedness.MOUTH:
            detector = mp_face.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            detector = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=MAX_NUM_HANDS,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

    last_bbox_ind = None
    x_diff = 0
    y_diff = 0
    window_size = 1

    for i, frame in enumerate(frames):
        # BGR -> RGB
        if convert_colors:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ДЕТЕКЦИЯ И КРОП РУКИ
        if crop:
            if handedness == Handedness.MOUTH:
                kwargs = {"face_detector": detector} if dynamic_crop else {}
                frame, bbox = crop_mouth_region(frame, **kwargs)
            else:
                kwargs = {"hands_detector": detector} if dynamic_crop else {}
                frame, bbox = crop_hand_region(frame, **kwargs)[handedness.value]
            if bbox is not None and last_bbox_ind is None:
                last_bbox_ind = len(crop_bboxes)
            crop_bboxes.append(bbox)

        frames[i] = frame

    for i in range(len(frames)):
        if crop and (crop_bboxes[i] is not None):
            last_bbox_ind = i
            for window_size, next_bbox in enumerate(crop_bboxes[i+1:], start=1):
                if next_bbox is not None:
                    x_diff = next_bbox[0] - crop_bboxes[i][0]
                    y_diff = next_bbox[1] - crop_bboxes[i][1]
                    break
            else:
                window_size = 1
                x_diff = 0
                y_diff = 0

        if crop and last_bbox_ind and (crop_bboxes[i] is None):
            multiplier = (i - last_bbox_ind) / window_size
            x_shift = round(x_diff * multiplier)
            y_shift = round(y_diff * multiplier)
            x_min, y_min, x_max, y_max = (
                crop_bboxes[last_bbox_ind][0] + x_shift,
                crop_bboxes[last_bbox_ind][1] + y_shift,
                crop_bboxes[last_bbox_ind][2] + x_shift,
                crop_bboxes[last_bbox_ind][3] + y_shift
            )
            frame_rgb = frames[i][y_min:y_max, x_min:x_max]
        else:
            frame_rgb = frames[i]

        # Resize
        frame_resized = cv2.resize(frame_rgb, (img_size, img_size))

        # Нормализация
        frame_norm = frame_resized.astype(np.float32) / 255.0
        # frame_norm = (frame_norm - NORM_MEAN) / NORM_STD

        # HWC -> CHW
        frame_chw = np.transpose(frame_norm, (2, 0, 1))

        frames[i] = frame_chw

    return frames

# ============================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================
def display_frames(frames: Union[np.ndarray, List], title: str = "Кадры видео", denorm: bool = False):
    if isinstance(frames, np.ndarray) and len(frames.shape) == 3:
        frames = [frames]
    MAX_COLS = 20
    MIN_COLS = 5
    cols = MAX_COLS
    while (cols - 1) * (cols - 2) >= len(frames) and cols >= MIN_COLS: cols -= 1
    # cols = 5 if len(frames) > 12 else 4 if len(frames) > 4 else len(frames)
    rows = (len(frames) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(2*cols, 2*rows), squeeze=False)
    for ax, f in zip(axs.flatten(), frames):
        ax: Axes
        if denorm: f = f * NORM_STD[:, None, None] + NORM_MEAN[:, None, None]
        f = np.transpose(f, (1, 2, 0))
        f = np.clip(f, 0, 1)
        ax.imshow(f)
        ax.set_axis_off()
    for ax in axs.flatten()[len(frames):]:
        ax.set_axis_off()
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(title)
    plt.show()
