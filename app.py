"""
Приложение Streamlit
"""
import os

# Set a writable directory for MediaPipe models
os.environ['MEDIAPIPE_DOWNLOAD_PATH'] = '/tmp/mediapipe'
os.makedirs(os.environ['MEDIAPIPE_DOWNLOAD_PATH'], exist_ok=True)

from HandReader.demo_KP.kp_proccesor import process_landmarks
from demo_KP.utils import get_vocab, getRuTokens
from src.utils import Decoder as CharDecoder
from HandReader.model_dev.load import load_kp_rgb_model
from HandReader.model_dev.preprocess import preprocess_frames

from streamlit_app.model import (
    load_weights_conf, FullKPRGBModel,
    make_full_model_for_inference, load_encoder_and_decoder, predict_video
)
from streamlit_app.runner import Runner

import streamlit as st
from functools import partial
import tempfile
import os
import torch

HAND_FRAME_SIZE = 224

@st.cache_data(show_spinner=False)
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = get_device()

@st.cache_resource(show_spinner=False)
def get_model():
    weights_path, conf = load_weights_conf()

    full_model = make_full_model_for_inference(conf)

    # encoder, decoder = load_kp_rgb_model(None, conf)
    # # encoder, decoder = load_kp_rgb_model(weights_path, conf)
    
    # chars = getRuTokens()
    # _, _, char_list = get_vocab(chars)
    # char_decoder = CharDecoder(char_list)

    # model = FullKPRGBModel(encoder, decoder, char_decoder).to(DEVICE)
    full_model.load_state_dict(torch.load(weights_path, DEVICE)["model_state_dict"])

    # return encoder, decoder

    # load_encoder_and_decoder(full_model, *weights_path)
    return full_model

@st.cache_data(show_spinner=False)
def run_for_file(file_path: str):
    # encoder, decoder = get_model()

    # print("Creating runner")
    # runner = Runner(
    #     encoder, decoder, file_path, process_landmarks,
    #     partial(preprocess_frames, img_size=HAND_FRAME_SIZE, dynamic_crop=True, convert_colors=True)
    # )
    # print("Inferencing through runner")
    # return runner.run()
    result, *_ = predict_video(file_path, get_model())
    return result

st.title("DactDetect — Распознавание дактиля")
uploaded_file = st.file_uploader("Загрузите видеофайл", type=["mp4", "avi", "mov", "mkv", "webm"])

if uploaded_file is not None:
    filename: str = uploaded_file.name
    suff_ind = filename.rfind(".")
    with st.spinner("Сохраняю видеофайл..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=filename[suff_ind:]) as tfile:
            tfile.write(uploaded_file.read())
            temp_file_path = tfile.name

    st.video(uploaded_file)

    with st.spinner("Подгружаю модель..."):
        model = get_model()
    
    with st.spinner("Извлекаю дактиль..."):
        result = run_for_file(temp_file_path)
    
    st.markdown(f"Результат: **{result.strip()}**")

    st.success("Обработка завершена!")
    
    os.unlink(temp_file_path)
else:
    st.info("Ожидаем загрузки видео...")
