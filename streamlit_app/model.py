# ============================================================
# INFERENCE FOR ONE VIDEO
# encoder from SupCon + decoder from CTC2 checkpoint
# MODE must be "KP_RGB"
# ============================================================

import os
import cv2
import json
import torch
import numpy as np
from pathlib import Path

from HandReader.demo_KP.utils import get_vocab, getRuTokens, Preprocessing
from HandReader.demo_KP.kp_proccesor import process_landmarks
from HandReader.src.utils import Decoder as CharDecoder

from HandReader.model_dev.load import load_kp_rgb_model
from HandReader.model_dev.preprocess import preprocess_frames, Handedness

HAND_FRAME_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_weights_conf():
    # ! TODO: UNCOMMENT
    from huggingface_hub import hf_hub_download 
    from omegaconf import OmegaConf
    import pathlib

    # ! TODO: UNCOMMENT
    model_tensors_path = hf_hub_download(repo_id="Ajno/DactDetect", filename="weights.pt")
    # model_tensors_path = (
    #     hf_hub_download(repo_id="Ajno/DactDetect", filename="weights_encoder.pt"),
    #     hf_hub_download(repo_id="Ajno/DactDetect", filename="weights_decoder.pt"),
    # )
    # WEIGHTS_FOLDER = pathlib.Path("HandReader") / "weights" / "znaki"
    # model_tensors_path = WEIGHTS_FOLDER / "kp_rgb" / "best.pt"
    # model_tensors_path = WEIGHTS_FOLDER / "KP_RGB_ctc_supcon_ctc.pt"
    # model_tensors_path = (
    #     WEIGHTS_FOLDER / "KP_RGB_ctc_supcon_ctc_encoder.pt",
    #     WEIGHTS_FOLDER / "KP_RGB_ctc_supcon_ctc_decoder.pt"
    # )
    # model_tensors_path = WEIGHTS_FOLDER / "KP_RGB_ctc_supcon_ctc_decoder.pt"
    # model_tensors_path = WEIGHTS_FOLDER / "KP_RGB_ctc_supcon_ctc_last.pt"

    OmegaConf.register_new_resolver("len", lambda x: len(x), replace=True)
    conf_path = pathlib.Path("HandReader") / "demo_KP" / "config_KP_RGB.yaml"
    conf = OmegaConf.load(str(conf_path))
    
    return model_tensors_path, conf

# ============================================================
# VOCAB / PROCESSOR
# ============================================================

chars = getRuTokens()
vocab_map, inv_vocab_map, char_list = get_vocab(chars)

processor = Preprocessing()

args_path = Path("HandReader") / "demo_KP" / "inference_args.json"
with open(args_path) as f:
    columns = json.load(f)["selected_columns"]

filtered_columns = [
    idx for idx, col in enumerate(columns)
    if any(s in col for s in ["pose", "hand"])
]

# ============================================================
# UTILS
# ============================================================

def clean_text(s):
    import re
    return "".join(
        re.findall(
            r"[а-я]",
            str(s).lower().replace("ё", "е")
        )
    )


def read_rgb_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames


def make_kp_from_raw_frames(raw_frames):
    landmarks_list = []

    for frame in raw_frames:
        frame = np.array(frame)

        if frame.ndim == 3 and frame.shape[0] == 3:
            frame = np.transpose(frame, (1, 2, 0))

        if frame.dtype == np.float32 or frame.dtype == np.float64:
            if frame.max() <= 2:
                frame = (frame * 255).clip(0, 255).astype(np.uint8)
            else:
                frame = frame.clip(0, 255).astype(np.uint8)

        landmarks_list.append(process_landmarks(frame))

    input_landmarks = np.squeeze(np.stack(landmarks_list, axis=0))

    return torch.unsqueeze(
        processor(torch.from_numpy(input_landmarks), filtered_columns),
        0
    ).float()


def cut_video(raw, max_frames=128):
    if len(raw) <= max_frames:
        return raw

    ids = np.linspace(0, len(raw) - 1, max_frames).astype(int)
    return [raw[i] for i in ids]


def decode_ctc_argmax(outputs):
    """
    outputs shape: [B, T, vocab_size]
    """
    pred_ids = outputs.argmax(dim=-1)
    result = []

    for seq in pred_ids:
        prev = None
        chars_out = []

        for token in seq.detach().cpu().tolist():
            if token != prev and token != 0:
                chars_out.append(inv_vocab_map.get(token, ""))

            prev = token

        result.append(clean_text("".join(chars_out)))

    return result


# ============================================================
# MODEL WRAPPER
# Требует, чтобы encoder, decoder, CharDecoder, FullKPRGBModel
# уже были созданы раньше в ноутбуке
# ============================================================

def make_full_model_for_inference(cfg):
    encoder, decoder = load_kp_rgb_model(None, cfg)

    _, _, char_list_local = get_vocab(getRuTokens())
    char_decoder = CharDecoder(char_list_local)

    return FullKPRGBModel(
        encoder,
        decoder,
        char_decoder
    ).to(DEVICE)


# ============================================================
# LOAD ENCODER + DECODER
# ============================================================

def load_encoder_and_decoder(full_model, encoder_path, decoder_path):
    # ----------------------------
    # 1. load SupCon encoder
    # ----------------------------

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Не найден encoder файл: {encoder_path}")

    enc_state = torch.load(encoder_path, map_location=DEVICE)

    print("\nENCODER FILE KEYS:")
    if isinstance(enc_state, dict):
        for i, k in enumerate(enc_state.keys()):
            print(" ", k)
            if i >= 10:
                break

    if isinstance(enc_state, dict) and "encoder_state_dict" in enc_state:
        print("\nФормат encoder: SupCon encoder_state_dict")
        full_model.encoder.load_state_dict(
            enc_state["encoder_state_dict"],
            strict=True
        )

    elif isinstance(enc_state, dict) and (
        any(str(k).startswith("encoder_rgb.") for k in enc_state.keys())
        or any(str(k).startswith("encoder_kp.") for k in enc_state.keys())
    ):
        print("\nФормат encoder: original HandReader encoder")
        full_model.encoder.load_state_dict(
            enc_state,
            strict=False
        )

    elif isinstance(enc_state, dict) and (
        any(str(k).startswith("encoder.") for k in enc_state.keys())
    ):
        print("\nФормат encoder: full_model или checkpoint с encoder.*")
        encoder_only = {}

        for k, v in enc_state.items():
            if str(k).startswith("encoder."):
                new_k = str(k).replace("encoder.", "", 1)
                encoder_only[new_k] = v

        full_model.encoder.load_state_dict(
            encoder_only,
            strict=True
        )

    else:
        raise ValueError("Не понял формат encoder файла")

    print("✅ encoder loaded")

    # ----------------------------
    # 2. load decoder
    # ----------------------------

    if not os.path.exists(decoder_path):
        raise FileNotFoundError(f"Не найден decoder файл: {decoder_path}")

    dec_state = torch.load(decoder_path, map_location=DEVICE)

    print("\nDECODER FILE KEYS:")
    if isinstance(dec_state, dict):
        for i, k in enumerate(dec_state.keys()):
            print(" ", k)
            if i >= 10:
                break

    # case A: checkpoint от run_ctc_finetune
    # там обычно лежит model_state_dict всей модели
    if isinstance(dec_state, dict) and "model_state_dict" in dec_state:
        print("\nФормат decoder: checkpoint with model_state_dict")

        model_state = dec_state["model_state_dict"]

        decoder_only = {}

        for k, v in model_state.items():
            if str(k).startswith("decoder_net."):
                new_k = str(k).replace("decoder_net.", "", 1)
                decoder_only[new_k] = v

        if len(decoder_only) == 0:
            raise ValueError("В checkpoint не нашёл ключи decoder_net.*")

        full_model.decoder_net.load_state_dict(
            decoder_only,
            strict=True
        )

    # case B: полный full_model.state_dict()
    elif isinstance(dec_state, dict) and any(str(k).startswith("decoder_net.") for k in dec_state.keys()):
        print("\nФормат decoder: full_model.state_dict()")

        decoder_only = {}

        for k, v in dec_state.items():
            if str(k).startswith("decoder_net."):
                new_k = str(k).replace("decoder_net.", "", 1)
                decoder_only[new_k] = v

        full_model.decoder_net.load_state_dict(
            decoder_only,
            strict=True
        )

    # case C: чистый decoder state_dict
    else:
        print("\nФормат decoder: decoder-only state_dict")
        full_model.decoder_net.load_state_dict(
            dec_state,
            strict=True
        )

    print("✅ decoder loaded")

    full_model.to(DEVICE)
    full_model.eval()

    return full_model


# ============================================================
# PREPARE ONE VIDEO
# ============================================================

def prepare_video_for_kp_rgb(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Не найдено видео: {video_path}")

    raw = read_rgb_frames(video_path)

    if len(raw) == 0:
        raise ValueError("Видео пустое или не прочиталось")

    # raw = cut_video(raw, MAX_FRAMES)

    frames = preprocess_frames(
        raw,
        HAND_FRAME_SIZE,
        dynamic_crop=True,
        handedness=Handedness.RIGHT,
        convert_colors=False,
    )

    x = torch.from_numpy(np.stack(frames)).float()

    if x.max() > 2:
        x = x / 255.0

    kp = make_kp_from_raw_frames(raw).float()

    input_lengths = torch.tensor([len(raw)]).long()

    # shapes:
    # inputs: [1, T, 3, 224, 224]
    # kp:     [1, T, 54, 3]
    # length: [1]
    inputs = x.unsqueeze(0)

    return (
        inputs.to(DEVICE),
        kp.to(DEVICE),
        input_lengths.to(DEVICE),
        len(raw)
    )


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def predict_video(video_path, full_model):
    inputs, landmarks, input_lengths, used_frames = prepare_video_for_kp_rgb(video_path)

    outputs = full_model(
        inputs,
        landmarks,
        input_lengths
    )

    pred = decode_ctc_argmax(outputs)[0]

    return pred, outputs, used_frames


# ============================================================
# RUN
# ============================================================

import torch.nn as nn

# ============================================================
# MODEL WRAPPER FOR KP_RGB
# ============================================================

class FullKPRGBModel(nn.Module):
    def __init__(self, encoder, decoder_net, char_decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder_net = decoder_net
        self.decoder = char_decoder

    def forward(self, inputs, landmarks, input_lengths):
        # inputs: [B, T, 3, 224, 224]
        # landmarks: [B, T, 54, 3]
        # в нашем коде BATCH_SIZE = 1

        if inputs.shape[0] != 1:
            raise ValueError("Пока поддерживается только batch_size=1")

        rgb = inputs[0]       # [T, 3, 224, 224]
        kp = landmarks        # [1, T, 54, 3]

        z = self.encoder(rgb, kp, input_lengths)
        outputs = self.decoder_net(z)

        return outputs
