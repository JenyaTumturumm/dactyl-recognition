import json
import pathlib

import cv2
import numpy as np
import torch
import torch.nn as nn

from demo_KP.utils import Preprocessing, get_vocab, getRuTokens
from src.utils import Decoder as CharDecoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseRecognition:
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        frames_list,
        landmarks_list,
        prediction_list,
        verbose,
        video_length: int,
    ):
        self.video_length = video_length
        self.verbose = verbose
        self.started = None
        self.encoder = encoder.eval()
        self.decoder = decoder.eval()

        self.frames_list = frames_list
        self.landmarks_list = landmarks_list

        self.prediction_list = prediction_list
        chars = getRuTokens()
        self.vocab_map, self.inv_vocab_map, self.char_list = get_vocab(chars)
        self.dec = CharDecoder(self.char_list)

        self.sentence = ""
        self.last_letter = "_"
        self.processor = Preprocessing()
        self.cur_tokens = []
        self.path_ids = None  # будет заполнено в run()

        args_path = pathlib.Path("HandReader") / "demo_KP" / "inference_args.json"
        with open(args_path) as f:
            columns = json.load(f)["selected_columns"]
        self.filtered_columns = [
            idx
            for idx, col in enumerate(columns)
            if any(substring in col for substring in ["pose", "hand"])
        ]

    def run_recognition(self):
        pass

    def run(self):
        """
        Run the recognition model.
        """
        # encoder part
        if (len(self.landmarks_list) < self.video_length - 1 or len(self.frames_list) < self.video_length - 1):
            return
        print("Start recognition")
        # list contains numpy arrays with shape [1, 390]
        forward_args = []
        input_landmarks = np.squeeze(
            np.stack(self.landmarks_list[:], axis=0)
        )  # whole video frames
        # shape is [1, video_length, n_landmarks, 3]
        data = torch.unsqueeze(
            self.processor(torch.from_numpy(input_landmarks), self.filtered_columns), 0
        ).float().to(DEVICE)
        # shape is [1, window_size_frames, 512]
        forward_args.append(data)
        # input_frames = np.squeeze(
        #     np.stack(self.frames_list[:], axis=0)
        # )  # whole video frames
        input_frames = np.stack(self.frames_list[:], axis=0)
        print(f"{input_frames.shape=}")
        # shape is [video_length, frame_size, frame_size, 3]
        # data = torch.unsqueeze(
        #     self.processor(torch.from_numpy(input_frames)), 0
        # ).float()
        data = self.processor(torch.from_numpy(input_frames), no_reshaping=True).float().to(DEVICE)
        print(f"{data.shape=}")
        # shape is [1, ]
        forward_args = [data] + forward_args
        forward_args.append(torch.tensor([len(data)]).to(DEVICE))

        encoder_outs = self.encoder(*forward_args)

        dec_outs = self.decoder(encoder_outs)
        print(f"{dec_outs.shape=}")

        # # === ДОБАВИТЬ ЭТИ 2 СТРОКИ ===
        # logits = dec_outs[0].detach().cpu().numpy()
        # self.path_ids = logits.argmax(axis=1)  # [T] — индекс буквы для каждого кадра
        # # ============================
        
        curr_pred = self.dec.greedy_decode(dec_outs[0].detach().cpu().numpy())
        print(curr_pred)

        self.prediction_list.extend(curr_pred)
        result = "".join(self.prediction_list)
        print("Result:", result)
        
        return result


class Recognition(BaseRecognition):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        frames_list,
        landmarks_list,
        prediction_list,
        verbose,
        video_length: int,
    ):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            frames_list=frames_list,
            landmarks_list=landmarks_list,
            prediction_list=prediction_list,
            verbose=verbose,
            video_length=video_length,
        )
        self.started = True

    def start(self):
        return self.run()


class Runner:
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        path_to_video: str,
        preprocess_landmarks,
        preprocess_frames,
        verbose: bool = False,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.verbose = verbose
        self.cap = cv2.VideoCapture(path_to_video)
        self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.preprocess_landmarks = preprocess_landmarks
        self.preprocess_frames = preprocess_frames

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.frames_list = []
        self.landmarks_list = []
        self.prediction_list = []
        self.recognizer = Recognition(
            encoder=self.encoder,
            decoder=self.decoder,
            frames_list=self.frames_list,
            landmarks_list=self.landmarks_list,
            prediction_list=self.prediction_list,
            verbose=self.verbose,
            video_length=self.video_length,
        )

    def run(self):
        """
        Run the runner.

        """

        print("Start reading video")
        for _ in range(self.video_length):
            ret, frame = self.cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frames_list.append(rgb_frame)
            landmarks = self.preprocess_landmarks(rgb_frame)
            self.landmarks_list.append(landmarks)

        print("Preprocessing frames")
        self.frames_list = self.preprocess_frames(self.frames_list)

        print("Releasing capture")
        self.cap.release()

        print("Starting recognizer")
        return self.recognizer.start()


# runner = Runner(
#     encoder, decoder,
#     path_to_video=(
#         FOLDER_PATH / "fingerspelling_dataset_from_channel_ni_dnya_bez_daktilya"
#         / "round_video_messages" / "file_4@06-06-2025_21-40-36.mp4"
#     )
# )
# runner.run()
