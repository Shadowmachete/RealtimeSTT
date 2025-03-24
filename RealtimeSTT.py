"""Realtime Text-To-Speech Application"""

from typing import Any, List, Tuple, cast
from faster_whisper import WhisperModel
import numpy as np
from numpy._typing import NDArray
import sounddevice as sd
import threading
import logging
import keyboard
import os
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
from transformers import logging as transformers_logging
import warnings
import time
from termcolor import colored
from silero_vad import load_silero_vad, get_speech_timestamps
# import pyautogui
# from colorama import Fore, init, Style

# init(autoreset=True)

formatter: logging.Formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)-8s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

console_handler: logging.Handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
logger.addHandler(console_handler)

logging.getLogger("faster_whisper").propagate = False

transformers_logging.set_verbosity_error()

warnings.filterwarnings(
    "ignore", message=".*Torch was not compiled with flash attention.*"
)

# create a audio stream
# keep reading audio until the user quits
# send 30ms - 50ms clips to silero
# while silero detects voice, save it to a buffer
# once voice not detected transcribe the buffer while still running the audio buffer thread
# print out the transcription and continue


class TTSHandler:
    def __init__(
        self,
        model_size: str = "large-v3",
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        audio_len: int = 2,
    ) -> None:
        self.model_size: str = model_size
        self.model, self.processor, self.device = self.whisper_setup()
        self.silero_vad = load_silero_vad()
        self.buffer: NDArray[np.float16] = np.zeros((0,), dtype=np.float16)
        self.sample_rate: int = sample_rate
        self.chunk_size: int = chunk_size
        self.audio_len: int = audio_len

        self.stop_event: threading.Event = threading.Event()
        self.recording_thread: threading.Thread = threading.Thread(
            target=self.rec_audio, daemon=True
        )
        self.transcription_thread: threading.Thread = threading.Thread(
            target=self.transcribe_audio, daemon=True
        )
        # self.clear_chat_thread: threading.Thread = threading.Thread(
        #     target=self.clear_chat, daemon=True
        # )

        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="english", task="transcribe"
        )

    def whisper_setup(self) -> Tuple[PreTrainedModel, WhisperProcessor, torch.device]:
        logger.info("Loading whisper model")
        # model: WhisperModel = WhisperModel(
        #     self.model_size, device="cuda", compute_type="float16"
        # )
        device: Any = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-base"
        ).to(device)
        model = cast(PreTrainedModel, model)

        processor = WhisperProcessor.from_pretrained(
            "openai/whisper-base", task="transcribe"
        )
        processor = cast(WhisperProcessor, processor)
        return model, processor, device

    # def transcribe_audio(self, audio_data: NDArray[np.float16]) -> None:
    #     """
    #     Perform transcription on the given audio data.
    #     """
    #     logger.info("Transcribing...")
    #     # segments, _ = self.model.transcribe(audio_data)
    #     # transcription: str = "".join(segment.text for segment in segments)
    #
    #     inputs: BatchFeature = self.processor(
    #         audio_data,
    #         sampling_rate=self.sample_rate,
    #         return_tensors="pt",
    #     ).to(self.device)
    #     input_features: torch.Tensor = inputs["input_features"]
    #
    #     generated_ids = self.model.generate(input_features)
    #     transcription: List[str] = self.processor.batch_decode(
    #         generated_ids, skip_special_tokens=True, language="en"
    #     )
    #     logger.info("Transcription: %s", transcription[0]) if transcription else None

    def transcribe_audio(self) -> None:
        """
        Perform transcription on the given audio data.
        """
        logger.info("Transcription thread started...")
        while not self.stop_event.is_set():
            if len(self.buffer) < self.sample_rate * self.audio_len:
                continue

            logger.info("Transcribing...")
            buffer_copy: NDArray[np.float16] = self.buffer.copy()
            self.buffer = np.zeros((0,), dtype=np.float16)
            if not self.is_speech(buffer_copy):
                continue

            print(" " * 14, end="\r")
            print("Transcribing...")
            # segments, _ = self.model.transcribe(audio_data)
            # transcription: str = "".join(segment.text for segment in segments)

            inputs: BatchFeature = self.processor(
                buffer_copy,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            ).to(self.device)
            input_features: torch.Tensor = inputs["input_features"]

            generated_ids = self.model.generate(
                input_features, forced_decoder_ids=self.forced_decoder_ids
            )
            transcription: List[str] = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            if transcription:
                # pyautogui.write(transcription[0].strip(), interval=0.01)
                print(" " * 14, end="\r")
                print(transcription[0].strip())
            # logger.info(
            #     "Transcription: %s", transcription[0]
            # ) if transcription else None

    def is_speech(self, audio: NDArray[np.float16]) -> bool:
        speech_timestamps = get_speech_timestamps(
            torch.tensor(audio), self.silero_vad, return_seconds=True
        )
        return bool(speech_timestamps)

    def audio_callback(self, indata, frames, time, status) -> None:
        self.buffer = np.concatenate((self.buffer, indata[:, 0]))
        # if len(self.buffer) > self.sample_rate * 10:
        #     self.buffer = self.buffer[-self.sample_rate * 10 :]

    def rec_audio(self) -> None:
        with sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size * 2,
        ):
            logger.info("Recording started...")
            print("Speak now...")
            while not self.stop_event.is_set():
                print(colored("\\", "green", on_color=None) + " Recording...", end="\r")
                time.sleep(0.2)
                print(colored("-", "green", on_color=None) + " Recording...", end="\r")
                time.sleep(0.2)
                print(colored("/", "green", on_color=None) + " Recording...", end="\r")
                time.sleep(0.2)
                print(colored("|", "green", on_color=None) + " Recording...", end="\r")
                time.sleep(0.2)
                # print(Fore.GREEN + "\\" + Style.RESET_ALL + " Recording...", end="\r")
                # time.sleep(0.2)
                # print(Fore.GREEN + "-" + Style.RESET_ALL + " Recording...", end="\r")
                # time.sleep(0.2)
                # print(Fore.GREEN + "/" + Style.RESET_ALL + " Recording...", end="\r")
                # time.sleep(0.2)
                # print(Fore.GREEN + "|" + Style.RESET_ALL + " Recording...", end="\r")
                # time.sleep(0.2)
                # print("\033[32m\\\033[0m Recording...", end="\r")
                # time.sleep(0.2)
                # print("\033[32m-\033[0m Recording...", end="\r")
                # time.sleep(0.2)
                # print("\033[32m/\033[0m Recording...", end="\r")
                # time.sleep(0.2)
                # print("\033[32m|\033[0m Recording...", end="\r")
                # time.sleep(0.2)

    # def check_transcription_condition(self) -> None:
    #     logger.info("Transcription thread started...")
    #     while not self.stop_event.is_set():
    #         buffer_copy: NDArray[np.float16] = np.zeros((0,), dtype=np.float16)
    #         while (
    #             len(self.buffer) < self.sample_rate * 5 and not self.stop_event.is_set()
    #         ):
    #             continue
    #
    #         if self.stop_event.is_set():
    #             break
    #
    #         buffer_copy = self.buffer[: self.sample_rate * 5].copy()
    #         self.buffer = self.buffer[self.sample_rate * 5 :]
    #
    #         self.transcribe_audio(buffer_copy)

    def clear_chat(self) -> None:
        while not self.stop_event.is_set():
            if keyboard.is_pressed("c"):
                os.system("cls" if os.name == "nt" else "clear")

    def start_threads(self) -> None:
        self.recording_thread.start()
        self.transcription_thread.start()
        # self.clear_chat_thread.start()

    def stop_threads(self) -> None:
        self.stop_event.set()
        self.recording_thread.join()
        self.transcription_thread.join()
        # self.clear_chat_thread.join()


if __name__ == "__main__":
    interval_len: int = 5
    print(f"Realtime speech to text with a {interval_len}s interval.")
    print("Wait until it says 'Speak now...'")
    handler: TTSHandler = TTSHandler(audio_len=interval_len)
    handler.start_threads()
    try:
        keyboard.wait("esc")
    except KeyboardInterrupt:
        logger.info("Exiting by interrupt")
    finally:
        logger.info("Stopping threads...")
        handler.stop_threads()
        print("\nStopping...")
