#!/usr/bin/env python3
import whisper
import sounddevice as sd
import numpy as np
import queue
import threading
import tempfile
import os
from scipy.io.wavfile import write
import datetime


# Config
CHUNK_DURATION = 10  # seconds
SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
CHANNELS = 1

EXIT_WORD = "exit"
TRANSCRIPTION_FILE = "/home/sacha/Documents/transcription.txt"

q = queue.Queue()
model = whisper.load_model(
    "small"
)  # you can use "tiny", "small", "medium", "base", "large", "turbo"


def audio_callback(indata, frames, time_info, status):
    q.put(indata.copy())


def announce(text):
    os.system(f"/home/sacha/bin/mac_say.sh '{text}'")


def record_loop():
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback
    ):
        print("🎙️  Listening... Press Ctrl+C to stop.")
        announce("Starting transcription. Say 'exit' to stop.")
        audio_buffer = np.empty((0, CHANNELS), dtype=np.float32)
        try:
            while True:
                audio_buffer = np.append(audio_buffer, q.get(), axis=0)

                if len(audio_buffer) / SAMPLE_RATE >= CHUNK_DURATION:
                    chunk = audio_buffer[: int(CHUNK_DURATION * SAMPLE_RATE)]
                    audio_buffer = audio_buffer[int(CHUNK_DURATION * SAMPLE_RATE) :]

                    threading.Thread(target=transcribe_chunk, args=(chunk,)).start()

        except KeyboardInterrupt:
            announce("Stopping transcription.")
            print("\n🛑 Stopped.")


def transcribe_chunk(chunk):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        write(f.name, SAMPLE_RATE, chunk)

        result = model.transcribe(f.name, fp16=False)
        text = result["text"].lower()
        if text != "":
            print(f"> {text}")
            if EXIT_WORD in text:
                announce("Stopping transcription.")
                print("🛑 Stopping transcription.")
                os._exit(0)

            timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            with open(TRANSCRIPTION_FILE, "a") as file:
                file.write(f"{timestamp}: {text}\n")

        os.remove(f.name)


if __name__ == "__main__":
    record_loop()
