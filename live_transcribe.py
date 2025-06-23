import whisper
import sounddevice as sd
import numpy as np
import queue
import threading
import tempfile
import os
import time
from scipy.io.wavfile import write
import requests
import re


# Config
CHUNK_DURATION = 7  # seconds
SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
CHANNELS = 1

TRIGGER_WORD = "venus"

q = queue.Queue()
model = whisper.load_model(
    "base.en"
)  # you can use "tiny", "small", "medium", "base", "large", "turbo"


def ask_ollama(command_text, model="phi4"):
    prompt = f"""
"{command_text}"
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120
    )

    result = response.json()
    reply = result.get("response", "").strip().lower()
    return reply


def audio_callback(indata, frames, time_info, status):
    q.put(indata.copy())


def record_loop():
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback
    ):
        print("🎙️  Listening... Press Ctrl+C to stop.")
        audio_buffer = np.empty((0, CHANNELS), dtype=np.float32)
        try:
            while True:
                audio_buffer = np.append(audio_buffer, q.get(), axis=0)

                if len(audio_buffer) / SAMPLE_RATE >= CHUNK_DURATION:
                    chunk = audio_buffer[: int(CHUNK_DURATION * SAMPLE_RATE)]
                    audio_buffer = audio_buffer[int(CHUNK_DURATION * SAMPLE_RATE) :]

                    threading.Thread(target=transcribe_chunk, args=(chunk,)).start()

        except KeyboardInterrupt:
            print("\n🛑 Stopped.")


def transcribe_chunk(chunk):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        write(f.name, SAMPLE_RATE, chunk)
        # print(f"\n🔍 Transcribing chunk...")
        result = model.transcribe(f.name, fp16=False)
        text = result['text'].lower()
        if text != "":
            print(f"> {text}")
            if TRIGGER_WORD in text:
                idx = text.find(TRIGGER_WORD)
                command = text[idx + len(TRIGGER_WORD):].lstrip(", ")
                print(f"🧠 Processing: {command}")
                response = ask_ollama(command)
                answer = re.sub(r'\\.', ' ', response)
                # print(f"{answer=}")
                os.system(f"say '{answer}'")

        os.remove(f.name)


if __name__ == "__main__":
    record_loop()
