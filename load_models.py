import whisper
import queue

q = queue.Queue()

for x in ["tiny", "small.en", "medium.en", "base.en", "large", "turbo"]:
    print(f"loading {x}")
    model = whisper.load_model(x)
