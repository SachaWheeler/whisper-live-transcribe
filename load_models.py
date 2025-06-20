import whisper
import queue

q = queue.Queue()

for x in ["tiny", "small", "medium", "base", "large", "turbo"]:
    print(f"loading {x}")
    model = whisper.load_model(x)
