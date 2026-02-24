import time
import numpy as np
import sounddevice as sd
from openwakeword.model import Model
from scipy.signal import resample_poly

MODEL_DIR = "/home/luna/wakeword"
WAKE_MODEL = f"{MODEL_DIR}/luna.onnx"
EMBED_MODEL = f"{MODEL_DIR}/embedding_model.onnx"
MELSPEC_MODEL = f"{MODEL_DIR}/melspectrogram.onnx"

CAPTURE_RATE = 48000
TARGET_RATE = 16000

BLOCK_MS = 80
CAPTURE_SAMPLES = int(CAPTURE_RATE * (BLOCK_MS / 1000.0))

THRESH = 0.15
REFRACTORY_S = 1.0
PRINT_EVERY_S = 0.5

PREFERRED_MATCH = ["AIRHUG 21", "USB Audio (hw:0,0)"]

def pick_input_device() -> int:
    devices = sd.query_devices()
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) <= 0:
            continue
        name = d.get("name", "")
        for token in PREFERRED_MATCH:
            if token.lower() in name.lower():
                return idx
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            return idx
    raise RuntimeError("No input devices available.")

def main():
    in_dev = pick_input_device()
    print("Using input device index:", in_dev, "name:", sd.query_devices()[in_dev]["name"])
    print("Capture rate:", CAPTURE_RATE, "Target rate:", TARGET_RATE)
    print("Loading wake model:", WAKE_MODEL)

    mdl = Model(
        wakeword_models=[WAKE_MODEL],
        inference_framework="onnx",
        embedding_model_path=EMBED_MODEL,
        melspec_model_path=MELSPEC_MODEL,
    )

    print("Listening... say 'luna' near the mic. Ctrl+C to stop.")
    last_fire = 0.0
    last_print = 0.0
    max_score = 0.0

    def callback(indata, frames, time_info, status):
        nonlocal last_fire, last_print, max_score

        xf = indata[:, 0].astype(np.float32)
        rms = float(np.sqrt(np.mean(xf * xf)) + 1e-12)

        # downsample 48k -> 16k via polyphase (up=1, down=3)
        y = resample_poly(xf, up=1, down=3).astype(np.float32)
        if y.size == 0:
            return

        x = np.clip(y * 32768.0, -32768.0, 32767.0).astype(np.int16)

        preds = mdl.predict(x)
        score = float(preds.get("luna", 0.0)) if preds else 0.0
        max_score = max(max_score, score)

        now = time.time()
        if (now - last_print) >= PRINT_EVERY_S:
            last_print = now
            print(f"score={score:.3f} max={max_score:.3f} rms={rms:.4f}")

        if score >= THRESH and (now - last_fire) > REFRACTORY_S:
            last_fire = now
            print(f"WAKEWORD FIRED: luna (score={score:.3f}, rms={rms:.4f})")

    with sd.InputStream(
        channels=1,
        samplerate=CAPTURE_RATE,
        blocksize=CAPTURE_SAMPLES,
        dtype="float32",
        callback=callback,
        device=in_dev,
    ):
        while True:
            time.sleep(0.25)

if __name__ == "__main__":
    main()
