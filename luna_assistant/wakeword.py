from __future__ import annotations

import os
import time
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openwakeword.model import Model


@dataclass
class OpenWakeWordConfig:
    audio_in: str
    model_dir: str
    threshold: float = 0.35
    block_ms: int = 80
    refractory_s: float = 1.2


class WakeWord:
    def __init__(self, mode: str, cmd: str = "", *, oww: OpenWakeWordConfig | None = None):
        self.mode = mode
        self.cmd = cmd.strip()
        self.oww = oww

    def wait(self) -> bool:
        """
        Returns:
          True  -> wake triggered
          False -> user requested quit (keyboard mode only)
        """
        if self.mode == "keyboard":
            s = input("Press Enter to talk (q then Enter to quit): ").strip().lower()
            return s != "q"

        if self.mode == "cmd":
            if not self.cmd:
                raise ValueError("WAKE_MODE=cmd requires WAKEWORD_CMD to be set")
            while True:
                rc = subprocess.call(self.cmd, shell=True)
                if rc == 0:
                    return True

        if self.mode == "openwakeword":
            if not self.oww:
                raise ValueError("WAKE_MODE=openwakeword requires OpenWakeWordConfig")
            return self._wait_openwakeword(self.oww)

        raise ValueError(f"Unknown WAKE_MODE: {self.mode}")

    def _wait_openwakeword(self, cfg: OpenWakeWordConfig) -> bool:
        model_dir = Path(os.path.expanduser(cfg.model_dir))
        wake_model = model_dir / "luna.onnx"
        embed_model = model_dir / "embedding_model.onnx"
        mels_model = model_dir / "melspectrogram.onnx"

        for f in (wake_model, embed_model, mels_model):
            if not f.exists():
                raise FileNotFoundError(f"Missing wakeword model: {f}")

        mdl = Model(
            wakeword_models=[str(wake_model)],
            inference_framework="onnx",
            embedding_model_path=str(embed_model),
            melspec_model_path=str(mels_model),
        )

        rate = 16000
        bytes_per_sample = 2  # S16_LE
        block_samples = int(rate * (cfg.block_ms / 1000.0))
        block_bytes = block_samples * bytes_per_sample

        arecord_cmd = [
            "arecord",
            "-D", cfg.audio_in,
            "-f", "S16_LE",
            "-r", str(rate),
            "-c", "1",
            "-t", "raw",
        ]

        last_fire = 0.0
        p = subprocess.Popen(arecord_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        assert p.stdout is not None

        try:
            while True:
                buf = p.stdout.read(block_bytes)
                if not buf or len(buf) < block_bytes:
                    continue

                x = np.frombuffer(buf, dtype=np.int16)
                preds = mdl.predict(x)
                score = float(preds.get("luna", 0.0)) if preds else 0.0

                now = time.time()
                if score >= cfg.threshold and (now - last_fire) >= cfg.refractory_s:
                    last_fire = now
                    return True
        finally:
            try:
                p.kill()
            except Exception:
                pass
            try:
                p.wait(timeout=1)
            except Exception:
                pass
