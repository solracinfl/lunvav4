from __future__ import annotations

import os
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PiperTTS:
    model_path: str
    audio_out: str
    venv_path: Optional[str] = None
    prefer_inproc: bool = True

    def __post_init__(self) -> None:
        os.environ.setdefault("ORT_DISABLE_DEVICE_DISCOVERY", "1")
        self._voice = None
        if self.prefer_inproc:
            self._try_load_inproc()

    def _try_load_inproc(self) -> None:
        try:
            from piper.voice import PiperVoice  # type: ignore
            self._voice = PiperVoice.load(self.model_path)
        except Exception:
            self._voice = None

    def warmup(self) -> None:
        try:
            self.synth_to_wav("Ready.", "/tmp/piper_warm.wav")
        except Exception:
            pass

    def synth_to_wav(self, text: str, out_wav: str) -> None:
        Path(out_wav).parent.mkdir(parents=True, exist_ok=True)

        if self._voice is not None:
            with wave.open(out_wav, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._voice.config.sample_rate)
                self._voice.synthesize(text, wf)
            return

        if not self.venv_path:
            raise RuntimeError("Piper in-proc unavailable and PIPER_VENV not configured")

        env = os.environ.copy()
        env["ORT_DISABLE_DEVICE_DISCOVERY"] = "1"

        python_bin = os.path.join(self.venv_path, "bin", "python")
        cmd = [
            python_bin,
            "-m", "piper",
            "--model", self.model_path,
            "--output_file", out_wav,
        ]
        p = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        assert p.stdin is not None
        p.stdin.write(text)
        p.stdin.write("\n")
        p.stdin.close()
        rc = p.wait()
        if rc != 0:
            err = (p.stderr.read() if p.stderr else "") or ""
            raise RuntimeError(f"piper failed rc={rc}: {err[:2000]}")
