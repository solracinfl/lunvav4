from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass


@dataclass
class WhisperCppASR:
    whisper_bin: str
    model_path: str
    threads: int = 4
    processors: int = 1
    beam_size: int = 1
    best_of: int = 1
    use_gpu: bool = True
    flash_attn: bool = True
    language: str = "en"  # force language to skip auto-detect latency

    def warmup(self) -> None:
        """Best-effort: run a tiny silent decode so first real turn is faster."""
        try:
            wav = self._make_silence_wav(seconds=0.3, rate=16000)
            _ = self.transcribe(wav, warm=True)
        except Exception:
            pass

    def _make_silence_wav(self, seconds: float, rate: int) -> str:
        import wave
        import struct

        nframes = int(seconds * rate)
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="whisper_warm_")
        os.close(fd)

        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(rate)
            silence = struct.pack("<h", 0) * nframes
            wf.writeframes(silence)

        return path

    def transcribe(self, wav_path: str, warm: bool = False) -> str:
        # Debug: detect empty/near-empty recordings that cause [BLANK_AUDIO]
        try:
            size = os.path.getsize(wav_path)
        except OSError:
            size = -1
        # print(f"DEBUG wav bytes={size} path={wav_path}")

        cmd = [
            self.whisper_bin,
            "-m", self.model_path,
            "-f", wav_path,
            "--no-timestamps",
            "-l", self.language,
            "-t", str(self.threads),
            "-p", str(self.processors),
            "-bs", str(self.beam_size),
            "-bo", str(self.best_of),
        ]

        if not self.use_gpu:
            cmd.append("-ng")
        if not self.flash_attn:
            cmd.append("-nfa")

        p = subprocess.run(cmd, check=True, capture_output=True, text=True)

        # whisper-cli often prints transcript to stdout; we want only the final non-empty line.
        last = ""
        for line in p.stdout.splitlines():
            s = line.strip()
            if s:
                last = s

        return last
