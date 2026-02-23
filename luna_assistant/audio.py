from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import webrtcvad

# DEBUG VAD
DEBUG_VAD = False  # set to False after tuning
DEBUG_EVERY_N_FRAMES = 25  # 25 * 20ms = ~0.5s when frame_ms=20


class AudioIO:
    def __init__(self, audio_in: str, audio_out: str):
        self.audio_in = audio_in
        self.audio_out = audio_out

    def record_until_vad_end(
        self,
        out_wav: str,
        rate: int = 16000,
        channels: int = 1,
        frame_ms: int = 20,
        vad_mode: int = 2,              # 0..3 (3 = most aggressive)
        start_trigger_ms: int = 200,    # speech needed to "start"
        end_trigger_ms: int = 450,      # silence needed to "end" after speech started
        max_seconds: float = 20.0,      # hard cap
        pre_roll_ms: int = 200,         # keep a bit before speech starts
        min_speech_ms: int = 350,       # reject clips shorter than this
        min_rms: float = 0.008,         # reject clips quieter than this (0..1)
    ) -> bool:
        """
        Records from ALSA and uses WebRTC VAD to stop shortly after speech ends.
        Returns True if speech was captured, False if no speech was detected.
        Output is a 16kHz mono WAV suitable for whisper.cpp.
        """
        if rate != 16000:
            raise ValueError("This VAD setup expects 16kHz audio (rate=16000).")
        if channels != 1:
            raise ValueError("This VAD setup expects mono audio (channels=1).")
        if frame_ms not in (10, 20, 30):
            raise ValueError("frame_ms must be 10, 20, or 30 for webrtcvad.")

        Path(out_wav).parent.mkdir(parents=True, exist_ok=True)

        vad = webrtcvad.Vad(vad_mode)

        dbg_frame = 0  # DEBUG_VAD local counter (per call)

        bytes_per_sample = 2  # S16_LE
        frame_samples = int(rate * frame_ms / 1000)
        frame_bytes = frame_samples * bytes_per_sample

        start_trigger_frames = max(1, start_trigger_ms // frame_ms)
        end_trigger_frames = max(1, end_trigger_ms // frame_ms)
        pre_roll_frames = max(0, pre_roll_ms // frame_ms)

        arecord_cmd = [
            "arecord",
            "-D", self.audio_in,
            "-f", "S16_LE",
            "-r", str(rate),
            "-c", str(channels),
            "-t", "raw",
        ]

        p = subprocess.Popen(arecord_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        assert p.stdout is not None

        ring: List[bytes] = []
        captured: List[bytes] = []
        speech_started = False
        speech_frames = 0
        silence_frames = 0

        t0 = time.time()
        try:
            while True:
                if (time.time() - t0) > max_seconds:
                    break

                chunk = p.stdout.read(frame_bytes)
                if not chunk or len(chunk) < frame_bytes:
                    break

                ring.append(chunk)
                if len(ring) > max(pre_roll_frames, 1):
                    ring.pop(0)

                is_speech = vad.is_speech(chunk, rate)

                if DEBUG_VAD:
                    # RMS in float32 for quick signal sanity check
                    import numpy as _np
                    _i16 = _np.frombuffer(chunk, dtype=_np.int16).astype(_np.float32)
                    _rms = float(_np.sqrt(_np.mean((_i16 / 32768.0) ** 2) + 1e-12))
                    # Print occasionally to avoid spam
                    dbg_frame += 1
                    if (dbg_frame % DEBUG_EVERY_N_FRAMES) == 0:
                        print(
                            f"VAD dbg rms={_rms:.4f} is_speech={int(is_speech)} "
                            f"started={int(speech_started)} speech_frames={speech_frames} silence_frames={silence_frames}"
                        )


                if not speech_started:
                    if is_speech:
                        speech_frames += 1
                    else:
                        speech_frames = max(0, speech_frames - 1)  # small decay

                    if speech_frames >= start_trigger_frames:
                        speech_started = True
                        # include pre-roll so we don't clip the first syllable
                        captured.extend(ring)
                        ring.clear()
                        silence_frames = 0
                else:
                    captured.append(chunk)
                    if is_speech:
                        silence_frames = 0
                    else:
                        silence_frames += 1
                        if silence_frames >= end_trigger_frames:
                            break
        finally:
            p.kill()
            p.wait()

        if not captured:
            if DEBUG_VAD:
                print("VAD dbg result: no speech captured (returning False).")
            return False

        # Convert to int16
        pcm = b"".join(captured)
        audio_i16 = np.frombuffer(pcm, dtype=np.int16)

        # Reject tiny/quiet clips (prevents BLANK_AUDIO loops)
        dur_ms = (len(audio_i16) / float(rate)) * 1000.0
        rms = float(np.sqrt(np.mean((audio_i16.astype(np.float32) / 32768.0) ** 2) + 1e-12))
        if dur_ms < float(min_speech_ms) or rms < float(min_rms):
            if DEBUG_VAD:
                print(f"VAD dbg reject: dur_ms={dur_ms:.0f} rms={rms:.4f} (too short/quiet)")
            return False

        # Write WAV
        sf.write(out_wav, audio_i16, rate, subtype="PCM_16")
        if DEBUG_VAD:
            import os as _os
            try:
                _size = _os.path.getsize(out_wav)
            except OSError:
                _size = -1
            print(f"VAD dbg result: captured frames={len(captured)} wav_bytes={_size}")
        return True

    def play_wav(self, wav_path: str) -> None:
        cmd = ["aplay", "-D", self.audio_out, wav_path]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
