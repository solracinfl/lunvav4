from __future__ import annotations

import os
import time

from app.audio import AudioIO
from app.asr_whispercpp import WhisperCppASR
from app.config import Config
from app.llm_ollama import OllamaLLM
from app.memory_store import MemoryStore, TurnRecord
from app.tts_piper import PiperTTS
from app.wakeword import WakeWord, OpenWakeWordConfig


def _strip_wake_phrase(text: str, wake_phrase: str) -> str | None:
    t = (text or "").strip()
    if not t:
        return None

    wp = (wake_phrase or "").strip().lower()
    if not wp:
        return t

    tl = t.lower()

    # Allow: "luna ..." and "luna, ..." and "luna: ..."
    if not tl.startswith(wp):
        return None

    rest = t[len(wp):].strip()
    # Remove leading punctuation/noise after wake word
    rest = rest.lstrip(" \t\r\n,.:;!?-")
    return rest or None


class LunaAssistant:
    def __init__(self):
        # Reduce ORT probing noise across the whole process
        os.environ.setdefault("ORT_DISABLE_DEVICE_DISCOVERY", "1")

        self.cfg = Config()

        # Audio IO (ALSA)
        self.audio = AudioIO(self.cfg.audio_in, self.cfg.audio_out)

        # ASR (whisper.cpp)
        self.asr = WhisperCppASR(
            whisper_bin=self.cfg.whisper_bin,
            model_path=self.cfg.whisper_model,
            threads=self.cfg.whisper_threads,
            processors=self.cfg.whisper_processors,
            beam_size=self.cfg.whisper_beam_size,
            best_of=self.cfg.whisper_best_of,
            use_gpu=self.cfg.whisper_use_gpu,
            flash_attn=self.cfg.whisper_flash_attn,
            language="en",
        )
        if self.cfg.warmup_enable:
            self.asr.warmup()

        # TTS (Piper)
        self.tts = PiperTTS(
            model_path=self.cfg.piper_model,
            audio_out=self.cfg.audio_out,
            venv_path=self.cfg.piper_venv,
            prefer_inproc=True,
        )
        if self.cfg.warmup_enable:
            self.tts.warmup()

        # LLM (Ollama)
        self.llm = OllamaLLM(
            url=self.cfg.ollama_url,
            model=self.cfg.ollama_model,
            system_prompt=self.cfg.system_prompt,
            keep_alive=self.cfg.ollama_keep_alive,
            timeout_s=self.cfg.ollama_timeout_s,
        )
        if self.cfg.warmup_enable:
            self.llm.warmup()

        # Memory
        self.store = MemoryStore(self.cfg.memory_db_path)

        # Wakeword
        oww_cfg = None
        if self.cfg.wake_mode == "openwakeword":
            oww_cfg = OpenWakeWordConfig(
                audio_in=self.cfg.audio_in,
                model_dir=self.cfg.wakeword_model_dir,
                threshold=self.cfg.wakeword_threshold,
                block_ms=self.cfg.wakeword_block_ms,
                refractory_s=self.cfg.wakeword_refractory_s,
            )
        self.wake = WakeWord(self.cfg.wake_mode, self.cfg.wakeword_cmd, oww=oww_cfg)

        # IMPORTANT: You explicitly want 5 minutes idle timeout unless you intentionally change it.
        # Treat low/invalid config values as misconfig and clamp to 300s.
        try:
            v = float(getattr(self.cfg, "awake_window_s", 300.0) or 300.0)
        except Exception:
            v = 300.0
        self.awake_window_s = v if v >= 60.0 else 300.0  # clamp anything <60s to 5 minutes

        # temp wavs
        self._rec_wav = "/tmp/luna_last.wav"
        self._tts_wav = "/tmp/luna_tts.wav"

    def _build_pinned_context(self) -> str:
        """Deterministic memory injection (pinned only)."""
        pinned = self.store.get_pinned_memories(limit=50)
        if not pinned:
            return ""

        lines = ["Pinned user facts (trusted):"]
        for k, v, score, pinned_flag, created_at in pinned:
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)

    def speak(self, text: str):
        self.tts.synth_to_wav(text, self._tts_wav)
        self.audio.play_wav(self._tts_wav)
        # Guard against recording our own TTS (speaker bleed) immediately after playback
        time.sleep(0.6)

    def _looks_like_self_tts(self, text: str) -> bool:
        t = (text or "").strip().lower()
        if not t:
            return True
        if "luna is listening" in t:
            return True
        return False

    def run(self):
        print("Luna ready. Sleeping...")

        wake_phrase = (self.cfg.wake_phrase or "").strip().lower()

        while True:
            woke = self.wake.wait()
            if not woke:
                print("Bye.")
                return

            self.speak("Luna is listening.")

            # Idle timer is based ONLY on commands that start with the wake phrase ("luna ...").
            # Use monotonic time to avoid clock jumps.
            last_luna_cmd_ts = time.monotonic()

            # Convenience: allow the first 2 commands after wake without requiring the prefix,
            # but they DO NOT reset the 5-minute sleep timer unless they actually start with "luna".
            free_commands_left = 2

            while (time.monotonic() - last_luna_cmd_ts) < self.awake_window_s:
                print("Listening...")

                ok = self.audio.record_until_vad_end(
                    out_wav=self._rec_wav,
                    vad_mode=self.cfg.vad_mode,
                    start_trigger_ms=self.cfg.vad_start_trigger_ms,
                    end_trigger_ms=self.cfg.vad_end_trigger_ms,
                    max_seconds=self.cfg.vad_max_seconds,
                    pre_roll_ms=self.cfg.vad_pre_roll_ms,
                    discard_ms=700,
                    min_speech_ms=300,
                    min_rms=0.006,
                )
                if not ok:
                    continue

                raw_text = (self.asr.transcribe(self._rec_wav) or "").strip()
                if not raw_text:
                    continue

                if self._looks_like_self_tts(raw_text):
                    continue

                raw_l = raw_text.strip().lower()
                starts_with_luna = bool(wake_phrase) and raw_l.startswith(wake_phrase)

                # Determine the command text to send to the LLM.
                if free_commands_left > 0:
                    # Accept as-is (no prefix requirement) for the first couple of turns.
                    stripped = raw_text
                    free_commands_left -= 1
                else:
                    # After that, require the wake phrase.
                    stripped = _strip_wake_phrase(raw_text, self.cfg.wake_phrase)
                    if not stripped:
                        continue
                    starts_with_luna = True  # by construction

                stripped = stripped.lstrip(" \t\r\n,.:;!?-")
                if not stripped:
                    continue

                print(f"You: {stripped}")

                # Reset sleep timer ONLY if the user actually said a "luna ..." command.
                if starts_with_luna:
                    last_luna_cmd_ts = time.monotonic()

                # Local memory command
                if stripped.strip().lower() in {"list memories", "memories", "show memories"}:
                    pinned = self.store.get_pinned_memories(limit=30)
                    recent = self.store.get_memories(limit=30)
                    if not pinned and not recent:
                        self.speak("No memories stored yet.")
                        continue

                    lines = []
                    if pinned:
                        lines.append("Pinned memories:")
                        for k, v, score, pinned_flag, created_at in pinned:
                            lines.append(f"- {k}: {v}")
                    if recent:
                        lines.append("Recent memories:")
                        for k, v, score, pinned_flag, created_at in recent:
                            if pinned_flag:
                                continue
                            lines.append(f"- {k}: {v}")

                    self.speak("\n".join(lines))
                    # Speaking can take time; do NOT affect idle timer unless the input was "luna ...".
                    continue

                extra_ctx = self._build_pinned_context()
                reply = self.llm.chat(stripped, extra_context=extra_ctx)

                print(f"Luna: {reply}")
                self.speak(reply)

                try:
                    self.store.add_turn(TurnRecord(user=stripped, assistant=reply))
                except Exception:
                    pass

            print("No 'luna ...' commands for 5 minutes. Sleeping...")
