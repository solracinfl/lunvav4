from __future__ import annotations

import os
import time

from luna_assistant.audio import AudioIO
from luna_assistant.asr_whispercpp import WhisperCppASR
from luna_assistant.config import Config
from luna_assistant.llm_ollama import OllamaLLM
from luna_assistant.memory_store import MemoryStore, TurnRecord
from luna_assistant.tts_piper import PiperTTS
from luna_assistant.wakeword import WakeWord, OpenWakeWordConfig


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
    rest = rest.lstrip(" \t\r\n,.:;!?-—")
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

        # Awake window: 5 minutes
        self.awake_window_s = 300.0

        # temp wavs
        self._rec_wav = "/tmp/luna_last.wav"
        self._tts_wav = "/tmp/luna_tts.wav"

    def _build_pinned_context(self) -> str:
        """
        Minimal, deterministic memory injection:
        Always include pinned facts as context for the LLM.
        """
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

    def run(self):
        print("Luna ready. Sleeping...")

        while True:
            woke = self.wake.wait()
            if not woke:
                print("Bye.")
                return

            self.speak("Luna is listening.")
            deadline = time.time() + self.awake_window_s
            first_command_free = True

            while time.time() < deadline:
                print("Listening...")  # requested: show listening whenever actively recording

                ok = self.audio.record_until_vad_end(
                    out_wav=self._rec_wav,
                    vad_mode=self.cfg.vad_mode,
                    start_trigger_ms=self.cfg.vad_start_trigger_ms,
                    end_trigger_ms=self.cfg.vad_end_trigger_ms,
                    max_seconds=self.cfg.vad_max_seconds,
                    pre_roll_ms=self.cfg.vad_pre_roll_ms,
                    min_speech_ms=450,
                    min_rms=0.010,
                )
                if not ok:
                    continue

                raw_text = self.asr.transcribe(self._rec_wav)
                raw_text = (raw_text or "").strip()
                if not raw_text:
                    continue

                # After a wakeword event, accept the first utterance as a command even if
                # it doesn't start with "luna". After that, require the prefix.
                if first_command_free:
                    stripped = raw_text
                    first_command_free = False
                else:
                    stripped = _strip_wake_phrase(raw_text, self.cfg.wake_phrase)
                    if not stripped:
                        continue

                # Normalize leading punctuation from STT (e.g., "luna, ...")
                stripped = stripped.lstrip(" \t\r\n,.:;!?-")
                if not stripped:
                    continue

                print(f"You: {stripped}")

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
                    continue

                # LLM with pinned memory injection (fixes birthday/location/etc)
                extra_ctx = self._build_pinned_context()
                reply = self.llm.chat(stripped, extra_context=extra_ctx)

                print(f"Luna: {reply}")
                self.speak(reply)

                try:
                    self.store.add_turn(TurnRecord(user=stripped, assistant=reply))
                except Exception:
                    pass

            print("No valid 'luna ...' command for 5 minutes. Sleeping...")