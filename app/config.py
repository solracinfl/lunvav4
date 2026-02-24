from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    v = os.environ.get(name, "").strip()
    return v if v else default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "y", "on")


@dataclass(frozen=True)
class Config:
    # ALSA devices
    audio_in: str = _env("AUDIO_IN", "plughw:CARD=A21,DEV=0")
    audio_out: str = _env("AUDIO_OUT", "plughw:CARD=Bar,DEV=0")

    # Paths
    whisper_bin: str = _env("WHISPER_BIN", os.path.expanduser("~/whisper.cpp/build/bin/whisper-cli"))
    whisper_model: str = _env("WHISPER_MODEL", os.path.expanduser("~/whisper.cpp/models/ggml-base.en.bin"))
    piper_model: str = _env("PIPER_MODEL", os.path.expanduser("~/voice/models/en_US-amy-medium.onnx"))
    piper_venv: str = _env("PIPER_VENV", os.path.expanduser("~/voice/.venv"))
    memory_db_path: str = "data/luna.db"
    memory_pinned_limit: int = 12

    # Ollama
    ollama_url: str = _env("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
    ollama_model: str = _env("OLLAMA_MODEL", "luna")
    ollama_keep_alive: str = _env("OLLAMA_KEEP_ALIVE", "10m")
    ollama_timeout_s: int = int(_env("OLLAMA_TIMEOUT_S", "120"))

    # Assistant behavior
    system_prompt: str = _env(
        "SYSTEM_PROMPT",
        ""
    )
    wake_phrase: str = _env("WAKE_PHRASE", "luna")

    # Whisper speed knobs
    whisper_threads: int = int(_env("WHISPER_THREADS", "6"))
    whisper_processors: int = int(_env("WHISPER_PROCESSORS", "1"))
    whisper_beam_size: int = int(_env("WHISPER_BEAM_SIZE", "1"))
    whisper_best_of: int = int(_env("WHISPER_BEST_OF", "1"))
    whisper_use_gpu: bool = _env_bool("WHISPER_USE_GPU", True)
    whisper_flash_attn: bool = _env_bool("WHISPER_FLASH_ATTN", True)

    # VAD knobs (latency vs completeness)
    vad_mode: int = int(_env("VAD_MODE", "1"))
    vad_start_trigger_ms: int = int(_env("VAD_START_MS", "200"))
    vad_end_trigger_ms: int = int(_env("VAD_END_MS", "700"))
    vad_pre_roll_ms: int = int(_env("VAD_PRE_ROLL_MS", "500"))
    vad_max_seconds: float = float(_env("VAD_MAX_SECONDS", "20.0"))

    # Warmup
    warmup_enable: bool = _env_bool("WARMUP", True)

    # Wake word mode
    # keyboard: press Enter to trigger
    # cmd: run WAKEWORD_CMD and wait for exit code 0 to trigger
    wake_mode: str = _env("WAKE_MODE", "keyboard")
    wakeword_cmd: str = _env("WAKEWORD_CMD", "")

    # openwakeword (ONNX) wake mode
    # Models live in: ~/wakeword (embedding_model.onnx, melspectrogram.onnx, luna.onnx)
    wakeword_model_dir: str = _env("WAKEWORD_MODEL_DIR", os.path.expanduser("~/wakeword"))
    wakeword_threshold: float = float(_env("WAKEWORD_THRESHOLD", "0.35"))
    wakeword_block_ms: int = int(_env("WAKEWORD_BLOCK_MS", "80"))  # audio chunk size for detector
    wakeword_refractory_s: float = float(_env("WAKEWORD_REFRACTORY_S", "1.2"))

    # After waking, stay awake this long waiting for a command that starts with "luna"
    awake_window_s: float = float(_env("AWAKE_WINDOW_S", "8.0"))
