# Luna Assistant v4

## Overview

Luna v4 builds on v3 and introduces a fully offline wake word system using ONNX models (OpenWakeWord-compatible).

This version adds:

- Offline wake word detection (ONNX)
- Hands-free activation
- Improved VAD gating (minimum duration + RMS)
- Stabilized audio capture loop
- Reduced blank audio loops

No cloud APIs. Fully local. Optimized for NVIDIA Jetson Orin Nano.

---

## What Changed From v3

Wake flow:

Sleeping → Wake Word Detection → "Luna is listening." → VAD Recording → Whisper → LLM → TTS → Return to Sleeping

Wake detection is now separate from conversational VAD.

---

## Wake Word Setup

### 1. Create Directory

```
~/wakeword
```

### 2. Place Required Files

The directory must contain:

```
embedding_model.onnx
melspectrogram.onnx
luna.onnx
```

All three files are required.

---

## Config Updates (config.py)

Set:

```
wake_mode = "openwakeword"
wakeword_model_dir = "~/wakeword"
wakeword_threshold = 0.35
wakeword_block_ms = 80
wakeword_refractory_s = 1.2
awake_window_s = 8.0
```

Environment variables are no longer required for wake word.

---

## VAD Improvements in v4

To prevent blank audio loops and noise-triggered captures:

- Minimum speech duration gate (450ms)
- Minimum RMS gate (0.010)
- Per-call VAD state isolation
- Rejection of short/quiet clips

These prevent:

- Rapid "Listening..." loops
- [BLANK_AUDIO]
- Phantom speech captures

---

## Audio Behavior

When wake word fires:

Luna says:

"Luna is listening."

She then listens for up to `awake_window_s` seconds.

If no valid speech is captured, she returns to sleep.

---

## Running v4

```
python run.py
```

Expected flow:

```
Luna ready. Sleeping...
Listening...
You: ...
Luna: ...
```

---

## Wake Sensitivity Tuning

If wake fires too often:

```
wakeword_threshold = 0.45
```

If wake misses:

```
wakeword_threshold = 0.25
```

---

## Jetson ONNX Warning

If you see:

```
ReadFileContents Failed to open file: "/sys/class/drm/card0/device/vendor"
```

This is ONNX GPU probing and is harmless.

---

Luna v4 enables fully hands-free offline interaction while preserving deterministic memory behavior from v3.
