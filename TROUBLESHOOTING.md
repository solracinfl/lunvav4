# Luna Assistant v4 – Troubleshooting

## Wake Word Not Firing

Verify wakeword directory:

```
ls ~/wakeword
```

Must contain:

- embedding_model.onnx
- melspectrogram.onnx
- luna.onnx

Test microphone:

```
arecord -D plughw:CARD=A21,DEV=0 -f S16_LE -r 16000 test.wav
aplay test.wav
```

If RMS is near 0.0000, increase mic gain.

---

## Wake Fires Too Often

Increase:

```
wakeword_threshold = 0.45
```

If false triggers continue:

```
wakeword_refractory_s = 2.0
```

---

## Blank Audio or Rapid Listening Loop

Cause:
Tiny or noise-only clips passing VAD.

Fix:
v4 adds minimum duration and RMS gating.

If still occurring, increase:

```
min_speech_ms = 600
min_rms = 0.015
```

---

## Listening Stops Responding

Cause:
awake_window_s too short.

Increase:

```
awake_window_s = 15.0
```

---

## Piper espeakbridge Import Error

Cause:
Incorrect piper installation in user site-packages.

Fix inside virtual environment:

```
pip uninstall piper -y
pip install piper-tts
```

Ensure config.py has correct piper_model and piper_venv paths.

---

## ONNX GPU Warning

Message:

```
ReadFileContents Failed to open file: "/sys/class/drm/card0/device/vendor"
```

This is harmless GPU probing by ONNX runtime.

---

## Full Reset

Reset memory database:

```
rm data/luna.db
```

Wake word models do not affect memory storage.
