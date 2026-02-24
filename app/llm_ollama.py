from __future__ import annotations

import json
from dataclasses import dataclass
import requests


@dataclass
class OllamaLLM:
    url: str
    model: str
    system_prompt: str
    keep_alive: str = "10m"
    timeout_s: int = 120

    def __post_init__(self) -> None:
        self._session = requests.Session()

    def warmup(self) -> None:
        try:
            _ = self.chat("Say: OK")
        except Exception:
            pass

    def chat(self, user_text: str, extra_context: str = "") -> str:
        # If caller provides extra context (memories, rules), include it above the user turn.
        if extra_context and extra_context.strip():
            prompt = f"{self.system_prompt}\n{extra_context.strip()}\nUser: {user_text}\nAssistant:"
        else:
            prompt = f"{self.system_prompt}\nUser: {user_text}\nAssistant:"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
        }
        r = self._session.post(
            self.url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()
