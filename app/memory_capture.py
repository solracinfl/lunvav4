from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class MemoryItem:
    key: str
    value: str
    confidence: float = 1.0


class MemoryCapture:
    """
    Fast, rule-based memory extraction.
    - No LLM calls
    - Captures only stable, high-signal facts
    - Keeps speed impact ~zero
    """

    def extract(self, text: str) -> list[MemoryItem]:
        t = (text or "").strip()
        if not t:
            return []

        out: list[MemoryItem] = []

        # Explicit remember directive
        m = re.search(r"\bremember\b[:\s-]*(.+)$", t, re.IGNORECASE)
        if m:
            val = m.group(1).strip().strip('"').strip("'")
            if val:
                out.append(MemoryItem("remember", val, 0.95))

        # Name
        m = re.search(r"\bmy name is\s+([A-Za-z0-9][A-Za-z0-9 _-]{1,48})\b", t, re.IGNORECASE)
        if m:
            out.append(MemoryItem("user_name", m.group(1).strip(), 0.95))

        # Location (keep short)
        m = re.search(r"\bi (?:live|am located)\s+in\s+(.+)$", t, re.IGNORECASE)
        if m:
            val = m.group(1).strip().rstrip(".")
            if 2 <= len(val) <= 80:
                out.append(MemoryItem("user_location", val, 0.8))

        # Wake phrase
        m = re.search(r"\bwake(?:\s+word|\s+phrase)?\s+is\s+([A-Za-z0-9][A-Za-z0-9 _-]{1,28})\b", t, re.IGNORECASE)
        if m:
            out.append(MemoryItem("wake_phrase", m.group(1).strip(), 0.9))

        # Audio routing hints (only if it contains concrete device strings)
        if "plughw:" in t or "hw:" in t or "AUDIO_OUT" in t:
            if len(t) <= 160:
                out.append(MemoryItem("audio_hint", t, 0.6))

        return self._dedupe(out)

    def _dedupe(self, items: Iterable[MemoryItem]) -> list[MemoryItem]:
        seen: set[tuple[str, str]] = set()
        out: list[MemoryItem] = []
        for it in items:
            k = (it.key or "").strip()
            v = (it.value or "").strip()
            if not k or not v:
                continue
            tup = (k, v)
            if tup in seen:
                continue
            seen.add(tup)
            out.append(it)
        return out
