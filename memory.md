# Memory and Knowledge — Implementation Guide

## Overview

Luna has **three subsystems** for memory and knowledge. Only the Memory Store is actively wired into the assistant loop today.

---

## 1. Memory Store (Active) — `app/memory_store.py`

Persists everything in a single **SQLite** database at `data/luna.db` (configurable via `MEMORY_DB_PATH`).

### Data Model — Three Tables

| Table      | Purpose                                                                   |
| ---------- | ------------------------------------------------------------------------- |
| `sessions` | One row per conversation session (id, timestamp, metadata)                |
| `turns`    | Every user/assistant exchange with timing metrics (ASR, LLM, TTS latency) |
| `memories` | Key-value facts about the user, with a `pinned` flag                      |

### Pinned vs. Non-Pinned Memories

- **Pinned** — permanent, trusted facts (name, location, birthday). Survive pruning. Always injected into the LLM prompt. Unique per key via `UNIQUE(k, pinned)` constraint; upserting the same key overwrites.
- **Non-pinned** — transient. A rolling cap of 500 (configurable) is enforced; older ones are deleted automatically via `_enforce_non_pinned_cap()`.

### How Memories Reach the LLM

Every conversation turn calls `_build_pinned_context()` in `assistant.py` (lines 122–129):

```python
def _build_pinned_context(self) -> str:
    pinned = self.store.get_pinned_memories(limit=50)
    if not pinned:
        return ""
    lines = ["Pinned user facts (trusted):"]
    for k, v, score, pinned_flag, created_at in pinned:
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)
```

This string is passed as `extra_context` to `OllamaLLM`, which inserts it between the system prompt and the user message in `_build_prompt()` (`app/llm_ollama.py`, lines 35–40):

```python
def _build_prompt(self, user_text: str, extra_context: str = "") -> str:
    sys_p = (self.system_prompt or "").strip()
    ctx = (extra_context or "").strip()
    if ctx:
        return f"{sys_p}\n{ctx}\nUser: {user_text}\nAssistant:".strip()
    return f"{sys_p}\nUser: {user_text}\nAssistant:".strip()
```

Final prompt sent to Ollama:

```
<system prompt>
Pinned user facts (trusted):
- name: Carlos
- birthday: 10/13/1969
- spouse: Ivette
...
User: <what the user said>
Assistant:
```

### Performance

Pinned memories are cached in-memory with a **15-second TTL** (`_pinned_cache_ttl_s`) to avoid hitting SQLite on every turn.

### Seeding Memories

`load_memories.py` bulk-loads pinned facts from `memories.csv`:

```
"name","Carlos"
"birthday","10/13/1969"
"location","642 Avila Place, Howey In The Hills, FL 34737"
"spouse","Ivette"
...
```

Batch-upserted with `score=3.0` and `pinned=1`.

### Voice Command

User can say "list memories" / "show memories" to hear all stored memories read aloud (`assistant.py`, lines 261–279).

---

## 2. Memory Capture — `app/memory_capture.py`

Rule-based regex extractor that pulls facts from user utterances without any LLM call.

### Matched Patterns

| Pattern                         | Key             | Example                              |
| ------------------------------- | --------------- | ------------------------------------ |
| `remember: ...`                 | `remember`      | "Remember my favorite color is blue" |
| `my name is X`                  | `user_name`     | "My name is Carlos"                  |
| `I live in X`                   | `user_location` | "I live in Florida"                  |
| `wake word is X`                | `wake_phrase`   | "Wake word is Luna"                  |
| `plughw:` / `hw:` / `AUDIO_OUT` | `audio_hint`    | Device routing strings               |

**Status:** The `MemoryCapture` class is never instantiated or called from `assistant.py`. It is ready to be integrated but is not yet active.

---

## 3. Knowledge Base — `app/knowledge_base.py`

Lightweight RAG using **BM25** keyword scoring (no embeddings, no external vector DB).

### Pipeline

1. **Ingest:** `ingest_text()` splits a document into ~1200-char chunks at paragraph boundaries, stores in `docs` + `chunks` SQLite tables.
2. **Index:** `rebuild_index()` loads all chunks, tokenizes (lowercase alphanumeric), builds an in-memory `BM25Okapi` index.
3. **Retrieve:** `retrieve(query, k=5)` tokenizes the query, scores all chunks via BM25, returns top-k above a minimum score threshold.

### Dependencies

- `rank-bm25==0.2.2`

**Status:** Not wired into the assistant loop. No code in `assistant.py` calls `KnowledgeBase`.

---

## Active Data Flow

```
memories.csv
    │  (load_memories.py)
    ▼
SQLite: data/luna.db
    │
    │  Runtime loop (assistant.py):
    │
    ├── Wake word detected
    ├── Record audio → ASR → user text
    ├── _build_pinned_context() reads pinned memories from DB (cached 15s)
    ├── Inject pinned context into LLM prompt
    ├── Ollama generates reply
    ├── store.add_turn() logs the exchange to DB
    └── Loop
```

### Key Files

| File                    | Role                                                |
| ----------------------- | --------------------------------------------------- |
| `app/memory_store.py`   | SQLite-backed storage for sessions, turns, memories |
| `app/memory_capture.py` | Regex-based memory extraction (dormant)             |
| `app/knowledge_base.py` | BM25-based RAG retrieval (dormant)                  |
| `app/assistant.py`      | Wires pinned memories into LLM prompts              |
| `app/llm_ollama.py`     | Builds the final prompt with extra context          |
| `app/config.py`         | `MEMORY_DB_PATH`, `MEMORY_PINNED_LIMIT` settings    |
| `load_memories.py`      | CLI script to seed pinned memories from CSV         |
| `memories.csv`          | User facts CSV                                      |
| `reset_memories.sh`     | Deletes all memories, turns, sessions; runs VACUUM  |
