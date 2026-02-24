from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class TurnRecord:
    session_id: str
    ts_unix: float
    user_text: str
    assistant_text: str
    asr_ms: int = 0
    llm_ms: int = 0
    tts_ms: int = 0
    total_ms: int = 0
    meta: Optional[dict[str, Any]] = None


class MemoryStore:
    """
    Lightweight persistent storage:
      - turns: raw conversation log + timings
      - memories: pinned facts (key/value) for fast recall

    Design goals:
      - one SQLite connection kept open
      - WAL for concurrency safety
      - fast inserts (single statement per turn)
    """

    def __init__(self, db_path: str = "data/luna.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self._migrate()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def _migrate(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
              id TEXT PRIMARY KEY,
              started_at REAL NOT NULL,
              meta_json TEXT
            );

            CREATE TABLE IF NOT EXISTS turns (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id TEXT NOT NULL,
              ts_unix REAL NOT NULL,
              user_text TEXT NOT NULL,
              assistant_text TEXT NOT NULL,
              asr_ms INTEGER NOT NULL DEFAULT 0,
              llm_ms INTEGER NOT NULL DEFAULT 0,
              tts_ms INTEGER NOT NULL DEFAULT 0,
              total_ms INTEGER NOT NULL DEFAULT 0,
              meta_json TEXT,
              FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS memories (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              created_at REAL NOT NULL,
              k TEXT NOT NULL,
              v TEXT NOT NULL,
              source_session_id TEXT,
              score REAL NOT NULL DEFAULT 1.0
            );

            CREATE INDEX IF NOT EXISTS idx_turns_session_ts ON turns(session_id, ts_unix);
            CREATE INDEX IF NOT EXISTS idx_memories_k ON memories(k);
            """
        )
        self.conn.commit()

    def start_session(self, session_id: str, meta: Optional[dict[str, Any]] = None) -> str:
        meta_json = json.dumps(meta or {}, ensure_ascii=False)
        self.conn.execute(
            "INSERT OR REPLACE INTO sessions(id, started_at, meta_json) VALUES(?,?,?)",
            (session_id, time.time(), meta_json),
        )
        self.conn.commit()
        return session_id

    def add_turn(self, t: TurnRecord) -> None:
        meta_json = json.dumps(t.meta or {}, ensure_ascii=False)
        self.conn.execute(
            """
            INSERT INTO turns(session_id, ts_unix, user_text, assistant_text,
                              asr_ms, llm_ms, tts_ms, total_ms, meta_json)
            VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                t.session_id,
                t.ts_unix,
                t.user_text,
                t.assistant_text,
                int(t.asr_ms),
                int(t.llm_ms),
                int(t.tts_ms),
                int(t.total_ms),
                meta_json,
            ),
        )
        self.conn.commit()

    def upsert_memory(self, key: str, value: str, session_id: Optional[str] = None, score: float = 1.0) -> None:
        # Simple strategy: keep latest value per key by deleting older entries.
        self.conn.execute("DELETE FROM memories WHERE k = ?", (key,))
        self.conn.execute(
            "INSERT INTO memories(created_at, k, v, source_session_id, score) VALUES(?,?,?,?,?)",
            (time.time(), key, value, session_id, float(score)),
        )
        self.conn.commit()

    def get_memories(self, limit: int = 30) -> list[tuple[str, str]]:
        cur = self.conn.execute(
            "SELECT k, v FROM memories ORDER BY score DESC, created_at DESC LIMIT ?",
            (int(limit),),
        )
        return [(r[0], r[1]) for r in cur.fetchall()]

    def get_recent_turns(self, session_id: str, limit: int = 6) -> list[tuple[str, str]]:
        cur = self.conn.execute(
            """
            SELECT user_text, assistant_text
            FROM turns
            WHERE session_id = ?
            ORDER BY ts_unix DESC
            LIMIT ?
            """,
            (session_id, int(limit)),
        )
        rows = cur.fetchall()
        rows.reverse()
        return [(r[0], r[1]) for r in rows]
