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


@dataclass
class MemoryItem:
    key: str
    value: str
    score: float = 1.0
    pinned: bool = False
    created_at: float = 0.0


class MemoryStore:
    """
    Persistent storage:
      - sessions, turns (conversation log + timings)
      - memories (key/value facts)

    Requirements:
      - injected/pinned facts are ALWAYS kept
      - after that, keep latest N (default 500) non-pinned facts
    """

    def __init__(self, db_path: str = "data/luna.db", keep_latest: int = 500) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.keep_latest = int(keep_latest)

        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        # Fast, safe-enough defaults for an embedded assistant DB.
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        # Negative = KB units. Tune as needed; 20MB is a solid default on Jetson.
        self.conn.execute("PRAGMA cache_size=-20000;")

        # Hot-path cache: pinned facts are read constantly, written rarely.
        self._pinned_cache_rows: list[tuple] | None = None
        self._pinned_cache_ts: float = 0.0
        self._pinned_cache_ttl_s: float = 15.0
        self._migrate()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def _migrate(self) -> None:
        # Create base tables if missing
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
              score REAL NOT NULL DEFAULT 1.0,
              pinned INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_turns_session_ts ON turns(session_id, ts_unix);
            CREATE INDEX IF NOT EXISTS idx_memories_k ON memories(k);
            CREATE INDEX IF NOT EXISTS idx_memories_pinned_created ON memories(pinned, created_at DESC, id DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC, id DESC);
            """
        )

        # Add pinned column if upgrading from older DB (best-effort)
        try:
            cols = [r[1] for r in self.conn.execute("PRAGMA table_info(memories);").fetchall()]
            if "pinned" not in cols:
                self.conn.execute("ALTER TABLE memories ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0;")
        except Exception:
            pass

        # Ensure a single row per (k, pinned) category for fast UPSERT semantics.
        # Best-effort: dedupe before creating the UNIQUE index.
        try:
            self.conn.execute(
                """
                DELETE FROM memories
                WHERE id NOT IN (
                  SELECT MAX(id)
                  FROM memories
                  GROUP BY k, pinned
                );
                """
            )
            self.conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_memories_k_pinned ON memories(k, pinned);")
        except Exception:
            pass

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

    def upsert_memory(
        self,
        key: str,
        value: str,
        session_id: Optional[str] = None,
        score: float = 1.0,
        pinned: bool = False,
    ) -> None:
        """
        - pinned=True: keep forever; unique per key (latest wins)
        - pinned=False: unique per key (latest wins) AND enforce keep_latest cap across ALL non-pinned
        """
        now = time.time()
        p = 1 if pinned else 0

        # Single-row per (k,pinned), with UNIQUE index enabling UPSERT.
        self.conn.execute(
            """
            INSERT INTO memories(created_at, k, v, source_session_id, score, pinned)
            VALUES(?,?,?,?,?,?)
            ON CONFLICT(k, pinned) DO UPDATE SET
              created_at=excluded.created_at,
              v=excluded.v,
              source_session_id=excluded.source_session_id,
              score=excluded.score;
            """,
            (now, key, value, session_id, float(score), p),
        )
        self.conn.commit()

        # Invalidate caches on write.
        self._invalidate_cache()

        if not pinned:
            self._enforce_non_pinned_cap()

    def _enforce_non_pinned_cap(self) -> None:
        """
        Keep only the newest self.keep_latest rows where pinned=0.
        Pinned rows are never deleted.
        """
        if self.keep_latest <= 0:
            return

        # Delete older non-pinned rows beyond the newest keep_latest by created_at
        self.conn.execute(
            f"""
            DELETE FROM memories
            WHERE pinned = 0
              AND id NOT IN (
                SELECT id
                FROM memories
                WHERE pinned = 0
                ORDER BY created_at DESC, id DESC
                LIMIT {self.keep_latest}
              )
            """
        )
        self.conn.commit()

        self._invalidate_cache()

    def pin_memory(self, key: str, value: str, session_id: Optional[str] = None, score: float = 1.0) -> None:
        self.upsert_memory(key=key, value=value, session_id=session_id, score=score, pinned=True)

    def unpin_memory(self, key: str) -> None:
        self.conn.execute("DELETE FROM memories WHERE k = ? AND pinned = 1", (key,))
        self.conn.commit()
        self._invalidate_cache()

    def forget_memory(self, key: str) -> None:
        """Remove both pinned and non-pinned entries for this key."""
        self.conn.execute("DELETE FROM memories WHERE k = ?", (key,))
        self.conn.commit()
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        self._pinned_cache_rows = None
        self._pinned_cache_ts = 0.0

    def get_pinned(self, limit: int = 30) -> list[MemoryItem]:
        cur = self.conn.execute(
            """
            SELECT k, v, score, pinned, created_at
            FROM memories
            WHERE pinned = 1
            ORDER BY score DESC, created_at DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        out: list[MemoryItem] = []
        for k, v, score, pinned, created_at in cur.fetchall():
            out.append(MemoryItem(str(k), str(v), float(score), bool(pinned), float(created_at)))
        return out

    # Assistant-facing APIs (rows, not objects) for speed and backward compatibility.
    def get_pinned_memories(self, limit: int = 30) -> list[tuple]:
        now = time.time()
        if self._pinned_cache_rows is not None and (now - self._pinned_cache_ts) <= self._pinned_cache_ttl_s:
            return self._pinned_cache_rows[: int(limit)]

        cur = self.conn.execute(
            """
            SELECT k, v, score, pinned, created_at
            FROM memories
            WHERE pinned = 1
            ORDER BY score DESC, created_at DESC, id DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        rows = cur.fetchall()
        self._pinned_cache_rows = rows
        self._pinned_cache_ts = now
        return rows

    def get_memories(self, limit: int = 200) -> list[tuple]:
        cur = self.conn.execute(
            """
            SELECT k, v, score, pinned, created_at
            FROM memories
            ORDER BY pinned DESC, score DESC, created_at DESC, id DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        return cur.fetchall()

    def get_all_memories(self, limit: int = 200) -> list[MemoryItem]:
        cur = self.conn.execute(
            """
            SELECT k, v, score, pinned, created_at
            FROM memories
            ORDER BY pinned DESC, score DESC, created_at DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        out: list[MemoryItem] = []
        for k, v, score, pinned, created_at in cur.fetchall():
            out.append(MemoryItem(str(k), str(v), float(score), bool(pinned), float(created_at)))
        return out

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
    def prune_non_pinned(self, limit: int = 500) -> int:
        """
        Keep ALL pinned memories.
        For non-pinned memories, keep only the most recent `limit` rows and delete the rest.
        Returns number of rows deleted.
        """
        limit = int(limit)
        if limit < 0:
            limit = 0

        cur = self.conn.execute(
            """
            SELECT id
            FROM memories
            WHERE pinned = 0
            ORDER BY created_at DESC, id DESC
            LIMIT -1 OFFSET ?
            """,
            (limit,),
        )
        ids = [r[0] for r in cur.fetchall()]
        if not ids:
            return 0

        deleted = 0
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            qmarks = ",".join(["?"] * len(chunk))
            self.conn.execute(f"DELETE FROM memories WHERE id IN ({qmarks})", chunk)
            deleted += len(chunk)

        self.conn.commit()
        self._invalidate_cache()
        return deleted
