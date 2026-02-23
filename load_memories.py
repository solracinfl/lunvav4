#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterable, Tuple


def _clean(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()

    # Normalize “smart quotes” to plain quotes
    s = (
        s.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )

    # Strip wrapping quotes if present
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()

    # Normalize separators in values (protect against sqlite CLI pipe display issues)
    s = s.replace("|", "; ")

    # Collapse repeated spaces
    s = " ".join(s.split())
    return s


def _iter_csv_rows(csv_path: Path) -> Iterable[Tuple[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) < 2:
                continue
            k = _clean(row[0])
            v = _clean(row[1])

            if not k or not v:
                continue

            # Skip header rows like: key,value
            if k.lower() == "key" and v.lower() == "value":
                continue

            yield k, v


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA foreign_keys=ON;")

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS memories (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          created_at REAL NOT NULL,
          k TEXT NOT NULL,
          v TEXT NOT NULL,
          source_session_id TEXT,
          score REAL NOT NULL DEFAULT 1.0,
          pinned INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_memories_k ON memories(k);
        CREATE INDEX IF NOT EXISTS idx_memories_pinned_created ON memories(pinned, created_at DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC, id DESC);
        """
    )

    # If an older DB exists without pinned, add it safely.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(memories);").fetchall()}
    if "pinned" not in cols:
        conn.execute("ALTER TABLE memories ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0;")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_pinned_created ON memories(pinned, created_at DESC, id DESC);")

    # Best-effort: enforce one row per (k,pinned) for fast UPSERT.
    try:
        conn.execute(
            """
            DELETE FROM memories
            WHERE id NOT IN (
              SELECT MAX(id)
              FROM memories
              GROUP BY k, pinned
            );
            """
        )
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_memories_k_pinned ON memories(k, pinned);")
    except Exception:
        pass

    conn.commit()


def _reset_memories(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM memories;")
    conn.commit()


def _upsert_pinned_batch(conn: sqlite3.Connection, rows: list[Tuple[str, str]], score: float = 3.0) -> int:
    """Batch UPSERT pinned memories. Returns count processed."""
    now = time.time()
    payload = [(now, k, v, None, float(score), 1) for (k, v) in rows]
    conn.executemany(
        """
        INSERT INTO memories(created_at, k, v, source_session_id, score, pinned)
        VALUES(?,?,?,?,?,?)
        ON CONFLICT(k, pinned) DO UPDATE SET
          created_at=excluded.created_at,
          v=excluded.v,
          source_session_id=excluded.source_session_id,
          score=excluded.score;
        """,
        payload,
    )
    return len(rows)


def _prune_non_pinned(conn: sqlite3.Connection, keep_latest: int) -> int:
    keep_latest = int(keep_latest)
    if keep_latest < 0:
        keep_latest = 0

    cur = conn.execute(
        """
        SELECT id
        FROM memories
        WHERE pinned = 0
        ORDER BY created_at DESC, id DESC
        LIMIT -1 OFFSET ?;
        """,
        (keep_latest,),
    )
    ids = [r[0] for r in cur.fetchall()]
    if not ids:
        return 0

    deleted = 0
    # Chunk deletes to avoid SQLite parameter limits
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        qmarks = ",".join(["?"] * len(chunk))
        conn.execute(f"DELETE FROM memories WHERE id IN ({qmarks});", chunk)
        deleted += len(chunk)

    conn.commit()
    return deleted


def main() -> int:
    ap = argparse.ArgumentParser(description="Load pinned memories from a CSV into Luna's SQLite DB.")
    ap.add_argument("csv_path", help='Path to memories.csv (layout: "key","value")')
    ap.add_argument("--db", default="data/luna.db", help="SQLite DB path (default: data/luna.db)")
    ap.add_argument("--keep", type=int, default=500, help="Keep latest N non-pinned memories (default: 500)")
    ap.add_argument("--reset", action="store_true", help="Delete ALL existing memories before loading CSV")
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    db_path = Path(args.db)

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 2

    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    try:
        _ensure_schema(conn)

        if args.reset:
            _reset_memories(conn)

        # One transaction for the whole import.
        rows = list(_iter_csv_rows(csv_path))
        conn.execute("BEGIN;")
        loaded = _upsert_pinned_batch(conn, rows, score=3.0)
        conn.commit()

        deleted = _prune_non_pinned(conn, keep_latest=args.keep)

        print(f"Loaded/updated pinned memories: {loaded}")
        print(f"Pruned non-pinned to latest: {args.keep} (deleted {deleted})")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

    
