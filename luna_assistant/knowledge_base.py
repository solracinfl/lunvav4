from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    # cheap tokenizer: words + numbers
    return re.findall(r"[a-z0-9']+", text)


@dataclass
class RetrievedChunk:
    doc_id: str
    chunk_id: int
    score: float
    text: str
    source: str


class KnowledgeBase:
    """
    Offline keyword retrieval (BM25) for fast RAG without embeddings.

    - Stores docs + chunks in SQLite
    - Builds an in-memory BM25 index on startup (or after ingest)
    """

    def __init__(self, db_path: str = "data/luna.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")
        self._migrate()

        self._bm25: Optional[BM25Okapi] = None
        self._chunk_rows: list[tuple[int, str, str, str]] = []  # (chunk_id, doc_id, source, text)
        self._tokens: list[list[str]] = []

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def _migrate(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS docs (
              id TEXT PRIMARY KEY,
              source TEXT NOT NULL,
              path TEXT,
              title TEXT,
              added_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              doc_id TEXT NOT NULL,
              chunk_index INTEGER NOT NULL,
              text TEXT NOT NULL,
              created_at REAL NOT NULL,
              FOREIGN KEY (doc_id) REFERENCES docs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
            """
        )
        self.conn.commit()

    def ingest_text(self, doc_id: str, source: str, text: str, title: str = "", chunk_chars: int = 1200) -> None:
        """
        Simple chunking by character count with boundary on paragraph breaks when possible.
        """
        self.conn.execute(
            "INSERT OR REPLACE INTO docs(id, source, path, title, added_at) VALUES(?,?,?,?,?)",
            (doc_id, source, "", title, time.time()),
        )
        self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

        parts: list[str] = []
        buf: list[str] = []
        buf_len = 0
        for para in text.splitlines():
            line = para.rstrip()
            if not line:
                line = "\n"
            if buf_len + len(line) + 1 > chunk_chars and buf:
                parts.append("\n".join(buf).strip())
                buf = []
                buf_len = 0
            buf.append(line)
            buf_len += len(line) + 1
        if buf:
            parts.append("\n".join(buf).strip())

        for i, chunk in enumerate([p for p in parts if p]):
            self.conn.execute(
                "INSERT INTO chunks(doc_id, chunk_index, text, created_at) VALUES(?,?,?,?)",
                (doc_id, i, chunk, time.time()),
            )
        self.conn.commit()

    def rebuild_index(self) -> None:
        cur = self.conn.execute(
            """
            SELECT c.id, c.doc_id, d.source, c.text
            FROM chunks c
            JOIN docs d ON d.id = c.doc_id
            ORDER BY c.id ASC
            """
        )
        self._chunk_rows = [(r[0], r[1], r[2], r[3]) for r in cur.fetchall()]
        self._tokens = [_tokenize(r[3]) for r in self._chunk_rows]
        self._bm25 = BM25Okapi(self._tokens) if self._tokens else None

    def retrieve(self, query: str, k: int = 5, min_score: float = 0.0) -> list[RetrievedChunk]:
        if not self._bm25:
            self.rebuild_index()
        if not self._bm25:
            return []

        q_tokens = _tokenize(query)
        scores = self._bm25.get_scores(q_tokens)

        # Top-k by score
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: int(k)]
        out: list[RetrievedChunk] = []
        for i in idxs:
            score = float(scores[i])
            if score <= min_score:
                continue
            chunk_id, doc_id, source, text = self._chunk_rows[i]
            out.append(RetrievedChunk(doc_id=doc_id, chunk_id=int(chunk_id), score=score, text=text, source=source))
        return out
