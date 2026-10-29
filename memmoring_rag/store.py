"""One SQLite file per user_id: the dated chunks, their vectors and a full-text index.

A user's file is the only place Search reads, so isolation by user_id is structural.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .chunk import chunks, required

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, payload_hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, request_id TEXT NOT NULL,
    msg INTEGER NOT NULL, part INTEGER NOT NULL, role TEXT NOT NULL,
    said_at INTEGER NOT NULL, text TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS vecs (id INTEGER PRIMARY KEY, v BLOB NOT NULL);
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
    text, content='chunks', content_rowid='id', tokenize='porter unicode61');
"""

NOTHING = {"duplicate": False, "chunks": 0, "calls": 0, "tokens": 0, "cost_yuan": 0.0}


class RequestConflict(ValueError):
    """A request_id was reused with different content for the same user."""


def user_path(root: Path, user_id: str) -> Path:
    return root / f"u_{hashlib.sha256(user_id.encode()).hexdigest()[:32]}.db"


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    db = sqlite3.connect(str(path), timeout=30)
    db.execute("PRAGMA busy_timeout=30000")
    try:
        with db:
            yield db
    finally:
        db.close()


def _payload_hash(body: dict[str, Any]) -> str:
    raw = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _previous(path: Path, request_id: str) -> str | None:
    with connect(path) as db:
        row = db.execute("SELECT payload_hash FROM requests WHERE request_id=?",
                         (request_id,)).fetchone()
    return row[0] if row else None


def add(root: Path, body: dict[str, Any], embedder) -> dict:
    """Persist one Add. Vectors are computed before the write, so a failed embedding
    leaves nothing half-written; the same request_id with the same body is a no-op."""
    request_id = required(body, "request_id")
    user_id = required(body, "user_id")
    session_id = required(body, "session_id")
    pieces = chunks(body)
    digest = _payload_hash(body)
    path = user_path(root, user_id)
    with connect(path) as db:
        db.executescript(SCHEMA)
    previous = _previous(path, request_id)
    if previous is not None:
        if previous != digest:
            raise RequestConflict("request_id already used with different content")
        return {**NOTHING, "duplicate": True}
    matrix, stats = embedder([c.text for c in pieces])
    with connect(path) as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("INSERT INTO requests VALUES (?, ?, ?)", (request_id, session_id, digest))
        for piece, vector in zip(pieces, matrix):
            cursor = db.execute(
                "INSERT INTO chunks (session_id, request_id, msg, part, role, said_at, text)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, request_id, piece.msg, piece.part, piece.role, piece.said_at, piece.text))
            db.execute("INSERT INTO vecs VALUES (?, ?)",
                       (cursor.lastrowid, vector.astype(np.float32).tobytes()))
            db.execute("INSERT INTO fts (rowid, text) VALUES (?, ?)", (cursor.lastrowid, piece.text))
    return {**NOTHING, "chunks": len(pieces), **stats}
