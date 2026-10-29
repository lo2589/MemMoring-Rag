"""text-embedding-v4 on Alibaba Cloud Model Studio, the only model this system calls.

Vectors are cached by the hash of their text, so a retried Add never pays for the
same chunk twice. The API key is read from a file and never logged.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np

URL = os.environ.get("MEMMORING_EMBED_URL",
                     "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings")
MODEL = "text-embedding-v4"
DIM = int(os.environ.get("MEMMORING_EMBED_DIM", "512"))
BATCH = 10
YUAN_PER_MILLION = 0.5


class RateLimited(Exception):
    """The provider asked us to slow down; the caller answers 429 with Retry-After."""

    def __init__(self, retry_after: int):
        super().__init__(f"embedding rate limited, retry after {retry_after}s")
        self.retry_after = min(max(retry_after, 1), 60)


class EmbedUnavailable(Exception):
    """The provider failed; the caller answers 503 and writes nothing."""


def key_file() -> Path:
    return Path(os.environ.get("MEMMORING_EMBED_KEY_FILE",
                               "~/.config/memmoring-rag/dashscope.key")).expanduser()


def ready() -> bool:
    path = key_file()
    return path.is_file() and path.stat().st_size > 0


def normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)


def call_api(texts: list[str]) -> tuple[np.ndarray, int]:
    """One request of at most BATCH texts -> (unit vectors, billed tokens)."""
    body = json.dumps({"model": MODEL, "input": texts, "dimensions": DIM,
                       "encoding_format": "float"}).encode()
    request = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {key_file().read_text().strip()}",
        "Content-Type": "application/json"})
    data = _post(request)
    rows = sorted(data["data"], key=lambda row: row["index"])
    matrix = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
    return normalize(matrix), int((data.get("usage") or {}).get("total_tokens") or 0)


def _post(request: urllib.request.Request, attempts: int = 3) -> dict:
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise RateLimited(int(exc.headers.get("Retry-After") or 30)) from exc
            if exc.code < 500 or attempt == attempts - 1:
                raise EmbedUnavailable(f"embedding provider answered HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == attempts - 1:
                raise EmbedUnavailable(f"embedding provider unreachable: {exc}") from exc
        time.sleep(1 + attempt)
    raise EmbedUnavailable("embedding provider failed")


class Embedder:
    """Cached embedding: only texts never seen before reach the provider."""

    def __init__(self, cache_path: str | Path,
                 call: Callable[[list[str]], tuple[np.ndarray, int]] = call_api):
        self.db = sqlite3.connect(str(cache_path), check_same_thread=False, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS vec (h TEXT PRIMARY KEY, v BLOB NOT NULL)")
        self.lock = threading.Lock()
        self.call = call

    def __call__(self, texts: list[str]) -> tuple[np.ndarray, dict]:
        hashes = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        known = self._lookup(set(hashes))
        missing = list(dict.fromkeys(h for h in hashes if h not in known))
        text_of = dict(zip(hashes, texts))
        calls = tokens = 0
        for i in range(0, len(missing), BATCH):
            batch = missing[i:i + BATCH]
            vectors, used = self.call([text_of[h] for h in batch])
            calls, tokens = calls + 1, tokens + used
            rows = [(h, v.astype(np.float32).tobytes()) for h, v in zip(batch, vectors)]
            with self.lock, self.db:
                self.db.executemany("INSERT OR REPLACE INTO vec VALUES (?, ?)", rows)
            known.update((h, v.astype(np.float32)) for h, v in zip(batch, vectors))
        matrix = np.stack([known[h] for h in hashes]) if hashes else np.zeros((0, DIM), np.float32)
        return matrix, {"calls": calls, "tokens": tokens,
                        "cost_yuan": tokens / 1e6 * YUAN_PER_MILLION}

    def _lookup(self, hashes: set[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        items = list(hashes)
        with self.lock:
            for i in range(0, len(items), 500):
                part = items[i:i + 500]
                marks = ",".join("?" * len(part))
                for h, blob in self.db.execute(f"SELECT h, v FROM vec WHERE h IN ({marks})", part):
                    found[h] = np.frombuffer(blob, dtype=np.float32)
        return found
