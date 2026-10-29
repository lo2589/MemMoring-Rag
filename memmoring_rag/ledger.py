"""Every Add and Search leaves one row: status, latency, size, embedding calls, tokens and cost.

The running total of cost is what the spending cap is checked against.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

COLUMNS = ("ts", "route", "status", "ms", "user_hash", "request_id", "messages", "bytes_in",
           "bytes_out", "chunks", "duplicate", "embed_calls", "embed_tokens", "cost_yuan", "error")


class Ledger:
    def __init__(self, path: Path):
        self.db = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(f"CREATE TABLE IF NOT EXISTS calls ({', '.join(COLUMNS)})")
        self.lock = threading.Lock()

    def write(self, **row) -> None:
        row.setdefault("ts", time.time())
        values = [row.get(c) for c in COLUMNS]
        with self.lock, self.db:
            self.db.execute(f"INSERT INTO calls VALUES ({','.join('?' * len(COLUMNS))})", values)

    def total_cost(self) -> float:
        with self.lock:
            return float(self.db.execute("SELECT COALESCE(SUM(cost_yuan), 0) FROM calls").fetchone()[0])

    def summary(self) -> dict:
        with self.lock:
            by_status = self.db.execute(
                "SELECT route, status, COUNT(*), ROUND(AVG(ms)), MAX(ms) FROM calls"
                " GROUP BY route, status ORDER BY route, status").fetchall()
            totals = self.db.execute(
                "SELECT COUNT(*), COALESCE(SUM(chunks),0), COALESCE(SUM(duplicate),0),"
                " COALESCE(SUM(embed_calls),0), COALESCE(SUM(embed_tokens),0),"
                " COALESCE(SUM(cost_yuan),0), COALESCE(SUM(bytes_in),0), COALESCE(SUM(bytes_out),0)"
                " FROM calls").fetchone()
        keys = ("requests", "chunks", "duplicates", "embed_calls", "embed_tokens", "cost_yuan",
                "bytes_in", "bytes_out")
        return {"by_route_status": [dict(zip(("route", "status", "count", "avg_ms", "max_ms"), r))
                                    for r in by_status],
                **dict(zip(keys, totals))}
