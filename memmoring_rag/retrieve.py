"""Search: a dated question takes three routes in turn, everything else the vector route.

When the question names a day or a month of a year, the top 20 of three routes are
interleaved (inside the time window, by vector, by keyword) and cut at 30. Otherwise
the 30 nearest chunks by vector are taken. Results are returned in time order so the
reader sees a timeline.
"""

from __future__ import annotations

import calendar
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .chunk import InputError, iso, required
from .store import connect, user_path

BUDGET, PER_ROUTE = 30, 20
EMBED_HEAD, EMBED_TAIL = 1000, 5000     # characters; text-embedding-v4 takes at most 8,192 tokens
DAY_MS = 86_400_000
VECTOR_SLOTS = threading.BoundedSemaphore(2)
MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}
_MN = "|".join(MONTHS)
_DAY_MONTH = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MN})\b", re.I)
_MONTH_DAY = re.compile(rf"\b({_MN})\s+(\d{{1,2}})(?!\d)", re.I)
_MONTH = re.compile(rf"\b({_MN})\b", re.I)
_YEAR = re.compile(r"\b((?:19|20)\d\d)\b")
STOP = set("""the and for are was were what when where which who whom how did does do you your
yours i me my mine we our they them their this that these those with from into about have has
had been being any some can could would should will shall may might than then there here
not just also very much many more most other such only own same too again ever never""".split())


def time_window(question: str) -> tuple[int, int, int | None] | None:
    """(start ms, end ms, centre ms or None) for a named day or month of a named year."""
    year = _YEAR.search(question)
    if not year:
        return None
    y = int(year.group(1))

    def at(month: int, day: int, end: bool = False) -> int:
        if end:
            return int(datetime(y, month, day, 23, 59, 59, tzinfo=timezone.utc).timestamp() * 1000) + 999
        return int(datetime(y, month, day, tzinfo=timezone.utc).timestamp() * 1000)

    dm, md = _DAY_MONTH.search(question), _MONTH_DAY.search(question)
    if dm:
        day, month = int(dm.group(1)), MONTHS[dm.group(2).lower()]
    elif md:
        month, day = MONTHS[md.group(1).lower()], int(md.group(2))
    else:
        m = _MONTH.search(question)
        if not m:
            return None
        month = MONTHS[m.group(1).lower()]
        return at(month, 1), at(month, calendar.monthrange(y, month)[1], True), None
    try:
        centre = at(month, day)
    except ValueError:
        return None
    return centre, centre + DAY_MS - 1, centre


def weave(*routes: list[int]) -> list[int]:
    """route1[0], route2[0], route3[0], route1[1], ... without repeats."""
    out, seen = [], set()
    for i in range(max((len(r) for r in routes), default=0)):
        for route in routes:
            if i < len(route) and route[i] not in seen:
                seen.add(route[i])
                out.append(route[i])
    return out


def lexical(db, question: str, limit: int) -> list[int]:
    terms = [w for w in re.findall(r"[a-z0-9]+", question.lower()) if len(w) > 2 and w not in STOP]
    if not terms:
        return []
    expression = " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))
    rows = db.execute("SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT ?",
                      (expression, limit)).fetchall()
    return [r[0] for r in rows]


def _vectors(db) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = db.execute("SELECT c.id, c.said_at, v.v FROM chunks c JOIN vecs v ON v.id = c.id").fetchall()
    if not rows:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 1), np.float32)
    ids = np.fromiter((r[0] for r in rows), np.int64, len(rows))
    said = np.fromiter((r[1] for r in rows), np.int64, len(rows))
    matrix = np.frombuffer(b"".join(r[2] for r in rows), np.float32).reshape(len(rows), -1)
    return ids, said, matrix


def for_embedding(question: str) -> str:
    """The part of a question the embedding model can take: its opening and its end, where a
    long prompt keeps the actual question. The time and keyword routes still see all of it."""
    if len(question) <= EMBED_HEAD + EMBED_TAIL:
        return question
    return question[:EMBED_HEAD] + "\n...\n" + question[-EMBED_TAIL:]


def _top_k(body: dict[str, Any]) -> int:
    top_k = body.get("top_k")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise InputError("top_k must be a positive integer")
    return top_k


def search(root: Path, body: dict[str, Any], embedder) -> tuple[list[dict], dict]:
    question, user_id, top_k = required(body, "query"), required(body, "user_id"), _top_k(body)
    path = user_path(root, user_id)
    if not path.exists():
        return [], {"calls": 0, "tokens": 0, "cost_yuan": 0.0}
    query_vec, stats = embedder([for_embedding(question)])
    with connect(path) as db:
        with VECTOR_SLOTS:
            ids, said, matrix = _vectors(db)
            if not len(ids):
                return [], stats
            sims = matrix @ query_vec[0]
        score = dict(zip(ids.tolist(), sims.tolist()))
        by_vector = [int(ids[i]) for i in np.argsort(-sims)[:BUDGET]]
        window = time_window(question)
        if window:
            lo, hi, centre = window
            when = dict(zip(ids.tolist(), said.tolist()))
            inside = [i for i, t in when.items() if lo <= t <= hi]
            by_time = sorted(inside, key=lambda i: (abs(when[i] - centre) // DAY_MS if centre else 0,
                                                    -score[i]))[:PER_ROUTE]
            picked = weave(by_time, by_vector[:PER_ROUTE], lexical(db, question, PER_ROUTE))[:BUDGET]
        else:
            picked = by_vector
        picked = picked[:min(top_k, BUDGET)]
        marks = ",".join("?" * len(picked))
        rows = db.execute(f"SELECT id, said_at, msg, part, text FROM chunks WHERE id IN ({marks})",
                          picked).fetchall() if picked else []
    rows.sort(key=lambda r: (r[1], r[2], r[3], r[0]))
    return [{"id": f"c{r[0]}", "content": r[4], "score": round(score.get(r[0], 0.0), 4),
             "created_at": iso(r[1])} for r in rows], stats
