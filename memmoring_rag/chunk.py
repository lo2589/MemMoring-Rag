"""Turn an Add request into dated chunks.

Each message is cut into 100-word windows with a stride of 80 (20 words overlap),
and every chunk starts with "[date] role:" so the reader can reason about time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

WORDS, STRIDE = 100, 80


class InputError(ValueError):
    """The caller sent a malformed Add or Search request."""


@dataclass(frozen=True)
class Chunk:
    msg: int
    part: int
    role: str
    said_at: int
    text: str


def day_label(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y/%m/%d (%a) %H:%M")


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def windows(text: str, words: int = WORDS, stride: int = STRIDE) -> list[str]:
    tokens = text.split()
    if not tokens:
        return []
    return [" ".join(tokens[k:k + words])
            for k in range(0, max(1, len(tokens) - (words - stride)), stride)]


def required(body: dict[str, Any], key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{key} must be a non-empty string")
    return value


def messages(body: dict[str, Any]) -> list[tuple[str, str, int]]:
    """Validated (role, content, said_at ms) per message, in order."""
    items = body.get("messages")
    if not isinstance(items, list) or not items:
        raise InputError("messages must be a non-empty array")
    now = int(time.time() * 1000)
    out = []
    for i, message in enumerate(items):
        if not isinstance(message, dict):
            raise InputError(f"messages[{i}] must be an object")
        role, content = message.get("role"), message.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            raise InputError(f"messages[{i}] needs role=user|assistant and non-empty string content")
        stamp = message.get("timestamp")
        if stamp is not None and (not isinstance(stamp, int) or isinstance(stamp, bool)):
            raise InputError(f"messages[{i}].timestamp must be Unix milliseconds")
        out.append((role, content, stamp if stamp is not None else now + i))
    return out


def chunks(body: dict[str, Any]) -> list[Chunk]:
    out = []
    for i, (role, content, said_at) in enumerate(messages(body)):
        prefix = f"[{day_label(said_at)}] {role}: "
        out.extend(Chunk(i, j, role, said_at, prefix + piece)
                   for j, piece in enumerate(windows(content)))
    return out
