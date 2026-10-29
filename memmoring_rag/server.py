"""AML-compatible Add / Search / Health over plain HTTP (standard library only)."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import shutil
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import embed
from .chunk import InputError
from .ledger import Ledger
from .retrieve import search
from .store import RequestConflict, add

MIN_FREE_BYTES = 3 * 1024 ** 3


class Unavailable(Exception):
    """Refuse the request with 503: spending cap reached, disk nearly full, provider down."""


class App:
    def __init__(self, data_dir: str | Path, api_key: str, budget_yuan: float, embedder=None):
        if not api_key:
            raise ValueError("API key must be non-empty")
        self.root = Path(data_dir)
        self.users = self.root / "users"
        self.users.mkdir(parents=True, exist_ok=True)
        self.check_key_file = embedder is None
        self.embedder = embedder or embed.Embedder(self.root / "embed_cache.db")
        self.ledger = Ledger(self.root / "calls.db")
        self.key, self.budget = api_key, budget_yuan
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._guard = threading.Lock()

    def health(self) -> str | None:
        """None when healthy, otherwise the reason."""
        if self.check_key_file and not embed.ready():
            return "embedding key file missing"
        if shutil.disk_usage(self.root).free < MIN_FREE_BYTES:
            return "disk almost full"
        return None

    def authorized(self, headers) -> bool:
        raw = headers.get("X-Api-Key", "")
        if not raw:
            match = re.fullmatch(r"(?:Bearer|Token)\s+(.+)", headers.get("Authorization", ""), re.I)
            raw = match.group(1) if match else ""
        return hmac.compare_digest(raw.encode(), self.key.encode())

    def add(self, body: dict[str, Any]) -> tuple[dict, dict]:
        if self.ledger.total_cost() >= self.budget:
            raise Unavailable(f"spending cap of {self.budget} yuan reached")
        reason = self.health()
        if reason:
            raise Unavailable(reason)
        with self._guard:
            lock = self._locks[str(body.get("user_id"))]
        with lock:
            stats = add(self.users, body, self.embedder)
        reply = {"success": True, **{k: body[k] for k in ("request_id", "user_id", "session_id")}}
        return reply, stats

    def search(self, body: dict[str, Any]) -> tuple[dict, dict]:
        data, stats = search(self.users, body, self.embedder)
        return {"data": data}, stats


def _row(route: str, body: Any, stats: dict) -> dict:
    body = body if isinstance(body, dict) else {}
    user = str(body.get("user_id", ""))
    return {"route": route, "user_hash": hashlib.sha256(user.encode()).hexdigest()[:16] if user else None,
            "request_id": body.get("request_id"),
            "messages": len(body["messages"]) if isinstance(body.get("messages"), list) else None,
            "chunks": stats.get("chunks"), "duplicate": int(bool(stats.get("duplicate"))),
            "embed_calls": stats.get("calls", 0), "embed_tokens": stats.get("tokens", 0),
            "cost_yuan": stats.get("cost_yuan", 0.0)}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict, headers: dict | None = None) -> int:
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)
            return len(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/aml/health":
                self._send(404, {"detail": "unknown path"})
                return
            reason = app.health()
            self._send(503 if reason else 200, {"ok": False, "detail": reason} if reason else {"ok": True})

        def do_POST(self) -> None:  # noqa: N802
            route = {"/aml/add": "add", "/aml/search": "search"}.get(self.path)
            if route is None:
                self._send(404, {"detail": "unknown path"})
                return
            start, body, stats, error, raw = time.time(), None, {}, None, b""
            status, payload, headers = 200, {}, {}
            try:
                if not app.authorized(self.headers):
                    status, payload = 401, {"detail": "bad or missing API key"}
                else:
                    length = int(self.headers.get("Content-Length", "-1"))
                    if length < 0:
                        raise InputError("Content-Length is required")
                    raw = self.rfile.read(length)
                    body = json.loads(raw)
                    if not isinstance(body, dict):
                        raise InputError("request body must be an object")
                    payload, stats = getattr(app, route)(body)
            except RequestConflict as exc:
                status, payload = 409, {"detail": str(exc)}
            except (InputError, ValueError) as exc:
                status, payload = 400, {"detail": str(exc)}
            except embed.RateLimited as exc:
                status, payload, headers = 429, {"detail": str(exc)}, {"Retry-After": str(exc.retry_after)}
            except (embed.EmbedUnavailable, Unavailable) as exc:
                status, payload = 503, {"detail": str(exc)}
            if status != 200:
                error = payload.get("detail")
            size = self._send(status, payload, headers)
            app.ledger.write(**_row(route, body, stats), status=status, ms=int((time.time() - start) * 1000),
                             bytes_in=len(raw), bytes_out=size, error=error)

        def log_message(self, fmt: str, *args) -> None:
            pass

    return Handler


def serve(app: App, host: str, port: int) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
