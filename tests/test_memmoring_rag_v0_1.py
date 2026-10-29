import json
import re
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pytest

from memmoring_rag import embed
from memmoring_rag.chunk import InputError, chunks, windows
from memmoring_rag.retrieve import search, time_window
from memmoring_rag.server import App, make_handler
from memmoring_rag.store import RequestConflict, add


class FakeAPI:
    """Bag-of-words vectors: deterministic, free, and close for texts sharing words."""

    def __init__(self, fail: Exception | None = None):
        self.calls, self.fail = 0, fail

    def __call__(self, texts):
        if self.fail:
            raise self.fail
        self.calls += 1
        out = np.zeros((len(texts), 64), np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()):
                out[i, hash(w) % 64] += 1
        return embed.normalize(out + 1e-3), sum(len(t.split()) for t in texts)


def ms(y, m, d):
    return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp() * 1000)


def body(user, req, *texts, when=ms(2024, 3, 5)):
    return {"request_id": req, "user_id": user, "session_id": "s1",
            "messages": [{"role": "user", "content": t, "timestamp": when + i} for i, t in enumerate(texts)]}


@pytest.fixture
def fake(tmp_path):
    api = FakeAPI()
    return api, embed.Embedder(tmp_path / "cache.db", call=api), tmp_path / "users"


def test_windows_overlap_and_prefix():
    assert len(windows(" ".join(f"w{i}" for i in range(250)))) == 3
    piece = chunks(body("u", "r", "hello there"))[0]
    assert piece.text.startswith("[2024/03/05 (Tue) 00:00] user: hello there")


def test_add_then_search_finds_the_right_chunk(fake):
    api, embedder, root = fake
    root.mkdir()
    add(root, body("u", "r1", "I adopted a cat named Miso", "The weather was rainy"), embedder)
    data, _ = search(root, {"query": "what is my cat named", "user_id": "u", "top_k": 1}, embedder)
    assert len(data) == 1 and "Miso" in data[0]["content"] and data[0]["created_at"].startswith("2024-03-05")


def test_same_request_is_idempotent_and_changed_request_conflicts(fake):
    api, embedder, root = fake
    root.mkdir()
    assert add(root, body("u", "r1", "alpha"), embedder)["chunks"] == 1
    assert add(root, body("u", "r1", "alpha"), embedder)["duplicate"] is True
    with pytest.raises(RequestConflict):
        add(root, body("u", "r1", "beta"), embedder)


def test_users_are_isolated(fake):
    api, embedder, root = fake
    root.mkdir()
    add(root, body("u", "r1", "secret plan"), embedder)
    data, _ = search(root, {"query": "secret plan", "user_id": "other", "top_k": 5}, embedder)
    assert data == []


def test_cache_bills_each_text_once(fake):
    api, embedder, root = fake
    root.mkdir()
    add(root, body("u1", "r", "shared history line"), embedder)
    before = api.calls
    stats = add(root, body("u2", "r", "shared history line"), embedder)
    assert api.calls == before and stats["tokens"] == 0


def test_time_window_routes_dated_questions(fake):
    api, embedder, root = fake
    root.mkdir()
    add(root, body("u", "a", "went hiking", when=ms(2024, 3, 5)), embedder)
    add(root, body("u", "b", "went hiking again", when=ms(2024, 7, 9)), embedder)
    assert time_window("What did I do in July 2024?")[0] == ms(2024, 7, 1)
    data, _ = search(root, {"query": "What did I do in July 2024?", "user_id": "u", "top_k": 1}, embedder)
    assert "again" in data[0]["content"]


def test_bad_input_is_rejected():
    with pytest.raises(InputError):
        chunks({"messages": [{"role": "system", "content": "x"}]})


def _client(app):
    from http.server import ThreadingHTTPServer
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # never route localhost via a proxy

    def call(path, payload=None, key="k"):
        req = urllib.request.Request(base + path, data=None if payload is None else json.dumps(payload).encode(),
                                     headers={"Authorization": f"Token {key}"})
        try:
            with opener.open(req) as r:
                return r.status, json.load(r), dict(r.headers)
        except urllib.error.HTTPError as e:
            return e.code, json.load(e), dict(e.headers)
    return httpd, call


def test_http_contract(tmp_path):
    app = App(tmp_path, "k", 100.0, embedder=embed.Embedder(tmp_path / "c.db", call=FakeAPI()))
    httpd, call = _client(app)
    try:
        assert call("/aml/health")[0] == 200
        assert call("/aml/add", body("u", "r", "hi"), key="bad")[0] == 401
        status, reply, _ = call("/aml/add", body("u", "r", "my cat Miso"))
        assert status == 200 and reply == {"success": True, "request_id": "r", "user_id": "u", "session_id": "s1"}
        status, reply, _ = call("/aml/search", {"query": "cat", "user_id": "u", "top_k": 3})
        assert status == 200 and reply["data"][0]["content"].endswith("my cat Miso")
        assert call("/aml/search", {"query": "cat", "user_id": "u", "top_k": 0})[0] == 400
        assert app.ledger.summary()["requests"] == 4
    finally:
        httpd.shutdown()


def test_rate_limit_and_spending_cap(tmp_path):
    limited = App(tmp_path / "a", "k", 100.0,
                  embedder=embed.Embedder(tmp_path / "a.db", call=FakeAPI(embed.RateLimited(7))))
    httpd, call = _client(limited)
    try:
        status, _, headers = call("/aml/add", body("u", "r", "x"))
        assert status == 429 and headers["Retry-After"] == "7"
    finally:
        httpd.shutdown()
    capped = App(tmp_path / "b", "k", 0.0, embedder=embed.Embedder(tmp_path / "b.db", call=FakeAPI()))
    httpd, call = _client(capped)
    try:
        assert call("/aml/add", body("u", "r", "x"))[0] == 503
    finally:
        httpd.shutdown()
