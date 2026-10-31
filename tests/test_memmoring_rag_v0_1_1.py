"""v0.1.1: a question longer than the embedding model takes is cut for the vector route only."""

import numpy as np

from memmoring_rag import embed
from memmoring_rag.retrieve import EMBED_HEAD, EMBED_TAIL, for_embedding, search
from memmoring_rag.store import add

from test_memmoring_rag_v0_1 import FakeAPI, body


def test_short_question_is_untouched():
    assert for_embedding("what is my cat named") == "what is my cat named"


def test_long_question_keeps_its_opening_and_its_end():
    long = "context " * 20000 + "QUESTION: what is my cat named?"
    cut = for_embedding(long)
    assert len(cut) <= EMBED_HEAD + EMBED_TAIL + 10
    assert cut.startswith("context") and cut.endswith("what is my cat named?")


def test_search_with_a_huge_query_embeds_only_the_cut(tmp_path):
    seen = []

    class Recording(FakeAPI):
        def __call__(self, texts):
            seen.extend(texts)
            return super().__call__(texts)

    embedder = embed.Embedder(tmp_path / "cache.db", call=Recording())
    root = tmp_path / "users"
    root.mkdir()
    add(root, body("u", "r1", "I adopted a cat named Miso"), embedder)
    seen.clear()
    query = "filler " * 30000 + "what is my cat named"
    data, _ = search(root, {"query": query, "user_id": "u", "top_k": 1}, embedder)
    assert "Miso" in data[0]["content"]
    assert len(seen) == 1 and len(seen[0]) <= EMBED_HEAD + EMBED_TAIL + 10


def test_refused_question_falls_back_to_keywords_instead_of_failing(tmp_path):
    embedder_ok = embed.Embedder(tmp_path / "cache.db", call=FakeAPI())
    root = tmp_path / "users"
    root.mkdir()
    add(root, body("u", "r1", "I adopted a cat named Miso", "The weather was rainy"), embedder_ok)

    def refusing(texts):
        raise embed.EmbedUnavailable("embedding provider answered HTTP 400")

    query = "zzz " * 50000 + "what is the name of my cat Miso"
    data, stats = search(root, {"query": query, "user_id": "u", "top_k": 5}, refusing)
    assert any("Miso" in d["content"] for d in data) and stats["calls"] == 0


def test_rate_limit_is_still_passed_up(tmp_path):
    root = tmp_path / "users"
    root.mkdir()
    add(root, body("u", "r1", "I adopted a cat named Miso"), embed.Embedder(tmp_path / "c.db", call=FakeAPI()))

    def limited(texts):
        raise embed.RateLimited(5)

    import pytest
    with pytest.raises(embed.RateLimited):
        search(root, {"query": "cat", "user_id": "u", "top_k": 1}, limited)
