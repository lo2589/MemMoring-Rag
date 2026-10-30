# memmoring-rag

Dated-chunk retrieval memory for the [Agent Memory Leaderboard](https://agentmemoryleaderboard.ai) (AML) Textual track, Open-source Methods division.

It calls one model: **`text-embedding-v4`** for embeddings. It makes no LLM calls on Add or on Search.

## How it works

**Add**: each message is cut into 100-word windows with a stride of 80. Each chunk starts with `[YYYY/MM/DD (Day) HH:MM] role:`, so the reader can reason about time. Chunks are embedded and stored in one SQLite file per `user_id`, together with an FTS5 keyword index. Vectors are cached by the hash of their text, so a retried or repeated Add is never billed twice.

**Search** has two cases:

- **The question names a day or month of a year** ("in July 2024", "on 5 March 2023"). Three routes are interleaved: chunks inside that time window, chunks by vector similarity, and chunks by BM25 keyword match. Each route contributes its top 20, and the result is cut at 30.
- **Any other question.** The 30 chunks nearest by vector are taken.

Results are returned in time order (`min(top_k, 30)` items), so the answer model reads a timeline.

## API

| Route | Auth | Body | Reply |
|---|---|---|---|
| `GET /aml/health` | none | – | `200 {"ok": true}`, or `503` when the embedding key is missing or the disk is nearly full |
| `POST /aml/add` | `Authorization: Token <key>` | `request_id, user_id, session_id, messages[{role, content, timestamp?}]` | `{"success": true, request_id, user_id, session_id}` after the write is durable |
| `POST /aml/search` | same | `query, user_id, top_k` | `{"data": [{id, content, score, created_at}]}` |

Errors and status codes:

| Status | Meaning |
|---|---|
| `400` | Malformed request |
| `401` | Bad or missing key |
| `409` | `request_id` reused with different content |
| `429` | Embedding provider is rate limiting; the response carries `Retry-After` |
| `503` | Provider down, spending cap reached, or disk nearly full |

## Install and run

Requires Python 3.10+ with SQLite FTS5. The only dependency is numpy.

```bash
git clone https://github.com/lo2589/memmoring-rag && cd memmoring-rag
python3 -m venv .venv && .venv/bin/pip install .

mkdir -p ~/.config/memmoring-rag && chmod 700 ~/.config/memmoring-rag
# Model Studio (DashScope) API key for text-embedding-v4:
(umask 077; cat > ~/.config/memmoring-rag/dashscope.key)
# Key that AML must send in "Authorization: Token ...":
(umask 077; python3 -c "import secrets; print(secrets.token_urlsafe(32))" > ~/.config/memmoring-rag/api.key)

.venv/bin/python -m memmoring_rag serve --data ~/memmoring-data \
    --key-file ~/.config/memmoring-rag/api.key --port 8013 --budget-yuan 400
.venv/bin/python -m memmoring_rag stats --data ~/memmoring-data   # call counts, tokens, cost
```

Put an HTTPS reverse proxy (for example Caddy) in front of the port.

Environment variables:

| Variable | Default |
|---|---|
| `MEMMORING_EMBED_KEY_FILE` | `~/.config/memmoring-rag/dashscope.key` |
| `MEMMORING_EMBED_DIM` | `512` |
| `MEMMORING_EMBED_URL` | DashScope's OpenAI-compatible embeddings endpoint |

## Cost

Embedding is billed per token, about ¥0.5 per million tokens. Chunk overlap and the date prefix make the embedded text about 1.26× the raw text. Every Add and Search is recorded in `calls.db`, along with its embedding calls, tokens and cost. Once the total cost reaches `--budget-yuan`, Add answers `503`.

## Tests

```bash
pip install pytest && python -m pytest -q tests
```

The tests use a local fake embedder. They make no network calls and cost nothing.

## Origin

This is the author's own original work. No third-party code is reused. The only runtime dependency is numpy, and the only model called is `text-embedding-v4` through Alibaba Cloud Model Studio.

## License

MIT
