"""A prompt that does not fit is an answer, not a mystery.

llama.cpp words the overflow two ways, and both are live: the older
"not enough tokens in the context window" and the current "request (N
tokens) exceeds the available context size (M tokens)". The second one
used to reach the client as a 500 with an unreadable body, and a
streaming client saw the stream die mid-turn instead of a status it
could act on - MiniMax Code compacts and retries on a 413 and gives up
on a 500.

The check has to happen before the response headers go out, and it has
to count the prompt the way llama-server builds it: the chat template
plus the tool schemas, which on an agent client are more tokens than
the conversation.
"""

import json

from fastapi.testclient import TestClient

from tokirecur.api.app import (
    _context_overflow_status,
    _openai_error,
    create_app,
)


class FakeBackend:
    """A backend that can count a prompt but serves nothing."""

    def __init__(self, tokens_per_char: int = 4) -> None:
        self.tokens_per_char = tokens_per_char
        self.applied: list[tuple] = []

    def tokenize(self, text: str) -> list[int]:
        return [0] * max(1, len(text) // self.tokens_per_char)

    def apply_template(self, messages, tools=None) -> str:
        self.applied.append((messages, tools))
        rendered = json.dumps(messages) + json.dumps(tools or [])
        return rendered

    def embed(self, text: str) -> list[float]:
        return [0.0] * 4

    def complete_openai(self, messages, **kw):
        raise AssertionError("the request should not have reached the model")

    async def complete_openai_stream(self, messages, **kw):
        raise AssertionError("the request should not have reached the model")
        yield  # pragma: no cover


class FakeIndex:
    def __init__(self) -> None:
        self.entries: dict[str, object] = {}

    def search(self, embedding):
        return None

    def insert(self, entry) -> None:
        self.entries[entry.cache_id] = entry

    def touch(self, cache_id: str) -> None:
        pass

    def stats(self) -> dict:
        return {"entries": len(self.entries), "bytes_on_disk": 0, "embedding_dim": 4}


def _app(backend, *, window: int | None):
    return create_app(
        backend=backend,
        index=FakeIndex(),
        cache_dir="/tmp/toki-tests",
        enable_auth=False,
        enable_rate_limit=False,
        max_context_tokens=window,
    )


def test_the_current_llama_cpp_wording_is_an_overflow():
    """The wording llama.cpp uses now, with its error code."""
    error = Exception(
        'llama-server 400: {"error":{"code":400,"message":"request (27488 '
        'tokens) exceeds the available context size (512 tokens), try '
        'increasing it","type":"exceed_context_size_error","n_prompt":27488}}'
    )
    assert _context_overflow_status(error) == 413


def test_the_older_wording_is_still_an_overflow():
    error = Exception("not enough tokens in the context window")
    assert _context_overflow_status(error) == 413


def test_an_unrelated_error_is_not_an_overflow():
    assert _context_overflow_status(Exception("connection reset by peer")) is None


def test_the_error_body_is_readable_by_openai_clients():
    """FastAPI's default body is ``{"detail": ...}``.

    Every OpenAI-compatible client looks for ``error.message``; one of
    ours arrives as an empty body, and the message that would have
    told the client to compact never gets read.
    """
    response = _openai_error(413, "too long", code="context_length_exceeded")
    body = json.loads(response.body)
    assert body["error"]["message"] == "too long"
    assert body["error"]["code"] == "context_length_exceeded"
    assert body["detail"] == "too long"


def test_a_prompt_that_does_not_fit_is_refused_before_the_stream():
    backend = FakeBackend()
    with TestClient(_app(backend, window=1000)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "x" * 20000}],
                "stream": True,
            },
        )
    assert response.status_code == 413
    body = response.json()
    assert body["error"]["code"] == "context_length_exceeded"
    assert "context window" in body["error"]["message"]


def test_a_prompt_that_fits_is_not_refused():
    backend = FakeBackend()
    with TestClient(_app(backend, window=100000)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "x" * 20000}],
                "stream": True,
            },
        )
    # It is not refused: the turn reaches the model (this fake serves
    # nothing, so the stream ends in an error - the point is that the
    # daemon did not call it an overflow). A prompt far from the
    # ceiling is not even measured.
    assert response.status_code != 413
    assert backend.applied == []


def test_the_tools_are_counted_with_the_prompt():
    """A client that ships 61 tools sends more tokens in the tool
    schemas than in the conversation, so the message text alone is a
    count that is short by the whole tool block."""
    backend = FakeBackend()
    tools = [
        {"type": "function", "function": {"name": f"tool{i}",
                                          "description": "d" * 4000}}
        for i in range(4)
    ]
    with TestClient(_app(backend, window=1000)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "short prompt"}],
                "tools": tools,
                "stream": True,
            },
        )
    assert response.status_code == 413
    # The server was asked to render the prompt, tools included.
    assert backend.applied
    assert backend.applied[0][1] == tools
