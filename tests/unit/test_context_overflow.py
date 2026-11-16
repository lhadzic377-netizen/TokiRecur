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
from types import SimpleNamespace

from fastapi.testclient import TestClient

from tokirecur.api.app import (
    _context_overflow_status,
    _estimate_prompt_tokens,
    _openai_error,
    _prompt_would_overflow,
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


class MixedBackend:
    """A backend whose prompts are denser at one end than the other.

    The overflow check estimates the token count from a sample of the
    prompt, so the sample has to be representative: the head and the
    tail are both taken, and a prompt that lands inside the margin is
    counted exactly.
    """

    def __init__(self, head_chars_per_token: int, tail_chars_per_token: int):
        self.head = head_chars_per_token
        self.tail = tail_chars_per_token
        self.tokenize_calls = 0
        self.full_calls = 0

    def _count(self, text: str) -> int:
        # "a" is dense, "b" is sparse, so a sample of the prompt can
        # count differently from the whole of it.
        return (text.count("a") // self.head) + (text.count("b") // self.tail)

    def tokenize(self, text: str) -> list[int]:
        self.tokenize_calls += 1
        return [0] * self._count(text)

    def apply_template(self, messages, tools=None) -> str:
        return "".join(
            str(m.get("content") or "") for m in messages
        ) + ("".join(str(t) for t in (tools or [])))


def test_the_estimate_holds_when_the_density_changes_along_the_prompt():
    """A prompt that does not fit is refused even when the sample
    counts it differently from the whole."""
    backend = MixedBackend(head_chars_per_token=8, tail_chars_per_token=2)
    engine = SimpleNamespace(backend=backend)
    # 40000 characters: "a" is sparse (8 per token), "b" is dense
    # (2 per token) - 12500 tokens in all, over an 8000 window.
    messages = [{"role": "user", "content": "a" * 20000 + "b" * 20000}]
    assert _prompt_would_overflow(engine, messages, None, ceiling=8000) is True


def test_a_prompt_that_fits_is_not_refused_when_the_sample_counts_it_low():
    backend = MixedBackend(head_chars_per_token=2, tail_chars_per_token=16)
    engine = SimpleNamespace(backend=backend)
    messages = [{"role": "user", "content": "a" * 20000 + "b" * 20000}]
    # 11250 tokens in all, inside a 12000 window: a prompt that fits
    # must not be refused.
    assert _prompt_would_overflow(engine, messages, None, ceiling=12000) is False


def test_the_sampled_estimate_is_close_to_the_true_count():
    """On a prompt of one density, the sample is within a few percent."""
    backend = MixedBackend(head_chars_per_token=4, tail_chars_per_token=4)
    engine = SimpleNamespace(backend=backend)
    text = "a" * 40000
    rendered = backend.apply_template([{"role": "user", "content": text}])
    estimate = _estimate_prompt_tokens(engine, rendered)
    true = len(backend.tokenize(rendered))
    assert abs(estimate - true) / true < 0.1, (estimate, true)
