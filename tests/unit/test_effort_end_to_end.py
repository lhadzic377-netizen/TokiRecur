"""The reasoning effort a turn runs at, end to end through the daemon.

llama-server reads "reasoning_effort" itself and hands it to the model's
chat template, so the daemon's job is to decide which effort applies and
put it on the wire. These tests watch what the engine was called with.
"""

import pytest
from fastapi.testclient import TestClient

from tokirecur.api.app import create_app
from tokirecur.backend import LlamaServerBackend
from tokirecur.core.recycling import RecyclingEngine, RecyclingResult
from tokirecur.index import Index


def _client(tmp_path, monkeypatch, observed):
    backend = LlamaServerBackend(base_url="http://llama.invalid")
    index = Index()
    app = create_app(
        backend=backend, index=index, cache_dir=tmp_path,
        enable_auth=False, enable_rate_limit=False,
    )

    def fake_serve_chat(self, messages, *, max_tokens, **kw):
        observed["effort"] = kw.get("reasoning_effort")
        observed["ctk"] = kw.get("chat_template_kwargs")
        return RecyclingResult(
            hit=False, soft_hit=False, cache_id=None, reuse_depth=0,
            embedding_similarity=0.0, prefilled_tokens=4, skipped_tokens=0,
            generated_tokens=12, ttft_ms=15.0, total_ms=80.0,
            prefill_tokens_per_sec=100.0, generated_tokens_per_sec=50.0,
            output_text="Hello!", reasoning="",
            slot_id=0, spilled=False, tool_calls=None, finish_reason="stop",
        )

    monkeypatch.setattr(RecyclingEngine, "serve_chat", fake_serve_chat)
    return TestClient(app)


def _post(client, **extra):
    payload = {
        "model": "test", "messages": [{"role": "user", "content": "hi"}],
    }
    payload.update(extra)
    return client.post("/v1/chat/completions", json=payload)


def test_an_explicit_effort_reaches_the_engine(tmp_path, monkeypatch):
    observed: dict = {}
    client = _client(tmp_path, monkeypatch, observed)
    res = _post(client, reasoning_effort="high")
    assert res.status_code == 200, res.text
    assert observed["effort"] == "high"


def test_the_thinking_flag_follows_the_effort(tmp_path, monkeypatch):
    """An effort of "none" is thinking off; every other level is on."""
    observed: dict = {}
    client = _client(tmp_path, monkeypatch, observed)

    _post(client, reasoning_effort="none")
    assert observed["ctk"]["enable_thinking"] is False

    _post(client, reasoning_effort="high")
    assert observed["ctk"]["enable_thinking"] is True


def test_no_effort_leaves_the_model_alone(tmp_path, monkeypatch):
    """A client that asks for nothing gets the model's own setting — the
    daemon does not pick a depth on its own."""
    observed: dict = {}
    client = _client(tmp_path, monkeypatch, observed)
    _post(client)
    assert observed["effort"] is None


def test_an_unknown_effort_is_ignored(tmp_path, monkeypatch):
    """A level nobody defined is dropped rather than forwarded: the
    template would answer with its own default, and forwarding a word it
    does not know changes nothing while pretending to."""
    observed: dict = {}
    client = _client(tmp_path, monkeypatch, observed)
    _post(client, reasoning_effort="enormous")
    assert observed["effort"] is None


@pytest.mark.parametrize("level", ["low", "medium", "high", "xhigh", "max"])
def test_every_level_is_forwarded(tmp_path, monkeypatch, level):
    observed: dict = {}
    client = _client(tmp_path, monkeypatch, observed)
    _post(client, reasoning_effort=level)
    assert observed["effort"] == level
