"""A wedged backend must be reported, not waited on forever.

llama-server keeps a cancelled task's slot flagged busy. After two
cancellations the whole server is wedged: ``/health`` answers 200,
``/v1/models`` lists the model, and every completion queues forever —
so a harness sees a transport error with no hint about why.

The daemon therefore watches the backend's slots and, when every slot has
produced *nothing at all* for a long time, fails fast with a 503 that
names the fix. ``toki doctor`` reports the same thing.

The hard part is not noticing a stall but not inventing one: a server
prefilling a long prompt reports no progress either (its prompt counter
moves in 2048-token blocks), so a busy backend is not a wedged one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from tokirecur.api.app import create_app
from tokirecur import diagnose


@dataclass
class FakeBackend:
    """Backend whose slot snapshot is frozen (every slot busy, no progress)."""

    base_url: str = "http://127.0.0.1:8112"
    slots: list = field(default_factory=list)
    chat_payloads: list = field(default_factory=list)

    def tokenize(self, text: str) -> list[int]:
        return [hash(w) % 100000 for w in text.split()]

    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]

    def clear_slot(self, slot_id: int) -> dict:
        return {}

    async def slot_snapshot(self) -> list[dict]:
        return list(self.slots)

    def complete_openai(self, messages, **kw):
        self.chat_payloads.append(messages)
        return {"text": "ok", "reasoning": "", "tool_calls": None,
                "finish_reason": "stop", "tokens_generated": 1,
                "prefill_tokens": 1, "ttft_ms": 1.0, "total_ms": 1.0,
                "prefill_tokens_per_sec": 0.0}

    async def complete_openai_stream(self, messages, **kw):
        yield {"kind": "delta", "delta": "ok"}
        yield {"kind": "result", "result": {"text": "ok", "reasoning": "",
                                             "tool_calls": None,
                                             "finish_reason": "stop",
                                             "tokens_generated": 1,
                                             "prefill_tokens": 1,
                                             "ttft_ms": 1.0, "total_ms": 1.0,
                                             "prefill_tokens_per_sec": 0.0}}


class FakeIndex:
    """Mirrors the index double used by the other API tests."""

    def __init__(self) -> None:
        self.entries: dict[str, object] = {}

    def search(self, embedding):
        return None

    def insert(self, entry) -> None:
        self.entries[entry.cache_id] = entry

    def touch(self, cache_id: str) -> None:
        pass

    def stats(self) -> dict:
        return {"entries": len(self.entries), "bytes_on_disk": 0,
                "embedding_dim": 4}


FROZEN = [
    {"id": 0, "busy": True, "task": 38434, "processed": 0,
     "decoded": 0, "remaining": 0},
    {"id": 1, "busy": True, "task": 38435, "processed": 0,
     "decoded": 0, "remaining": 0},
]


def _client(backend) -> TestClient:
    return TestClient(create_app(
        backend=backend,
        index=FakeIndex(),
        cache_dir="/tmp/toki-tests",
        enable_auth=False,
        enable_rate_limit=False,
    ))


def _watch(app, snapshot, idle_for: float):
    """Put the daemon's slot watcher in a given state."""
    app.state.slot_watch = {
        "snapshot": list(snapshot),
        "last_progress": time.time() - idle_for,
    }


def _ask(client, app, snapshot, idle_for: float):
    _watch(app, snapshot, idle_for)
    return client.post("/v1/chat/completions", json={
        "model": "x",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
    })


def test_chat_completion_fails_fast_when_backend_is_wedged():
    backend = FakeBackend(slots=FROZEN)
    with _client(backend) as client:
        app = client.app
        r = _ask(client, app, FROZEN, idle_for=600)
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert "wedged" in detail
    assert "toki model use" in detail
    assert backend.chat_payloads == []  # never forwarded to the dead backend


def test_chat_completion_proceeds_when_slots_are_free():
    backend = FakeBackend()
    free = [
        {"id": 0, "busy": False, "task": None, "processed": 0,
         "decoded": 0, "remaining": 0},
        {"id": 1, "busy": True, "task": 7, "processed": 12,
         "decoded": 3, "remaining": 5},
    ]
    with _client(backend) as client:
        r = _ask(client, client.app, free, idle_for=600)
    assert r.status_code == 200
    assert backend.chat_payloads  # forwarded


def test_chat_completion_proceeds_while_a_slot_is_prefilling():
    """A server prefilling a long prompt reports no decoded tokens. It is
    working, not wedged — calling it wedged refuses valid requests."""
    backend = FakeBackend()
    prefilling = [
        {"id": 0, "busy": True, "task": 3, "processed": 20480,
         "decoded": 0, "remaining": 2048},
    ]
    with _client(backend) as client:
        r = _ask(client, client.app, prefilling, idle_for=600)
    assert r.status_code == 200
    assert backend.chat_payloads


def test_chat_completion_proceeds_when_a_slot_is_generating():
    """One busy slot that is decoding is normal load, not a wedge."""
    backend = FakeBackend()
    generating = [
        {"id": 0, "busy": True, "task": 7, "processed": 12,
         "decoded": 3, "remaining": 5},
        {"id": 1, "busy": True, "task": 8, "processed": 9,
         "decoded": 1, "remaining": 2},
    ]
    with _client(backend) as client:
        r = _ask(client, client.app, generating, idle_for=600)
    assert r.status_code == 200


def test_chat_completion_proceeds_while_the_stall_is_recent():
    """Nothing produced yet, but only just: not a wedge yet."""
    backend = FakeBackend()
    with _client(backend) as client:
        r = _ask(client, client.app, FROZEN, idle_for=5)
    assert r.status_code == 200


def _fake_slots(monkeypatch, payload):
    import httpx

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return payload

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())


def test_backend_wedged_reports_frozen_slots(monkeypatch):
    _fake_slots(monkeypatch, [
        {"id": 0, "is_processing": True, "id_task": 38434,
         "n_prompt_tokens_processed": 0,
         "next_token": [{"n_decoded": 0, "n_remain": 0}]},
        {"id": 1, "is_processing": True, "id_task": 38435,
         "n_prompt_tokens_processed": 0,
         "next_token": [{"n_decoded": 0, "n_remain": 0}]},
    ])
    out = diagnose.backend_wedged("http://127.0.0.1:8112", gap=0.01)
    assert out and "wedged" in out
    assert "38434" in out


def test_backend_wedged_ignores_a_server_that_is_prefilling(monkeypatch):
    """The prompt counter moves in 2048-token blocks; a server working
    through a long prompt must not be called wedged."""
    _fake_slots(monkeypatch, [
        {"id": 0, "is_processing": True, "id_task": 3,
         "n_prompt_tokens_processed": 20480,
         "next_token": [{"n_decoded": 0, "n_remain": 2048}]},
    ])
    assert diagnose.backend_wedged("http://127.0.0.1:8112", gap=0.01) is None


def test_backend_wedged_returns_none_when_slots_are_idle(monkeypatch):
    _fake_slots(monkeypatch, [
        {"id": 0, "is_processing": False, "id_task": None,
         "n_prompt_tokens_processed": 0, "next_token": []},
    ])
    assert diagnose.backend_wedged("http://127.0.0.1:8112", gap=0.01) is None


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])


class SlowBackend(FakeBackend):
    """A backend that takes its time before the first token — a long
    prefill on a slow CPU."""

    def __init__(self, delay: float = 0.6, **kw) -> None:
        super().__init__(**kw)
        self.delay = delay

    async def complete_openai_stream(self, messages, **kw):
        import asyncio
        await asyncio.sleep(self.delay)
        yield {"kind": "delta", "delta": "ok"}
        yield {"kind": "result", "result": {"text": "ok", "reasoning": "",
                                             "tool_calls": None,
                                             "finish_reason": "stop",
                                             "tokens_generated": 1,
                                             "prefill_tokens": 1,
                                             "ttft_ms": 1.0, "total_ms": 1.0,
                                             "prefill_tokens_per_sec": 0.0}}


def test_chat_stream_keeps_the_wire_alive_while_the_backend_is_busy(monkeypatch):
    """A client with a stream-idle timeout must not see a dead
    connection while the backend prefills."""
    import tokirecur.api.app as app_mod

    monkeypatch.setattr(app_mod, "STREAM_HEARTBEAT_SECONDS", 0.2)
    backend = SlowBackend(delay=0.7)
    with _client(backend) as client:
        app = client.app
        _watch(app, [], idle_for=0)
        with client.stream(
            "POST", "/v1/chat/completions",
            json={"model": "x",
                  "messages": [{"role": "user", "content": "hi"}],
                  "max_tokens": 8, "stream": True},
        ) as response:
            chunks = [line for line in response.iter_lines()
                      if line.startswith("data:")]
    # At least one heartbeat arrived before the model's own chunk.
    assert len(chunks) >= 2, chunks
    assert chunks[-1].strip() == "data: [DONE]", chunks[-1]
