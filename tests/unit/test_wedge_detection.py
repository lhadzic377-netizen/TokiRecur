"""A wedged backend must be reported, not waited on forever.

llama-server keeps a cancelled task's slot flagged busy. After two
cancellations the whole server is wedged: ``/health`` answers 200,
``/v1/models`` lists the model, and every completion queues forever —
so a harness sees a transport error with no hint about why.

The daemon therefore samples the backend's slots and, when every slot is
busy while nothing about them moves, fails fast with a 503 that names
the fix. ``toki doctor`` reports the same thing.
"""

from __future__ import annotations

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
    moving: bool = False
    _ticks: int = 0

    def tokenize(self, text: str) -> list[int]:
        return [hash(w) % 100000 for w in text.split()]

    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]

    def clear_slot(self, slot_id: int) -> dict:
        return {}

    async def slot_snapshot(self) -> list[dict]:
        if not self.moving:
            return list(self.slots)
        # A slot that is decoding advances on every sample.
        self._ticks += 1
        return [
            {**s, "decoded": s.get("decoded", 0) + self._ticks}
            for s in self.slots
        ]

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


def test_chat_completion_fails_fast_when_backend_is_wedged():
    backend = FakeBackend(slots=FROZEN)
    client = _client(backend)
    with client:
        r = client.post("/v1/chat/completions", json={
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        })
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert "wedged" in detail
    assert "toki model use" in detail
    assert backend.chat_payloads == []  # never forwarded to the dead backend


def test_chat_completion_proceeds_when_slots_are_free():
    backend = FakeBackend(slots=[
        {"id": 0, "busy": False, "task": None, "processed": 0,
         "decoded": 0, "remaining": 0},
        {"id": 1, "busy": True, "task": 7, "processed": 12,
         "decoded": 3, "remaining": 5},
    ])
    client = _client(backend)
    with client:
        r = client.post("/v1/chat/completions", json={
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        })
    assert r.status_code == 200
    assert backend.chat_payloads  # forwarded


def test_chat_completion_proceeds_when_a_slot_is_generating():
    """One busy slot that is moving is normal load, not a wedge."""
    backend = FakeBackend(moving=True, slots=[
        {"id": 0, "busy": True, "task": 7, "processed": 12,
         "decoded": 3, "remaining": 5},
        {"id": 1, "busy": True, "task": 8, "processed": 9,
         "decoded": 1, "remaining": 2},
    ])
    client = _client(backend)
    with client:
        r = client.post("/v1/chat/completions", json={
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        })
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


def test_backend_wedged_returns_none_when_a_slot_moves(monkeypatch):
    """A slot that is decoding is generating, not stuck."""
    state = {"n": 0}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            state["n"] += 1
            return [{
                "id": 0, "is_processing": True, "id_task": 7,
                "n_prompt_tokens_processed": 12,
                "next_token": [{"n_decoded": state["n"], "n_remain": 5}],
            }]

    import httpx
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())
    assert diagnose.backend_wedged("http://127.0.0.1:8112", gap=0.01) is None


def test_backend_wedged_returns_none_when_slots_are_idle(monkeypatch):
    _fake_slots(monkeypatch, [
        {"id": 0, "is_processing": False, "id_task": None,
         "n_prompt_tokens_processed": 0, "next_token": []},
    ])
    assert diagnose.backend_wedged("http://127.0.0.1:8112", gap=0.01) is None


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
