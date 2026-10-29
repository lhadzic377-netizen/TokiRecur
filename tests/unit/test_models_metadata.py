"""The ids a daemon advertises must be the ids clients are configured with.

Every client config TokiRecur writes points at the registry's
``model_id`` (``minicpm5-1b-q4_k_m``), while llama-server reports the id
of the weights it loaded (``MiniCPM5-1B-Q4_K_M``). A harness that looks
its configured id up in ``/v1/models`` — codex does, and warns "Model
metadata not found ... this can degrade performance" — finds nothing and
falls back to guessed metadata.

So the daemon advertises the registry id, and keeps the served id listed
beside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fastapi.testclient import TestClient

from tokirecur.api.app import create_app


@dataclass
class FakeBackend:
    base_url: str = "http://127.0.0.1:8112"
    served_id: str = "MiniCPM5-1B-Q4_K_M"
    chat_payloads: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self._client = _FakeClient(self.served_id)

    def tokenize(self, text: str) -> list[int]:
        return [hash(w) % 100000 for w in text.split()]

    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]

    def clear_slot(self, slot_id: int) -> dict:
        return {}

    def _get(self, path: str) -> dict:
        return {}

    def _post(self, path: str, payload: dict) -> dict:
        return {}

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


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    def __init__(self, served_id: str) -> None:
        self.served_id = served_id

    def get(self, path: str, timeout: float = 3.0):
        return _FakeResponse({"data": [{"id": self.served_id}]})


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
        return {"entries": len(self.entries), "bytes_on_disk": 0,
                "embedding_dim": 4}


def _client(backend) -> TestClient:
    return TestClient(create_app(
        backend=backend,
        index=FakeIndex(),
        cache_dir="/tmp/toki-tests",
        enable_auth=False,
        enable_rate_limit=False,
    ))


def test_models_list_advertises_the_registry_model_id(monkeypatch):
    """The id a client is configured with must be in /v1/models."""
    import tokirecur.api.app as app_mod

    monkeypatch.setattr(app_mod, "_active_model_id", lambda: "minicpm5-1b-q4_k_m")

    with _client(FakeBackend()) as client:
        r = client.get("/v1/models")
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["data"]]
    assert "minicpm5-1b-q4_k_m" in ids, ids
    # The id of the weights stays listed too.
    assert "MiniCPM5-1B-Q4_K_M" in ids, ids


def test_models_list_falls_back_to_the_served_id(monkeypatch):
    import tokirecur.api.app as app_mod

    monkeypatch.setattr(app_mod, "_active_model_id", lambda: None)

    with _client(FakeBackend()) as client:
        r = client.get("/v1/models")
    ids = [m["id"] for m in r.json()["data"]]
    assert ids == ["MiniCPM5-1B-Q4_K_M"], ids
