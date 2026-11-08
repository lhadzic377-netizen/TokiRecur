"""A slow client must not be abandoned, and must be told the truth.

Claude Code gives a local endpoint ten minutes (API_TIMEOUT_MS) and
treats 300 seconds of silence as a dead stream. On a CPU-only laptop a
prefill can spend that on its own: the client gave up on a turn the
daemon was still working on, and neither side said a number.

Three things have to hold, and they are tested here:

* the daemon outlasts the client (the upstream timeout);
* the client is told to wait (the launcher's budgets);
* both say how long the wait will be (the prefill estimate and the
  measured prefill).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from fastapi.testclient import TestClient

from tokirecur import ctl, hardware, selector
from tokirecur.api.app import create_app
from tokirecur.backend import LlamaServerBackend, backend_timeout

# ── the daemon outlasts the client ───────────────────────────────────


def test_backend_timeout_defaults_to_thirty_minutes(monkeypatch):
    monkeypatch.delenv("TOKI_BACKEND_TIMEOUT", raising=False)
    assert backend_timeout() == 1800.0


def test_backend_timeout_is_configurable(monkeypatch):
    monkeypatch.setenv("TOKI_BACKEND_TIMEOUT", "90")
    assert backend_timeout() == 90.0
    monkeypatch.setenv("TOKI_BACKEND_TIMEOUT", "not a number")
    assert backend_timeout() == 1800.0
    monkeypatch.setenv("TOKI_BACKEND_TIMEOUT", "0")
    assert backend_timeout() == 1800.0


def test_the_backend_client_uses_the_configured_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKI_BACKEND_TIMEOUT", "123")
    backend = LlamaServerBackend("http://127.0.0.1:1", cache_dir=tmp_path)
    assert backend.timeout == 123.0


# ── the client is told to wait ───────────────────────────────────────


def _profile(backend: str):
    return type("P", (), {"recommended": {"backend": backend}})()


def test_claude_gets_a_long_budget_on_a_cpu_only_box(monkeypatch):
    monkeypatch.delenv("TOKI_CLAUDE_TIMEOUT_MS", raising=False)
    monkeypatch.setattr(hardware, "load_profile", lambda: _profile("cpu"))

    env = selector._claude_timeout_env()

    assert env["API_TIMEOUT_MS"] == "1500000"
    assert env["CLAUDE_STREAM_IDLE_TIMEOUT_MS"] == "1500000"


def test_claude_keeps_the_defaults_on_a_gpu_box(monkeypatch):
    monkeypatch.delenv("TOKI_CLAUDE_TIMEOUT_MS", raising=False)
    monkeypatch.setattr(hardware, "load_profile", lambda: _profile("rocm"))

    assert selector._claude_timeout_env() == {}


def test_the_claude_timeout_override_wins(monkeypatch):
    monkeypatch.setenv("TOKI_CLAUDE_TIMEOUT_MS", "900000")
    monkeypatch.setattr(hardware, "load_profile", lambda: _profile("cpu"))

    env = selector._claude_timeout_env()

    assert env["API_TIMEOUT_MS"] == "900000"
    assert env["CLAUDE_STREAM_IDLE_TIMEOUT_MS"] == "900000"


def test_the_launch_plan_carries_the_budget(tmp_path, monkeypatch):
    monkeypatch.delenv("TOKI_CLAUDE_TIMEOUT_MS", raising=False)
    monkeypatch.setenv("TOKI_CLAUDE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("TOKI_CLAUDE_CONFIG", str(tmp_path / ".claude.json"))
    monkeypatch.setenv("TOKI_AUTH_TOKEN", "test-token")
    monkeypatch.setattr(hardware, "load_profile", lambda: _profile("cpu"))

    spec = ctl.ModelSpec(
        name="m", label="m", gguf="/tmp/m.gguf", llama_port=1, daemon_port=2,
        ctx_size=8192, parallel=1, kv="f16", max_tokens=4096, model_id="m",
    )
    plan = selector.launch_cmd("chat", "claude", spec)

    assert plan["env"]["API_TIMEOUT_MS"] == "1500000"
    assert plan["env"]["CLAUDE_STREAM_IDLE_TIMEOUT_MS"] == "1500000"


# ── the numbers ──────────────────────────────────────────────────────


@dataclass
class FakeBackend:
    prefill_tokens: int = 100
    ttft_ms: float = 1000.0
    stream_chunks: list[dict] = field(default_factory=list)

    def tokenize(self, text):
        return text.split()

    def embed(self, text):
        return [0.0, 0.0, 0.0]

    def complete_openai(self, messages, **kwargs):
        return {
            "text": "done",
            "tool_calls": None,
            "finish_reason": "stop",
            "tokens_generated": 2,
            "prefill_tokens": self.prefill_tokens,
            "ttft_ms": self.ttft_ms,
            "total_ms": self.ttft_ms + 10,
            "prefill_tokens_per_sec": self.prefill_tokens / (self.ttft_ms / 1000),
        }

    async def complete_openai_stream(self, messages, **kwargs):
        for item in self.stream_chunks:
            yield item


class FakeIndex:
    def search(self, embedding):
        return None

    def insert(self, entry):
        pass


def _client(backend, tmp_path):
    return TestClient(create_app(
        backend=backend,
        index=FakeIndex(),
        cache_dir=tmp_path,
        enable_auth=False,
        enable_rate_limit=False,
    ))


def test_the_daemon_reports_the_prefill_it_measured(tmp_path):
    """A finished turn carries what the prefill actually cost."""
    backend = FakeBackend(prefill_tokens=120, ttft_ms=600.0)
    with _client(backend, tmp_path) as client:
        response = client.post("/anthropic/v1/messages", json={
            "model": "local",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hello"}],
        })

    assert response.status_code == 200
    prefill = response.json()["toki_prefill"]
    assert prefill["prefill_tokens"] == 120
    assert prefill["prefill_ms"] == 600.0
    assert prefill["prefill_tokens_per_second"] == 200.0


def test_the_daemon_projects_the_next_prefill_before_it_waits(tmp_path):
    """The second turn is told how long it will take, in the first
    event of the stream — before the silence the client mistakes for a
    dead connection."""
    backend = FakeBackend(prefill_tokens=100, ttft_ms=1000.0)
    backend.stream_chunks = [
        {"kind": "delta", "delta": "hi"},
        {"kind": "finish", "finish_reason": "stop"},
        {"kind": "result", "result": {
            "text": "hi",
            "reasoning": "",
            "tool_calls": [],
            "finish_reason": "stop",
            "tokens_generated": 1,
            "prefill_tokens": 100,
            "ttft_ms": 1000.0,
            "total_ms": 1010.0,
            "prefill_tokens_per_sec": 100.0,
        }},
    ]
    long_prompt = "x " * 20000  # ~20k tokens
    with _client(backend, tmp_path) as client:
        # First turn: teaches the daemon how fast this machine prefills.
        client.post("/anthropic/v1/messages", json={
            "model": "local",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hello"}],
        })
        # Second turn: the estimate rides along in message_start.
        response = client.post("/anthropic/v1/messages", json={
            "model": "local",
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": long_prompt}],
        })

    assert response.status_code == 200
    first = response.text.split("\n\n")[0]
    payload = json.loads(first.split("data: ", 1)[1])
    estimate = payload["message"]["toki_prefill_estimate"]
    assert estimate["prefill_tokens_per_second"] == 100.0
    assert estimate["prompt_tokens_estimated"] > 5000
    assert estimate["prefill_seconds_estimated"] > 50


def test_a_slow_prefill_is_said_out_loud(tmp_path, caplog):
    """The number reaches the log before the wait, not after the client
    has already given up."""
    backend = FakeBackend(prefill_tokens=100, ttft_ms=1000.0)
    backend.stream_chunks = [
        {"kind": "delta", "delta": "hi"},
        {"kind": "finish", "finish_reason": "stop"},
        {"kind": "result", "result": {
            "text": "hi",
            "reasoning": "",
            "tool_calls": [],
            "finish_reason": "stop",
            "tokens_generated": 1,
            "prefill_tokens": 100,
            "ttft_ms": 1000.0,
            "total_ms": 1010.0,
            "prefill_tokens_per_sec": 100.0,
        }},
    ]
    long_prompt = "x " * 20000
    with _client(backend, tmp_path) as client:
        client.post("/anthropic/v1/messages", json={
            "model": "local",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hello"}],
        })
        with caplog.at_level(logging.WARNING):
            client.post("/anthropic/v1/messages", json={
                "model": "local",
                "max_tokens": 64,
                "stream": True,
                "messages": [{"role": "user", "content": long_prompt}],
            })

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("slow prefill expected" in w for w in warnings), warnings
    assert any("tok/s" in w for w in warnings), warnings
