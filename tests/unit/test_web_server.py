"""Unit tests for the web UI server (`tokirecur.web.server`).

These exercise the message/payload builders, the SSE stream parser, the
Prometheus metrics parsing, and the WebSocket chat protocol using fake
httpx responses — no live backends required.
"""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from tokirecur.web import server as server_mod
from tokirecur.web.server import (
    _build_messages,
    _completion_payload,
    _iter_sse_chunks,
    _parse_prometheus_metrics,
    create_web_app,
)

# ── Fakes ──────────────────────────────────────────────────────────────


def sse(*objs):
    return "".join(f"data: {json.dumps(o)}\n\n" for o in objs) + "data: [DONE]\n\n"


class FakeStreamResponse:
    def __init__(self, sse_text, status_code=200):
        self.status_code = status_code
        self._sse_text = sse_text
        self._chunks = sse_text.splitlines(keepends=True)

    async def aiter_text(self):
        for c in self._chunks:
            await asyncio.sleep(0.001)
            yield c

    async def aread(self):
        return self._sse_text.encode()


class FakeStreamContext:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *a):
        return False


class FakeResponse:
    def __init__(self, payload=None, status_code=200, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload

    async def aread(self):
        return self.text.encode()


class FakeHttpx:
    """Drop-in for httpx.AsyncClient: records calls, returns fake responses."""

    def __init__(self, *, stream_response=None, post_response=None, get_response=None):
        self.stream_response = stream_response
        self.post_response = post_response
        self.get_response = get_response
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def stream(self, method, url, **kw):
        self.calls.append(("stream", method, url, kw))
        return FakeStreamContext(self.stream_response)

    async def post(self, url, **kw):
        self.calls.append(("post", url, kw))
        return self.post_response

    async def get(self, url, **kw):
        self.calls.append(("get", url, kw))
        return self.get_response


@pytest.fixture
def fake_async_client(monkeypatch):
    """Patch httpx.AsyncClient with a configurable fake; returns a factory."""

    def _install(fake):
        monkeypatch.setattr(server_mod.httpx, "AsyncClient", lambda *a, **k: fake)
        return fake

    return _install


# ── Message / payload builders ─────────────────────────────────────────


class TestBuildMessages:
    def test_system_history_and_current(self):
        messages = _build_messages(
            [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
            ],
            system="be terse",
            current_prompt="p2",
        )
        assert messages == [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "p2"},
        ]

    def test_invalid_roles_filtered(self):
        messages = _build_messages(
            [{"role": "tool", "content": "x"}, {"role": "", "content": "y"}],
            system=None,
            current_prompt="p",
        )
        assert messages == [{"role": "user", "content": "p"}]


class TestCompletionPayload:
    def test_raw_uses_llama_server_openai_with_cache_prompt(self, monkeypatch):
        # The product disables cache_prompt on Windows on purpose
        # (the CPU build can hang on the multi-turn path). The
        # documented override makes the intent explicit here.
        monkeypatch.setenv("TOKI_CACHE_PROMPT", "1")
        endpoint, payload = _completion_payload(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=16,
            temperature=0.0,
            mode="raw",
            raw_url="http://raw:8088",
            daemon_url="http://daemon:8787",
            session_id="s1",
        )
        assert endpoint == "http://raw:8088/v1/chat/completions"
        assert payload["cache_prompt"] is True
        # Thinking mode is decided by the backend (daemon / llama-server
        # template default), not the web client.
        assert "chat_template_kwargs" not in payload
        assert "session_id" not in payload
        assert payload["messages"] == [{"role": "user", "content": "hi"}]

    def test_cached_uses_daemon_with_session(self):
        endpoint, payload = _completion_payload(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=16,
            temperature=0.5,
            mode="cached",
            raw_url="http://raw:8088",
            daemon_url="http://daemon:8787",
            session_id="s1",
        )
        assert endpoint == "http://daemon:8787/v1/chat/completions"
        assert payload["session_id"] == "s1"
        assert payload["max_slots"] == 4
        assert "chat_template_kwargs" not in payload
        assert "cache_prompt" not in payload

    def test_raw_disables_cache_prompt_on_windows(self, monkeypatch):
        # Windows CPU llama-server can hang on the multi-turn KV-reuse
        # path (first message works, second spins without tokens), so
        # the raw path must not send cache_prompt there by default.
        monkeypatch.setattr("os.name", "nt")
        _, payload = _completion_payload(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=16,
            temperature=0.0,
            mode="raw",
            raw_url="http://raw:8088",
            daemon_url="http://daemon:8787",
            session_id="s1",
        )
        assert payload["cache_prompt"] is False

    def test_toki_cache_prompt_env_overrides_default(self, monkeypatch):
        monkeypatch.setattr("os.name", "nt")
        monkeypatch.setenv("TOKI_CACHE_PROMPT", "1")
        _, payload = _completion_payload(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=16,
            temperature=0.0,
            mode="raw",
            raw_url="http://raw:8088",
            daemon_url="http://daemon:8787",
            session_id="s1",
        )
        assert payload["cache_prompt"] is True
        monkeypatch.setattr("os.name", "posix")
        monkeypatch.setenv("TOKI_CACHE_PROMPT", "0")
        _, payload = _completion_payload(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=16,
            temperature=0.0,
            mode="raw",
            raw_url="http://raw:8088",
            daemon_url="http://daemon:8787",
            session_id="s1",
        )
        assert payload["cache_prompt"] is False


# ── SSE parser ─────────────────────────────────────────────────────────


class TestIterSseChunks:
    @pytest.mark.asyncio
    async def test_content_then_usage_and_cache(self):
        text = sse(
            {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {"content": "!"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                "usage": {"prompt_tokens": 25, "completion_tokens": 2, "total_tokens": 27},
                "cache": {"hit": False, "slot_id": 1, "skipped_tokens": 0},
            },
        )
        events = [ev async for ev in _iter_sse_chunks(FakeStreamResponse(text))]
        assert events[0] == {"content": "hi"}
        assert events[1] == {"content": "!"}
        assert events[2]["usage"]["prompt_tokens"] == 25
        assert events[2]["cache"]["slot_id"] == 1

    @pytest.mark.asyncio
    async def test_ignores_garbage_and_empty_deltas(self):
        text = (
            "data: {not json}\n\n"
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":null}]}\n\n'
            'data: {"content":""}\n\n'
            "data: [DONE]\n\n"
        )
        events = [ev async for ev in _iter_sse_chunks(FakeStreamResponse(text))]
        assert events == []

    @pytest.mark.asyncio
    async def test_llama_server_completion_format(self):
        text = sse({"content": "legacy"})
        events = [ev async for ev in _iter_sse_chunks(FakeStreamResponse(text))]
        assert events == [{"content": "legacy"}]

    @pytest.mark.asyncio
    async def test_extracts_top_level_reasoning_content(self):
        """Qwen3 / DeepSeek-R1 templates stream reasoning at the top
        level via ``reasoning_content``. The parser should pull it
        out as a ``reasoning`` event, distinct from ``content``."""
        text = sse(
            {"reasoning_content": "let me think about this..."},
            {"reasoning_content": " step two."},
            {"content": "the answer"},
        )
        events = [ev async for ev in _iter_sse_chunks(FakeStreamResponse(text))]
        assert events == [
            {"reasoning": "let me think about this..."},
            {"reasoning": " step two."},
            {"content": "the answer"},
        ]

    @pytest.mark.asyncio
    async def test_extracts_reasoning_content_from_delta(self):
        """Some templates nest reasoning under choices[0].delta.
        The parser should pick that up too."""
        text = sse(
            {"choices": [{"index": 0, "delta": {"reasoning_content": "thinking..."}}]},
            {"choices": [{"index": 0, "delta": {"content": "answer"}}]},
        )
        events = [ev async for ev in _iter_sse_chunks(FakeStreamResponse(text))]
        assert events == [
            {"reasoning": "thinking..."},
            {"content": "answer"},
        ]

    @pytest.mark.asyncio
    async def test_reasoning_only_no_visible_content(self):
        """A response that streams reasoning but no answer should
        yield only reasoning events. The caller is responsible for
        flagging the no-answer case on the client; the parser just
        reports what it saw."""
        text = sse(
            {"reasoning_content": "thinking really hard"},
            {"reasoning_content": " about the answer..."},
        )
        events = [ev async for ev in _iter_sse_chunks(FakeStreamResponse(text))]
        assert events == [
            {"reasoning": "thinking really hard"},
            {"reasoning": " about the answer..."},
        ]
        assert all("content" not in ev for ev in events)


# ── Prometheus metrics parsing ─────────────────────────────────────────


class TestParsePrometheusMetrics:
    def test_labeled_counters(self):
        text = (
            "# HELP toki_requests_total total\n"
            "# TYPE toki_requests_total counter\n"
            "toki_requests_total 12\n"
            'toki_cache_hits_total{kind="hit"} 5\n'
            'toki_cache_hits_total{kind="miss"} 3\n'
            "toki_tokens_skipped_total 42\n"
            "toki_active_slots 2\n"
            "toki_auth_failures_total 1\n"
        )
        out = _parse_prometheus_metrics(text)
        assert out["requests_total"] == 12
        assert out["cache_hits"] == 5
        assert out["cache_misses"] == 3
        assert out["tokens_skipped"] == 42
        assert out["active_slots"] == 2
        assert out["auth_failures"] == 1

    def test_empty_text(self):
        assert _parse_prometheus_metrics("") == {
            "requests_total": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "tokens_skipped": 0,
            "active_slots": 0,
            "auth_failures": 0,
        }

    def test_metrics_endpoint_parses_daemon_text(self, fake_async_client):
        text = (
            "toki_requests_total 3\n"
            'toki_cache_hits_total{kind="hit"} 1\n'
            'toki_cache_hits_total{kind="miss"} 2\n'
        )
        fake = FakeHttpx(get_response=FakeResponse(payload=None, text=text))
        fake_async_client(fake)
        app = create_web_app(raw_url="http://raw:8088", daemon_url="http://daemon:8787")
        with TestClient(app) as client:
            r = client.get("/v1/metrics/live")
        assert r.json()["cache_hits"] == 1
        assert r.json()["cache_misses"] == 2
        assert r.json()["requests_total"] == 3
        assert fake.calls[0][1] == "http://daemon:8787/metrics"


# ── Precision proxy ────────────────────────────────────────────────────


class TestPrecisionProxy:
    def test_proxies_daemon_endpoint(self, fake_async_client):
        body = {
            "baseline_deterministic": 1.0,
            "recycled_deterministic": 1.0,
            "byte_identical_to_baseline": 1.0,
            "prompts_tested": 3,
            "per_prompt": [],
        }
        fake = FakeHttpx(post_response=FakeResponse(payload=body))
        fake_async_client(fake)
        app = create_web_app(raw_url="http://raw:8088", daemon_url="http://daemon:8787")
        with TestClient(app) as client:
            r = client.post("/v1/precision", json={"max_tokens": 32})
        assert r.status_code == 200
        assert r.json() == body
        assert fake.calls[0][1] == "http://daemon:8787/v1/precision"

    def test_daemon_unreachable_returns_error(self, fake_async_client):
        fake = FakeHttpx(post_response=FakeResponse(payload={}, status_code=503))
        fake_async_client(fake)
        app = create_web_app(raw_url="http://raw:8088", daemon_url="http://daemon:8787")
        with TestClient(app) as client:
            r = client.post("/v1/precision", json={})
        assert "error" in r.json()


# ── WebSocket chat ─────────────────────────────────────────────────────


def _chat_client(monkeypatch, fake):
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", lambda *a, **k: fake)
    return create_web_app(raw_url="http://raw:8088", daemon_url="http://daemon:8787")


class TestWsChat:
    def test_cached_done_carries_usage_and_cache(self, fake_async_client):
        fake = FakeHttpx(
            stream_response=FakeStreamResponse(
                sse(
                    {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
                    {"choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]},
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                    {
                        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                        "usage": {"prompt_tokens": 25, "completion_tokens": 1, "total_tokens": 26},
                        "cache": {"hit": False, "slot_id": 1, "skipped_tokens": 0},
                    },
                )
            )
        )
        fake_async_client(fake)
        app = create_web_app(raw_url="http://raw:8088", daemon_url="http://daemon:8787")
        with TestClient(app) as client:
            with client.websocket_connect("/ws/chat") as ws:
                ws.send_json({
                    "mode": "cached", "prompt": "hi", "max_tokens": 8,
                    "temperature": 0.0, "history": [], "session_id": "s1",
                })
                done = None
                while True:
                    msg = ws.receive_json()
                    if msg["type"] == "done":
                        done = msg
                        break
        assert done["text"] == "Hello"
        assert done["metrics"]["prompt_tokens"] == 25
        assert done["metrics"]["tokens_generated"] == 1
        assert done["cache"]["slot_id"] == 1
        stream_calls = [c for c in fake.calls if c[0] == "stream"]
        assert stream_calls[0][2] == "http://daemon:8787/v1/chat/completions"
        sent = stream_calls[0][3]["json"]
        assert sent["session_id"] == "s1"
        assert "chat_template_kwargs" not in sent

    def test_raw_uses_llama_server_openai_with_cache_prompt(self, fake_async_client, monkeypatch):
        # The product disables cache_prompt on Windows on purpose
        # (the CPU build can hang on the multi-turn path). The
        # documented override makes the intent explicit here.
        monkeypatch.setenv("TOKI_CACHE_PROMPT", "1")
        fake = FakeHttpx(
            stream_response=FakeStreamResponse(
                sse({"choices": [{"index": 0, "delta": {"content": "yo"}, "finish_reason": None}]})
            )
        )
        fake_async_client(fake)
        app = create_web_app(raw_url="http://raw:8088", daemon_url="http://daemon:8787")
        with TestClient(app) as client:
            with client.websocket_connect("/ws/chat") as ws:
                ws.send_json({
                    "mode": "raw", "prompt": "hi", "max_tokens": 8,
                    "temperature": 0.0, "history": [], "session_id": "s1",
                })
                done = None
                while True:
                    msg = ws.receive_json()
                    if msg["type"] == "done":
                        done = msg
                        break
        assert done["text"] == "yo"
        stream_calls = [c for c in fake.calls if c[0] == "stream"]
        assert stream_calls[0][2] == "http://raw:8088/v1/chat/completions"
        sent = stream_calls[0][3]["json"]
        assert sent["cache_prompt"] is True
        assert "chat_template_kwargs" not in sent
        assert "session_id" not in sent

    def test_abort_stops_stream_and_reports_partial(self, fake_async_client):
        chunks = [
            {"choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}]}
            for _ in range(50)
        ]
        fake = FakeHttpx(stream_response=FakeStreamResponse(sse(*chunks)))
        fake_async_client(fake)
        app = create_web_app(raw_url="http://raw:8088", daemon_url="http://daemon:8787")
        with TestClient(app) as client:
            with client.websocket_connect("/ws/chat") as ws:
                ws.send_json({
                    "mode": "cached", "prompt": "long", "max_tokens": 8,
                    "temperature": 0.0, "history": [], "session_id": "s1",
                })
                done = None
                saw_token = False
                for _ in range(300):
                    msg = ws.receive_json()
                    if msg["type"] == "token" and not saw_token:
                        saw_token = True
                        ws.send_json({"type": "abort"})
                    if msg["type"] == "done":
                        done = msg
                        break
        assert done is not None
        assert done["aborted"] is True
        assert done["text"]  # partial output preserved

    def test_socket_open_without_any_session(self, fake_async_client):
        """The UI is always open: no login, no session cookie needed —
        even when a daemon token is configured on this machine."""
        fake = FakeHttpx(
            stream_response=FakeStreamResponse(
                sse({"choices": [{"index": 0, "delta": {"content": "yo"}, "finish_reason": None}]})
            )
        )
        fake_async_client(fake)
        app = create_web_app(raw_url="http://raw:8088", daemon_url="http://daemon:8787")
        with TestClient(app) as client:
            with client.websocket_connect("/ws/chat") as ws:
                ws.send_json({"mode": "raw", "prompt": "hi", "max_tokens": 8})
                done = None
                while True:
                    msg = ws.receive_json()
                    if msg["type"] == "done":
                        done = msg
                        break
        assert done["text"] == "yo"
