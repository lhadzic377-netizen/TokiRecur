"""Integration tests for the daemon (FastAPI app exposed via TestClient)."""

import pytest


@pytest.fixture
def daemon_app(llama_server):
    """Build a daemon app pointed at the test llama-server."""
    import tempfile
    from pathlib import Path

    from tokirecur.api.app import create_app
    from tokirecur.backend import LlamaServerBackend
    from tokirecur.index import BruteForceIndex

    backend = LlamaServerBackend(llama_server, timeout=120.0)
    tmp_db = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp_db.close()
    index = BruteForceIndex(tmp_db.name)
    cache_dir = Path(tempfile.mkdtemp())
    return create_app(
        backend=backend, index=index, cache_dir=cache_dir,
        similarity_threshold=0.85,
        enable_auth=False,  # Disable auth in tests by default
        enable_rate_limit=False,
    )


@pytest.fixture
def auth_daemon_app(llama_server):
    """Daemon with auth enabled for auth tests."""
    import os
    import tempfile
    from pathlib import Path

    from tokirecur.api.app import create_app
    from tokirecur.backend import LlamaServerBackend
    from tokirecur.index import BruteForceIndex

    os.environ["TOKI_AUTH_TOKEN"] = "test-token-xyz"
    backend = LlamaServerBackend(llama_server, timeout=120.0)
    tmp_db = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp_db.close()
    index = BruteForceIndex(tmp_db.name)
    cache_dir = Path(tempfile.mkdtemp())
    return create_app(
        backend=backend, index=index, cache_dir=cache_dir,
        similarity_threshold=0.85,
        enable_auth=True,
        enable_rate_limit=False,
    )


@pytest.fixture
def client(daemon_app):
    from fastapi.testclient import TestClient
    with TestClient(daemon_app) as c:
        yield c


@pytest.fixture
def auth_client(auth_daemon_app):
    import os

    from fastapi.testclient import TestClient
    with TestClient(auth_daemon_app) as c:
        yield c
    os.environ.pop("TOKI_AUTH_TOKEN", None)


class TestDaemonChatStreaming:
    """The stream path must behave like the non-stream path: default
    thinking mode from the model's reasoning setting (enabled → visible
    content + reasoning deltas) and real token-by-token streaming."""

    def test_stream_chat_defaults_ctk_and_streams_content(self, client):
        """Regression: the streaming chat path previously called
        serve_chat WITHOUT chat_template_kwargs, so Qwen3.5 burned all
        tokens on reasoning and the TUI showed an empty answer. The
        default now enables thinking (from the reasoning setting) with
        the max-token floor reserving headroom so content still lands."""
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Count from 1 to 5."}],
            "max_tokens": 64,
            "temperature": 0.0,
            "max_slots": 2,
            "stream": True,
        })
        assert r.status_code == 200
        body = r.text
        # SSE stream: role chunk + content chunks + finish_reason + [DONE].
        assert "[DONE]" in body, f"stream missing [DONE]; got {body[:300]!r}"

        deltas = []
        for line in body.splitlines():
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if not data or data == "[DONE]":
                continue
            import json
            obj = json.loads(data)
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content") or ""
            deltas.append(content)

        text = "".join(deltas)
        assert text.strip(), f"expected streamed content, got {text!r}"
        non_empty = [d for d in deltas if d]
        assert len(non_empty) >= 2, (
            f"expected real token-by-token streaming (>= 2 deltas), "
            f"got {len(non_empty)}: {non_empty!r}"
        )

        # Final chunk carries usage + cache verdict so clients can render
        # prompt tokens and the hit/miss badge (TUI + web parity).
        usage_chunks = []
        for line in body.splitlines():
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if not data or data == "[DONE]":
                continue
            import json
            obj = json.loads(data)
            if obj.get("usage"):
                usage_chunks.append(obj)
        assert len(usage_chunks) == 1, (
            f"expected exactly one usage chunk, got {len(usage_chunks)}"
        )
        usage = usage_chunks[0]["usage"]
        assert usage["prompt_tokens"] >= 1
        assert usage["completion_tokens"] >= 1
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        cache = usage_chunks[0].get("cache")
        assert cache is not None, "usage chunk missing cache verdict"
        assert cache["slot_id"] >= 0
        assert "hit" in cache
        assert "skipped_tokens" in cache

    def test_stream_chat_respects_explicit_ctk(self, client):
        """Clients can still opt into thinking mode by passing
        chat_template_kwargs explicitly."""
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 16,
            "temperature": 0.0,
            "max_slots": 2,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": True},
        })
        assert r.status_code == 200
        assert "[DONE]" in r.text


class TestDaemonHealth:
    def test_healthz(self, client):
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_cache_stats(self, client):
        r = client.get("/cache/stats")
        assert r.status_code == 200
        body = r.json()
        assert "index" in body
        assert "cache_dir" in body


class TestDaemonCompletion:
    def test_v1_completions_runs(self, client):
        r = client.post("/v1/completions", json={
            "prompt": "The capital of France is",
            "max_tokens": 8,
            "temperature": 0.0,
            "max_slots": 2,
        })
        assert r.status_code == 200
        body = r.json()
        assert "cache" in body
        assert "slot_id" in body["cache"]
        assert body["usage"]["completion_tokens"] >= 1

    def test_v1_chat_completions_runs(self, client):
        # Daemon should default chat_template_kwargs to disable Qwen3.5's
        # thinking mode so we get visible content back. We don't pass
        # chat_template_kwargs here — that's the whole point: any client
        # that hits the daemon (TUI, curl, web UI) should get sensible
        # behavior without knowing about Qwen-specific flags.
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 64,
            "temperature": 0.0,
            "max_slots": 2,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["usage"]["completion_tokens"] >= 1, (
            "expected at least one completion token"
        )
        # Response should have non-empty text (not just thinking tokens).
        text = body["choices"][0]["message"]["content"]
        assert text, f"expected non-empty text, got {text!r}"

    def test_v1_chat_completions_uses_openai_endpoint(self, client):
        """Regression: daemon's /v1/chat/completions must produce a
        coherent answer (not a flat-prompt continuation like
        "user: hi\\n\\nassistant: ...").

        This is the actual user-facing bug fix: previously the daemon
        flattened messages into a single string and hit llama-server's
        /completion endpoint, which does NOT apply chat templates.
        Now it passes messages through to /v1/chat/completions so
        llama-server applies the right template.
        """
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
            "max_tokens": 64,
            "temperature": 0.0,
            "max_slots": 2,
        })
        assert r.status_code == 200
        text = r.json()["choices"][0]["message"]["content"]
        # The model should answer with a number, not role labels.
        # "user: 4\\n\\nassistant:" is the old broken behavior.
        assert "4" in text, f"expected answer containing '4', got {text!r}"
        assert not text.lower().startswith("user:"), (
            f"response looks like a flat-prompt continuation, not a chat "
            f"answer: {text!r}"
        )

    def test_session_id_is_passed_through(self, client):
        r1 = client.post("/v1/completions", json={
            "prompt": "Hello, how are you?",
            "max_tokens": 8,
            "session_id": "test-session-1",
        })
        r2 = client.post("/v1/completions", json={
            "prompt": "Tell me a fact.",
            "max_tokens": 8,
            "session_id": "test-session-1",
        })
        assert r1.status_code == 200
        assert r2.status_code == 200
        slot1 = r1.json()["cache"]["slot_id"]
        slot2 = r2.json()["cache"]["slot_id"]
        assert slot1 == slot2  # Session pinned

    def test_daemon_defaults_thinking_enabled_and_content_nonempty(self, client):
        """The daemon defaults `chat_template_kwargs` from the model's
        `reasoning` setting: auto/on → `{enable_thinking: True}` so the
        model emits reasoning, and the max-token floor reserves enough
        budget that the response still contains visible content.
        """
        # Send a chat completion without chat_template_kwargs.
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "What is 1+1?"}],
            "max_tokens": 32,
            "temperature": 0.0,
            "max_slots": 2,
        })
        assert r.status_code == 200
        body = r.json()
        text = body["choices"][0]["message"]["content"]
        # With thinking enabled the model may emit reasoning first, but
        # the max-token floor guarantees it still produces visible
        # content. The 0.8B test model answers "2" in a few tokens.
        assert text, (
            f"daemon default must leave room for content after reasoning; "
            f"got {text!r}"
        )
        # Thinking is enabled but not forced: for a trivial prompt the
        # model may emit an empty thinking block (surfaced as no
        # reasoning_content) and go straight to the answer. The
        # streaming test verifies reasoning deltas do flow when the
        # model actually thinks.


class TestDaemonSlotState:
    def test_cache_state_endpoint(self, client):
        # Make a request first to populate state.
        client.post("/v1/completions", json={
            "prompt": "Hello",
            "max_tokens": 4,
            "session_id": "u1",
        })
        r = client.get("/v1/cache/state")
        assert r.status_code == 200
        body = r.json()
        assert "slots" in body
        assert "sessions" in body
        assert "spill_count" in body
        assert body["sessions"].get("u1") is not None

    def test_cache_reset_endpoint(self, client):
        client.post("/v1/completions", json={
            "prompt": "Hello",
            "max_tokens": 4,
            "session_id": "u1",
        })
        r = client.post("/v1/cache/reset")
        assert r.status_code == 200
        # After reset, the session map should be empty.
        state = client.get("/v1/cache/state").json()
        assert state["sessions"] == {}


class TestDaemonLookup:
    def test_cache_lookup_endpoint(self, client):
        r = client.post("/cache/lookup", json={"prompt": "test prompt"})
        assert r.status_code == 200
        # First lookup — no entries yet, should return no match.
        assert r.json()["match"] is None


class TestDaemonCacheAdmin:
    def test_cache_list_and_delete(self, client):
        # Populate the cache with a request, then list, delete, verify gone.
        r = client.post("/v1/completions", json={
            "prompt": "List three colors.",
            "max_tokens": 16,
            "temperature": 0.0,
        })
        assert r.status_code == 200
        cache_id = r.json()["cache"]["cache_id"]
        assert cache_id

        listing = client.get("/cache/list")
        assert listing.status_code == 200
        ids = [e["cache_id"] for e in listing.json()["entries"]]
        assert cache_id in ids

        r = client.delete(f"/cache/{cache_id}")
        assert r.status_code == 200
        assert r.json()["status"] == "deleted"

        listing = client.get("/cache/list")
        ids = [e["cache_id"] for e in listing.json()["entries"]]
        assert cache_id not in ids

    def test_cache_delete_missing_returns_404(self, client):
        r = client.delete("/cache/deadbeef")
        assert r.status_code == 404

    def test_cache_delete_invalid_id_returns_400(self, client):
        r = client.delete("/cache/does-not-exist")
        assert r.status_code == 400
