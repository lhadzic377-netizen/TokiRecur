"""Unit tests for the TUI client (DaemonClient + SSE parser + auth)."""

import pytest

from tokirecur.tui.client import (
    DaemonClient,
    _extract_reasoning,
    _extract_text,
    _parse_sse_line,
    _resolve_auth_token,
    _split_sse,
)


class TestExtractText:
    def test_llama_server_content(self):
        assert _extract_text({"content": "hi"}) == "hi"

    def test_llama_server_response(self):
        assert _extract_text({"response": "hi"}) == "hi"

    def test_openai_delta(self):
        obj = {"choices": [{"delta": {"content": "hello"}}]}
        assert _extract_text(obj) == "hello"

    def test_openai_empty_delta(self):
        assert _extract_text({"choices": [{"delta": {}}]}) == ""

    def test_unknown_format(self):
        assert _extract_text({"foo": "bar"}) == ""

    def test_non_dict(self):
        assert _extract_text("not a dict") == ""
        assert _extract_text(None) == ""


class TestExtractReasoning:
    def test_top_level(self):
        assert _extract_reasoning({"reasoning_content": "thinking…"}) == "thinking…"

    def test_openai_delta(self):
        obj = {"choices": [{"delta": {"reasoning_content": "step 1"}}]}
        assert _extract_reasoning(obj) == "step 1"

    def test_empty_delta(self):
        assert _extract_reasoning({"choices": [{"delta": {}}]}) == ""

    def test_unknown(self):
        assert _extract_reasoning({"foo": "bar"}) == ""

    def test_non_dict(self):
        assert _extract_reasoning("not a dict") == ""


class TestParseSseLine:
    def test_valid_json(self):
        obj = _parse_sse_line('{"content": "hi"}')
        assert obj == {"content": "hi"}

    def test_invalid_json(self):
        assert _parse_sse_line("not json") is None

    def test_empty(self):
        assert _parse_sse_line("") is None


class TestSplitSse:
    def test_single_line(self):
        lines, remainder = _split_sse("data: foo\n")
        assert lines == ["data: foo"]
        assert remainder == ""

    def test_multiple_lines(self):
        lines, remainder = _split_sse("data: a\ndata: b\ndata: c\n")
        assert lines == ["data: a", "data: b", "data: c"]
        assert remainder == ""

    def test_partial_line_held_back(self):
        # No trailing newline → the partial line must NOT be returned
        # as a complete line (parsing it as JSON would fail and the
        # token would be silently dropped). It is held back so the
        # caller can prepend the next chunk.
        lines, remainder = _split_sse("data: partial")
        assert lines == []
        assert remainder == "data: partial"

    def test_partial_completed_by_next_chunk(self):
        lines, remainder = _split_sse("data: part")
        assert lines == []
        assert remainder == "data: part"
        lines, remainder = _split_sse(remainder + 'ial"}\n')
        assert lines == ['data: partial"}']
        assert remainder == ""


class TestResolveAuthToken:
    """Verify the auth-token lookup order used by DaemonClient.

    Lookup order:
      1. explicit arg
      2. $TOKI_AUTH_TOKEN env var
      3. ~/.config/tokirecur/auth (first line)
      4. /etc/toki-recur/auth (system-wide)
    """

    def test_explicit_arg_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TOKI_AUTH_TOKEN", "env-token")
        monkeypatch.setattr(
            "tokirecur.tui.client.Path.home", lambda: tmp_path,
        )
        # Even with a file present, explicit arg takes priority.
        assert _resolve_auth_token("explicit") == "explicit"

    def test_env_var(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TOKI_AUTH_TOKEN", "env-token")
        monkeypatch.setattr(
            "tokirecur.tui.client.Path.home", lambda: tmp_path,
        )
        assert _resolve_auth_token(None) == "env-token"

    def test_file_fallback(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TOKI_AUTH_TOKEN", raising=False)
        monkeypatch.setattr(
            "tokirecur.tui.client.Path.home", lambda: tmp_path,
        )
        # No file → None.
        assert _resolve_auth_token(None) is None

        # Write file → returns its content.
        cfg = tmp_path / ".config" / "tokirecur"
        cfg.mkdir(parents=True)
        (cfg / "auth").write_text("file-token\n")
        assert _resolve_auth_token(None) == "file-token"

    def test_file_with_whitespace(self, monkeypatch, tmp_path):
        """File content is stripped; surrounding whitespace ignored."""
        monkeypatch.delenv("TOKI_AUTH_TOKEN", raising=False)
        monkeypatch.setattr(
            "tokirecur.tui.client.Path.home", lambda: tmp_path,
        )
        cfg = tmp_path / ".config" / "tokirecur"
        cfg.mkdir(parents=True)
        (cfg / "auth").write_text("  \n   file-token   \n  \n")
        assert _resolve_auth_token(None) == "file-token"

    def test_empty_file_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TOKI_AUTH_TOKEN", raising=False)
        monkeypatch.setattr(
            "tokirecur.tui.client.Path.home", lambda: tmp_path,
        )
        cfg = tmp_path / ".config" / "tokirecur"
        cfg.mkdir(parents=True)
        (cfg / "auth").write_text("  \n  \n")  # whitespace only
        assert _resolve_auth_token(None) is None

    def test_no_source_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TOKI_AUTH_TOKEN", raising=False)
        monkeypatch.setattr(
            "tokirecur.tui.client.Path.home", lambda: tmp_path,
        )
        assert _resolve_auth_token(None) is None


class TestDaemonClientHeaders:
    """Verify auth header construction without making network calls."""

    def test_no_token_no_header(self, monkeypatch, tmp_path):
        # Clear env and point Path.home at an empty tmp dir so the
        # file fallback also finds nothing.
        monkeypatch.delenv("TOKI_AUTH_TOKEN", raising=False)
        monkeypatch.setattr(
            "tokirecur.tui.client.Path.home", lambda: tmp_path,
        )
        c = DaemonClient(auth_token=None)
        assert c._auth_headers() == {}

    def test_explicit_token(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TOKI_AUTH_TOKEN", raising=False)
        monkeypatch.setattr(
            "tokirecur.tui.client.Path.home", lambda: tmp_path,
        )
        c = DaemonClient(auth_token="my-token")
        assert c._auth_headers() == {"Authorization": "Bearer my-token"}

    def test_env_var_token(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TOKI_AUTH_TOKEN", "env-token")
        monkeypatch.setattr(
            "tokirecur.tui.client.Path.home", lambda: tmp_path,
        )
        c = DaemonClient(auth_token=None)
        assert c._auth_headers() == {"Authorization": "Bearer env-token"}

    def test_explicit_overrides_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TOKI_AUTH_TOKEN", "env-token")
        monkeypatch.setattr(
            "tokirecur.tui.client.Path.home", lambda: tmp_path,
        )
        c = DaemonClient(auth_token="explicit-token")
        assert c._auth_headers() == {"Authorization": "Bearer explicit-token"}

    def test_file_fallback_in_client(self, monkeypatch, tmp_path):
        """DaemonClient reads ~/.config/tokirecur/auth when no flag/env set.

        This is the bug fix: previously the TUI silently dropped the token
        if the user hadn't set TOKI_AUTH_TOKEN. Now it picks up the same
        auth file the daemon reads.
        """
        monkeypatch.delenv("TOKI_AUTH_TOKEN", raising=False)
        monkeypatch.setattr(
            "tokirecur.tui.client.Path.home", lambda: tmp_path,
        )
        cfg = tmp_path / ".config" / "tokirecur"
        cfg.mkdir(parents=True)
        (cfg / "auth").write_text("file-token")
        c = DaemonClient(auth_token=None)
        assert c._auth_token == "file-token"
        assert c._auth_headers() == {"Authorization": "Bearer file-token"}


class TestDaemonClientErrors:
    """Verify graceful error handling when backends are unreachable."""

    @pytest.mark.asyncio
    async def test_health_llama_unreachable(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "tokirecur.tui.client.Path.home", lambda: tmp_path,
        )
        c = DaemonClient(raw_url="http://127.0.0.1:1")  # unreachable port
        result = await c.health_llama()
        assert result == "unreachable"

    @pytest.mark.asyncio
    async def test_health_daemon_unreachable(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "tokirecur.tui.client.Path.home", lambda: tmp_path,
        )
        c = DaemonClient(daemon_url="http://127.0.0.1:1")
        result = await c.health_daemon()
        assert result == "unreachable"

    @pytest.mark.asyncio
    async def test_cache_state_unreachable(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "tokirecur.tui.client.Path.home", lambda: tmp_path,
        )
        c = DaemonClient(daemon_url="http://127.0.0.1:1")
        result = await c.cache_state()
        assert result["slots"] == []
        assert result["daemon_status"] == "unreachable"

    @pytest.mark.asyncio
    async def test_metrics_live_unreachable(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "tokirecur.tui.client.Path.home", lambda: tmp_path,
        )
        c = DaemonClient(daemon_url="http://127.0.0.1:1")
        result = await c.metrics_live()
        assert result["requests_total"] == 0

    @pytest.mark.asyncio
    async def test_precision_check_unreachable(self, monkeypatch, tmp_path):
        """Precision check fails cleanly when llama-server is unreachable.

        NOTE: hits the daemon (not raw llama-server) since the daemon
        proxies the precision check internally. To make this test
        reliably fail, we point at an unreachable port.
        """
        monkeypatch.setattr(
            "tokirecur.tui.client.Path.home", lambda: tmp_path,
        )
        c = DaemonClient(daemon_url="http://127.0.0.1:1",
                        raw_url="http://127.0.0.1:1")
        result = await c.precision_check()
        assert result["prompts_tested"] == 0
        assert "error" in result


class TestChatStreamRouting:
    """Verify chat_stream hits the correct URL for each mode.

    Regression test for Bug B: raw mode previously POSTed to
    /completion with a hand-rolled prompt, causing chat-tuned models
    to produce garbage continuations. Now both modes use
    /v1/chat/completions so llama-server applies the right chat
    template.
    """

    @pytest.mark.asyncio
    async def test_raw_mode_uses_openai_endpoint(self, monkeypatch, tmp_path):
        """Raw mode must POST to llama-server's OpenAI-compatible endpoint."""
        monkeypatch.setattr(
            "tokirecur.tui.client.Path.home", lambda: tmp_path,
        )

        captured_url: list[str] = []
        captured_payload: list[dict] = []

        class FakeStream:
            def __init__(self, *a, **kw):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            async def aiter_text(self):
                return
                yield ""  # never reached; makes this an async generator

        class FakeResponse:
            status_code = 200
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            def aiter_text(self):
                return iter([])

        class FakeClient:
            def __init__(self, *a, **kw):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            def stream(self, method, url, json):
                captured_url.append(url)
                captured_payload.append(json)
                return FakeResponse()

        import tokirecur.tui.client as client_mod
        monkeypatch.setattr(client_mod, "_make_client", lambda *a, **kw: FakeClient())

        c = DaemonClient(
            daemon_url="http://daemon:8787",
            raw_url="http://llama:8088",
            auth_token=None,
        )
        async for _ in c.chat_stream(
            "hi", mode="raw", session_id="x", max_tokens=8,
        ):
            pass

        assert len(captured_url) == 1, f"expected 1 request, got {len(captured_url)}"
        # Must be the OpenAI-compatible endpoint, NOT /completion.
        assert captured_url[0] == "http://llama:8088/v1/chat/completions", (
            f"raw mode hit {captured_url[0]!r}; expected "
            "http://llama:8088/v1/chat/completions"
        )
        # Payload must be OpenAI messages format, NOT a flat prompt string.
        payload = captured_payload[0]
        assert "messages" in payload, (
            f"raw mode payload missing 'messages'; got keys: {list(payload)}"
        )
        assert "prompt" not in payload, (
            "raw mode should not use 'prompt' field; that was the bug"
        )
        assert payload["messages"][-1]["role"] == "user"
        assert payload["messages"][-1]["content"] == "hi"

    @pytest.mark.asyncio
    async def test_neither_mode_overrides_thinking_mode(
        self, monkeypatch, tmp_path,
    ):
        """The client must not force `enable_thinking` in either mode:
        thinking is decided by the backend (daemon `reasoning` setting
        in raw->llama-server and cached->daemon paths), so reasoning
        deltas flow through to the TUI's thinking renderer.
        """
        monkeypatch.setattr(
            "tokirecur.tui.client.Path.home", lambda: tmp_path,
        )

        for mode in ("raw", "cached"):
            captured_payloads: list[dict] = []

            class FakeResponse:
                status_code = 200
                async def __aenter__(self):
                    return self
                async def __aexit__(self, *a):
                    return False
                def aiter_text(self):
                    return iter([])

            class FakeClient:
                def __init__(self, *a, **kw):
                    pass
                async def __aenter__(self):
                    return self
                async def __aexit__(self, *a):
                    return False
                # ``captured_payloads`` is a loop variable in the enclosing
                # scope (re-initialised on every ``mode``). Bind it via a
                # default-arg to avoid the B023 late-binding warning.
                def stream(self, method, url, json, _sink=captured_payloads):
                    _sink.append(json)
                    return FakeResponse()

            import tokirecur.tui.client as client_mod
            monkeypatch.setattr(
                client_mod, "_make_client", lambda *a, **kw: FakeClient(),
            )

            c = DaemonClient(
                daemon_url="http://daemon:8787",
                raw_url="http://llama:8088",
                auth_token=None,
            )
            async for _ in c.chat_stream(
                "What is 2+2?", mode=mode, session_id="x", max_tokens=8,
            ):
                pass

            assert len(captured_payloads) == 1
            payload = captured_payloads[0]
            assert "chat_template_kwargs" not in payload, (
                f"{mode} mode must not pin enable_thinking (the daemon "
                f"decides from the model's reasoning setting); got keys: "
                f"{list(payload.keys())}"
            )

    @pytest.mark.asyncio
    async def test_raw_mode_emits_applicable_false_cache_marker(
        self, monkeypatch, tmp_path,
    ):
        """Raw mode should emit cache:{applicable: False, mode: 'raw'}
        on the done event so the TUI can render a non-misleading
        badge. Cached mode should emit cache:{hit: bool, ...} instead.
        """
        monkeypatch.setattr(
            "tokirecur.tui.client.Path.home", lambda: tmp_path,
        )

        for mode, expected_cache_key in [
            ("raw", "applicable"),
            ("cached", "hit"),
        ]:
            seen_done = []

            class FakeResponse:
                status_code = 200
                async def __aenter__(self):
                    return self
                async def __aexit__(self, *a):
                    return False
                def aiter_text(self):
                    return iter([])

            class FakeClient:
                def __init__(self, *a, **kw):
                    pass
                async def __aenter__(self):
                    return self
                async def __aexit__(self, *a):
                    return False
                def stream(self, method, url, json):
                    return FakeResponse()

            import tokirecur.tui.client as client_mod
            monkeypatch.setattr(
                client_mod, "_make_client", lambda *a, **kw: FakeClient(),
            )

            c = DaemonClient(
                daemon_url="http://daemon:8787",
                raw_url="http://llama:8088",
                auth_token=None,
            )
            async for ev in c.chat_stream(
                "hi", mode=mode, session_id="x", max_tokens=8,
            ):
                if ev["event"] == "done":
                    seen_done.append(ev)

            assert len(seen_done) == 1, f"{mode} mode: expected 1 done event"
            cache = seen_done[0]["cache"]
            assert expected_cache_key in cache, (
                f"{mode} mode done event cache should have "
                f"{expected_cache_key!r} key; got {cache!r}"
            )
            if mode == "raw":
                assert cache["applicable"] is False
                assert cache["mode"] == "raw"
            else:
                assert "hit" in cache
                assert "skipped_tokens" in cache

    @pytest.mark.asyncio
    async def test_raw_mode_does_not_send_auth_header(
        self, monkeypatch, tmp_path,
    ):
        """Raw mode bypasses the daemon, so no Authorization header
        should be sent even when a token is configured. (Daemon's
        /v1/chat/completions requires auth; raw hits llama-server
        directly which doesn't.)
        """
        monkeypatch.setattr(
            "tokirecur.tui.client.Path.home", lambda: tmp_path,
        )

        captured_headers: list[dict] = []

        class FakeResponse:
            status_code = 200
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            def aiter_text(self):
                return iter([])

        class FakeClient:
            def __init__(self, *a, **kw):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            def stream(self, method, url, json, headers=None):
                captured_headers.append(headers or {})
                return FakeResponse()

        import tokirecur.tui.client as client_mod
        monkeypatch.setattr(
            client_mod, "_make_client", lambda *a, **kw: FakeClient(),
        )

        c = DaemonClient(
            daemon_url="http://daemon:8787",
            raw_url="http://llama:8088",
            auth_token="should-not-leak",  # noqa: S105 (test fixture)
        )
        async for _ in c.chat_stream("hi", mode="raw", session_id="x"):
            pass

        assert len(captured_headers) == 1
        assert "Authorization" not in captured_headers[0], (
            f"raw mode must not send Authorization; got "
            f"{captured_headers[0]!r}"
        )

    @pytest.mark.asyncio
    async def test_cached_mode_uses_daemon_endpoint(self, monkeypatch, tmp_path):
        """Cached mode POSTs to the daemon's /v1/chat/completions."""
        monkeypatch.setattr(
            "tokirecur.tui.client.Path.home", lambda: tmp_path,
        )

        captured_url: list[str] = []

        class FakeResponse:
            status_code = 200
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            def aiter_text(self):
                return iter([])

        class FakeClient:
            def __init__(self, *a, **kw):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            def stream(self, method, url, json):
                captured_url.append(url)
                return FakeResponse()

        import tokirecur.tui.client as client_mod
        monkeypatch.setattr(client_mod, "_make_client", lambda *a, **kw: FakeClient())

        c = DaemonClient(
            daemon_url="http://daemon:8787",
            raw_url="http://llama:8088",
            auth_token="my-token",
        )
        async for _ in c.chat_stream(
            "hi", mode="cached", session_id="x", max_tokens=8,
        ):
            pass

        assert len(captured_url) == 1
        assert captured_url[0] == "http://daemon:8787/v1/chat/completions"

class TestChatStreamRawCacheFlag:
    """Raw mode must send cache_prompt=true so llama-server reuses the
    slot KV prefix across multi-turn messages (fast multi-turn without
    re-prefilling the whole history)."""

    @pytest.mark.asyncio
    async def test_raw_payload_requests_cache_prompt(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "tokirecur.tui.client.Path.home", lambda: tmp_path,
        )
        captured: list[dict] = []

        class FakeResponse:
            status_code = 200
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            def aiter_text(self):
                return iter([])

        class FakeClient:
            def __init__(self, *a, **kw):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            def stream(self, method, url, json):
                captured.append(json)
                return FakeResponse()

        import tokirecur.tui.client as client_mod
        monkeypatch.setattr(client_mod, "_make_client", lambda *a, **kw: FakeClient())

        c = DaemonClient(raw_url="http://llama:8088", auth_token=None)
        async for _ in c.chat_stream(
            "second question",
            mode="raw",
            session_id="s",
            max_tokens=8,
            history=[{"role": "user", "content": "first question"}],
        ):
            pass

        assert len(captured) == 1
        payload = captured[0]
        assert payload.get("cache_prompt") is True, (
            f"raw mode must request KV reuse; payload keys: {list(payload)}"
        )
        # History must be included as messages for context awareness.
        roles = [m["role"] for m in payload["messages"]]
        assert roles == ["user", "user"], f"expected history+prompt, got {roles}"


class TestChatStreamPromptTokens:
    """The final SSE usage chunk must surface as prompt_tokens in the
    done event, so the TUI can render 'in N tok' per message."""

    @pytest.mark.asyncio
    async def test_done_event_carries_prompt_tokens(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "tokirecur.tui.client.Path.home", lambda: tmp_path,
        )

        body = "\n".join([
            'data: {"choices": [{"delta": {"role": "assistant", "content": ""}}]}',
            'data: {"choices": [{"delta": {"content": "Four"}}]}',
            'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}',
            'data: {"choices": [], "usage": {"prompt_tokens": 214, '
            '"completion_tokens": 1, "total_tokens": 215}}',
            "data: [DONE]",
            "",
        ])

        class FakeResponse:
            status_code = 200
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            async def aiter_text(self):
                yield body

        class FakeClient:
            def __init__(self, *a, **kw):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            def stream(self, method, url, json):
                return FakeResponse()

        import tokirecur.tui.client as client_mod
        monkeypatch.setattr(client_mod, "_make_client", lambda *a, **kw: FakeClient())

        c = DaemonClient(raw_url="http://llama:8088", auth_token=None)
        done_metrics: dict | None = None
        text = ""
        async for ev in c.chat_stream(
            "what is 2+2?", mode="raw", session_id="s", max_tokens=8,
        ):
            if ev["event"] == "token":
                text += ev["content"]
            elif ev["event"] == "done":
                done_metrics = ev.get("metrics", {})

        assert text == "Four"
        assert done_metrics is not None
        assert done_metrics.get("prompt_tokens") == 214, (
            f"expected prompt_tokens=214 in done metrics, got {done_metrics!r}"
        )


class TestChatStreamSseChunking:
    """JSON events split across HTTP chunks must be reassembled, not
    dropped. Regression for the _split_sse bug: a partial line was
    returned as if complete, failed json.loads, and the token was
    silently lost."""

    @pytest.mark.asyncio
    async def test_json_split_across_chunks_not_dropped(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "tokirecur.tui.client.Path.home", lambda: tmp_path,
        )

        # One JSON object split mid-stream across three chunks.
        chunk1 = 'data: {"choices": [{"delta": {"content": "Fo'
        chunk2 = 'ur"}}]}\n\n'
        chunk3 = 'data: {"choices": [], "usage": {"prompt_tokens": 7, '
        chunk4 = '"completion_tokens": 1, "total_tokens": 8}}\n\ndata: [DONE]\n\n'

        class FakeResponse:
            status_code = 200
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            async def aiter_text(self):
                yield chunk1
                yield chunk2
                yield chunk3
                yield chunk4

        class FakeClient:
            def __init__(self, *a, **kw):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            def stream(self, method, url, json):
                return FakeResponse()

        import tokirecur.tui.client as client_mod
        monkeypatch.setattr(client_mod, "_make_client", lambda *a, **kw: FakeClient())

        c = DaemonClient(raw_url="http://llama:8088", auth_token=None)
        text = ""
        done_metrics: dict | None = None
        async for ev in c.chat_stream(
            "hi", mode="raw", session_id="s", max_tokens=8,
        ):
            if ev["event"] == "token":
                text += ev["content"]
            elif ev["event"] == "done":
                done_metrics = ev.get("metrics", {})

        assert text == "Four", f"tokens lost across chunk boundary: {text!r}"
        assert done_metrics is not None
        assert done_metrics.get("prompt_tokens") == 7, (
            f"usage chunk split across boundary not reassembled: {done_metrics!r}"
        )


class TestChatStreamCacheVerdict:
    """The daemon's terminal chunk carries the cache verdict (hit,
    soft_hit, cache_id, reuse_depth, skipped_tokens, ...). The TUI must
    surface it instead of hardcoding a miss — the old behaviour made
    every cached response render as '✗ miss'."""

    @pytest.mark.asyncio
    async def test_cached_mode_carries_daemon_verdict(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "tokirecur.tui.client.Path.home", lambda: tmp_path,
        )

        body = "\n".join([
            'data: {"choices": [{"delta": {"role": "assistant", "content": ""}}]}',
            'data: {"choices": [{"delta": {"content": "Four"}}]}',
            'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}',
            'data: {"choices": [], "usage": {"prompt_tokens": 12, '
            '"completion_tokens": 1, "total_tokens": 13}, "cache": '
            '{"hit": true, "soft_hit": false, "cache_id": "abc123", '
            '"reuse_depth": 3, "skipped_tokens": 120, "ttft_ms": 4.2, '
            '"total_ms": 9.8, "slot_id": 0, "spilled": false}}',
            "data: [DONE]",
            "",
        ])

        class FakeResponse:
            status_code = 200
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            async def aiter_text(self):
                yield body

        class FakeClient:
            def __init__(self, *a, **kw):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            def stream(self, method, url, json):
                return FakeResponse()

        import tokirecur.tui.client as client_mod
        monkeypatch.setattr(client_mod, "_make_client", lambda *a, **kw: FakeClient())

        c = DaemonClient(
            daemon_url="http://daemon:8787",
            raw_url="http://llama:8088",
            auth_token=None,
        )
        cache: dict | None = None
        async for ev in c.chat_stream(
            "hi", mode="cached", session_id="s", max_tokens=8,
        ):
            if ev["event"] == "done":
                cache = ev.get("cache", {})

        assert cache is not None, "no done event seen"
        assert cache.get("hit") is True, f"daemon hit verdict lost: {cache!r}"
        assert cache.get("soft_hit") is False
        assert cache.get("cache_id") == "abc123"
        assert cache.get("reuse_depth") == 3
        assert cache.get("skipped_tokens") == 120
        assert cache.get("ttft_ms") == 4.2
        assert cache.get("total_ms") == 9.8

    @pytest.mark.asyncio
    async def test_cached_mode_miss_without_verdict_chunk(self, monkeypatch, tmp_path):
        """No verdict chunk arrives (e.g. upstream closed early) → the
        done event must still carry a conservative miss marker."""
        monkeypatch.setattr(
            "tokirecur.tui.client.Path.home", lambda: tmp_path,
        )

        body = "\n".join([
            'data: {"choices": [{"delta": {"role": "assistant", "content": ""}}]}',
            'data: {"choices": [{"delta": {"content": "Four"}}]}',
            'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}',
            "data: [DONE]",
            "",
        ])

        class FakeResponse:
            status_code = 200
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            async def aiter_text(self):
                yield body

        class FakeClient:
            def __init__(self, *a, **kw):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            def stream(self, method, url, json):
                return FakeResponse()

        import tokirecur.tui.client as client_mod
        monkeypatch.setattr(client_mod, "_make_client", lambda *a, **kw: FakeClient())

        c = DaemonClient(
            daemon_url="http://daemon:8787",
            raw_url="http://llama:8088",
            auth_token=None,
        )
        cache: dict | None = None
        async for ev in c.chat_stream(
            "hi", mode="cached", session_id="s", max_tokens=8,
        ):
            if ev["event"] == "done":
                cache = ev.get("cache", {})

        assert cache is not None, "no done event seen"
        assert cache.get("hit") is False
        assert cache.get("skipped_tokens") == 0


class TestChatStreamReasoning:
    """Thinking-mode deltas arrive as separate ``reasoning_content``
    SSE chunks from the daemon; the client surfaces them as
    ``{"event": "reasoning", ...}`` so the TUI can render them dimmed.
    """

    @pytest.mark.asyncio
    async def test_reasoning_deltas_yield_reasoning_events(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setattr(
            "tokirecur.tui.client.Path.home", lambda: tmp_path,
        )

        body = "\n".join([
            'data: {"choices": [{"delta": {"role": "assistant", "content": ""}}]}',
            'data: {"choices": [{"delta": {"reasoning_content": "let me think..."}}]}',
            'data: {"choices": [{"delta": {"reasoning_content": " step 2"}}]}',
            'data: {"choices": [{"delta": {"content": "Four"}}]}',
            'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}',
            "data: [DONE]",
            "",
        ])

        class FakeResponse:
            status_code = 200
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            async def aiter_text(self):
                yield body

        class FakeClient:
            def __init__(self, *a, **kw):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            def stream(self, method, url, json):
                return FakeResponse()

        import tokirecur.tui.client as client_mod
        monkeypatch.setattr(client_mod, "_make_client", lambda *a, **kw: FakeClient())

        c = DaemonClient(raw_url="http://llama:8088", auth_token=None)
        reasoning: list[str] = []
        text: list[str] = []
        done_reasoning = ""
        async for ev in c.chat_stream(
            "what is 2+2?", mode="raw", session_id="s", max_tokens=8,
        ):
            if ev["event"] == "reasoning":
                reasoning.append(ev["content"])
            elif ev["event"] == "token":
                text.append(ev["content"])
            elif ev["event"] == "done":
                done_reasoning = ev.get("reasoning", "")

        assert "".join(reasoning) == "let me think... step 2"
        assert "".join(text) == "Four"
        assert done_reasoning == "let me think... step 2"
