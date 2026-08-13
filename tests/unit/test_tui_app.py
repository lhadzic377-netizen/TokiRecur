"""Textual Pilot tests for the TUI App.

Uses Textual's `run_test` to drive the app headlessly. We mock the
`DaemonClient` so we don't need a live backend.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tokirecur.tui.app import TokiApp


def _mock_client():
    """Build an AsyncMock DaemonClient with sensible defaults."""
    client = MagicMock()
    client.health_llama = AsyncMock(return_value="ok")
    client.health_daemon = AsyncMock(return_value="ok")
    client.cache_state = AsyncMock(return_value={
        "slots": [], "sessions": {}, "spill_count": 0,
        "max_context_tokens": 4096, "daemon_status": "ok",
    })
    client.metrics_live = AsyncMock(return_value={
        "requests_total": 0, "cache_hits": 0, "cache_misses": 0,
        "tokens_skipped": 0, "active_slots": 0, "auth_failures": 0,
    })
    client.precision_check = AsyncMock(return_value={
        "baseline_deterministic": 1.0,
        "recycled_deterministic": 1.0,
        "byte_identical_to_baseline": 1.0,
        "prompts_tested": 3,
        "per_prompt": [],
    })
    client.reset_cache = AsyncMock(return_value=True)
    # chat_stream is an async generator; default to a quick completion.
    async def fake_stream(*args, **kwargs):
        yield {"event": "start", "mode": "cached", "target": "cached",
               "session_id": "x"}
        yield {"event": "token", "content": "Hello "}
        yield {"event": "token", "content": "world"}
        yield {"event": "done", "text": "Hello world",
               "metrics": {"ttft_ms": 50.0, "total_ms": 200.0,
                          "tokens_generated": 2},
               "cache": {"hit": False, "skipped_tokens": 0}}
    client.chat_stream = fake_stream
    client.raw_url = "http://127.0.0.1:8088"
    client.daemon_url = "http://127.0.0.1:8787"
    return client


class TestTokiApp:
    @pytest.mark.asyncio
    async def test_app_mounts(self):
        client = _mock_client()
        with patch("tokirecur.tui.app.DaemonClient", return_value=client):
            app = TokiApp()
            async with app.run_test() as pilot:
                # App mounts and widgets are queryable.
                assert pilot.app.query_one("Sidebar") is not None
                assert pilot.app.query_one("ChatLog") is not None
                assert pilot.app.query_one("StatusBar") is not None
                assert pilot.app.query_one("InfoPanel") is not None
                assert pilot.app.query_one("Input") is not None

    @pytest.mark.asyncio
    async def test_header_rendered_on_mount(self):
        from tokirecur.tui import art

        client = _mock_client()
        with patch("tokirecur.tui.app.DaemonClient", return_value=client):
            app = TokiApp()
            async with app.run_test():
                rendered = "\n".join(str(line) for line in app.chat.lines)
        assert art.WORDMARK in rendered
        assert art.TAGLINE in rendered

    @pytest.mark.asyncio
    async def test_mode_switching(self):
        client = _mock_client()
        with patch("tokirecur.tui.app.DaemonClient", return_value=client):
            app = TokiApp()
            async with app.run_test() as pilot:
                # Default mode is cached.
                assert app.mode == "cached"
                # Ctrl+1 → raw
                await pilot.press("ctrl+1")
                assert app.mode == "raw"
                # Ctrl+2 → cached
                await pilot.press("ctrl+2")
                assert app.mode == "cached"
                # Ctrl+3 → compare
                await pilot.press("ctrl+3")
                assert app.mode == "compare"

    @pytest.mark.asyncio
    async def test_cached_mode_blocked_when_daemon_down(self):
        client = _mock_client()
        client.health_daemon = AsyncMock(return_value="unreachable")
        with patch("tokirecur.tui.app.DaemonClient", return_value=client):
            app = TokiApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                # App auto-switched to raw because daemon is unreachable.
                assert app.mode == "raw"
                # Trying Ctrl+2 (cached) should be rejected.
                await pilot.press("ctrl+2")
                assert app.mode == "raw"  # unchanged

    @pytest.mark.asyncio
    async def test_send_message_renders_assistant(self):
        client = _mock_client()
        with patch("tokirecur.tui.app.DaemonClient", return_value=client):
            app = TokiApp()
            async with app.run_test() as pilot:
                # Type a message and submit.
                pilot.app.query_one("Input").value = "hi there"
                await pilot.press("enter")
                # Let the async work finish.
                await pilot.pause()
                await pilot.pause()
                # ChatLog should have content (user + assistant + status).
                chat = pilot.app.query_one("ChatLog")
                assert chat.lines  # RichLog stores lines

    @pytest.mark.asyncio
    async def test_slash_command_quit(self):
        client = _mock_client()
        with patch("tokirecur.tui.app.DaemonClient", return_value=client):
            app = TokiApp()
            async with app.run_test() as pilot:
                pilot.app.query_one("Input").value = "/quit"
                await pilot.press("enter")
                # App should exit.
                await pilot.pause()
                # Verify the quit command was handled (no exception).

    @pytest.mark.asyncio
    async def test_slash_command_clear(self):
        client = _mock_client()
        with patch("tokirecur.tui.app.DaemonClient", return_value=client):
            app = TokiApp()
            async with app.run_test() as pilot:
                # Send a message.
                pilot.app.query_one("Input").value = "hello"
                await pilot.press("enter")
                await pilot.pause()
                await pilot.pause()
                # Clear.
                pilot.app.query_one("Input").value = "/clear"
                await pilot.press("enter")
                await pilot.pause()
                # ChatLog should be empty (or only the "cleared" system line).
                pilot.app.query_one("ChatLog")
                # After clear, only the system message remains.
                # RichLog.lines is the internal store.

    @pytest.mark.asyncio
    async def test_action_new_session(self):
        client = _mock_client()
        with patch("tokirecur.tui.app.DaemonClient", return_value=client):
            app = TokiApp()
            async with app.run_test() as pilot:
                original = app._session_id
                await pilot.press("ctrl+u")
                assert app._session_id != original

    @pytest.mark.asyncio
    async def test_status_bar_shows_backend_health(self):
        client = _mock_client()
        client.health_daemon = AsyncMock(return_value="auth_required")
        with patch("tokirecur.tui.app.DaemonClient", return_value=client):
            app = TokiApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                # App should reflect daemon auth_required.
                assert app.daemon_status == "auth_required"
                # StatusBar reactive should be set.
                assert pilot.app.query_one("StatusBar").daemon_status == "auth_required"

    @pytest.mark.asyncio
    async def test_raw_mode_does_not_increment_hit_miss_counters(self):
        """Regression: raw mode emits cache:{applicable:False}, which
        should NOT bump _hits or _misses (those are cache-layer metrics).
        """
        client = _mock_client()

        async def raw_stream(*args, **kwargs):
            # Simulate the new raw-mode done event with the
            # `applicable: False` marker.
            yield {"event": "start", "mode": "raw", "target": "raw",
                   "session_id": "x"}
            yield {"event": "token", "content": "Hello"}
            yield {"event": "done", "text": "Hello",
                   "metrics": {"ttft_ms": 50, "total_ms": 100,
                               "tokens_generated": 1},
                   "cache": {"applicable": False, "mode": "raw"}}
        client.chat_stream = raw_stream

        with patch("tokirecur.tui.app.DaemonClient", return_value=client):
            app = TokiApp()
            async with app.run_test() as pilot:
                hits_before = app._hits
                misses_before = app._misses
                pilot.app.query_one("#input").value = "test"
                await pilot.press("enter")
                for _ in range(30):
                    await pilot.pause()
                    if not app.busy:
                        break
                # Both should be unchanged — raw mode bypasses cache.
                assert app._hits == hits_before, (
                    f"raw mode should not increment _hits; "
                    f"was {hits_before}, now {app._hits}"
                )
                assert app._misses == misses_before, (
                    f"raw mode should not increment _misses; "
                    f"was {misses_before}, now {app._misses}"
                )

    @pytest.mark.asyncio
    async def test_cached_mode_increments_counters(self):
        """Cached mode still bumps _hits / _misses correctly."""
        client = _mock_client()

        async def cached_stream(*args, **kwargs):
            yield {"event": "start", "mode": "cached", "target": "cached",
                   "session_id": "x"}
            yield {"event": "token", "content": "Hello"}
            yield {"event": "done", "text": "Hello",
                   "metrics": {"ttft_ms": 50, "total_ms": 100,
                               "tokens_generated": 1},
                   "cache": {"hit": True, "skipped_tokens": 50}}
        client.chat_stream = cached_stream

        with patch("tokirecur.tui.app.DaemonClient", return_value=client):
            app = TokiApp()
            async with app.run_test() as pilot:
                hits_before = app._hits
                pilot.app.query_one("#input").value = "test"
                await pilot.press("enter")
                for _ in range(30):
                    await pilot.pause()
                    if not app.busy:
                        break
                # _hits should have incremented by 1.
                assert app._hits == hits_before + 1, (
                    f"cached-mode hit should increment _hits; "
                    f"was {hits_before}, now {app._hits}"
                )


class TestTokiContextAndMaxTokens:
    """Multi-turn context: in-session turns must be fed back into the
    API request; max-token budget must be configurable in-TUI."""

    @pytest.mark.asyncio
    async def test_history_syncs_user_and_assistant_turns(self, monkeypatch, tmp_path):
        # Redirect JSONL history so prior test runs can't leak in.
        monkeypatch.setattr(
            "tokirecur.tui.history.history_path",
            lambda: tmp_path / "history.jsonl",
        )
        client = _mock_client()
        sent_histories: list[list[dict]] = []

        async def recording_stream(*args, **kwargs):
            sent_histories.append(kwargs.get("history", []))
            yield {"event": "start", "mode": "cached", "target": "cached",
                   "session_id": "x"}
            yield {"event": "token", "content": "Answer 1"}
            yield {"event": "done", "text": "Answer 1",
                   "metrics": {"ttft_ms": 10, "total_ms": 20,
                               "tokens_generated": 2},
                   "cache": {"hit": False, "skipped_tokens": 0}}
        client.chat_stream = recording_stream

        with patch("tokirecur.tui.app.DaemonClient", return_value=client):
            app = TokiApp()
            async with app.run_test() as pilot:
                # Turn 1.
                pilot.app.query_one("#input").value = "question one"
                await pilot.press("enter")
                for _ in range(30):
                    await pilot.pause()
                    if not app.busy:
                        break
                assert app._history == [
                    {"role": "user", "content": "question one"},
                    {"role": "assistant", "content": "Answer 1"},
                ], f"in-memory history not synced: {app._history!r}"

                # Turn 2 must send turn 1 in history.
                pilot.app.query_one("#input").value = "question two"
                await pilot.press("enter")
                for _ in range(30):
                    await pilot.pause()
                    if not app.busy:
                        break
                assert len(sent_histories) == 2
                second = sent_histories[1]
                assert second[-1] == {"role": "user", "content": "question two"}
                assert {"role": "user", "content": "question one"} in second
                assert {"role": "assistant", "content": "Answer 1"} in second

    @pytest.mark.asyncio
    async def test_history_capped_at_20_messages(self):
        client = _mock_client()
        with patch("tokirecur.tui.app.DaemonClient", return_value=client):
            app = TokiApp()
            async with app.run_test():
                for i in range(30):
                    app._history.append({"role": "user", "content": f"msg {i}"})
                sent = app._history_for_api()
                assert len(sent) == 20
                assert sent[0] == {"role": "user", "content": "msg 10"}
                assert sent[-1] == {"role": "user", "content": "msg 29"}

    @pytest.mark.asyncio
    async def test_max_tokens_default_is_2048(self):
        client = _mock_client()
        with patch("tokirecur.tui.app.DaemonClient", return_value=client):
            app = TokiApp()
            async with app.run_test() as pilot:
                assert app._max_tokens() == 2048
                assert pilot.app.query_one("StatusBar").max_tokens == 2048

    @pytest.mark.asyncio
    async def test_max_tokens_slash_command(self):
        client = _mock_client()
        with patch("tokirecur.tui.app.DaemonClient", return_value=client):
            app = TokiApp()
            async with app.run_test() as pilot:
                pilot.app.query_one("#input").value = "/max_tokens 4096"
                await pilot.press("enter")
                await pilot.pause()
                assert app._max_tokens() == 4096
                assert pilot.app.query_one("StatusBar").max_tokens == 4096

                # Invalid values rejected.
                pilot.app.query_one("#input").value = "/max_tokens 999999"
                await pilot.press("enter")
                await pilot.pause()
                assert app._max_tokens() == 4096  # unchanged

    @pytest.mark.asyncio
    async def test_ctrl_t_cycles_max_tokens(self):
        client = _mock_client()
        with patch("tokirecur.tui.app.DaemonClient", return_value=client):
            app = TokiApp()
            async with app.run_test() as pilot:
                assert app._max_tokens() == 2048
                await pilot.press("ctrl+t")
                assert app._max_tokens() == 4096
                await pilot.press("ctrl+t")
                assert app._max_tokens() == 512  # wraps around


class TestTokiInfoAndScroll:
    """De-bloated single InfoPanel + scrollable ChatLog."""

    @pytest.mark.asyncio
    async def test_info_panel_toggle_with_ctrl_m(self):
        client = _mock_client()
        with patch("tokirecur.tui.app.DaemonClient", return_value=client):
            app = TokiApp()
            async with app.run_test() as pilot:
                info = pilot.app.query_one("InfoPanel")
                assert info.display is True
                await pilot.press("ctrl+m")
                assert info.display is False
                await pilot.press("ctrl+m")
                assert info.display is True

    @pytest.mark.asyncio
    async def test_prompt_tokens_render_in_assistant_line(self):
        """The done event's prompt_tokens must appear as 'in N tok'."""
        client = _mock_client()

        async def stream_with_prompt_tokens(*args, **kwargs):
            yield {"event": "token", "content": "Four"}
            yield {"event": "done", "text": "Four",
                   "metrics": {"ttft_ms": 50, "total_ms": 100,
                               "tokens_generated": 1,
                               "prompt_tokens": 214},
                   "cache": {"hit": True, "skipped_tokens": 200}}
        client.chat_stream = stream_with_prompt_tokens

        with patch("tokirecur.tui.app.DaemonClient", return_value=client):
            app = TokiApp()
            async with app.run_test() as pilot:
                pilot.app.query_one("#input").value = "what is 2+2?"
                await pilot.press("enter")
                for _ in range(30):
                    await pilot.pause()
                    if not app.busy:
                        break
                chat = pilot.app.query_one("ChatLog")
                rendered = "".join(
                    (t.plain if hasattr(t, "plain") else str(t)) for t in chat.lines
                )
                assert "in 214 tok" in rendered, (
                    f"prompt token count not rendered: {rendered!r}"
                )

    @pytest.mark.asyncio
    async def test_session_stats_update_after_turn(self):
        client = _mock_client()
        with patch("tokirecur.tui.app.DaemonClient", return_value=client):
            app = TokiApp()
            async with app.run_test() as pilot:
                pilot.app.query_one("#input").value = "hello"
                await pilot.press("enter")
                for _ in range(30):
                    await pilot.pause()
                    if not app.busy:
                        break
                assert app._messages == 1
                assert app._generated_total == 2
                info = pilot.app.query_one("InfoPanel")
                stats = info.session_stats
                assert stats.get("messages") == 1
                assert stats.get("tokens_generated") == 2

    @pytest.mark.asyncio
    async def test_chat_scroll_actions_work(self):
        client = _mock_client()
        with patch("tokirecur.tui.app.DaemonClient", return_value=client):
            app = TokiApp()
            async with app.run_test() as pilot:
                # Fill the chat with enough lines to overflow.
                for i in range(40):
                    app.chat.write_system(f"line {i}", style="dim")
                await pilot.pause()
                assert app.chat.scroll_y >= 0
                # Page up then back to bottom.
                await pilot.press("pageup")
                await pilot.pause()
                assert app.chat.scroll_y < app.chat.max_scroll_y
                await pilot.press("end")
                await pilot.pause()
                assert app.chat.scroll_y >= app.chat.max_scroll_y - 1


class TestChatLogStreamingLive:
    """Tokens must appear on the in-flight line as they arrive, not
    only after the response completes.

    Regression: append_assistant_token only mutated internal state and
    re-rendered a frozen snapshot, so the TUI showed '…' until
    end_assistant — the answer popped in all at once instead of
    streaming.
    """

    @pytest.mark.asyncio
    async def test_partial_tokens_visible_mid_stream(self):
        from tokirecur.tui.widgets import ChatLog

        log = ChatLog()
        log.begin_assistant(message_id=1, mode="cached")

        log.append_assistant_token("Hello ")
        rendered = _line_text(log.lines[-1])
        assert "Hello " in rendered, (
            f"token not visible on live line mid-stream: {rendered!r}"
        )
        assert "…" not in rendered.split("▸ ")[-1], (
            f"placeholder should be replaced once text streams: {rendered!r}"
        )

        log.append_assistant_token("world")
        rendered = _line_text(log.lines[-1])
        assert "Hello world" in rendered, (
            f"accumulated text not on live line: {rendered!r}"
        )

    @pytest.mark.asyncio
    async def test_reasoning_visible_live_then_final(self):
        from tokirecur.tui.widgets import ChatLog

        log = ChatLog()
        log.begin_assistant(message_id=1, mode="cached")

        log.append_assistant_reasoning("think ")
        rendered = _line_text(log.lines[-1])
        assert "think think " in rendered, (
            f"reasoning not visible mid-stream: {rendered!r}"
        )

        log.append_assistant_token("answer")
        log.end_assistant(
            mode="cached", hit=False, soft_hit=False,
            ttft_ms=5.0, total_ms=20.0, tokens_generated=1,
            prompt_tokens=3,
        )
        final = "\n".join(_line_text(ln) for ln in log.lines)
        assert "answer" in final
        assert "think › think" in final


class TestChatLogReasoning:
    """Thinking-mode deltas render dim alongside the live answer, then
    end up on their own indented dim line at end_assistant time."""

    @pytest.mark.asyncio
    async def test_reasoning_and_content_both_rendered(self):
        from tokirecur.tui.widgets import ChatLog

        log = ChatLog()
        log.begin_assistant(message_id=1, mode="cached")
        log.append_assistant_reasoning("thinking…")
        log.append_assistant_token("Hello")
        log.append_assistant_reasoning(" more")
        log.append_assistant_token(" world")
        log.end_assistant(
            mode="cached", hit=False, soft_hit=False,
            ttft_ms=10.0, total_ms=50.0, tokens_generated=2,
            prompt_tokens=4,
        )

        # The streaming line is replaced by a final block: a dim
        # indented "think ›" line followed by the answer line.
        rendered = "\n".join(_line_text(ln) for ln in log.lines)
        assert "think › thinking… more" in rendered
        assert "Hello world" in rendered
        # And the content line should appear AFTER the think line.
        think_idx = rendered.find("think ›")
        answer_idx = rendered.find("Hello world")
        assert 0 <= think_idx < answer_idx

    @pytest.mark.asyncio
    async def test_reasoning_end_renders_both_lines_to_surface(self):
        """Regression: end_assistant for a thinking-mode response
        used to replace the streaming line with the think summary
        (which calls _rerender) and then append the answer line
        without rerendering. The answer silently disappeared from the
        screen until some other event triggered a rerender."""
        from tokirecur.tui.widgets import ChatLog

        log = ChatLog()
        log.begin_assistant(message_id=1, mode="cached")
        log.append_assistant_reasoning("thinking…")
        log.append_assistant_token("the answer")

        # Capture every surface update so we can inspect what the
        # screen actually sees after end_assistant returns.
        updates = []
        log._surface.update = lambda renderable: updates.append(renderable)

        log.end_assistant(
            mode="cached", hit=False, soft_hit=False,
            ttft_ms=10.0, total_ms=50.0, tokens_generated=1,
            prompt_tokens=4,
        )

        # The final surface update must contain both the think line
        # and the answer line. If only the think line is there, the
        # answer was appended but never rendered.
        assert updates, "no surface updates captured during end_assistant"
        final = updates[-1]
        from rich.console import Console, Group

        # Group holds renderables; flatten to plain text.
        renderables = (
            list(final.renderables)
            if isinstance(final, Group)
            else [final]
        )
        console = Console(width=200, record=True)
        for r in renderables:
            console.print(r)
        rendered = console.export_text()
        assert "think › thinking…" in rendered, (
            f"think summary missing from final surface render: {rendered!r}"
        )
        assert "the answer" in rendered, (
            f"answer missing from final surface render: {rendered!r}"
        )


def _line_text(line) -> str:
    """Plain-text rendering of a rich Text line for assertions."""
    from rich.text import Text
    if isinstance(line, Text):
        return line.plain
    return str(line)
