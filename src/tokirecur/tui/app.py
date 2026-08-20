"""The Toki Recur TUI application.

A Textual-based terminal UI for chatting with a local LLM through
the toki daemon. Supports three modes (raw, cached, compare) with
session pinning, live metrics, and a precision-check panel.

Run via:
    toki tui
    toki tui --daemon-url http://127.0.0.1:8787 --auth-token ...
"""

from __future__ import annotations

import uuid

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.reactive import reactive
from textual.widgets import Input

from . import _cleanup as _tui_cleanup
from . import history
from .client import DaemonClient
from .widgets import (
    ChatLog,
    InfoPanel,
    Sidebar,
    StatusBar,
)


class TokiApp(App):
    """Production TUI chat client for Toki Recur."""

    CSS_PATH = "theme.tcss"

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
        Binding("ctrl+r", "reprobe", "Re-probe backends", priority=True),
        Binding("ctrl+l", "clear_chat", "Clear chat", priority=True),
        Binding("ctrl+u", "new_session", "New session", priority=True),
        Binding("ctrl+1", "mode_raw", "Raw", priority=True),
        Binding("ctrl+2", "mode_cached", "Cached", priority=True),
        Binding("ctrl+3", "mode_compare", "Compare", priority=True),
        Binding("ctrl+p", "run_precision", "Precision check", priority=True),
        Binding("ctrl+m", "toggle_info_panel", "Toggle info panel", priority=True),
        Binding("ctrl+t", "cycle_max_tokens", "Cycle max tokens", priority=True),
        Binding("pageup", "chat_scroll_up", "Scroll chat up", priority=True, show=False),
        Binding("pagedown", "chat_scroll_down", "Scroll chat down", priority=True, show=False),
        Binding("home", "chat_scroll_top", "Scroll chat to top", priority=True, show=False),
        Binding("end", "chat_scroll_bottom", "Scroll chat to bottom", priority=True, show=False),
    ]

    # ── Reactive state ───────────────────────────────────────────────

    mode: reactive[str] = reactive("cached")
    session_id: reactive[str] = reactive("—")
    llama_status: reactive[str] = reactive("unknown")
    daemon_status: reactive[str] = reactive("unknown")
    auth_status: reactive[str] = reactive("n/a")
    busy: reactive[bool] = reactive(False)

    def __init__(
        self,
        daemon_url: str = "http://127.0.0.1:8787",
        raw_url: str = "http://127.0.0.1:8088",
        auth_token: str | None = None,
    ):
        super().__init__()
        self.client = DaemonClient(
            daemon_url=daemon_url,
            raw_url=raw_url,
            auth_token=auth_token,
        )
        # Generate a stable session ID per TUI session.
        self._session_id = uuid.uuid4().hex
        self._history: list[dict] = []
        self._message_counter = 0
        self._last_ttft_ms = 0.0
        self._hits = 0
        self._misses = 0
        self._skipped_total = 0
        self._generated_total = 0
        self._messages = 0
        # Max tokens for completions: env override, else 2048 default.
        self._max_tokens_value = self._env_max_tokens()
        self._max_tokens_presets = [512, 1024, 2048, 4096]

    # ── Compose ───────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        self.sidebar = Sidebar(id="sidebar")
        self.chat = ChatLog(id="chat")
        self.info = InfoPanel(id="info")
        self.status_bar = StatusBar(id="status-bar")
        self.input = Input(
            placeholder="Type a message and press Enter…  (try /help)",
            id="input",
        )
        yield Horizontal(
            self.sidebar,
            Container(self.chat, id="chat-container"),
            self.info,
            id="main-area",
        )
        yield self.input
        yield self.status_bar

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def on_mount(self) -> None:
        self.session_id = self._session_id
        self.status_bar.session_id = self._session_id
        self.status_bar.mode = self.mode
        self.status_bar.llama_status = self.llama_status
        self.status_bar.daemon_status = self.daemon_status
        self.status_bar.auth_status = self.auth_status
        self.status_bar.max_tokens = self._max_tokens_value
        self.status_bar.ctx_pct = 0
        self.sidebar.auth_status = self.auth_status
        self.info.set_ctx(0)

        # Brand header first, then history below it.
        self.chat.write_header()

        # Load history (last N messages).
        loaded = history.load(limit=100)
        for m in loaded:
            self._history.append(m)
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "user":
                self.chat.write_user(content)
            elif role == "assistant":
                # Re-render as a finished assistant line.
                self._message_counter += 1
                self.chat._current_assistant_text = content
                self.chat.end_assistant(
                    mode=m.get("mode", "cached") or "cached",
                    hit=m.get("meta", {}).get("hit", False),
                    ttft_ms=m.get("meta", {}).get("ttft_ms", 0),
                    total_ms=m.get("meta", {}).get("total_ms", 0),
                    tokens_generated=m.get("meta", {}).get("tokens", 0),
                    skipped_tokens=m.get("meta", {}).get("skipped", 0),
                )
        if loaded:
            self.chat.write_system(f"loaded {len(loaded)} messages from history", style="dim")

        # Initial probe.
        await self._probe_all()
        # Refresh metrics + cache state every 2s.
        self.set_interval(2.0, self._refresh_live)
        self.set_interval(3.0, self._probe_all)
        self.input.focus()

    # ── Actions ───────────────────────────────────────────────────────

    async def action_reprobe(self) -> None:
        await self._probe_all()
        self.chat.write_system("backend health re-probed", style="dim")

    async def action_clear_chat(self) -> None:
        self.chat.clear()
        self._history.clear()
        history.clear()
        self.chat.write_system("chat cleared", style="dim")

    async def action_new_session(self) -> None:
        self._session_id = uuid.uuid4().hex
        self.session_id = self._session_id
        self.status_bar.session_id = self._session_id
        self.chat.write_system(f"new session: {self._session_id[:12]}…", style="cyan")

    def action_mode_raw(self) -> None:
        self._set_mode("raw")

    def action_mode_cached(self) -> None:
        self._set_mode("cached")

    def action_mode_compare(self) -> None:
        self._set_mode("compare")

    def action_toggle_info_panel(self) -> None:
        self.info.display = not self.info.display

    async def action_chat_scroll_up(self) -> None:
        self.chat.scroll_page_up(animate=False)

    async def action_chat_scroll_down(self) -> None:
        self.chat.scroll_page_down(animate=False)

    async def action_chat_scroll_top(self) -> None:
        self.chat.scroll_home(animate=False)

    async def action_chat_scroll_bottom(self) -> None:
        self.chat.scroll_end(animate=False)

    async def action_run_precision(self) -> None:
        if self.daemon_status != "ok" and self.llama_status != "ok":
            self.chat.write_system("cannot run precision: both backends down", style="red")
            return
        self.chat.write_system("running precision check…", style="cyan")
        result = await self.client.precision_check(max_tokens=32)
        if "error" in result and result.get("prompts_tested", 0) == 0:
            self.chat.write_system(f"precision failed: {result['error']}", style="red")
            return
        self.info.set_precision(result)
        bd = (result.get("baseline_deterministic", 0) or 0) * 100
        rd = (result.get("recycled_deterministic", 0) or 0) * 100
        xd = (result.get("byte_identical_to_baseline", 0) or 0) * 100
        self.chat.write_system(
            f"precision: baseline={bd:.0f}%  recycled={rd:.0f}%  match={xd:.0f}%",
            style="green",
        )

    # ── Mode helper ───────────────────────────────────────────────────

    def _set_mode(self, mode: str) -> None:
        if self.daemon_status == "unreachable" and mode in ("cached", "compare"):
            self.chat.write_system(
                f"cannot switch to {mode}: daemon unreachable",
                style="yellow",
            )
            return
        self.mode = mode
        self.status_bar.mode = mode
        self.sidebar.mode = mode
        self.chat.write_system(f"mode → {mode}", style="cyan")

    # ── Input handling ────────────────────────────────────────────────

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if self.busy:
            return
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""
        await self._handle_input(text)

    async def _handle_input(self, text: str) -> None:
        # Slash commands.
        if text.startswith("/"):
            await self._handle_slash(text)
            return

        if self.mode == "compare":
            await self._do_compare(text)
        else:
            await self._do_chat(text, mode=self.mode)

    async def _handle_slash(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        if cmd == "/help":
            self.chat.write_system(self._help_text(), style="cyan")
        elif cmd == "/quit" or cmd == "/exit":
            self.exit()
        elif cmd == "/clear":
            await self.action_clear_chat()
        elif cmd in ("/raw", "/cached", "/compare"):
            self._set_mode(cmd[1:])
        elif cmd == "/session":
            if arg:
                self._session_id = arg
                self.session_id = arg
                self.status_bar.session_id = arg
                self.chat.write_system(f"session set to {arg}", style="cyan")
            else:
                self.chat.write_system(
                    f"current session: {self._session_id}", style="cyan",
                )
        elif cmd == "/session-new" or cmd == "/newsession":
            await self.action_new_session()
        elif cmd == "/precision":
            await self.action_run_precision()
        elif cmd == "/metrics":
            m = await self.client.metrics_live()
            self.info.update_metrics(m)
        elif cmd == "/slots":
            state = await self.client.cache_state()
            self.sidebar.slots = state.get("slots", [])
            self.sidebar.session_to_slot = state.get("sessions", {})
            self.chat.write_system(f"slots: {len(state.get('slots', []))}", style="cyan")
        elif cmd == "/reset":
            ok = await self.client.reset_cache()
            self.chat.write_system(
                "cache reset ok" if ok else "cache reset failed",
                style="green" if ok else "red",
            )
        elif cmd == "/mode":
            if arg in ("raw", "cached", "compare"):
                self._set_mode(arg)
            else:
                self.chat.write_system(f"current mode: {self.mode}", style="cyan")
        elif cmd == "/reprobe":
            await self.action_reprobe()
        elif cmd == "/max_tokens" or cmd == "/maxtokens":
            self._set_max_tokens(arg)
        elif cmd == "/help" or cmd == "/?":
            self.chat.write_system(self._help_text(), style="cyan")
        else:
            self.chat.write_system(
                f"unknown command: {cmd}  (try /help)", style="red",
            )

    def _help_text(self) -> str:
        return (
            "Commands: /help /quit /clear /mode [raw|cached|compare] "
            "/session [id] /session-new /precision /metrics /slots /reset /reprobe "
            "/max_tokens [n]\n"
            "Keys:    Ctrl+1 raw · Ctrl+2 cached · Ctrl+3 compare · "
            "Ctrl+L clear · Ctrl+R reprobe · Ctrl+P precision · "
            "Ctrl+U new session · Ctrl+M toggle metrics panel · "
            "Ctrl+T cycle max tokens"
        )

    def action_cycle_max_tokens(self) -> None:
        """Ctrl+T: cycle the completion token budget through presets."""
        current = self._max_tokens_value
        next_value = self._max_tokens_presets[0]
        for preset in self._max_tokens_presets:
            if preset > current:
                next_value = preset
                break
        if next_value == current:
            next_value = self._max_tokens_presets[0]
        self._set_max_tokens(str(next_value))

    # ── Chat dispatch ─────────────────────────────────────────────────

    async def _do_chat(self, prompt: str, *, mode: str) -> None:
        if self.busy:
            return
        self.busy = True
        self.input.disabled = True

        # Render user message.
        self.chat.write_user(prompt)
        history.append("user", prompt)
        self._history.append({"role": "user", "content": prompt})

        # Begin assistant line.
        self._message_counter += 1
        mid = self._message_counter
        self.chat.begin_assistant(mid, mode=mode)

        accumulated = ""
        metrics: dict = {}
        error_msg: str | None = None
        hit = False
        soft_hit = False
        skipped = 0
        prompt_tokens = 0
        cache_applicable = True
        try:
            async for event in self.client.chat_stream(
                prompt,
                mode=mode,
                session_id=self._session_id,
                max_tokens=self._max_tokens(),
                temperature=self._temperature(),
                history=self._history_for_api(),
            ):
                ev = event.get("event")
                if ev == "token":
                    chunk = event.get("content", "")
                    accumulated += chunk
                    # Live-render the in-progress line on every token.
                    self.chat.append_assistant_token(chunk)
                elif ev == "reasoning":
                    # Thinking-mode delta: streamed dim alongside the
                    # answer, then placed on its own indented dim line
                    # when the message finalises.
                    self.chat.append_assistant_reasoning(
                        event.get("content", "")
                    )
                elif ev == "done":
                    metrics = event.get("metrics", {}) or {}
                    cache = event.get("cache", {}) or {}
                    prompt_tokens = int(metrics.get("prompt_tokens", 0) or 0)
                    # Raw mode emits {"applicable": False, "mode": "raw"}.
                    # In that case there's no hit/miss verdict — the
                    # request bypassed the cache entirely. Don't pollute
                    # the hit-rate counters.
                    cache_applicable = cache.get("applicable", True)
                    hit = cache_applicable and bool(cache.get("hit"))
                    soft_hit = cache_applicable and bool(cache.get("soft_hit"))
                    skipped = int(cache.get("skipped_tokens", 0))
                elif ev == "error":
                    error_msg = event.get("message")
        except Exception as e:  # noqa: BLE001
            error_msg = f"{type(e).__name__}: {e}"

        # Finalize assistant line. The widget interprets the badge based
        # on the (hit, soft_hit, mode) tuple: in raw mode both are False
        # and the widget should render the "↯ raw" badge instead.
        self.chat.end_assistant(
            mode=mode,
            hit=hit,
            soft_hit=soft_hit,
            error=error_msg,
            ttft_ms=metrics.get("ttft_ms", 0),
            total_ms=metrics.get("total_ms", 0),
            tokens_generated=metrics.get("tokens_generated", 0),
            prompt_tokens=prompt_tokens,
            skipped_tokens=skipped,
        )

        # Update counters. Raw mode is exempt — no hit/miss verdict.
        if not error_msg and metrics and cache_applicable:
            self._last_ttft_ms = metrics.get("ttft_ms", 0)
            self.info.push_ttft(self._last_ttft_ms)
            self._generated_total += int(metrics.get("tokens_generated", 0))
            self._messages += 1
            if hit:
                self._hits += 1
            else:
                self._misses += 1
            self._skipped_total += skipped
            self._refresh_status_bar()
            self._refresh_session_stats()

        # Save assistant message.
        if not error_msg:
            history.append(
                "assistant", accumulated, mode=mode,
                metadata={
                    "hit": hit, "ttft_ms": metrics.get("ttft_ms", 0),
                    "total_ms": metrics.get("total_ms", 0),
                    "tokens": metrics.get("tokens_generated", 0),
                    "skipped": skipped,
                },
            )
            # In-memory mirror so later turns in this session carry
            # context (the JSONL file is only read once at startup).
            self._history.append({"role": "assistant", "content": accumulated})

        self.busy = False
        self.input.disabled = False
        self.input.focus()

    async def _do_compare(self, prompt: str) -> None:
        if self.busy:
            return
        self.busy = True
        self.input.disabled = True

        self.chat.write_user(prompt + "  [compare]")
        history.append("user", prompt)
        self._history.append({"role": "user", "content": prompt})

        # Run raw + cached back-to-back.
        raw_text = ""
        raw_metrics: dict = {}
        cached_text = ""
        cached_metrics: dict = {}

        # Raw first.
        try:
            async for ev in self.client.chat_stream(
                prompt, mode="raw",
                session_id=self._session_id,
                max_tokens=self._max_tokens(),
                temperature=self._temperature(),
                history=self._history_for_api(),
            ):
                if ev["event"] == "done":
                    raw_text = ev.get("text", "")
                    raw_metrics = ev.get("metrics", {})
        except Exception as e:
            self.chat.write_system(f"raw failed: {e}", style="red")

        # Cached.
        try:
            async for ev in self.client.chat_stream(
                prompt, mode="cached",
                session_id=self._session_id,
                max_tokens=self._max_tokens(),
                temperature=self._temperature(),
                history=self._history_for_api(),
            ):
                if ev["event"] == "done":
                    cached_text = ev.get("text", "")
                    cached_metrics = ev.get("metrics", {})
        except Exception as e:
            self.chat.write_system(f"cached failed: {e}", style="red")

        # Compute diff + speedup.
        identical = raw_text == cached_text
        if identical:
            diff_chars = 0
            first_diff = -1
        else:
            diff_chars = sum(
                1 for a, b in zip(raw_text, cached_text, strict=False) if a != b
            ) + abs(len(raw_text) - len(cached_text))
            first_diff = 0
            for i in range(min(len(raw_text), len(cached_text))):
                if raw_text[i] != cached_text[i]:
                    first_diff = i
                    break
        speedup = None
        if raw_metrics.get("total_ms") and cached_metrics.get("total_ms"):
            speedup = raw_metrics["total_ms"] / max(cached_metrics["total_ms"], 1e-9)

        self.chat.write_compare_pair(
            raw_text=raw_text,
            cached_text=cached_text,
            raw_m=raw_metrics,
            cached_m=cached_metrics,
            speedup=speedup,
            identical=identical,
            first_diff=first_diff,
            diff_chars=diff_chars,
        )
        if cached_metrics:
            self.info.push_ttft(cached_metrics.get("ttft_ms", 0))
            self._last_ttft_ms = cached_metrics.get("ttft_ms", 0)
            self._refresh_status_bar()

        # Keep context for subsequent turns (use the raw answer — it is
        # the complete one).
        if raw_text:
            history.append("assistant", raw_text, mode="compare")
            self._history.append({"role": "assistant", "content": raw_text})

        self.busy = False
        self.input.disabled = False
        self.input.focus()

    # ── Helpers ───────────────────────────────────────────────────────

    def _max_tokens(self) -> int:
        return self._max_tokens_value

    def _env_max_tokens(self) -> int:
        import os
        try:
            value = int(os.environ.get("TOKI_MAX_TOKENS", "2048"))
        except ValueError:
            value = 2048
        return max(64, min(value, 8192))

    def _set_max_tokens(self, arg: str) -> None:
        """Apply /max_tokens <n>: validated 64..8192."""
        try:
            value = int(arg)
        except ValueError:
            self.chat.write_system(
                f"current max_tokens: {self._max_tokens_value}  (use /max_tokens <64-8192>)",
                style="cyan",
            )
            return
        if not 64 <= value <= 8192:
            self.chat.write_system(
                f"max_tokens must be 64-8192; got {value}", style="red",
            )
            return
        self._max_tokens_value = value
        self.status_bar.max_tokens = value
        self.chat.write_system(f"max_tokens set to {value}", style="cyan")

    def _temperature(self) -> float:
        import os
        try:
            return float(os.environ.get("TOKI_TEMPERATURE", "0.0"))
        except ValueError:
            return 0.0

    def _history_for_api(self) -> list[dict]:
        # Cap at the last 20 messages so long sessions stay well under
        # the 6144-token context budget.
        return list(self._history)[-20:]

    def _refresh_status_bar(self) -> None:
        total = self._hits + self._misses
        rate = f"{(self._hits / total * 100):.0f}% ({self._hits}/{total})" if total else "—"
        self.status_bar.hit_rate = rate
        self.status_bar.session_id = self._session_id

    def _refresh_session_stats(self) -> None:
        total = self._hits + self._misses
        rate = f"{(self._hits / total * 100):.0f}% ({self._hits}/{total})" if total else "—"
        self.info.update_session({
            "messages": self._messages,
            "tokens_generated": self._generated_total,
            "tokens_reused": self._skipped_total,
            "hit_rate": rate,
        })

    # ── Periodic refresh ──────────────────────────────────────────────

    async def _refresh_live(self) -> None:
        try:
            metrics = await self.client.metrics_live()
            self.info.update_metrics(metrics)
            state = await self.client.cache_state()
            max_ctx = int(state.get("max_context_tokens", 0) or 0)
            # Per-slot payloads omit max_context_tokens; backfill the
            # engine-wide budget so slots render real percentages.
            slots = []
            for s in state.get("slots", []):
                s = dict(s)
                if max_ctx:
                    s["max_context_tokens"] = max_ctx
                slots.append(s)
            self.sidebar.slots = slots
            self.sidebar.session_to_slot = state.get("sessions", {})
            # Context utilization for this session's slot.
            slot_id = state.get("sessions", {}).get(self._session_id)
            pct = 0
            if slot_id is not None and max_ctx:
                for s in slots:
                    if s.get("slot_id") == slot_id:
                        ctx = s.get("context_tokens", 0)
                        pct = min(100, ctx * 100 // max_ctx)
                        break
            self.status_bar.ctx_pct = pct
            self.info.set_ctx(pct)
        except Exception:
            pass

    async def _probe_all(self) -> None:
        llama = await self.client.health_llama()
        daemon = await self.client.health_daemon()
        self.llama_status = llama
        self.daemon_status = daemon

        # Compute auth state: did the daemon reject our token?
        # "auth_required" means daemon has a token but we didn't send one
        # (or our token was wrong). We can distinguish by checking if
        # we have a configured token at all.
        if daemon == "ok":
            self.auth_status = "ok"
        elif daemon == "auth_required":
            # The daemon requires auth. Did we send one?
            if self.client._auth_token:
                self.auth_status = "wrong"
            else:
                self.auth_status = "required"
        elif daemon == "unreachable":
            self.auth_status = "n/a" if self.daemon_status == "unknown" else "n/a"
        else:
            self.auth_status = "n/a"

        self.status_bar.llama_status = llama
        self.status_bar.daemon_status = daemon
        self.status_bar.auth_status = self.auth_status
        self.sidebar.llama_status = llama
        self.sidebar.daemon_status = daemon
        self.sidebar.auth_status = self.auth_status
        self.sidebar.raw_url = self.client.raw_url
        self.sidebar.daemon_url = self.client.daemon_url

        # If daemon unreachable and mode requires daemon, drop back to raw.
        if daemon == "unreachable" and self.mode in ("cached", "compare"):
            self.mode = "raw"
            self.status_bar.mode = "raw"
            self.sidebar.mode = "raw"
            self.chat.write_system(
                "daemon unreachable — mode auto-changed to raw",
                style="yellow",
            )
        elif daemon == "auth_required" and self.mode in ("cached", "compare"):
            self.chat.write_system(
                "daemon requires auth — set TOKI_AUTH_TOKEN and restart",
                style="yellow",
            )


def run(
    *,
    daemon_url: str = "http://127.0.0.1:8787",
    raw_url: str = "http://127.0.0.1:8088",
    auth_token: str | None = None,
) -> None:
    """Entry point invoked by the CLI subcommand."""
    # See tui/_cleanup.py — defensive cleanup so the terminal is
    # restored after `toki tui` quits even if Textual's own cleanup
    # is skipped by a slow callback or an OS signal.
    _tui_cleanup.install()
    app = TokiApp(
        daemon_url=daemon_url, raw_url=raw_url, auth_token=auth_token,
    )
    app.run()
