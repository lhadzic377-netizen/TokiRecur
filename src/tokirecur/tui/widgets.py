"""Custom Textual widgets for the Toki Recur TUI.

Widgets:
- ChatLog: scrollable transcript with role badges + inline metrics.
  Built on VerticalScroll + a Static surface so the line list can be
  streamed in place (RichLog cannot replace lines) and the widget can
  actually scroll (plain Static has no scrollable region).
- StatusBar: one-line footer: backends + auth + mode + session +
  hit rate + max tokens + context utilization.
- InfoPanel: single right-side panel: live metrics + TTFT sparkline +
  session counters + last precision results.
- Sidebar: slim left panel: mode, session, backends, auth, slots.

All widgets are designed for live updates without flickering.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from collections import deque
from typing import Any

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.widgets import Static

from .art import AMBER, RED, YELLOW

# ── ChatLog ──────────────────────────────────────────────────────────



def _format_stamp(ts: float | None = None) -> str:
    """The time to print on a line.

    A replayed line carries the moment it happened, so yesterday's
    conversation does not open looking like it happened just now. Today's
    lines stay as a bare clock time; older ones say which day they were.
    """
    now = datetime.now()
    when = datetime.fromtimestamp(ts) if ts else now
    if when.date() == now.date():
        return when.strftime("%H:%M:%S")
    if when.date() == now.date() - timedelta(days=1):
        return "yesterday " + when.strftime("%H:%M")
    return when.strftime("%d %b %H:%M")


class ChatLog(VerticalScroll):
    """Scrollable transcript.

    Each user message is rendered as a left-aligned line; each
    assistant message with role badge, cache-hit badge and inline
    metrics. The in-flight assistant line is mutated on every streamed
    token (cheap in-place re-render).

    Built on ``VerticalScroll`` + an inner ``Static`` surface: the
    surface grows with the transcript and the container provides real
    scrolling (wheel + PgUp/PgDn via app bindings). When the view is
    already pinned to the bottom it auto-follows new content; if the
    user has scrolled up, streaming does not yank the view.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._surface = Static("", markup=False)
        self._lines: list[Text] = []
        self._current_assistant_id: int | None = None
        self._current_assistant_text: str = ""
        self._current_assistant_reasoning: str = ""
        self._current_assistant_mode: str = "cached"
        self._streaming_line_idx: int | None = None

    def compose(self) -> Any:
        yield self._surface

    # ── Public API (same as the previous RichLog-based widget) ───────

    @property
    def lines(self) -> list[Text]:
        """Expose the internal line store (used by tests + app)."""
        return self._lines

    def write(self, text: RenderableType) -> None:
        self._lines.append(text)
        self._trim()
        self._rerender()

    def clear(self) -> None:
        self._lines.clear()
        self._current_assistant_id = None
        self._current_assistant_text = ""
        self._current_assistant_reasoning = ""
        self._streaming_line_idx = None
        self._rerender()

    def write_user(self, content: str, *, ts: float | None = None) -> None:
        stamp = _format_stamp(ts)
        text = Text()
        text.append(f"[{stamp}] ", style="dim")
        text.append("you ", style="bold cyan")
        text.append("▸ ", style="cyan")
        text.append(content, style="white")
        self.write(text)

    def write_header(self) -> None:
        """Render the typographic brand header as the first lines."""
        from . import art

        self.write(Text(art.WORDMARK, style=f"bold {art.YELLOW}"))
        self.write(Text(art.TAGLINE, style="dim"))

    def begin_assistant(self, message_id: int, mode: str = "cached") -> None:
        self._current_assistant_id = message_id
        self._current_assistant_mode = mode
        self._current_assistant_text = ""
        self._current_assistant_reasoning = ""
        self._lines.append(self._streaming_text())
        self._streaming_line_idx = len(self._lines) - 1
        self._trim()
        self._rerender()

    def append_assistant_token(self, token: str) -> None:
        """Tokens arrive fast; buffer and re-render the live line."""
        self._current_assistant_text += token
        self._refresh_streaming_line()

    def append_assistant_reasoning(self, token: str) -> None:
        """Thinking-mode delta: streamed dimmed on the live line, rendered
        separately at end_assistant for the final transcript."""
        self._current_assistant_reasoning += token
        self._refresh_streaming_line()

    def end_assistant(
        self,
        *,
        mode: str = "cached",
        hit: bool = False,
        soft_hit: bool = False,
        error: str | None = None,
        ttft_ms: float = 0.0,
        total_ms: float = 0.0,
        tokens_generated: int = 0,
        prompt_tokens: int = 0,
        skipped_tokens: int = 0,
        ts: float | None = None,
    ) -> None:
        """Render the final assistant line with badge + metrics."""
        stamp = _format_stamp(ts)
        if error:
            text = Text()
            text.append(f"[{stamp}] ", style="dim")
            text.append("ai  ", style="bold red")
            text.append(f"[{mode}] ", style="red")
            text.append("▸ ", style="red")
            text.append(f"[error] {error}", style="red")
            self._replace_streaming_line(text)
            self._current_assistant_id = None
            self._current_assistant_text = ""
            self._current_assistant_reasoning = ""
            self._streaming_line_idx = None
            return

        if mode == "raw":
            # Raw mode bypasses the cache entirely; there's no hit/miss
            # verdict. Use a neutral badge that signals "intentional
            # bypass" rather than the misleading "✗ miss".
            badge_text, badge_style = "↯ raw", "bold blue"
        elif hit:
            badge_text, badge_style = "✓ hit", "bold green"
        elif soft_hit:
            badge_text, badge_style = "~ soft", "yellow"
        else:
            badge_text, badge_style = "✗ miss", "red"

        text = Text()
        text.append(f"[{stamp}] ", style="dim")
        text.append("ai  ", style="bold green")
        text.append(f"[{mode}] ", style="green")
        text.append(badge_text + " ", style=badge_style)
        text.append(self._current_assistant_text or "(empty)", style="white")
        if tokens_generated > 0:
            text.append(
                f"  [in {prompt_tokens} tok · ttft {ttft_ms:.0f}ms · "
                f"total {total_ms:.0f}ms · {tokens_generated} tok",
                style="dim",
            )
            if skipped_tokens > 0:
                text.append(f" · reused {skipped_tokens}", style="dim cyan")
            text.append("]", style="dim")
        if self._current_assistant_reasoning:
            think_line = Text()
            think_line.append(f"    think › {self._current_assistant_reasoning}",
                              style="dim italic")
            # Mutate _lines directly instead of going through
            # _replace_streaming_line, because that helper rerenders
            # on its own and the answer append below would then miss
            # the render — the screen would show only the dim
            # think-summary line and the answer would silently
            # disappear until some other event triggered a rerender.
            if (self._streaming_line_idx is not None
                    and self._streaming_line_idx < len(self._lines)):
                self._lines[self._streaming_line_idx] = think_line
            else:
                self._lines.append(think_line)
            self._lines.append(text)
            self._trim()
            self._rerender()
        else:
            self._replace_streaming_line(text)
        self._current_assistant_id = None
        self._current_assistant_text = ""
        self._current_assistant_reasoning = ""
        self._streaming_line_idx = None

    def write_compare_pair(
        self,
        raw_text: str,
        cached_text: str,
        raw_m: dict,
        cached_m: dict,
        speedup: float | None,
        identical: bool,
        first_diff: int,
        diff_chars: int,
    ) -> None:
        """Render a side-by-side compare summary."""
        ts = time.strftime("%H:%M:%S")
        # Header
        head = Text()
        head.append(f"[{ts}] ", style="dim")
        head.append("compare ", style="bold magenta")
        if speedup is None:
            head.append("[one side failed]", style="red")
        else:
            speedup_color = (
                "green" if speedup > 1.05 else
                "yellow" if speedup > 0.95 else "red"
            )
            head.append(f"[{speedup:.2f}× ", style=f"bold {speedup_color}")
            head.append("cached vs raw", style="dim")
            head.append("]", style=f"bold {speedup_color}")
        head.append(
            "  raw=" + f"{raw_m.get('ttft_ms', 0):.0f}/{raw_m.get('total_ms', 0):.0f}ms",
            style="dim",
        )
        head.append(
            "  cached=" + f"{cached_m.get('ttft_ms', 0):.0f}/{cached_m.get('total_ms', 0):.0f}ms",
            style="dim",
        )
        self.write(head)

        if speedup is None:
            return

        # Body
        body = Text()
        if identical:
            body.append("  outputs byte-identical", style="green")
        else:
            body.append(
                f"  outputs differ: {diff_chars} chars (first diff at char {first_diff})",
                style="yellow",
            )
        self.write(body)

    def write_system(self, message: str, style: str = "yellow") -> None:
        ts = time.strftime("%H:%M:%S")
        text = Text()
        text.append(f"[{ts}] ", style="dim")
        text.append("system ", style="bold magenta")
        text.append("▸ ", style="magenta")
        text.append(message, style=style)
        self.write(text)

    # ── Internals ────────────────────────────────────────────────────

    def _streaming_text(self) -> Text:
        ts = time.strftime("%H:%M:%S")
        text = Text()
        text.append(f"[{ts}] ", style="dim")
        text.append("ai  ", style="bold green")
        text.append(f"[{self._current_assistant_mode}] ", style="green")
        text.append("▸ ", style="green")
        if self._current_assistant_reasoning:
            # Live: show reasoning dimmed so the user sees the model
            # thinking while it streams, followed by any visible content.
            text.append("think ", style="dim italic")
            text.append(self._current_assistant_reasoning, style="dim")
        if self._current_assistant_text:
            if self._current_assistant_reasoning:
                text.append("\n    ", style="white")
            text.append(self._current_assistant_text, style="white")
        elif not self._current_assistant_reasoning:
            text.append("…", style="dim")
        return text

    def _refresh_streaming_line(self) -> None:
        """Rebuild the in-flight assistant line in place so the user
        sees tokens as they stream, not only after the response ends."""
        if self._streaming_line_idx is not None and self._streaming_line_idx < len(self._lines):
            self._lines[self._streaming_line_idx] = self._streaming_text()
        self._rerender()

    def _replace_streaming_line(self, final_text: Text) -> None:
        if self._streaming_line_idx is not None and self._streaming_line_idx < len(self._lines):
            self._lines[self._streaming_line_idx] = final_text
        else:
            self._lines.append(final_text)
        self._trim()
        self._rerender()

    def _trim(self) -> None:
        """Keep the transcript bounded so very long chats don't grow
        the renderable without limit."""
        if len(self._lines) > 500:
            drop = len(self._lines) - 400
            self._lines = self._lines[drop:]
            if self._streaming_line_idx is not None:
                self._streaming_line_idx -= drop
                if self._streaming_line_idx < 0:
                    self._streaming_line_idx = 0

    def _rerender(self) -> None:
        self._surface.update(Group(*self._lines))
        try:
            # Follow new content only when already pinned to the bottom,
            # so streaming never yanks the view away from history.
            if self.is_vertical_scroll_end:
                self.scroll_end(animate=False)
        except Exception:
            pass


# ── StatusBar ───────────────────────────────────────────────────────


class StatusBar(Static):
    """One-line footer with backend health + auth + mode + session."""

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 1;
        background: $boost;
        color: $text;
        padding: 0 1;
    }
    """

    mode: reactive[str] = reactive("cached")
    session_id: reactive[str] = reactive("—")
    llama_status: reactive[str] = reactive("unknown")
    daemon_status: reactive[str] = reactive("unknown")
    auth_status: reactive[str] = reactive("n/a")  # ok | required | wrong | n/a
    hit_rate: reactive[str] = reactive("—")
    max_tokens: reactive[int] = reactive(2048)
    ctx_pct: reactive[int] = reactive(0)

    def render(self) -> RenderableType:
        text = Text()
        text.append(" ● ", style="bold")
        text.append(_status_glyph(self.llama_status), style=_status_color(self.llama_status))
        text.append("llama ", style="dim")
        text.append(_status_glyph(self.daemon_status), style=_status_color(self.daemon_status))
        text.append("daemon ", style="dim")
        text.append(_auth_glyph(self.auth_status), style=_auth_color(self.auth_status))
        text.append("auth │ ", style="dim")
        text.append(f"mode={self.mode} ", style=f"bold {YELLOW}")
        text.append(f"session={self.session_id[:10]} ", style="dim")
        text.append(f"ctx={self.ctx_pct}% ", style="bold cyan")
        text.append(f"maxtok={self.max_tokens} ", style=f"bold {AMBER}")
        text.append(f"hits={self.hit_rate} ", style="dim cyan")
        from .. import tap

        if tap.is_enabled():
            text.append("● rec ", style=f"bold {RED}")
        text.append("│ Ctrl+R · Ctrl+T · /help", style="dim")
        return text


def _auth_glyph(s: str) -> str:
    return {"ok": "●", "required": "◐", "wrong": "✗", "n/a": "—"}.get(s, "—")


def _auth_color(s: str) -> str:
    return {"ok": "green", "required": "yellow",
            "wrong": "red", "n/a": "dim"}.get(s, "dim")


def _status_glyph(s: str) -> str:
    return {"ok": "●", "auth_required": "◐", "unreachable": "○",
            "unknown": "?"}.get(s, "?")


def _status_color(s: str) -> str:
    return {"ok": "green", "auth_required": "yellow",
            "unreachable": "red", "unknown": "dim"}.get(s, "dim")


# ── InfoPanel ───────────────────────────────────────────────────────


class InfoPanel(Static):
    """Single right-side panel.

    Sections (top to bottom):
    - Live metrics from the daemon's Prometheus endpoint + TTFT sparkline
    - This session's counters (messages, generated/reused tokens, hit rate)
      and context utilization
    - Last precision-check results (Ctrl+P) or a hint
    """

    DEFAULT_CSS = """
    InfoPanel {
        dock: right;
        width: 28;
        padding: 1;
        background: $panel;
        border-left: solid $primary;
    }
    """

    metrics: reactive[dict] = reactive(dict)
    session_stats: reactive[dict] = reactive(dict)
    ctx_pct: reactive[int] = reactive(0)
    precision: reactive[dict] = reactive(dict)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ttft_samples: deque[float] = deque(maxlen=40)

    def push_ttft(self, ttft_ms: float) -> None:
        if ttft_ms > 0:
            self._ttft_samples.append(ttft_ms)

    def update_metrics(self, m: dict) -> None:
        self.metrics = dict(m)

    def update_session(self, stats: dict) -> None:
        self.session_stats = dict(stats)

    def set_ctx(self, pct: int) -> None:
        self.ctx_pct = pct

    def set_precision(self, d: dict) -> None:
        self.precision = dict(d)

    def render(self) -> RenderableType:
        panel = Text()

        # Live metrics.
        panel.append("Live\n", style=f"bold {YELLOW}")
        m = self._metrics
        reqs = int(m.get("requests_total", 0))
        hits = int(m.get("cache_hits", 0))
        misses = int(m.get("cache_misses", 0))
        skipped = int(m.get("tokens_skipped", 0))
        active = int(m.get("active_slots", 0))
        auth_fail = int(m.get("auth_failures", 0))

        total = hits + misses
        hr = f"{(hits / total * 100):.0f}%" if total else "—"
        panel.append(f"  req={reqs} hit={hits} miss={misses}\n", style="white")
        panel.append(f"  rate={hr} skip={skipped}tok slots={active}\n", style="cyan")
        if auth_fail:
            panel.append(f"  auth_fail={auth_fail}\n", style="red")
        if self._ttft_samples:
            panel.append("  ttft " + _ascii_sparkline(self._ttft_samples) + "\n", style="dim")
        panel.append("\n")

        # Session counters (client-side).
        panel.append("Session\n", style=f"bold {YELLOW}")
        s = self._session_stats
        msgs = int(s.get("messages", 0))
        gen = int(s.get("tokens_generated", 0))
        reused = int(s.get("tokens_reused", 0))
        shr = s.get("hit_rate", "—")
        panel.append(f"  msgs={msgs} gen={gen}tok reused={reused}tok\n", style="white")
        panel.append(f"  hits={shr} ctx={self.ctx_pct}%\n", style="cyan")
        panel.append("\n")

        # Precision results.
        panel.append("Precision\n", style=f"bold {YELLOW}")
        d = self.precision
        if not d:
            panel.append("  (Ctrl+P to run)\n", style="dim")
        else:
            n = d.get("prompts_tested", 0)
            for label, key in [
                ("baseline", "baseline_deterministic"),
                ("recycled", "recycled_deterministic"),
                ("match", "byte_identical_to_baseline"),
            ]:
                pct = (d.get(key, 0) or 0) * 100
                color = "green" if pct >= 99 else "yellow" if pct >= 80 else "red"
                bar = _pct_bar(pct, width=14)
                panel.append(f"  {label:<8} {bar} {pct:.0f}%\n", style=color)
            if n:
                panel.append(f"  ({n} prompts)\n", style="dim")
        return Panel(panel, border_style="dim", padding=(0, 1))

    @property
    def _metrics(self) -> dict:
        return self.metrics

    @property
    def _session_stats(self) -> dict:
        return self.session_stats


def _ascii_sparkline(values) -> str:
    """Tiny ASCII bar chart, last 20 samples."""
    if not values:
        return ""
    bars = " ▁▂▃▄▅▆▇█"
    last = list(values)[-20:]
    if not last:
        return ""
    mx = max(last) or 1.0
    return "".join(bars[min(8, int(v / mx * 8))] for v in last)


# ── Sidebar ──────────────────────────────────────────────────────────


class Sidebar(Static):
    """Slim left-side info panel: mode, session, backends, slots."""

    DEFAULT_CSS = """
    Sidebar {
        dock: left;
        width: 22;
        padding: 1;
        background: $panel;
        border-right: solid $primary;
    }
    """

    mode: reactive[str] = reactive("cached")
    session_id: reactive[str] = reactive("—")
    llama_status: reactive[str] = reactive("unknown")
    daemon_status: reactive[str] = reactive("unknown")
    auth_status: reactive[str] = reactive("n/a")
    raw_url: reactive[str] = reactive("—")
    daemon_url: reactive[str] = reactive("—")
    slots: reactive[list[dict]] = reactive(list)
    session_to_slot: reactive[dict] = reactive(dict)

    def render(self) -> RenderableType:
        panel = Text()
        panel.append("Toki\n", style=f"bold {YELLOW}")
        panel.append("─" * 20 + "\n", style="dim")

        # Mode
        mode_style = {
            "raw": "red", "cached": "green", "compare": "magenta",
        }.get(self.mode, "white")
        panel.append(f"mode {self.mode}\n", style=f"bold {mode_style}")

        # Session
        sid = self.session_id
        if len(sid) > 16:
            sid = sid[:8] + "…" + sid[-4:]
        panel.append(f"ses  {sid}\n", style=YELLOW)

        # Backends
        panel.append("─" * 20 + "\n", style="dim")
        panel.append(_status_glyph(self.llama_status),
                     style=_status_color(self.llama_status))
        panel.append(" llama " + _short_url(self.raw_url) + "\n", style="dim")
        panel.append(_status_glyph(self.daemon_status),
                     style=_status_color(self.daemon_status))
        panel.append(" daemon " + _short_url(self.daemon_url) + "\n", style="dim")
        panel.append(_auth_glyph(self.auth_status),
                     style=_auth_color(self.auth_status))
        panel.append(" auth ", style="dim")
        if self.auth_status == "required":
            panel.append("(token needed)", style="yellow")
        elif self.auth_status == "wrong":
            panel.append("(rejected)", style="red")
        elif self.auth_status == "ok":
            panel.append("ok", style="dim")
        else:
            panel.append("n/a", style="dim")
        panel.append("\n")

        # Slots
        panel.append("─" * 20 + "\n", style="dim")
        panel.append("Slots\n", style=f"bold {YELLOW}")
        slots = self.slots or []
        if not slots:
            panel.append("  (none)\n", style="dim")
        else:
            for s in slots[:6]:
                pct = 0
                if s.get("context_tokens") and s.get("max_context_tokens"):
                    pct = min(100, s["context_tokens"] * 100 // s["max_context_tokens"])
                marker = "★" if s.get("reserved") else "·"
                sess = self.session_to_slot.get(s["slot_id"])
                sess_label = ""
                if sess:
                    short = sess[:5] if len(sess) > 5 else sess
                    sess_label = f" {short}"
                color = "red" if pct > 75 else "cyan" if pct > 25 else "dim"
                panel.append(f"  {marker} s{s['slot_id']} {pct}%{sess_label}\n", style=color)
            if len(slots) > 6:
                panel.append(f"  +{len(slots) - 6} more\n", style="dim")
        panel.append("\nCtrl+1/2/3 mode · Ctrl+U session", style="dim")
        return Panel(panel, border_style="dim", padding=(0, 1))


def _short_url(url: str) -> str:
    """'http://127.0.0.1:8088' -> '127.0.0.1:8088' (no scheme)."""
    if not url:
        return "—"
    stripped = url.split("://", 1)[-1]
    return stripped if stripped else url


def _pct_bar(pct: float, width: int = 16) -> str:
    """Render a percentage as a horizontal bar."""
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)
