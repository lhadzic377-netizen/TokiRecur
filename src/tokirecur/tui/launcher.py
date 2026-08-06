"""Ollama-style full-screen launcher for the toki stack.

A ``Textual`` app that mirrors the Ollama launcher: a single centered
column of apps-to-launch, each bound to the pinned model, with a
``→`` key to flip to the model picker and back. A typographic TOKI
wordmark heads the pane (no ASCII art). Footer bar hints the keys.

    ↑/↓ navigate   enter launch   → configure model   ← back · q/esc quit
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from ..ctl import load_registry, model_status
from . import art


@dataclass(frozen=True)
class LaunchApp:
    """A single launcher entry: a client opened with an interface."""

    key: str
    title: str
    description: str
    interface: str
    client: str
    model_binding: bool = True


LAUNCH_APPS: tuple[LaunchApp, ...] = (
    LaunchApp(
        "toki",
        "Chat",
        "Chat with models in the Toki TUI",
        "chat", "toki",
    ),
    LaunchApp(
        "opencode",
        "Launch OpenCode",
        "Anomaly's open-source coding agent",
        "chat", "opencode",
    ),
    LaunchApp(
        "hermes",
        "Launch Hermes Agent",
        "Self-improving AI agent built by Nous Research",
        "chat", "hermes",
    ),
    LaunchApp(
        "openclaw",
        "Launch OpenClaw",
        "Personal AI with 100+ skills",
        "chat", "openclaw",
    ),
    LaunchApp(
        "web",
        "Launch Web Dashboards",
        "browser UIs: toki · openclaw · hermes",
        "web", "toki", model_binding=False,
    ),
)

CSS = f"""
Screen {{
    background: {art.BG};
    color: {art.TEXT};
}}

#pane {{
    width: 100%;
    height: 1fr;
    color: {art.MUTED};
    padding: 0;
}}

#footer {{
    dock: bottom;
    height: 1;
    background: {art.BG};
    color: {art.MUTED};
    border-top: round {art.BORDER_HI};
    padding: 0 1;
}}
"""


class LauncherApp(App[dict | None]):
    """Keyboard-driven launcher: pick an app, pin a model, Enter to run."""

    TITLE = "Toki Recur launcher"

    BINDINGS = [
        Binding("up", "move(-1)", "Up", show=False, priority=True),
        Binding("down", "move(1)", "Down", show=False, priority=True),
        Binding("k", "move(-1)", "", show=False, priority=True),
        Binding("j", "move(1)", "", show=False, priority=True),
        Binding("1", "jump(0)", "1", show=False, priority=True),
        Binding("2", "jump(1)", "2", show=False, priority=True),
        Binding("3", "jump(2)", "3", show=False, priority=True),
        Binding("4", "jump(3)", "4", show=False, priority=True),
        Binding("5", "jump(4)", "5", show=False, priority=True),
        Binding("enter", "launch", "Launch", priority=True),
        Binding("right", "switch", "Configure", show=False, priority=True),
        Binding("left", "back", "Back", show=False, priority=True),
        Binding("tab", "switch", "Tab", priority=True),
        Binding("space", "pin", "Pin", priority=True),
        Binding("escape", "quit_app", "Quit", priority=True),
        Binding("ctrl+c", "quit_app", "Quit", show=False, priority=True),
        Binding("q", "quit_app", "quit", show=False, priority=True),
        Binding("s", "settings", "Settings", priority=True),
    ]

    focus_pane: reactive[str] = reactive("launchers")  # launchers | models | settings
    settings_idx: reactive[int] = reactive(0)
    _refresh: reactive[int] = reactive(0)

    def __init__(
        self,
        on_launch: Callable[[str, str, str], int] | None = None,
    ) -> None:
        """`on_launch(interface, client, model_name)` returns exit code."""
        super().__init__()
        self._on_launch = on_launch
        self.specs: list[Any] = []
        self.launch_idx = 0
        self.model_idx = 0
        self.pinned: str | None = None
        self._statuses: dict[str, Any] = {}
        self._error: str | None = None
        self._launching: str | None = None

    # ── Compose ───────────────────────────────────────────────────────

    def on_mount(self) -> None:
        self.specs = load_registry()
        if not self.specs:
            self._error = "no models in registry — run `toki model list`"
        self._refresh_pinned()
        self.set_interval(3.0, self._poll_statuses)

    def compose(self) -> ComposeResult:
        self.pane = Static("", id="pane", markup=False)
        self.footer = Static("", id="footer", markup=False)
        yield self.pane
        yield self.footer
        self.refresh(recompose=False)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _refresh_pinned(self) -> None:
        if not self.specs:
            return
        active = self._read_active()
        if self.pinned is None or self.pinned not in {s.name for s in self.specs}:
            self.pinned = active or self.specs[0].name
        self.model_idx = max(0, min(self.model_idx, len(self.specs) - 1))

    def _read_active(self) -> str | None:
        try:
            from ..ctl import read_active

            return read_active()
        except Exception:
            return None

    def _poll_statuses(self) -> None:
        for spec in self.specs:
            try:
                st = model_status(spec)
                self._statuses[spec.name] = {
                    "llama": st["llama"]["reachable"],
                    "daemon": st["daemon"]["reachable"],
                    "ctx": st["llama"].get("slot_ctx") or spec.ctx_size,
                }
            except Exception:
                self._statuses[spec.name] = {}
        self._refresh += 1

    def watch__refresh(self, _: int) -> None:
        self.pane.update(self._render())
        self.footer.update(self._render_help())

    # ── Actions ────────────────────────────────────────────────────────

    def action_move(self, delta: int) -> None:
        if self.focus_pane == "launchers":
            self.launch_idx = (self.launch_idx + delta) % len(LAUNCH_APPS)
        elif self.focus_pane == "models":
            if not self.specs:
                return
            self.model_idx = (self.model_idx + delta) % len(self.specs)
        elif self.focus_pane == "settings":
            rows = _settings_rows(self.specs)
            if not rows:
                return
            self.settings_idx = (self.settings_idx + delta) % len(rows)
        self._refresh += 1

    def action_jump(self, index: int) -> None:
        """1-5: jump straight to an app (or model)."""
        if self.focus_pane == "launchers":
            if 0 <= index < len(LAUNCH_APPS):
                self.launch_idx = index
        else:
            if self.specs and 0 <= index < len(self.specs):
                self.model_idx = index
        self._refresh += 1

    def action_switch(self) -> None:
        """→ / tab: open the model picker."""
        if self.focus_pane == "launchers" and self.specs:
            self.focus_pane = "models"
        self._refresh += 1

    def action_back(self) -> None:
        """←: leave the model picker (or settings) back to the launcher list."""
        if self.focus_pane in ("models", "settings"):
            self.focus_pane = "launchers"
        self._refresh += 1

    def action_settings(self) -> None:
        """s: open the settings pane (edit any registry field)."""
        if self.focus_pane == "settings":
            return
        self.focus_pane = "settings"
        self.settings_idx = 0
        self._refresh += 1

    def _current_setting(self) -> tuple[str, str, str] | None:
        rows = _settings_rows(self.specs)
        if not rows:
            return None
        idx = max(0, min(self.settings_idx, len(rows) - 1))
        return rows[idx]

    def action_edit_setting(self) -> None:
        """Enter in settings: open the edit modal for the highlighted field."""
        if self.focus_pane != "settings":
            return
        current = self._current_setting()
        if current is None:
            return
        name, field, value = current
        screen = SettingsEditScreen(name, field, value)
        self.push_screen(screen, self._on_edit_dismissed)

    def _on_edit_dismissed(self, new_value: str | None) -> None:
        """Commit the edit (or no-op on cancel). Called by the modal."""
        if new_value is None:
            self.notify("edit cancelled", timeout=2)
            return
        current = self._current_setting()
        if current is None:
            return
        name, field, _ = current
        from .. import ctl as ctl_mod
        try:
            ctl_mod.update_spec_field(name, field, new_value)
        except (ValueError, KeyError) as e:
            self.notify(str(e), timeout=4, severity="error")
            return
        # Reload specs so the row shows the new value.
        self.specs = ctl_mod.load_registry()
        invalidate_settings_rows()
        self.notify(
            f"{name}.{field} = {new_value}", timeout=3,
        )
        self._refresh += 1

    def action_pin(self) -> None:
        """space: pin the highlighted model (or open the picker)."""
        if self.focus_pane == "launchers":
            if self.specs:
                self.focus_pane = "models"
        else:
            name = self.specs[self.model_idx].name
            fallback = self.specs[0].name if self.specs else None
            self.pinned = name if self.pinned != name else fallback
            from ..ctl import set_active

            try:
                set_active(self.pinned or name)
            except Exception:
                pass
        self._refresh += 1

    async def action_launch(self) -> None:
        if self.focus_pane == "settings":
            self.action_edit_setting()
            return
        app = LAUNCH_APPS[self.launch_idx]
        if self.focus_pane == "models" and self.specs:
            # Enter in the model picker pins and returns to the launcher.
            self.action_pin()
            self.focus_pane = "launchers"
            self._refresh += 1
            return
        if not self.pinned:
            return
        if self._on_launch is not None:
            self._launching = app.title
            self._refresh += 1
            code = await asyncio.to_thread(
                self._on_launch, app.interface, app.client, self.pinned
            )
            self._launching = None
            self._refresh += 1
            if not code:
                self.notify(f"launched {app.title} with {self.pinned}", timeout=3)
        else:
            self.exit({"interface": app.interface, "client": app.client, "model": self.pinned})

    def action_quit_app(self) -> None:
        # Escape from settings/models goes back to the launcher list,
        # not out of the app. Only quit when we're on the main pane.
        if self.focus_pane in ("settings", "models"):
            self.action_back()
            return
        self.exit(None)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    _GUT = "  "  # shared left gutter for every line

    def _render(self) -> Text:
        # The pane fills the screen (padding: 0); reserve a 1-col right
        # buffer so chips don't kiss the screen edge.
        width = max(24, self.size.width - 1)
        if self.focus_pane == "models":
            lines = self._render_models(width)
        elif self.focus_pane == "settings":
            lines = self._render_settings(width)
        else:
            lines = self._render_launchers(width)
            lines = _render_wordmark() + lines
        status = self._render_status(width)
        # Compose a single Text from the per-line Text parts using
        # append_text, which avoids the previous `body.append(str(ln))`
        # round-trip (serialize → re-parse) that caused input lag.
        body = Text(end="")
        first = True
        for ln in lines:
            if not first:
                body.append("\n")
            body.append_text(ln)
            first = False
        body.append("\n\n")
        body.append_text(status)
        return body

    def _render_status(self, width: int) -> Text:
        """One-line footer: pinned model + backend dots + key hint."""
        line = Text(self._GUT, end="")
        if self.pinned:
            line.append(f"● {self.pinned}", style=f"bold {art.YELLOW}")
            st = self._statuses.get(self.pinned, {})
            llama = st.get("llama")
            daemon = st.get("daemon")
            if llama is not None:
                line.append("  llama ", style="dim")
                line.append("●", style=art.GREEN if llama else art.RED)
            if daemon is not None:
                line.append("  daemon ", style="dim")
                line.append("●", style=art.GREEN if daemon else art.RED)
        else:
            line.append("no model pinned", style="dim")
        line.append("    ", style="dim")
        line.append("1-5 jump · enter launch · esc quit", style="dim")
        return line

    def _render_launchers(self, width: int) -> list[Text]:
        out: list[Text] = []
        for i, app in enumerate(LAUNCH_APPS):
            active = i == self.launch_idx
            chip = (
                f"({self.pinned})"
                if (app.model_binding and self.pinned)
                else ""
            )

            if active:
                # Active row: yellow accent + bold title + chip pinned
                # tight to the title so the row reads as one unit
                # instead of stretching the title and the chip across
                # the whole width with empty space in the middle.
                row = Text(self._GUT, end="")
                row.append("▌ ", style=f"bold {art.YELLOW}")
                row.append(app.title, style=f"bold {art.YELLOW}")
                if chip:
                    row.append("  ", style="dim")
                    row.append(chip, style=f"bold {art.AMBER}")
                out.append(row)
                # Description: tight under the title, aligned to the gutter
                # so the `▌` bar stays the visual anchor.
                out.append(Text(self._GUT + app.description, style="dim"))
            else:
                row = Text(self._GUT, end="")
                row.append(app.title, style=art.MUTED)
                if chip:
                    row.append("  ", style="dim")
                    row.append(chip, style=f"dim {art.AMBER}")
                out.append(row)
        return out

    def _render_models(self, width: int) -> list[Text]:
        out: list[Text] = []
        out.append(Text(self._GUT + "Select model", style=f"bold {art.YELLOW}"))

        for i, spec in enumerate(self.specs):
            st = self._statuses.get(spec.name, {})
            llama = st.get("llama", False)
            daemon = st.get("daemon", False)
            focus = i == self.model_idx
            pinned = spec.name == self.pinned
            # laguna is the fallback stack — it has no chat backends, so
            # the llama/daemon health dots do not apply to it.
            show_dots = spec.name != "laguna"

            row = Text(self._GUT, end="")
            if focus:
                row.append("▌ ", style=f"bold {art.YELLOW}")
                row.append(f"{spec.name:<6}", style=f"bold {art.YELLOW}")
                if pinned:
                    row.append(" <~ pinned", style=f"bold {art.AMBER}")
                if show_dots:
                    row.append("  ", style="dim")
                    row.append("●", style=art.GREEN if llama else art.RED)
                    row.append(" ", style="dim")
                    row.append("●", style=art.GREEN if daemon else art.RED)
            else:
                row.append(f"{spec.name:<6}", style=art.TEXT)
                if pinned:
                    row.append(" <~ pinned", style=f"dim {art.AMBER}")
                if show_dots:
                    row.append("  ", style="dim")
                    row.append("●", style=art.GREEN if llama else art.RED)
                    row.append(" ", style="dim")
                    row.append("●", style=art.GREEN if daemon else art.RED)
            out.append(row)
            if focus:
                detail = (
                    f"      :{spec.llama_port}/:{spec.daemon_port}  "
                    f"ctx={st.get('ctx', spec.ctx_size)}  {spec.label}"
                )
                out.append(Text(detail, style="dim"))
        return out

    def _render_settings(self, width: int) -> list[Text]:
        """Flat list of (model, field, value) rows — the highlighted row
        gets the yellow accent so the user can see what Enter will edit."""
        out: list[Text] = []
        out.append(Text(self._GUT + "Settings", style=f"bold {art.YELLOW}"))
        rows = _settings_rows(self.specs)
        if not rows:
            out.append(Text(self._GUT + "(no models in registry)",
                            style="dim"))
            return out

        last_name: str | None = None
        idx = max(0, min(self.settings_idx, len(rows) - 1))
        for i, (name, field, value) in enumerate(rows):
            # Insert a blank separator + name header when a new model
            # starts — keeps the list readable.
            if name != last_name:
                if last_name is not None:
                    out.append(Text(""))
                out.append(Text(self._GUT + name, style=f"bold {art.YELLOW}"))
                last_name = name
            indent = self._GUT + "    "
            line = Text(indent, end="")
            if i == idx:
                line.append("▌ ", style=f"bold {art.YELLOW}")
                line.append(f"{field:<14}", style=f"bold {art.YELLOW}")
                line.append(value, style="white")
            else:
                line.append(f"{field:<14}", style=art.MUTED)
                line.append(value, style="dim")
            out.append(line)
        return out

    def _render_help(self) -> Text:
        if self._launching:
            return Text(self._GUT + "⟳ starting/selecting model …", style=art.YELLOW)
        if self.focus_pane == "models":
            hint = "↑/↓ choose · enter pin · ← back · esc quit"
        elif self.focus_pane == "settings":
            hint = "↑/↓ field · enter edit · esc back · q quit"
        else:
            hint = "1-5 jump · ↑/↓ navigate · enter launch · s settings · esc quit"
        return Text(self._GUT + hint, style="dim")


def _render_wordmark() -> list[Text]:
    """Big yellow TOKI wordmark at the top of the launcher pane.

    Rendered with pyfiglet's ``standard`` font so the letters look like
    actual letterforms (not solid block approximations). Cached so we
    don't re-rasterise on every key press — figlet rendering is the
    expensive part; the output is a fixed tuple of Text lines.
    """
    return list(_wordmark_lines())


@lru_cache(maxsize=1)
def _wordmark_lines() -> tuple[Text, ...]:
    import pyfiglet

    raw = pyfiglet.figlet_format("TOKI", font="standard").rstrip("\n")
    lines = [ln for ln in raw.split("\n") if ln.strip()]
    return tuple(
        Text(LauncherApp._GUT + line, style=f"bold {art.YELLOW}")
        for line in lines
    )


# ── Settings pane ──────────────────────────────────────────────────────
# Order of fields shown in the launcher settings list — must match the
# dataclass field order the user sees. Anything not editable is rendered
# dim and grayed-out as "(read-only)".
_SETTINGS_FIELDS: tuple[str, ...] = (
    "gguf",
    "llama_port",
    "daemon_port",
    "ctx_size",
    "parallel",
    "kv",
    "max_tokens",
    "model_id",
    "binary",
    "always_on",
    "boot",
    "reasoning",
)


class SettingsEditScreen(ModalScreen[str | None]):
    """Modal that asks for a new value for one editable field.

    Dismisses with the new value on Enter (or the ``Save`` button),
    or ``None`` on Escape / cancel. The launcher commits the value via
    ``update_spec_field`` after dismiss.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    DEFAULT_CSS = """
    SettingsEditScreen {
        align: center middle;
    }
    #settings-dialog {
        width: 70;
        height: auto;
        padding: 1 2;
        background: $boost;
        border: round $primary;
    }
    #settings-dialog Label {
        width: 100%;
        content-align: center middle;
    }
    #settings-input {
        margin-top: 1;
    }
    #settings-buttons {
        width: 100%;
        height: auto;
        margin-top: 1;
        align: right middle;
    }
    #settings-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(self, model_name: str, field: str, current: str) -> None:
        super().__init__()
        self._model_name = model_name
        self._field = field
        self._current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="settings-dialog"):
            yield Label(
                f"Edit  [b]{self._model_name}[/b]  ·  [b]{self._field}[/b]"
            )
            yield Label(f"current: {self._current}", id="settings-current")
            yield Input(value=self._current, id="settings-input")
            with Horizontal(id="settings-buttons"):
                yield Button("Cancel", id="settings-cancel")
                yield Button("Save", id="settings-save", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#settings-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "settings-save":
            self.dismiss(self.query_one("#settings-input", Input).value)
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


_SETTINGS_ROWS_CACHE: dict[int, tuple[tuple[str, str, str], ...]] = {}


def _settings_rows(specs: Any) -> tuple[tuple[str, str, str], ...]:
    """Flatten every editable spec field into (name, field, value) rows.

    Read-only spec fields (``name``, ``source``) are skipped so the
    settings pane only exposes things the user can actually change.

    Cached by the `specs` list's identity (`id(specs)`) so re-renders
    with the same registry state are O(1). The launcher invalidates
    the cache after `update_spec_field` reloads the registry.
    """
    key = id(specs)
    cached = _SETTINGS_ROWS_CACHE.get(key)
    if cached is not None:
        return cached
    from .. import ctl as ctl_mod

    rows: list[tuple[str, str, str]] = []
    for spec in specs:
        for field in _SETTINGS_FIELDS:
            if field not in ctl_mod._EDITABLE_FIELDS:
                continue
            value = getattr(spec, field, None)
            rows.append((spec.name, field, "" if value is None else str(value)))
    out = tuple(rows)
    _SETTINGS_ROWS_CACHE[key] = out
    return out


def invalidate_settings_rows() -> None:
    """Drop the cached settings rows after a registry mutation."""
    _SETTINGS_ROWS_CACHE.clear()


def run():
    """Shim entry used by ``toki`` when bare in a TTY."""
    from ..ctl import use_model
    from ..selector import _launch_detached, _launch_foreground, launch_cmd

    def on_launch(interface: str, client: str, model: str) -> int:
        try:
            spec = use_model(model)
        except (KeyError, FileNotFoundError, RuntimeError) as e:
            from rich.console import Console

            Console().print(f"[red]error: {e}[/red]")
            return 1
        plan = launch_cmd(interface, client, spec)
        argv = plan["argv"]
        if client == "openclaw" and argv[:2] == ["openclaw", "tui"]:
            from ..selector import _gateway_up

            if not _gateway_up():
                print("OpenClaw gateway is not running; falling back to local TUI.")
                argv = ["openclaw", "chat"]
        if plan["detach"]:
            return _launch_detached(interface, client, argv, plan["env"], plan["url"])
        return _launch_foreground(argv, plan["env"])

    from rich.console import Console

    if not __import__("sys").stdout.isatty():
        Console().print("error: toki requires an interactive terminal (TTY).", style="red")
        return 2
    app = LauncherApp(on_launch=on_launch)
    return app.run() or 0
