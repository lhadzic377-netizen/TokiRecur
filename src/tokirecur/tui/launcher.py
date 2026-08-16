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
from textual.widgets import Button, Input, Label, Select, Static

from ..ctl import load_registry, model_status, read_client_modes, write_client_mode
from . import art

# Fields whose change requires a llama-server restart (the spec's
# launch window changes; the running server can't honour the new
# value). Mirrors ctl._RESTART_REQUIRED_FIELDS — kept here so the
# launcher doesn't have to import a private symbol.
_RESTART_REQUIRED_FIELDS = frozenset({
    "ctx_size", "kv", "parallel", "binary", "gguf",
    "llama_port", "daemon_port",
})

# Fields whose change replaces the running model with a *different*
# one (not just a tuning). Triggers a harder-stop confirmation.
_MODEL_SWAP_FIELDS = frozenset({"gguf", "binary"})

# Fields whose change only affects CLI client configs (opencode.json
# advertises contextWindow/limit.output). No daemon restart needed.
_SYNC_ONLY_FIELDS = frozenset({"max_tokens", "model_id", "reasoning"})


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
        "codex",
        "Launch Codex",
        "OpenAI CLI coding agent (Responses API)",
        "chat", "codex",
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
        Binding("m", "toggle_mode", "Mode", priority=True),
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
        self._client_modes: dict[str, str] = {}

    # ── Compose ───────────────────────────────────────────────────────

    def on_mount(self) -> None:
        self.specs = load_registry()
        if not self.specs:
            self._error = "no models in registry — run `toki model list`"
        self._refresh_pinned()
        self._client_modes = read_client_modes()
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
        # Live values change when llama-server boots/restarts; the
        # settings row cache must reflect that, otherwise we'd render
        # stale "live: 8192" forever.
        invalidate_settings_rows()
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
            rows = _settings_rows(self.specs, app=self)
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
        rows = _settings_rows(self.specs, app=self)
        if not rows:
            return None
        idx = max(0, min(self.settings_idx, len(rows) - 1))
        return rows[idx]

    def action_edit_setting(self) -> None:
        """Enter in settings: open the edit modal (dropdown or text) for
        the highlighted field."""
        if self.focus_pane != "settings":
            return
        current = self._current_setting()
        if current is None:
            return
        name, field, value = current
        # Look up the spec so the dropdown can be spec-aware (e.g.
        # ctx_size filtered against the model's max + RAM clamp).
        spec = next((s for s in self.specs if s.name == name), None)
        choices = (
            _choices_for_field(spec, field)
            if spec is not None
            else _SETTINGS_CHOICES.get(field)
        )
        # For the text-editor path, always use the registry value —
        # never the "reg:X · live:Y" display string. The user types a
        # value, not a sentence; feeding the display string back makes
        # update_spec_field's int() parser reject it as invalid.
        edit_value = (
            str(getattr(spec, field, ""))
            if spec is not None
            else value
        )
        if choices is not None:
            screen = SettingsChoiceScreen(name, field, edit_value, choices)
            self.push_screen(screen, self._on_choice_dismissed)
        else:
            screen = SettingsEditScreen(name, field, edit_value)
            self.push_screen(screen, self._on_edit_dismissed)

    def _on_choice_dismissed(self, result: str | None) -> None:
        """Commit a dropdown pick, or chain into the text modal for the
        ``custom…`` option."""
        if result is None:
            self.notify("edit cancelled", timeout=2)
            return
        if result == _CUSTOM_VALUE:
            current = self._current_setting()
            if current is None:
                return
            name, field, value = current
            spec = next((s for s in self.specs if s.name == name), None)
            edit_value = (
                str(getattr(spec, field, ""))
                if spec is not None
                else value
            )
            self.push_screen(
                SettingsEditScreen(name, field, edit_value), self._on_edit_dismissed
            )
            return
        # result is a real choice value (not the custom sentinel).
        # Skip the choice-modal and commit directly.
        self._on_edit_dismissed(result)
        return

    def _on_edit_dismissed(self, new_value: str | None) -> None:
        """Commit the edit (or no-op on cancel). Called by the modal.

        Branches on the field class to do the right downstream work:
          * Restart-required (ctx_size, kv, parallel, gguf, binary,
            llama_port, daemon_port) → confirmation modal, then
            update_spec_field(apply=True) which restarts the daemon.
          * Sync-only (max_tokens, model_id, reasoning) → write the
            registry and re-sync opencode/openclaw/hermes/codex
            configs without touching the running daemon.
          * No-op (always_on, boot) → registry write only.

        The legacy behaviour ("registry only, no daemon work") used to
        be the only path and is what made the panel lie: the panel
        showed the new value but opencode.json still advertised the
        old contextWindow, and llama-server still ran with the old
        -c flag. See `update_spec_field(apply=...)` for the contract.
        """
        if new_value is None:
            self.notify("edit cancelled", timeout=2)
            return
        current = self._current_setting()
        if current is None:
            return
        name, field, _ = current
        if field in _RESTART_REQUIRED_FIELDS:
            # Modal: ask first. Use a different label for the model-swap
            # fields (gguf / binary) — those change the model entirely,
            # not just its tuning.
            if field in _MODEL_SWAP_FIELDS:
                title = f"{name}.{field} → {new_value}"
                body = (
                    f"This replaces the running model with a different one.\n\n"
                    f"Current model: {name}\n"
                    f"New {field}: {new_value}\n\n"
                    f"llama-server will be stopped and restarted."
                )
            else:
                title = f"{name}.{field} = {new_value}"
                body = (
                    "llama-server will be stopped and restarted with the new setting.\n"
                    "Approximate downtime: a few seconds.\n\n"
                    "Proceed?"
                )
            self.push_screen(
                ConfirmRestartScreen(title, body),
                lambda ok: self._commit_edit(name, field, new_value, apply=bool(ok)),
            )
            return
        # Sync-only and no-op fields go straight to the commit step.
        self._commit_edit(name, field, new_value, apply=field in _SYNC_ONLY_FIELDS)

    def _commit_edit(self, name: str, field: str, new_value: str, apply: bool) -> None:
        """Write the registry, optionally apply the downstream effect,
        and surface what happened to the user."""
        from .. import ctl as ctl_mod
        try:
            ctl_mod.update_spec_field(name, field, new_value, apply=apply)
        except (ValueError, KeyError) as e:
            self.notify(str(e), timeout=4, severity="error")
            return
        except (RuntimeError, OSError) as e:
            # Restart required but failed (binary not built, RAM too
            # low, etc.). Registry still holds the new value; tell the
            # user the restart failed and the next launch will use it.
            self.specs = ctl_mod.load_registry()
            invalidate_settings_rows()
            self._refresh += 1
            self.notify(
                f"{name}.{field} = {new_value}  (saved, but restart failed: {e})",
                timeout=6, severity="warning",
            )
            return
        # Reload specs so the row shows the new value.
        self.specs = ctl_mod.load_registry()
        invalidate_settings_rows()
        if field in _RESTART_REQUIRED_FIELDS:
            self.notify(
                f"{name}.{field} = {new_value}  ·  restarted, configs synced",
                timeout=3,
            )
        elif field in _SYNC_ONLY_FIELDS:
            self.notify(
                f"{name}.{field} = {new_value}  ·  configs synced",
                timeout=3,
            )
        else:
            self.notify(
                f"{name}.{field} = {new_value}", timeout=3,
            )
        self._refresh += 1

    def action_toggle_mode(self) -> None:
        """m: cycle the highlighted client's inference mode raw <-> managed.

        codex is pinned to managed: its Responses API shim only exists on
        the daemon, so a raw provider would fail every request.
        """
        if self.focus_pane != "launchers":
            self.action_back()
            return
        app = LAUNCH_APPS[self.launch_idx]
        if not app.model_binding or app.client == "codex":
            self.notify(
                f"{app.title}: always uses the daemon (managed)", timeout=3
            )
            return
        cur = self._client_modes.get(app.client, "managed")
        nxt = "raw" if cur == "managed" else "managed"
        self._client_modes = write_client_mode(app.client, nxt)
        if self.pinned:
            try:
                from ..ctl import get_model, sync_cli_configs
                sync_cli_configs(get_model(self.pinned))
                self.notify(f"{app.client}: {cur} -> {nxt} (configs synced)",
                            timeout=3)
            except (KeyError, RuntimeError, OSError):
                self.notify(f"{app.client}: {cur} -> {nxt} (re-sync on "
                            f"next start)", timeout=3)
        else:
            self.notify(f"{app.client}: {cur} -> {nxt}", timeout=3)
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
        line.append("1-5 jump · m mode · enter launch · esc quit", style="dim")
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
            mode_tag = ""
            if app.model_binding and app.client in ("opencode", "openclaw",
                                                    "hermes", "codex"):
                mode = self._client_modes.get(app.client, "managed")
                if app.client == "codex":
                    mode_tag = " · daemon"
                else:
                    mode_tag = f" · {mode}"

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
                out.append(
                    Text(self._GUT + app.description + mode_tag, style="dim")
                )
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
        rows = _settings_rows(self.specs, app=self)
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
            dropdown = field in _SETTINGS_CHOICES
            if i == idx:
                line.append("▌ ", style=f"bold {art.YELLOW}")
                line.append(f"{field:<14}", style=f"bold {art.YELLOW}")
                line.append(value, style="white")
            else:
                line.append(f"{field:<14}", style=art.MUTED)
                line.append(value, style="dim")
            if dropdown:
                line.append("  ▾", style="dim")
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


# Sentinel carried by the ``custom…`` dropdown option; the launcher
# chains into the free-text editor when a user picks it.
_CUSTOM_VALUE = "__custom__"

# Fields that edit through a dropdown of presets instead of raw text.
# Maps field -> {option label: value}. The ``custom…`` option opens the
# text editor. Fields absent here keep the plain text modal.
_SETTINGS_CHOICES: dict[str, dict[str, str]] = {
    "ctx_size": {
        "2048": "2048",
        "4096": "4096",
        "8192": "8192",
        "16384": "16384",
        "32768": "32768",
        "65536": "65536",
        "131072": "131072",
        "custom…": _CUSTOM_VALUE,
    },
    "max_tokens": {
        "256": "256",
        "512": "512",
        "1024": "1024",
        "2048": "2048",
        "4096": "4096",
        "8192": "8192",
        "custom…": _CUSTOM_VALUE,
    },
    "parallel": {
        "1": "1",
        "2": "2",
        "4": "4",
        "8": "8",
        "custom…": _CUSTOM_VALUE,
    },
    "kv": {
        "f16": "f16",
        "f8": "f8",
        "q8_0": "q8_0",
        "q4_0": "q4_0",
        "i8": "i8",
        "i4": "i4",
        "custom…": _CUSTOM_VALUE,
    },
    "reasoning": {
        "auto": "auto",
        "on": "on",
        "off": "off",
    },
    "always_on": {
        "true": "true",
        "false": "false",
    },
    "boot": {
        "true": "true",
        "false": "false",
    },
}


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


class ConfirmRestartScreen(ModalScreen[bool]):
    """Two-button confirmation modal: are you sure you want to restart
    llama-server? Dismisses with True on confirm, False on cancel.

    Used before any settings edit that requires restarting the daemon
    (ctx_size, kv, parallel, gguf, binary, llama_port, daemon_port).
    The model-swap fields (gguf, binary) use a different body copy.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    DEFAULT_CSS = """
    ConfirmRestartScreen {
        align: center middle;
    }
    #confirm-dialog {
        width: 70;
        height: auto;
        padding: 1 2;
        background: $boost;
        border: round $warning;
    }
    #confirm-dialog Label {
        width: 100%;
        content-align: center middle;
    }
    #confirm-title {
        margin-bottom: 1;
    }
    #confirm-body {
        text-align: center;
    }
    #confirm-buttons {
        width: 100%;
        height: auto;
        margin-top: 1;
        align: right middle;
    }
    #confirm-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(self, title: str, body: str) -> None:
        super().__init__()
        self._title = title
        self._body = body

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Label(self._title, id="confirm-title")
            yield Label(self._body, id="confirm-body")
            with Horizontal(id="confirm-buttons"):
                yield Button("Cancel", id="confirm-cancel")
                yield Button("Proceed", id="confirm-ok", variant="warning")

    def on_mount(self) -> None:
        self.query_one("#confirm-ok", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm-ok")

    def action_cancel(self) -> None:
        self.dismiss(False)


class SettingsChoiceScreen(ModalScreen[str | None]):
    """Modal that asks for a new value from preset choices.

    Dismisses with the selected value on Save (or the ``custom…``
    sentinel when the user wants to type their own value), or ``None``
    on Cancel / Escape. The launcher commits the value via
    ``update_spec_field`` after dismiss.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    DEFAULT_CSS = """
    SettingsChoiceScreen {
        align: center middle;
    }
    #settings-choice-dialog {
        width: 70;
        height: auto;
        padding: 1 2;
        background: $boost;
        border: round $primary;
    }
    #settings-choice-dialog Label {
        width: 100%;
        content-align: center middle;
    }
    #settings-choice-select {
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

    def __init__(
        self,
        model_name: str,
        field: str,
        current: str,
        choices: dict[str, str],
    ) -> None:
        super().__init__()
        self._model_name = model_name
        self._field = field
        self._current = current
        self._choices = choices

    def compose(self) -> ComposeResult:
        options = list(self._choices.items())
        # Preselect the current value; if it is not one of the presets
        # (case-insensitive), surface it as its own option so the user
        # can keep it without touching anything.
        selected = self._current
        for _label, value in self._choices.items():
            if value.lower() == self._current.lower():
                selected = value
                break
        else:
            options.insert(0, (self._current, self._current))
        with Vertical(id="settings-choice-dialog"):
            yield Label(
                f"Edit  [b]{self._model_name}[/b]  ·  [b]{self._field}[/b]"
            )
            yield Label(f"current: {self._current}", id="settings-choice-current")
            yield Select(
                options,
                value=selected,
                id="settings-choice-select",
                allow_blank=False,
            )
            with Horizontal(id="settings-buttons"):
                yield Button("Cancel", id="settings-cancel")
                yield Button("Save", id="settings-save", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#settings-choice-select", Select).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "settings-save":
            selected = self.query_one("#settings-choice-select", Select).value
            self.dismiss(None if selected is Select.BLANK else str(selected))
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


_SETTINGS_ROWS_CACHE: dict[int, tuple[tuple[str, str, str], ...]] = {}


def _live_value(app: LauncherApp, spec: Any, field: str) -> Any:
    """Effective value for the field — registry value is the plan, this
    is what's actually running. Falls back to the registry on first
    boot or when the model isn't loaded.

    Uses the status poll that already runs every 3 seconds (slot_ctx
    is captured there).
    """
    if field in ("ctx_size", "max_tokens"):
        st = app._statuses.get(spec.name, {}) if app is not None else {}
        slot = st.get("ctx")
        if slot:
            return slot
    return getattr(spec, field, None)


def _choices_for_field(spec: Any, field: str) -> dict[str, str] | None:
    """Return the dropdown choices for ``field`` on ``spec``.

    Static fields (parallel, kv, reasoning, always_on, boot) get the
    same choices for every model. Dynamic fields (ctx_size) get a
    spec-aware list filtered against the GGUF's max context and the
    RAM clamp from ``_fit_plan`` — so the user can never pick a value
    that would be rejected by ``_check_ram`` at launch.

    The ``custom…`` option is always appended; power users can still
    type any value through the text editor (the text-modal path then
    reuses ``update_spec_field`` which still validates).
    """
    base = _SETTINGS_CHOICES.get(field)
    if base is None or field != "ctx_size":
        return base
    # Spec-aware ctx_size: filter the preset list against the model's
    # GGUF max context (e.g. a 9B at 40960 max shouldn't offer 131072).
    from .. import ctl as ctl_mod
    cap = spec.ctx_size
    model_max = ctl_mod._model_max_ctx(spec)
    if model_max:
        cap = min(cap, model_max)
    # The RAM clamp is the smaller of the two. We approximate it with
    # the live `_fit_plan` value (what llama-server would actually
    # boot with on this box). If we can't read RAM, fall back to the
    # registry cap so the user at least sees what's configured.
    try:
        plan_ctx = ctl_mod._fit_plan(spec)[0]
        if plan_ctx:
            cap = min(cap, plan_ctx)
    except (RuntimeError, OSError, ValueError):
        pass
    filtered: dict[str, str] = {}
    for label, value in base.items():
        if value == _CUSTOM_VALUE:
            continue
        try:
            if int(value) <= cap:
                filtered[label] = value
        except (ValueError, TypeError):
            filtered[label] = value
    filtered["custom…"] = _CUSTOM_VALUE
    return filtered


def _settings_rows(
    specs: Any,
    app: LauncherApp | None = None,
) -> tuple[tuple[str, str, str], ...]:
    """Flatten every editable spec field into (name, field, value) rows.

    Read-only spec fields (``name``, ``source``) are skipped so the
    settings pane only exposes things the user can actually change.

    ``value`` is the LIVE value when one is available (slot context,
    running kv precision). The render step compares it to the
    registry value and surfaces a two-column layout when they differ,
    so the user can see "what's planned" vs "what's running".

    Cached by the `specs` list's identity (`id(specs)`) so re-renders
    with the same registry state are O(1). The launcher invalidates
    the cache after `update_spec_field` reloads the registry. The
    cache also drops on every status poll refresh (live values change
    when llama-server boots or restarts).
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
            reg_value = getattr(spec, field, None)
            live_value = _live_value(app, spec, field)
            # When the live value agrees with the registry, render the
            # plain value. When they differ, render "registry · live"
            # so the user can see what's actually running.
            mismatched = (
                live_value is not None
                and reg_value is not None
                and str(live_value) != str(reg_value)
            )
            if mismatched:
                value = f"reg:{reg_value} · live:{live_value}"
            else:
                value = "" if reg_value is None else str(reg_value)
            rows.append((spec.name, field, value))
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
