"""Interactive launcher selector for the toki stack.

Bare ``toki`` opens a three-level menu: interface (chat / tui / web) ->
client (toki / opencode / openclaw / hermes) -> model (from the
registry). The chosen model stack is started via :func:`use_model`
(which also points opencode / openclaw / hermes at it), and the client
is launched with the model pre-selected.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .ctl import DEFAULT_AUTH, model_status, read_active

console = Console()

INTERFACES = ["chat", "tui", "web"]
CLIENTS = ["toki", "opencode", "openclaw", "hermes", "codex"]


def _auth_token() -> str | None:
    """Daemon auth token, mirroring ``api.app._get_auth_token``.

    Read order: TOKI_AUTH_TOKEN env var → ~/.config/tokirecur/auth →
    /etc/toki-recur/auth. ``None`` when no token is configured (daemon
    runs open on loopback).
    """
    tok = os.environ.get("TOKI_AUTH_TOKEN")
    if tok:
        return tok
    for path in (
        Path.home() / ".config" / "tokirecur" / "auth",
        Path("/etc/toki-recur/auth"),
    ):
        try:
            if path.is_file():
                content = path.read_text().strip()
                if content:
                    return content
        except OSError:
            pass
    return None

INTERFACE_HINTS = {
    "chat": "open the client's interactive UI / TUI",
    "tui": "full-screen terminal interface",
    "web": "browser-based interface",
}
CLIENT_HINTS = {
    "toki": "toki's own client",
    "opencode": "opencode",
    "openclaw": "openclaw",
    "hermes": "hermes",
    "codex": "codex (OpenAI CLI)",
}

WEB_HINTS = {
    "toki": "http://127.0.0.1:7777  (toki web UI)",
    "openclaw": "http://127.0.0.1:18789  (OpenClaw Control)",
    "hermes": "http://127.0.0.1:9119  (hermes dashboard)",
    "codex": "codex app-server (experimental)",
}


def launch_cmd(interface: str, client: str, spec: Any) -> dict:
    """Build the launch plan for an (interface, client, model) combo.

    Returns a dict with keys: ``argv`` (command + args), ``detach``
    (bool: run in background), ``url`` (browser URL to open, or None),
    ``env`` (extra environment). The chosen model is passed explicitly
    where the client supports it; otherwise the configs written by
    ``toki model use`` select it.
    """
    llama_url = spec.llama_url
    daemon_url = spec.daemon_url
    provider = f"tokirecur-{spec.name}"
    model_ref = f"{provider}/{spec.model_id}"
    env = dict(os.environ)
    if os.environ.get("TOKI_AUTH_TOKEN"):
        env["TOKI_AUTH_TOKEN"] = os.environ["TOKI_AUTH_TOKEN"]
    if client == "codex":
        # Codex resolves the provider key from a real environment
        # variable (`env` in the provider config only names it; there is
        # no inline-key form). Inject the daemon's token so codex works
        # without the user exporting anything — and when no token is
        # configured, inject DEFAULT_AUTH, the same value the other
        # three clients already embed (the daemon accepts anything on
        # loopback when no token is configured).
        tok = _auth_token() or DEFAULT_AUTH
        env["TOKI_AUTH_TOKEN"] = tok

    if interface == "chat":
        if client == "toki":
            # The launcher's Chat entry opens the production TUI chat
            # client (`toki tui`).
            argv = ["toki", "tui", "--raw-url", llama_url, "--daemon-url", daemon_url]
            if env.get("TOKI_AUTH_TOKEN"):
                argv += ["--auth-token", env["TOKI_AUTH_TOKEN"]]
            return {"argv": argv, "detach": False, "url": None, "env": env}
        if client == "opencode":
            return {"argv": ["opencode", "-m", model_ref],
                    "detach": False, "url": None, "env": env}
        if client == "openclaw":
            return {"argv": ["openclaw", "tui"], "detach": False, "url": None,
                    "env": env}
        if client == "hermes":
            # hermes recognizes "llamacpp" as a built-in alias for the
            # custom local-server provider; the actual base_url lives
            # in ~/.hermes/config.yaml's custom_providers list, written
            # by ctl.write_hermes_config on every `toki model use`.
            return {"argv": ["hermes", "chat", "--tui", "--provider", "llamacpp",
                             "-m", spec.model_id],
                    "detach": False, "url": None, "env": env}
        if client == "codex":
            # Bare `codex` is the interactive TUI; model/provider come
            # from ~/.codex/config.toml (written by `toki model use`).
            return {"argv": ["codex"], "detach": False, "url": None, "env": env}
    elif interface == "tui":
        if client == "toki":
            argv = ["toki", "tui", "--raw-url", llama_url, "--daemon-url", daemon_url]
            if env.get("TOKI_AUTH_TOKEN"):
                argv += ["--auth-token", env["TOKI_AUTH_TOKEN"]]
            return {"argv": argv, "detach": False, "url": None, "env": env}
        if client == "opencode":
            return {"argv": ["opencode", "-m", model_ref], "detach": False,
                    "url": None, "env": env}
        if client == "openclaw":
            return {"argv": ["openclaw", "chat"], "detach": False, "url": None,
                    "env": env}
        if client == "hermes":
            return {"argv": ["hermes", "--tui", "chat", "--provider", "llamacpp",
                             "-m", spec.model_id],
                    "detach": False, "url": None, "env": env}
        if client == "codex":
            return {"argv": ["codex"], "detach": False, "url": None, "env": env}
    elif interface == "web":
        if client == "toki":
            return {"argv": ["toki", "web", "--detach",
                             "--raw-url", llama_url, "--daemon-url", daemon_url],
                    "detach": True, "url": "http://127.0.0.1:7777",
                    "env": env}
        if client == "openclaw":
            return {"argv": [], "detach": True, "url": "http://127.0.0.1:18789",
                    "env": env}
        if client == "hermes":
            return {"argv": ["hermes", "dashboard", "--port", "9119"],
                    "detach": True, "url": "http://127.0.0.1:9119",
                    "env": env}
        if client == "codex":
            # Experimental app server (no web URL yet).
            return {"argv": ["codex", "app-server"], "detach": True,
                    "url": None, "env": env}
    raise ValueError(f"unsupported combo: {interface} + {client}")


def _pick(title: str, options: list[tuple[str, str]], default: str | None = None) -> str:
    """Numbered single-choice menu; returns the chosen label."""
    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_column(width=3), table.add_column(style="bold"), table.add_column(style="dim")
    for i, (label, hint) in enumerate(options, 1):
        suffix = "  [green]<- active[/green]" if label == default else ""
        table.add_row(f"{i})", label, hint + suffix)
    console.print(Panel(table, title=title, border_style="#b8860b", expand=False))
    while True:
        raw = Prompt.ask("choose", default=str(default_index(options, default)) if default else "")
        if raw in ("q", "quit", "exit"):
            raise KeyboardInterrupt
        try:
            idx = int(raw)
        except ValueError:
            console.print("[red]enter a number, or q to quit[/red]")
            continue
        if 1 <= idx <= len(options):
            return options[idx - 1][0]
        console.print(f"[red]enter 1..{len(options)}[/red]")


def default_index(options: list[tuple[str, str]], default: str | None) -> int:
    for i, (label, _) in enumerate(options, 1):
        if label == default:
            return i
    return 1


def menu_lines(specs: list) -> list[tuple[str, str]]:
    """Build the model menu rows from registry specs + live status."""
    active = read_active()
    rows: list[tuple[str, str]] = []
    for spec in specs:
        st = model_status(spec)
        llama = "up" if st["llama"]["reachable"] else "down"
        daemon = "up" if st["daemon"]["reachable"] else "down"
        mark = "*" if spec.name == active else " "
        ctx = st["llama"].get("slot_ctx") or spec.ctx_size
        rows.append((spec.name,
                     f"{mark} {spec.label}  llama={llama} daemon={daemon} "
                     f"ctx={ctx}  :{spec.llama_port}/:{spec.daemon_port}"))
    return rows


def _open_browser(url: str) -> None:
    console.print(f"[green]opening {url}[/green]")
    try:
        if os.name == "nt":
            os.startfile(url)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", url], start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(["xdg-open", url], start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, OSError):
        console.print(f"[yellow]open manually: {url}[/yellow]")


def _gateway_up() -> bool:
    try:
        import httpx
        return httpx.get("http://127.0.0.1:18789/", timeout=2.0).status_code == 200
    except Exception:
        return False


def run() -> int:
    """Open the launcher and run the chosen client.

    On Linux/macOS this is the Ollama-style Textual launcher. The
    classic Windows console renders Textual TUIs slowly and glitches,
    so on Windows bare ``toki`` opens the plain CLI launcher menu
    (``clilauncher``) instead — same apps, numbered pick, with Chat
    running the CLI chat; Linux and macOS are unchanged.
    """
    if os.name == "nt":
        from .clilauncher import run_cli_launcher

        return run_cli_launcher()

    from .tui.launcher import run as launcher_run
    return launcher_run()


def _launch_foreground(argv: list[str], env: dict) -> int:
    console.print(Panel(" ".join(shlex.quote(a) for a in argv),
                        title="launching", border_style="green", expand=False))
    try:
        os.execvpe(argv[0], argv, env)
    except FileNotFoundError:
        console.print(f"[red]client not found: {argv[0]}[/red]")
        return 1
    return 1


def _launch_detached(interface: str, client: str, argv: list[str],
                     env: dict, url: str | None) -> int:
    if interface == "web" and client == "openclaw":
        if _gateway_up():
            _open_browser(url or "http://127.0.0.1:18789")
        else:
            console.print("[yellow]OpenClaw gateway is not running.[/yellow]")
            console.print("  start it with:  systemctl --user start openclaw-gateway.service")
            console.print("  then re-run:    toki")
        return 0
    console.print(Panel(" ".join(shlex.quote(a) for a in argv),
                        title="launching in background", border_style="green", expand=False))
    log_dir = Path.home() / ".local/state/tokirecur"
    log_path = str(log_dir / f"{client}-{interface}.log")
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    console.print(f"[dim]pid={proc.pid}  log: {log_path}[/dim]")
    if url:
        import time
        time.sleep(2.0)
        _open_browser(url)
    return 0
