"""Interactive launcher selector for the toki stack.

Bare ``toki`` opens a three-level menu: interface (chat / tui / web) ->
client (the shared launch catalog) -> model (from the registry).
Model-bound clients use the selected stack; launch-only clients use
their own provider configuration.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .ctl import (
    _effective_ctx,
    _resolve_mode,
    auth_token,
    client_mode,
    cline_config_dir,
    load_registry,
    model_status,
    read_active,
    write_claude_config,
    write_cline_config,
    write_codex_config,
    write_droid_config,
    write_dsh_config,
    write_grok_config,
    write_hermes_config,
    write_kilo_config,
    write_kimi_config,
    write_mimo_config,
    write_omp_config,
    write_openclaw_config,
    write_opencode_config,
    write_ornithcode_config,
    write_pi_config,
    write_qwen_config,
)
from .launch_catalog import get_launch_app
from .tui._cleanup import restore_now

console = Console()
_CLINE_AI_SDK_PRELOAD = Path(__file__).with_name("tui") / "cline-ai-sdk-preload.js"

INTERFACES = ["chat", "tui", "web"]
CLIENTS = ["toki", "opencode", "mimo", "openclaw", "hermes", "codex"]


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
    "mimo": "mimo code",
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


def _bound_endpoint(spec: Any, client: str) -> str:
    base = spec.llama_url if client_mode(client) == "raw" else spec.daemon_url
    return f"{base}/v1"


def _client_limits(spec: Any) -> tuple[int, int]:
    try:
        context = max(1, int(_effective_ctx(spec)))
    except (AttributeError, IndexError, KeyError, OSError, TypeError, ValueError):
        context = max(1, int(getattr(spec, "ctx_size", 8192)))
    configured = getattr(spec, "max_output_tokens", None)
    output = max(1, min(int(configured or 8192), max(1, context // 2)))
    return context, output


# Clients that own a config file the daemon has to be told about. They do not
# read OPENAI_BASE_URL from the environment - each one has its own settings
# format - so the only way they learn which endpoint and model to use is by
# that file being written. See _refresh_client_config.
_CLIENT_CONFIG_WRITERS = {
    "opencode": write_opencode_config,
    "mimo": write_mimo_config,
    "openclaw": write_openclaw_config,
    "hermes": write_hermes_config,
    "codex": write_codex_config,
    "qwen": write_qwen_config,
    "droid": write_droid_config,
    "cline": write_cline_config,
    "ornithcode": write_ornithcode_config,
    "dsh": write_dsh_config,
    "grok": write_grok_config,
    "kilo": write_kilo_config,
    "pi": write_pi_config,
    "kimi": write_kimi_config,
    "omp": write_omp_config,
}


def _refresh_client_config(spec: Any, client: str) -> None:
    """Rewrite a client's config to match the stack it is about to use.

    dsh was the visible symptom: its settings.yaml was pinned to a model and
    a port that no longer existed, so it reported a bare
    ``TRANSPORT: Connection error`` and its TUI profile sat there rendering
    nothing. Nothing about dsh was broken - the file was stale, because it was
    only written by ``toki model use``. Switch the active model through any
    other path, add or remove a model, and every client whose config was not
    rewritten silently points at a dead endpoint.

    Launching is the last moment where the answer is known and cannot go stale
    afterwards, so that is where it is written. Failures are swallowed on
    purpose: a client that will not pick up the new endpoint is far better than
    a launcher that refuses to start because a config file is read-only.
    """
    writer = _CLIENT_CONFIG_WRITERS.get(client)
    if writer is None:
        return
    try:
        writer(spec, load_registry(), mode=_resolve_mode(client, None))
    except Exception as e:  # defensive: never block a launch on a config write
        console.print(
            f"[yellow]could not refresh {client} config: {e}[/yellow]  "
            f"it may still be pointing at the previous model"
        )


def _bound_env(spec: Any, client: str) -> dict[str, str]:
    env = dict(os.environ)
    endpoint = _bound_endpoint(spec, client)
    new_clients = {
        "qwen", "droid", "cline", "dsh", "grok", "kilo", "pi",
        "copilot", "kimi", "omp", "ornithcode",
    }
    if client == "codex" or client in new_clients:
        token = _auth_token() or auth_token()
        env["TOKI_AUTH_TOKEN"] = token
    else:
        token = env.get("TOKI_AUTH_TOKEN", "")
    if client in {
        "qwen", "cline", "droid", "dsh", "grok", "kilo", "pi", "kimi", "omp",
        "ornithcode",
    }:
        env["OPENAI_API_KEY"] = token
        env["OPENAI_BASE_URL"] = endpoint
    if client == "grok":
        env["GROK_DEFAULT_MODEL"] = f"tokirecur-{spec.name}"
        env["GROK_MODELS_BASE_URL"] = endpoint
    if client == "copilot":
        context, output = _client_limits(spec)
        env["COPILOT_PROVIDER_BASE_URL"] = endpoint
        env["COPILOT_PROVIDER_TYPE"] = "openai"
        env["COPILOT_PROVIDER_API_KEY"] = token
        env["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"] = str(context)
        env["COPILOT_PROVIDER_MAX_OUTPUT_TOKENS"] = str(output)
        env["COPILOT_MODEL"] = spec.model_id
    if client == "cline":
        config_dir = cline_config_dir()
        env["CLINE_DATA_DIR"] = str(config_dir / "data")
        preload = f"--preload={shlex.quote(str(_CLINE_AI_SDK_PRELOAD))}"
        existing = env.get("BUN_OPTIONS", "").strip()
        env["BUN_OPTIONS"] = f"{existing} {preload}".strip()
    return env


def launch_cmd(interface: str, client: str, spec: Any | None) -> dict:
    """Build the launch plan for an (interface, client, model) combo."""
    launch = get_launch_app(client)
    if launch is not None and launch.interface == "chat" and not launch.model_binding:
        return {
            "argv": [launch.command or launch.client, *launch.args],
            "detach": False,
            "url": None,
            "env": dict(os.environ),
        }
    if spec is None:
        raise ValueError(f"client {client!r} requires an active model")
    if client == "claude":
        settings_path = write_claude_config(spec)
        token = _auth_token() or auth_token()
        env = dict(os.environ)
        env.pop("ANTHROPIC_AUTH_TOKEN", None)
        env.update({
            "ANTHROPIC_BASE_URL": f"{spec.daemon_url}/anthropic",
            "ANTHROPIC_API_KEY": token,
            "ANTHROPIC_MODEL": spec.model_id,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": spec.model_id,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": spec.model_id,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": spec.model_id,
        })
        argv = ["claude", "--settings", str(settings_path), "--model", spec.model_id]
        # Opt-in levers. A full setup (48 plugins, MCP servers, hooks, LSP)
        # makes every prompt tens of thousands of tokens, and extended
        # thinking is punishingly slow on a local 35B, so both are optional.
        if os.environ.get("TOKI_CLAUDE_BARE") == "1":
            argv.append("--bare")
        if os.environ.get("TOKI_CLAUDE_MCP") == "none":
            argv += ["--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}']
        effort = os.environ.get("TOKI_CLAUDE_EFFORT", "").strip()
        if effort:
            argv += ["--effort", effort]
        return {
            "argv": argv,
            "detach": False,
            "url": None,
            "env": env,
        }
    llama_url = spec.llama_url
    daemon_url = spec.daemon_url
    provider = f"tokirecur-{spec.name}"
    model_ref = f"{provider}/{spec.model_id}"
    _refresh_client_config(spec, client)
    env = _bound_env(spec, client)
    endpoint = _bound_endpoint(spec, client)

    if interface in {"chat", "tui"}:
        if client == "toki":
            argv = ["toki", "tui", "--raw-url", llama_url, "--daemon-url", daemon_url]
            return {"argv": argv, "detach": False, "url": None, "env": env}
        if client == "ornithcode":
            return {"argv": ["ornithcode"], "detach": False, "url": None, "env": env}
        if client == "opencode":
            return {"argv": ["opencode", "-m", model_ref], "detach": False,
                    "url": None, "env": env}
        if client == "mimo":
            return {"argv": ["mimo", "-m", model_ref], "detach": False,
                    "url": None, "env": env}
        if client == "openclaw":
            command = "tui" if interface == "chat" else "chat"
            return {"argv": ["openclaw", command], "detach": False,
                    "url": None, "env": env}
        if client == "hermes":
            if interface == "chat":
                argv = ["hermes", "chat", "--tui", "--provider", "llamacpp",
                        "-m", spec.model_id]
            else:
                argv = ["hermes", "--tui", "chat", "--provider", "llamacpp",
                        "-m", spec.model_id]
            return {"argv": argv, "detach": False, "url": None, "env": env}
        if client == "codex":
            return {"argv": ["codex"], "detach": False, "url": None, "env": env}
        if client == "qwen":
            return {"argv": ["qwen", "--model", spec.model_id,
                             "--openai-base-url", endpoint,
                             "--auth-type", "openai"],
                    "detach": False, "url": None, "env": env}
        if client == "droid":
            return {"argv": ["droid"], "detach": False, "url": None, "env": env}
        if client == "cline":
            config_dir = cline_config_dir()
            return {"argv": ["cline", "--config", str(config_dir),
                             "--data-dir", str(config_dir / "data"),
                             "--tui", "--provider", "openai-compatible", "--model", spec.model_id],
                    "detach": False, "url": None, "env": env}

        if client == "dsh":
            return {"argv": ["dsh", "--profile", "tui"], "detach": False,
                    "url": None, "env": env}
        if client == "grok":
            return {"argv": ["grok", "--model", provider], "detach": False,
                    "url": None, "env": env}
        if client == "kilo":
            return {"argv": ["kilo", "--model", f"openai-compatible/{spec.model_id}"],
                    "detach": False, "url": None, "env": env}
        if client == "pi":
            return {"argv": ["pi", "--provider", "tokirecur",
                             "--model", spec.model_id],
                    "detach": False, "url": None, "env": env}
        if client == "copilot":
            return {"argv": ["copilot", "--model", spec.model_id],
                    "detach": False, "url": None, "env": env}
        if client == "kimi":
            return {"argv": ["kimi", "--model", f"tokirecur/{spec.model_id}"],
                    "detach": False, "url": None, "env": env}
        if client == "omp":
            return {"argv": ["omp", "--provider", "tokirecur",
                             "--model", spec.model_id],
                    "detach": False, "url": None, "env": env}
    elif interface == "web":
        if client == "dsh":
            # --no-open because we do the opening, and only once we have read
            # the tokenised URL it prints on boot.
            return {"argv": ["dsh", "web", "--no-open"], "detach": True,
                    "url": None, "env": env}
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
    restore_now()
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
    if interface == "web" and client == "dsh":
        found = _await_dsh_web_url(proc, log_path)
        if found is None:
            console.print("[yellow]dsh did not print a URL - it may have failed to start.[/yellow]")
            console.print(f"  see: {log_path}")
            return 0
        _open_browser(found)
        return 0
    if url:
        import time
        time.sleep(2.0)
        _open_browser(url)
    return 0


# dsh serves its UI behind a one-time token and prints the full URL on boot.
# Opening the bare port gets a 401, so the URL has to be read from its output
# rather than assumed.
_DSH_URL_RE = re.compile(r"https?://127\.0\.0\.1:\d+/\?token=[A-Za-z0-9_=-]+")


def _await_dsh_web_url(proc: subprocess.Popen, log_path: str, timeout: float = 25.0) -> str | None:
    """Wait for dsh to announce its tokenised URL, or give up.

    Fails fast when the process dies rather than sitting out the timeout, so a
    broken launch says so immediately instead of appearing to hang.
    """
    import time

    deadline = time.time() + timeout
    seen = 0
    while time.time() < deadline:
        proc.poll()
        if proc.returncode is not None:
            return None
        try:
            data = Path(log_path).read_bytes()
        except OSError:
            data = b""
        # Only re-scan what is new, so a long log is not re-read every tick.
        chunk = data[seen:].decode("utf-8", "replace")
        seen = len(data)
        found = _DSH_URL_RE.search(chunk)
        if found:
            return found.group(0)
        time.sleep(0.25)
    return None
