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

from . import winproc
from .adapters import active_mode, active_mode_id
from .ctl import (
    _effective_ctx,
    _resolve_mode,
    auth_token,
    client_mode,
    cline_config_dir,
    load_registry,
    model_status,
    qwen_model_ref,
    qwen_uses_custom_providers,
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
        "copilot", "kimi", "omp",
    }
    if client == "codex" or client in new_clients:
        token = _auth_token() or auth_token()
        env["TOKI_AUTH_TOKEN"] = token
    else:
        token = env.get("TOKI_AUTH_TOKEN", "")
    if client in {
        "qwen", "cline", "droid", "dsh", "grok", "kilo", "pi", "kimi", "omp",
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
        preload_path = str(_CLINE_AI_SDK_PRELOAD)
        if winproc.IS_WINDOWS:
            preload = f'--preload="{preload_path}"'
        else:
            preload = f"--preload={shlex.quote(preload_path)}"
        existing = env.get("BUN_OPTIONS", "").strip()
        env["BUN_OPTIONS"] = f"{existing} {preload}".strip()
    return env


#: How long Claude Code should wait for a local turn, in milliseconds.
#:
#: Claude Code aborts a request after ten minutes (API_TIMEOUT_MS) and
#: treats 300 seconds of silence as a dead stream. On a CPU-only box a
#: prefill can spend that on its own — the client then gives up on a
#: turn the daemon is still working on. Twenty-five minutes is five
#: short of the daemon's own thirty, so the daemon is never the side
#: that gives up first.
CLAUDE_CLIENT_TIMEOUT_MS = 1_500_000


def _claude_timeout_env() -> dict[str, str]:
    """How long Claude Code should wait for a local turn.

    Only a machine the profile calls CPU-only gets the long budget: on
    a GPU box the defaults are plenty, and a generous timeout on a fast
    box only hides a real hang.
    """
    override = os.environ.get("TOKI_CLAUDE_TIMEOUT_MS", "").strip()
    if override:
        try:
            ms = int(override)
        except ValueError:
            ms = 0
        if ms > 0:
            return {
                "API_TIMEOUT_MS": str(ms),
                "CLAUDE_STREAM_IDLE_TIMEOUT_MS": str(ms),
            }
    try:
        from .hardware import load_profile

        profile = load_profile()
    except Exception:
        profile = None
    backend = ""
    if profile is not None:
        backend = str((profile.recommended or {}).get("backend", ""))
    if backend == "cpu":
        return {
            "API_TIMEOUT_MS": str(CLAUDE_CLIENT_TIMEOUT_MS),
            "CLAUDE_STREAM_IDLE_TIMEOUT_MS": str(CLAUDE_CLIENT_TIMEOUT_MS),
        }
    return {}


def launch_cmd(interface: str, client: str, spec: Any | None) -> dict:
    """Build the launch plan for an (interface, client, model) combo."""
    launch = get_launch_app(client)
    if launch is not None and launch.interface == "chat" and not launch.model_binding:
        return {
            "argv": [launch.command or launch.client, *launch.args],
            "detach": launch.detach,
            "url": None,
            "env": dict(os.environ),
        }
    if spec is None:
        raise ValueError(f"client {client!r} requires an active model")
    if client == "claude":
        settings_path = write_claude_config(spec)
        token = _auth_token() or auth_token()
        env = dict(os.environ)
        # The token travels as ANTHROPIC_AUTH_TOKEN. An inherited
        # ANTHROPIC_API_KEY would be preferred by Claude Code, and it
        # belongs to somebody else: the daemon would answer 401 for it.
        env.pop("ANTHROPIC_API_KEY", None)
        env.update({
            "ANTHROPIC_BASE_URL": f"{spec.daemon_url}/anthropic",
            "ANTHROPIC_AUTH_TOKEN": token,
            "ANTHROPIC_MODEL": spec.model_id,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": spec.model_id,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": spec.model_id,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": spec.model_id,
        })
        env.update(_claude_timeout_env())
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

    if interface in {"chat", "tui"}:
        if client == "toki":
            argv = ["toki", "tui", "--raw-url", llama_url, "--daemon-url", daemon_url]
            return {"argv": argv, "detach": False, "url": None, "env": env}
        if client == "opencode":
            # No model flag: this build of opencode has no -m/--model, and
            # passing one it does not know makes it print its help and exit
            # instead of starting. The model comes from the config, which
            # was just written with the same reference.
            return {"argv": ["opencode"], "detach": False,
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
            # Qwen Code 1.0 dropped --openai-base-url and --auth-type
            # (both exit 1 as unknown options) and keys a custom
            # provider's model as "<provider>/<model>", so --model needs
            # the qualified name there and the bare id before it.
            if qwen_uses_custom_providers():
                argv = ["qwen", "--model", qwen_model_ref(spec)]
            else:
                argv = ["qwen", "--model", spec.model_id,
                        "--openai-base-url", _bound_endpoint(spec, client),
                        "--auth-type", "openai"]
            return {"argv": argv, "detach": False, "url": None, "env": env}
        if client == "droid":
            return {"argv": ["droid"], "detach": False, "url": None, "env": env}
        if client == "cline":
            config_dir = cline_config_dir()
            return {"argv": ["cline", "--config", str(config_dir),
                             "--data-dir", str(config_dir / "data"),
                             "--tui", "--provider", "openai-compatible", "--model", spec.model_id],
                    "detach": False, "url": None, "env": env}

        if client == "dsh":
            # The web UI is the same model over the same daemon, so launching
            # the TUI also brings the browser UI up. The TUI execs and takes
            # the terminal, so the companion has to be started first - which
            # is what "companion" is for.
            return {"argv": ["dsh", "--profile", "tui"], "detach": False,
                    "url": None, "env": env,
                    "companion": {"client": "dsh", "argv": ["dsh", "web", "--no-open"]}}
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
        # Vision models are not interchangeable with text ones: without their
        # projector they answer every image as though it were blank.
        mmproj = getattr(spec, "mmproj", "") or ""
        if mmproj and not Path(mmproj).exists():
            badge = "[no projector]"
        elif mmproj:
            badge = "[vision]"
        else:
            badge = ""
        # The active adapter mode is how the stack actually runs, so it
        # belongs on the row: a missing adapter is a stack that will not
        # start, and a stale mode id is a catalog edit nobody finished.
        mode = active_mode(spec)
        if mode is not None:
            mark = f"[{mode.badge}]" if mode.exists else f"[no {mode.badge}]"
        elif active_mode_id(spec):
            mark = f"[{active_mode_id(spec)} missing]"
        else:
            mark = ""
        if mark:
            badge = (badge + " " + mark) if badge else mark
        rows.append((spec.name,
                     f"{mark} {spec.label}  llama={llama} daemon={daemon} "
                     f"ctx={ctx}  :{spec.llama_port}/:{spec.daemon_port}"
                     + (f"  {badge}" if badge else "")))
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

    The full-screen Textual launcher is the launcher on every platform,
    Windows included (Windows Terminal and the Windows 10+ console both
    host it). Set ``TOKI_CLI=1`` for the plain numbered menu
    (``clilauncher``) — a remote shell, a screen reader, or a console
    that cannot host a full-screen app. If Textual cannot be imported,
    the plain menu is used automatically.
    """
    if os.environ.get("TOKI_CLI") == "1":
        from .clilauncher import run_cli_launcher

        return run_cli_launcher()
    try:
        from .tui.launcher import run as launcher_run
    except ImportError as e:
        console.print(
            f"[yellow]full-screen launcher unavailable ({e}); "
            f"using the plain menu[/yellow]"
        )
        from .clilauncher import run_cli_launcher

        return run_cli_launcher()
    return launcher_run()


def start_companion(companion: dict, env: dict) -> None:
    """Start a detached side-surface and open it in a browser.

    Runs before any exec, because the foreground launch replaces this process
    and anything not already spawned dies with it.
    """
    argv = list(companion.get("argv") or [])
    if not argv:
        return
    client = companion.get("client") or ""
    log_dir = Path.home() / ".local/state/tokirecur"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(log_dir / f"{client}-companion.log")
    try:
        with open(log_path, "ab") as log:
            proc = subprocess.Popen(
                winproc.resolve_argv(argv), env=env, stdout=log,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                **winproc.detach_kwargs(),
            )
    except OSError as e:
        console.print(f"[yellow]could not start the {client} web UI: {e}[/yellow]")
        return
    found = _await_dsh_web_url(proc, log_path, start=_log_size(log_path))
    if found is None:
        existing = _existing_dsh_web_url(log_path)
        if existing is not None:
            console.print(f"[dim]{client} web UI is already running[/dim]")
            _open_browser(existing)
            return
        console.print(f"[yellow]{client} web UI did not start - see {log_path}[/yellow]")
        return
    console.print(f"[dim]{client} web UI: {found}[/dim]")
    _open_browser(found)


def _launch_foreground(argv: list[str], env: dict) -> int:
    console.print(Panel(winproc.format_cmdline(argv),
                        title="launching", border_style="green", expand=False))
    restore_now()
    if winproc.IS_WINDOWS:
        # os.execvpe exists on Windows but cannot run a .cmd shim and
        # does not replace the process the way it does on POSIX, so run
        # the client as a child and return its exit code.
        resolved = winproc.resolve_argv(argv)
        try:
            return subprocess.call(resolved, env=env)
        except FileNotFoundError:
            console.print(f"[red]client not found: {argv[0]}[/red]")
            return 1
        except OSError as e:
            console.print(f"[red]could not launch {argv[0]}: {e}[/red]")
            return 1
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
            if winproc.IS_WINDOWS:
                console.print("  start the OpenClaw gateway from its own app, then re-run: toki")
            else:
                console.print("  start it with:  systemctl --user start openclaw-gateway.service")
                console.print("  then re-run:    toki")
        return 0
    console.print(Panel(winproc.format_cmdline(argv),
                        title="launching in background", border_style="green", expand=False))
    log_dir = Path.home() / ".local/state/tokirecur"
    log_path = str(log_dir / f"{client}-{interface}.log")
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(
            winproc.resolve_argv(argv), env=env, stdout=log,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            **winproc.detach_kwargs(),
        )
    console.print(f"[dim]pid={proc.pid}  log: {log_path}[/dim]")
    if interface == "web" and client == "dsh":
        start = _log_size(log_path)
        found = _await_dsh_web_url(proc, log_path, start=start)
        if found is not None:
            _open_browser(found)
            return 0
        # A second `dsh web` dies with EADDRINUSE and prints no URL, even
        # though the first one is serving perfectly well. Reuse it instead of
        # reporting a failure the user cannot act on.
        existing = _existing_dsh_web_url(log_path)
        if existing is not None:
            console.print("[dim]dsh web is already running - opening that one[/dim]")
            _open_browser(existing)
            return 0
        console.print("[yellow]dsh web did not start.[/yellow]")
        console.print(f"  see: {log_path}")
        console.print("  common cause: port 3080 is taken by something that is not dsh")
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


def _log_size(path: str) -> int:
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def _existing_dsh_web_url(log_path: str) -> str | None:
    """The most recent URL a previous dsh web wrote, if it is still serving."""
    found = None
    try:
        for line in Path(log_path).read_text("utf-8", "replace").splitlines():
            match = _DSH_URL_RE.search(line)
            if match:
                found = match.group(0)
    except OSError:
        return None
    if found is None:
        return None
    return found


def _await_dsh_web_url(proc: subprocess.Popen, log_path: str, timeout: float = 25.0,
                       start: int | None = None) -> str | None:
    """Wait for dsh to announce its tokenised URL, or give up.

    Fails fast when the process dies rather than sitting out the timeout, so a
    broken launch says so immediately instead of appearing to hang.
    """
    import time

    deadline = time.time() + timeout
    # Start past anything a previous run left in the log. The file is opened in
    # append mode, so scanning from 0 matches last run's URL first and hands
    # the browser a token for a session that is long gone.
    seen = _log_size(log_path) if start is None else start
    while time.time() < deadline:
        proc.poll()
        if proc.returncode is not None:
            return None
        try:
            data = Path(log_path).read_bytes()
        except OSError:
            data = b""
        if len(data) < seen:
            # Truncated and rewritten; start over rather than slice nonsense.
            seen = 0
        chunk = data[seen:].decode("utf-8", "replace")
        seen = len(data)
        found = _DSH_URL_RE.search(chunk)
        if found:
            return found.group(0)
        time.sleep(0.25)
    return None
