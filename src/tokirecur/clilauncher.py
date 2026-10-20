"""Ollama-style CLI launcher for Windows.

The Textual launcher renders slowly and glitches on the classic
Windows console, so on Windows bare ``toki`` opens this plain CLI
menu instead. It keeps the Textual launcher's capabilities: pick an
app, switch the active model, and see live stack + GPU status.
Linux and macOS keep the Textual launcher unchanged.

The app catalog comes from ``launch_catalog`` and is kept as plain
 data here so this module needs no Textual import.

"""

from __future__ import annotations

import os
import shutil
import subprocess
import webbrowser

from .launch_catalog import LAUNCH_APPS, get_launch_app, unavailable_reason

_AGENT_APPS = tuple(
    app for app in LAUNCH_APPS if app.interface == "chat" and app.client != "toki"
)
AGENTS: tuple[tuple[str, str, str], ...] = tuple(
    (app.client, app.title, app.description) for app in _AGENT_APPS
)


def _availability_reason(launch) -> str | None:
    reason = unavailable_reason(launch)
    command = launch.command or launch.client
    if reason == f"{command} not installed" and _find_binary(command) is not None:
        return None
    return reason


def _agent_available(app: str) -> bool:
    launch = get_launch_app(app)
    return launch is not None and _availability_reason(launch) is None

WEB_URLS: tuple[str, ...] = (
    "http://127.0.0.1:7777",   # toki web UI
    "http://127.0.0.1:18789",  # openclaw gateway
    "http://127.0.0.1:9119",   # hermes dashboard
)

# Windows-only flag; absent on POSIX where the agent branch detaches
# via start_new_session instead.
_NEW_CONSOLE = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)


def _ctl():
    from . import ctl

    return ctl


_BACKEND_CACHE: dict[str, str | None] = {}


def _gpu_label(spec, llama_up: bool) -> str:
    """GPU status for a model: log evidence, else build evidence."""
    ctl = _ctl()
    gpu = ctl.gpu_state(spec)
    if gpu["backend"]:
        bits = [gpu["backend"]]
        if gpu["layers"]:
            bits.append(gpu["layers"] + " layers")
        return " · ".join(bits)
    if not llama_up:
        return "stopped"
    binary = ctl._resolve_binary(spec)
    if binary not in _BACKEND_CACHE:
        from . import diagnose

        _BACKEND_CACHE[binary] = diagnose.binary_backend(binary)
    kind = _BACKEND_CACHE[binary]
    if kind and kind != "CPU":
        return f"log terse (build: {kind})"
    return "CPU only"


def _active_spec():
    """The active registry spec, or None when none is set."""
    ctl = _ctl()
    try:
        active = ctl.read_active()
        if active:
            return ctl.get_model(active)
    except Exception:
        return None
    return None


def _header() -> str:
    """One-line header: model, stack state, GPU."""
    ctl = _ctl()
    spec = _active_spec()
    if spec is None:
        return "model: none set — use `2) Switch model`"
    st = ctl.model_status(spec)
    llama = "up" if st["llama"]["reachable"] else "down"
    daemon = "up" if st["daemon"]["reachable"] else "down"
    return (f"model: {spec.name}  llama:{llama}  daemon:{daemon}  "
            f"gpu: {_gpu_label(spec, st['llama']['reachable'])}")


def _pick(lo: int, hi: int, prompt: str = "pick") -> str:
    while True:
        try:
            raw = input(f"{prompt} ({lo}-{hi}, 0 to quit)> ").strip().lower()
        except EOFError:
            print()
            return "0"
        if not raw:
            continue
        return raw


def _status_row(spec) -> str:
    ctl = _ctl()
    st = ctl.model_status(spec)
    llama = "up" if st["llama"]["reachable"] else "down"
    daemon = "up" if st["daemon"]["reachable"] else "down"
    gpu = _gpu_label(spec, st["llama"]["reachable"])
    row = f"llama={llama} daemon={daemon} gpu={gpu}"
    # A vision model without its projector answers images as though it saw
    # nothing, so the row says which it is rather than leaving it a surprise.
    mmproj = getattr(spec, "mmproj", "") or ""
    if mmproj and not os.path.exists(mmproj):
        row += " [no projector]"
    elif mmproj:
        row += " [vision]"
    return row


def switch_model() -> None:
    """List models, pick one (or type a name) to make active."""
    ctl = _ctl()
    try:
        specs = ctl.load_registry()
    except Exception as e:
        print(f"error reading registry: {e}")
        return
    if not specs:
        print("no models installed — run `toki model pull <owner>/<repo>-GGUF` "
              "first.")
        return
    active = ctl.read_active()
    print()
    print("models:")
    for i, spec in enumerate(specs, 1):
        mark = " *active" if spec.name == active else ""
        print(f"  {i}) {spec.name:<16} {spec.label}  "
              f"{_status_row(spec)}{mark}")
    print()
    print("enter a number or a model name (blank keeps the active model)")
    try:
        raw = input("model> ").strip()
    except EOFError:
        print()
        return
    if not raw:
        return
    chosen = None
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(specs):
            chosen = specs[idx].name
    except ValueError:
        if any(s.name == raw for s in specs):
            chosen = raw
    if chosen is None:
        print(f"  unknown model: {raw}")
        return
    if chosen == active:
        print(f"  {chosen} is already active.")
        return
    print(f"  starting {chosen} ...")
    try:
        spec = ctl.use_model(chosen)
    except (KeyError, FileNotFoundError, RuntimeError, OSError) as e:
        print(f"  error: {e}")
        return
    print(f"  {spec.name} active  ({_status_row(spec)}).")


def run_cli_launcher() -> int:
    """Plain CLI stand-in for the Textual launcher. Returns exit code."""
    if os.name == "nt":
        os.system("")
    from . import diagnose
    from .clichat import run_cli_chat

    agents_available = {app: _agent_available(app) for app, _, _ in AGENTS}

    while True:
        print("toki launcher")
        print(_header())
        print()
        print("  1) Chat                  chat with the active model")
        print("  2) Switch model          list models, make one active")
        for i, (app, title, desc) in enumerate(AGENTS, 3):
            mark = "" if agents_available.get(app) else "  [not installed]"
            print(f"  {i}) {title:<22} {desc}{mark}")
        print(f"  {len(AGENTS) + 3}) Web Dashboards       browser UIs: toki · "
              "openclaw · hermes")
        print(f"  {len(AGENTS) + 4}) Doctor                GPU / build / stack "
              "diagnostics")
        print("  0) Quit")
        print()
        hi = len(AGENTS) + 4
        pick = _pick(1, hi)
        if pick in ("0", "q", "quit", "exit"):
            print("bye.")
            return 0
        if pick == "1":
            try:
                run_cli_chat()
            except SystemExit:
                continue
            continue
        if pick == "2":
            switch_model()
            agents_available = {app: _agent_available(app) for app, _, _ in AGENTS}
            continue
        if pick == str(len(AGENTS) + 3):
            _open_web_dashboards()
            continue
        if pick == str(len(AGENTS) + 4):
            print()
            print(diagnose.run())
            input("press Enter to return to the launcher> ")
            continue
        try:
            idx = int(pick) - 3
        except ValueError:
            print(f"  invalid pick — enter a number 1-{hi}.")
            continue
        if not 0 <= idx < len(AGENTS):
            print(f"  invalid pick — enter a number 1-{hi}.")
            continue
        _launch_agent(AGENTS[idx][0])


def _open_web_dashboards() -> None:
    """Start the toki web UI (and openclaw gateway when available),
    then open the browser tabs.

    The launcher used to only open tabs — on Windows nothing ever
    listened on 7777, so the browser said "can't reach page". Now we
    bring the toki UI up first (it needs the active model stack) and
    wait for it to answer before opening the tab.
    """
    urls: list[str] = []
    ctl = _ctl()
    spec = _active_spec()
    if spec is not None:
        try:
            if not ctl.daemon_reachable(spec):
                print(f"starting '{spec.name}' stack...")
                ctl.use_model(spec.name)
                spec = ctl.get_model(spec.name)
        except Exception as e:
            print(f"failed to start '{spec.name}' stack: {e}")
        try:
            from .selector import launch_cmd
            plan = launch_cmd("web", "toki", spec)
            argv = plan.get("argv") or []
            url = plan.get("url")
        except Exception as e:
            print(f"failed to build web launch plan: {e}")
            argv, url = [], None
        if url and not _port_open(_url_port(url)):
            print(f"starting toki web UI ({url})...")
            binary = _find_binary("toki")
            if binary:
                try:
                    subprocess.Popen(
                        [binary] + argv[1:],
                        stdin=subprocess.DEVNULL,
                        env=plan.get("env"),
                        creationflags=_NEW_CONSOLE,
                    )
                except OSError as e:
                    print(f"failed to launch toki web: {e}")
            if _wait_for_port(_url_port(url), timeout=15):
                urls.append(url)
            else:
                print("toki web UI did not answer on time; check the "
                      "stack is up (1) Chat).")
        elif url:
            urls.append(url)
    for url in WEB_URLS:
        if url not in urls and _port_open(_url_port(url)):
            urls.append(url)
    for url in urls:
        webbrowser.open(url)
    print(f"opened {len(urls)} dashboards in your browser.")


def _url_port(url: str) -> int:
    from urllib.parse import urlparse

    return urlparse(url).port


def _port_open(port: int) -> bool:
    import socket

    with socket.socket() as s:
        s.settimeout(0.4)
        try:
            s.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _wait_for_port(port: int, timeout: float) -> bool:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_open(port):
            return True
        time.sleep(0.5)
    return _port_open(port)


def _find_binary(app: str) -> str | None:
    """Locate an agent executable, including Windows install dirs.

    `shutil.which` handles PATH, but agents installed via `pipx`,
    npm `-g`, or native installers often live outside it; probe the
    usual user-local spots before giving up.
    """
    found = shutil.which(app)
    if found:
        return found
    if os.name != "nt":
        return None
    exts = ("", ".exe", ".cmd", ".bat")
    dirs = (
        os.environ.get("APPDATA", "") + "/npm",
        os.environ.get("LOCALAPPDATA", "") + "/Programs",
        os.path.expanduser("~/.local/bin"),
        os.path.expanduser("~/AppData/Roaming/npm"),
    )
    for d in dirs:
        if not d:
            continue
        for ext in exts:
            cand = os.path.join(d, app + ext)
            if os.path.isfile(cand):
                return cand
    return None


def _launch_agent(app: str) -> None:
    launch = get_launch_app(app)
    if launch is None:
        print(f"unknown launcher: {app}")
        return
    reason = _availability_reason(launch)
    if reason is not None:
        print(f"cannot launch {app}: {reason}")
        return
    ctl = _ctl()
    spec = _active_spec() if launch.model_binding else None
    if launch.model_binding:
        if spec is None:
            print("no active model — pick one with 2) Switch model first.")
            return
        try:
            if not ctl.daemon_reachable(spec):
                print(f"starting '{spec.name}' stack...")
                ctl.use_model(spec.name)
                spec = ctl.get_model(spec.name)
        except Exception as e:
            print(f"failed to start '{spec.name}' stack: {e}")
            return
    try:
        from .selector import launch_cmd
        plan = launch_cmd("chat", app, spec)
    except Exception as e:
        print(f"failed to build launch plan for {app}: {e}")
        return
    argv = plan.get("argv") or []
    if not argv:
        print(f"no launch plan for {app}.")
        return
    if app == "openclaw" and argv[:2] == ["openclaw", "tui"]:
        from .selector import _gateway_up
        if not _gateway_up():
            print("OpenClaw gateway is not running; falling back to local TUI.")
            argv = ["openclaw", "chat"]
    binary = _find_binary(launch.command or app)
    if not binary:
        print(f"not found in PATH: {launch.command or app} — install it, or pick another entry.")
        return
    args = argv[1:]
    if os.name == "nt":
        try:
            if binary.lower().endswith((".cmd", ".bat")):
                argv = ["cmd.exe", "/c", binary] + args
            else:
                argv = [binary] + args
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                env=plan.get("env"),
                creationflags=_NEW_CONSOLE,
            )
            print(f"launching {app} in a new console (pid {proc.pid}).")
        except OSError as e:
            print(f"failed to launch {app}: {e}")
        return
    try:
        proc = subprocess.Popen(
            [binary] + args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=plan.get("env"),
            start_new_session=True,
        )
        print(f"launching {app} in the background (pid {proc.pid}).")
    except OSError as e:
        print(f"failed to launch {app}: {e}")
