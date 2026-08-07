"""Ollama-style CLI launcher for Windows.

The Textual launcher renders slowly and glitches on the classic
Windows console, so on Windows bare ``toki`` opens this plain CLI
menu instead: a numbered app list, a model line, and foreground
chat / detached-console launches. Linux and macOS keep the Textual
launcher unchanged.

The app catalog mirrors ``tui.launcher.LAUNCH_APPS`` (kept as plain
data here so this module needs no Textual import).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import webbrowser

APPS: tuple[tuple[str, str, str], ...] = (
    ("toki", "Chat", "chat with models in the CLI chat"),
    ("opencode", "Launch OpenCode", "Anomaly's open-source coding agent"),
    ("hermes", "Launch Hermes Agent", "self-improving AI agent"),
    ("openclaw", "Launch OpenClaw", "personal AI with 100+ skills"),
    ("codex", "Launch Codex", "OpenAI CLI coding agent (Responses API)"),
)

WEB_URLS: tuple[str, ...] = (
    "http://127.0.0.1:7777",   # toki web UI
    "http://127.0.0.1:18789",  # openclaw gateway
    "http://127.0.0.1:9119",   # hermes dashboard
)


def _model_line() -> str:
    try:
        from . import ctl

        active = ctl.read_active()
    except Exception:
        active = None
    return f"model: {active}" if active else "model: none set (chat will ask)"


def _pick(lo: int, hi: int) -> str:
    while True:
        try:
            raw = input(f"pick ({lo}-{hi}, 0 to quit)> ").strip().lower()
        except EOFError:
            print()
            return "0"
        if not raw:
            continue
        return raw


def run_cli_launcher() -> int:
    """Plain CLI stand-in for the Textual launcher. Returns exit code."""
    if os.name == "nt":
        os.system("")
    from .clichat import run_cli_chat

    while True:
        print("toki launcher (CLI — the Textual launcher is replaced here "
              "on Windows)")
        print(_model_line())
        print()
        for i, (_, title, desc) in enumerate(APPS, 1):
            print(f"  {i}) {title:<22} {desc}")
        print(f"  {len(APPS) + 1}) Web Dashboards       browser UIs: toki · "
              "openclaw · hermes")
        print("  0) Quit")
        print()
        pick = _pick(1, len(APPS) + 1)
        if pick in ("0", "q", "quit", "exit"):
            print("bye.")
            return 0
        if pick == str(len(APPS) + 1):
            for url in WEB_URLS:
                webbrowser.open(url)
            print(f"opened {len(WEB_URLS)} dashboards in your browser.")
            continue
        try:
            idx = int(pick) - 1
        except ValueError:
            print(f"  invalid pick — enter a number 1-{len(APPS) + 1}.")
            continue
        if not 0 <= idx < len(APPS):
            print(f"  invalid pick — enter a number 1-{len(APPS) + 1}.")
            continue
        app, title, _ = APPS[idx]
        if app == "toki":
            try:
                run_cli_chat()
            except SystemExit:
                continue
            continue
        _launch_agent(app)


def _launch_agent(app: str) -> None:
    binary = shutil.which(app)
    if not binary:
        print(f"not found in PATH: {app} — install it, or pick another entry.")
        return
    try:
        if os.name == "nt":
            proc = subprocess.Popen(
                [app],
                stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
            print(f"launching {app} in a new console (pid {proc.pid}).")
        else:
            proc = subprocess.Popen(
                [app],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            print(f"launching {app} in the background (pid {proc.pid}).")
    except OSError as e:
        print(f"failed to launch {app}: {e}")
