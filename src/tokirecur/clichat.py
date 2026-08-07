"""Ollama-style interactive CLI chat for Windows.

Windows' classic console renders Textual TUIs slowly and glitches
(no WT_SESSION), so on Windows ``toki tui`` routes here: a plain,
streaming REPL modelled on ``ollama run`` — ``>>> `` prompt, live
token output, ``/bye`` ``/clear`` ``/help``, and ``\"\"\"`` multiline
prompts. Linux and macOS keep the full TUI. (Bare ``toki`` on Windows
opens the CLI launcher menu — ``clilauncher`` — whose Chat entry runs
this REPL.)

The chat runs in ``cached`` mode through the selected model's daemon
so the KV-cache reuse layer is exercised exactly like the TUI does.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from typing import Any

_DIM = "\x1b[2m"
_RESET = "\x1b[0m"


def _resolve_model(name: str | None):
    """Pick the model to chat with.

    Explicit name wins; otherwise the active marker, then a lone
    registry entry; several entries prompt for a number.
    """
    from . import ctl

    specs = ctl.load_registry()
    if not specs:
        print("no models installed — run `toki model pull <owner>/<repo>-GGUF` first.")
        raise SystemExit(1)
    if name:
        return ctl.get_model(name)
    active = ctl.read_active()
    if active and any(s.name == active for s in specs):
        return ctl.get_model(active)
    if len(specs) == 1:
        return specs[0]
    print("pick a model:")
    for i, s in enumerate(specs, 1):
        print(f"  {i}) {s.name}")
    while True:
        try:
            raw = input("1-" + str(len(specs)) + "> ").strip()
        except EOFError:
            raise SystemExit(1) from None
        if not raw:
            continue
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(specs):
                return specs[idx]
        except ValueError:
            pass
        print(f"  invalid pick — enter a number 1-{len(specs)}.")


def _max_tokens() -> int:
    try:
        return int(os.environ.get("TOKI_MAX_TOKENS", "2048"))
    except ValueError:
        return 2048


def _install_readline() -> None:
    """Give the REPL arrow-key history + line editing where available."""
    try:
        import readline  # noqa: F401  (POSIX; Python 3.13+ also on Windows)
    except ImportError:
        pass


async def _read_prompt(prompt: str) -> str | None:
    """One prompt, supporting ``\"\"\"`` multiline blocks like ollama.

    Returns None on EOF. A line starting with ``\"\"\"`` collects
    continuation lines until one ends with ``\"\"\"``; the quotes are
    stripped from the result.
    """
    try:
        line = await asyncio.to_thread(input, prompt)
    except EOFError:
        print()
        return None
    stripped = line.strip()
    if not stripped.startswith('"""'):
        return line
    parts = [line]
    while True:
        try:
            more = await asyncio.to_thread(input, "... ")
        except EOFError:
            print()
            break
        parts.append(more)
        if more.strip().endswith('"""'):
            break
    joined = "\n".join(parts)
    if joined.startswith('"""'):
        joined = joined[3:]
        if joined.startswith("\n"):
            joined = joined[1:]
    if joined.endswith('"""'):
        joined = joined[:-3]
        if joined.endswith("\n"):
            joined = joined[:-1]
    return joined


async def _stream_response(
    client: Any, prompt: str, session_id: str, history: list[dict],
) -> dict[str, Any] | None:
    """Stream one completion; prints tokens live. Returns the done event."""
    done: dict[str, Any] | None = None
    printed = False
    async for ev in client.chat_stream(
        prompt,
        mode="cached",
        session_id=session_id,
        max_tokens=_max_tokens(),
        temperature=0.0,
        history=history,
    ):
        e = ev.get("event")
        if e == "token":
            chunk = ev.get("content", "")
            if chunk:
                print(chunk, end="", flush=True)
                printed = True
        elif e == "done":
            done = ev
    if printed:
        print()
    if done:
        error = done.get("error")
        if error:
            print(f"error: {error}")
            return None
        cache = done.get("cache") or {}
        skipped = int(cache.get("skipped_tokens") or 0)
        if cache.get("hit") or cache.get("soft_hit"):
            print(f"{_DIM}✓ cache reuse · {skipped} tokens skipped{_RESET}")
    return done


_HELP = """\
Available commands:
  /help            show this help
  /clear           clear the conversation and start a new session
  /bye  /exit      quit (Ctrl+C also quits)
  \"\"\"             start a multi-line prompt; close it with \"""
Multi-turn context is kept automatically; the cache layer reuses the
KV prefix across turns."""


async def _repl(spec: Any, daemon_url: str, raw_url: str,
                auth_token: str | None) -> None:
    from .tui.client import DaemonClient

    client = DaemonClient(daemon_url, raw_url, auth_token)
    session_id = uuid.uuid4().hex
    history: list[dict[str, str]] = []
    while True:
        line = await _read_prompt(">>> ")
        if line is None:
            return
        if not line.strip():
            continue
        if line.strip().startswith("/"):
            cmd = line.strip().split(maxsplit=1)[0].lower()
            if cmd in ("/bye", "/exit", "/quit"):
                return
            if cmd == "/clear":
                history = []
                session_id = uuid.uuid4().hex
                print(f"{_DIM}(cleared — new session){_RESET}")
                continue
            if cmd == "/help":
                print(_HELP)
                continue
            print(f"unknown command '{cmd}' — /help lists the commands.")
            continue
        history.append({"role": "user", "content": line})
        done = await _stream_response(client, line, session_id, history)
        if done:
            text = done.get("text") or ""
            if text:
                history.append({"role": "assistant", "content": text})


def run_cli_chat(
    model_name: str | None = None,
    *,
    daemon_url: str | None = None,
    raw_url: str | None = None,
    auth_token: str | None = None,
) -> int:
    """Entry point: resolve the model, ensure its stack is up, chat.

    Returns a process exit code (0 on clean quit).
    """
    if os.name == "nt":
        # Enable ANSI (used by the dim cache line) in the legacy
        # console; a no-op elsewhere.
        os.system("")
    from . import ctl

    spec = _resolve_model(model_name)
    # The TUI-style defaults point at the first stack; the selected
    # model's own daemon/llama are the authoritative target.
    daemon_url = daemon_url or spec.daemon_url
    raw_url = raw_url or spec.llama_url

    if not ctl.daemon_reachable(spec):
        try:
            ctl.use_model(spec.name)
        except (KeyError, FileNotFoundError, RuntimeError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        if not ctl.daemon_reachable(spec):
            print(
                f"error: {spec.name} stack did not come up — "
                f"check `toki model status`.",
                file=sys.stderr,
            )
            return 1

    _install_readline()
    print(f"toki chat — {spec.name}  (llama :{spec.llama_port} · "
          f"daemon :{spec.daemon_port})")
    print("type /help for commands, /bye to quit.")
    try:
        asyncio.run(_repl(spec, daemon_url, raw_url, auth_token))
    except KeyboardInterrupt:
        print()
    return 0
