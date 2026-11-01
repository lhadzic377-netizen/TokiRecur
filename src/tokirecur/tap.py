"""Opt-in recording of the turns that pass through the daemon.

One JSONL line per completed turn: the messages the model was given, the
answer it gave, the reasoning it produced, and the numbers that describe
the turn (tokens, cache verdict, timings, effort). The shape is flat
enough to load with ``json.loads`` and rich enough to train on without a
second pass over the server.

Off unless asked for. A recorder that switches itself on would turn
people's prompts — their code, their keys, their customers — into files
without them choosing it. ``toki tap on`` writes a flag file the daemon
checks per request, so turning it on never means restarting a stack.

Where the data lands: ``$XDG_STATE_HOME/tokirecur/tap`` (usually
``~/.local/state/tokirecur/tap``), one file per UTC day, 0600. Nothing
here is ever sent anywhere; it is a file on the machine that ran the
model.

Not recorded, deliberately:

- images as pixels. A turn's images are captioned for a text-only model
  before the engine sees them, and the caption is what is recorded,
  because the caption is what the model actually read.
- token ids. llama-server does not return them for chat completions, and
  re-tokenising the text is not guaranteed to reproduce the model's own
  split. The text is here; the boundaries are not.
- raw-path traffic from the TUI, which talks to llama-server directly.
  The TUI writes its own tap for that path (``raw_turn``).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: One line per turn. Bumped when a field changes meaning, so a reader can
#: tell an old file from a new one without guessing.
RECORD_VERSION = 1

#: Directory ceiling. Oldest files are dropped to stay under it; the file
#: being written is never dropped. 512 MiB is roughly a million turns of
#: short agent traffic.
DEFAULT_MAX_MB = 512.0

#: Per-file ceiling before the day's file is rolled (``YYYY-MM-DD.1.jsonl``).
DEFAULT_FILE_MB = 64.0

_TRUE = frozenset({"1", "true", "yes", "on", "y", "t"})
_FALSE = frozenset({"0", "false", "no", "off", "n", "f"})

_write_lock = threading.Lock()
_warned: set[str] = set()


def _env_flag(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    return None


def _env_mb(name: str, default: float) -> int:
    raw = os.environ.get(name)
    if not raw:
        return int(default * 1024 * 1024)
    try:
        return max(0, int(float(raw) * 1024 * 1024))
    except ValueError:
        return int(default * 1024 * 1024)


def tap_dir() -> Path:
    """Where the recordings live. ``TOKI_TAP_DIR`` wins for tests."""
    override = os.environ.get("TOKI_TAP_DIR")
    if override:
        return Path(override).expanduser()
    state = os.environ.get("XDG_STATE_HOME")
    base = Path(state).expanduser() if state else Path.home() / ".local" / "state"
    return base / "tokirecur" / "tap"


def flag_path() -> Path:
    """The switch. Present means on; the daemon stats it per turn."""
    override = os.environ.get("TOKI_TAP_FLAG")
    if override:
        return Path(override).expanduser()
    config = os.environ.get("XDG_CONFIG_HOME")
    base = Path(config).expanduser() if config else Path.home() / ".config"
    return base / "tokirecur" / "tap"


def is_enabled() -> bool:
    """True when the tap is on.

    The environment wins in both directions, so a one-off run can record
    (``TOKI_TAP=1 toki serve``) or stay quiet (``TOKI_TAP=0``) regardless
    of the switch on disk.
    """
    env = _env_flag("TOKI_TAP")
    if env is not None:
        return env
    try:
        return flag_path().is_file()
    except OSError:
        return False


def set_enabled(on: bool) -> Path:
    """Write or remove the switch. Returns the flag path either way."""
    path = flag_path()
    if on:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "on\n"
            "# `toki tap off` removes this file. The daemon checks it per turn.\n"
        )
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    return path


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _base_path(day: str | None = None) -> Path:
    return tap_dir() / f"{day or _today()}.jsonl"


def _active_path() -> Path:
    """The file being appended to: today's, or the next roll of it.

    A file over the per-file ceiling moves to ``<day>.1.jsonl`` and the
    fresh one becomes active, so no single file grows without bound and
    no turn is ever split across two files.
    """
    limit = _env_mb("TOKI_TAP_FILE_MB", DEFAULT_FILE_MB)
    base = _base_path()
    if limit <= 0:
        return base
    try:
        if base.exists() and base.stat().st_size >= limit:
            index = 1
            while True:
                rolled = base.with_name(f"{base.stem}.{index}.jsonl")
                if not rolled.exists():
                    os.replace(base, rolled)
                    return base
                index += 1
    except OSError:
        pass
    return base


def _files() -> list[Path]:
    d = tap_dir()
    try:
        return sorted(
            (p for p in d.glob("*.jsonl") if p.is_file()),
            key=lambda p: (p.stat().st_mtime, p.name),
        )
    except OSError:
        return []


def _enforce_cap(active: Path) -> None:
    """Drop the oldest files until the directory is under its ceiling.

    The active file is never dropped: stopping mid-day would leave a tap
    that looks on and records nothing, which is worse than a full disk
    that says so.
    """
    budget = _env_mb("TOKI_TAP_MAX_MB", DEFAULT_MAX_MB)
    if budget <= 0:
        return
    try:
        sizes = [(p, p.stat().st_size) for p in _files()]
    except OSError:
        return
    total = sum(size for _, size in sizes)
    for path, size in sizes:
        if total <= budget:
            break
        if path == active:
            continue
        try:
            path.unlink()
            total -= size
        except OSError:
            continue


def _warn_once(key: str, message: str) -> None:
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(message)


def _write(record: dict[str, Any]) -> bool:
    """Append one record. Never raises into a request."""
    try:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as e:
        _warn_once("encode", f"tap: could not encode a record: {e}")
        return False
    payload = (line + "\n").encode("utf-8")
    with _write_lock:
        try:
            tap_dir().mkdir(parents=True, exist_ok=True)
            path = _active_path()
            fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                os.write(fd, payload)
            finally:
                os.close(fd)
        except OSError as e:
            _warn_once("write", f"tap: could not write a record: {e}")
            return False
        _enforce_cap(path)
    return True


def _seconds(value: Any) -> float | None:
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


def build_record(
    *,
    endpoint: str,
    model: str | None,
    result: Any = None,
    messages: list[dict] | None = None,
    prompt: str | None = None,
    effort: str | None = None,
    session_id: str | None = None,
    params: dict | None = None,
    mode: str = "cached",
    error: str | None = None,
) -> dict[str, Any]:
    """Shape one turn for disk. Pure, so the tests can pin the contract."""
    reused = int(getattr(result, "reuse_depth", 0) or 0)
    prefilled = int(getattr(result, "prefilled_tokens", 0) or 0)
    generated = int(getattr(result, "generated_tokens", 0) or 0)
    record: dict[str, Any] = {
        "v": RECORD_VERSION,
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "endpoint": endpoint,
        "mode": mode,
        "model": model,
        "effort": effort,
        "session_id": session_id,
        "params": params or {},
        "completion": getattr(result, "output_text", "") or "",
        "reasoning": getattr(result, "reasoning", "") or "",
        "finish_reason": getattr(result, "finish_reason", None),
        "usage": {
            # The whole prompt: what the cache served plus what the
            # backend had to decode. The daemon's wire `usage` counts
            # only the latter, which under-reports a cached turn.
            "prompt_tokens": reused + prefilled,
            "completion_tokens": generated,
            "total_tokens": reused + prefilled + generated,
        },
        "cache": {
            "hit": bool(getattr(result, "hit", False)),
            "soft_hit": bool(getattr(result, "soft_hit", False)),
            "reused_tokens": reused,
            "prefilled_tokens": prefilled,
            "embedding_similarity": _seconds(
                getattr(result, "embedding_similarity", None)
            ),
            "slot_id": getattr(result, "slot_id", None),
            "spilled": bool(getattr(result, "spilled", False)),
        },
        "timings": {
            "ttft_ms": _seconds(getattr(result, "ttft_ms", None)),
            "total_ms": _seconds(getattr(result, "total_ms", None)),
            "generated_tokens_per_sec": _seconds(
                getattr(result, "generated_tokens_per_sec", None)
            ),
        },
    }
    if messages is not None:
        record["messages"] = messages
    if prompt is not None:
        record["prompt"] = prompt
    tool_calls = getattr(result, "tool_calls", None)
    if tool_calls:
        record["tool_calls"] = tool_calls
    if error:
        record["error"] = error
    return record


def record(
    *,
    endpoint: str,
    model: str | None,
    result: Any = None,
    messages: list[dict] | None = None,
    prompt: str | None = None,
    effort: str | None = None,
    session_id: str | None = None,
    params: dict | None = None,
    mode: str = "cached",
    error: str | None = None,
) -> bool:
    """Record one turn if the tap is on. Returns whether it was written.

    Cheap when off (one ``stat`` and a return), and a failure to write is
    logged once and swallowed: a recorder must never be the reason a
    completion fails.
    """
    if "PYTEST_CURRENT_TEST" in os.environ and not os.environ.get("TOKI_TAP_DIR"):
        return False
    if not is_enabled():
        return False
    return _write(build_record(
        endpoint=endpoint,
        model=model,
        result=result,
        messages=messages,
        prompt=prompt,
        effort=effort,
        session_id=session_id,
        params=params,
        mode=mode,
        error=error,
    ))


def raw_turn(
    *,
    model: str | None,
    messages: list[dict] | None = None,
    prompt: str | None = None,
    completion: str = "",
    reasoning: str = "",
    effort: str | None = None,
    params: dict | None = None,
    error: str | None = None,
) -> bool:
    """Record a turn that bypassed the daemon (the TUI's raw path).

    Same file and same shape as the daemon's records, with whatever
    numbers the client can honestly supply — which, on the raw path, is
    nothing about caching: there was no cache in front of it.
    """
    if "PYTEST_CURRENT_TEST" in os.environ and not os.environ.get("TOKI_TAP_DIR"):
        return False
    if not is_enabled():
        return False
    record = build_record(
        endpoint="chat.completions",
        model=model,
        messages=messages,
        prompt=prompt,
        effort=effort,
        params=params,
        mode="raw",
        error=error,
    )
    record["completion"] = completion
    record["reasoning"] = reasoning
    return _write(record)


def iter_records(
    *, since: str | None = None, include_errors: bool = True,
) -> Iterator[dict[str, Any]]:
    """Every record on disk, oldest file first. ``since`` is a date
    (``YYYY-MM-DD``) and keeps files from that day onward."""
    for path in _files():
        if since and path.name[:10] < since:
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not include_errors and item.get("error"):
                        continue
                    yield item
        except OSError:
            continue


def shelf() -> dict[str, int]:
    """What is on the shelf, without counting the lines.

    The launcher draws this in a row that re-renders on every poll, and
    counting the lines of a 512 MiB tape on every frame is not free. The
    file count and the bytes are a stat each.
    """
    files = _files()
    total = 0
    for path in files:
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return {"files": len(files), "bytes": total}


def stats() -> dict[str, Any]:
    """What the switch says and what is actually on disk."""
    files = _files()
    total = 0
    lines = 0
    for path in files:
        try:
            total += path.stat().st_size
            with path.open("r", encoding="utf-8") as fh:
                lines += sum(1 for _ in fh)
        except OSError:
            continue
    return {
        "enabled": is_enabled(),
        "flag": str(flag_path()),
        "dir": str(tap_dir()),
        "files": len(files),
        "records": lines,
        "bytes": total,
        "max_bytes": _env_mb("TOKI_TAP_MAX_MB", DEFAULT_MAX_MB),
        "newest": str(files[-1]) if files else None,
    }


def clear() -> tuple[int, int]:
    """Delete every recording. Returns (files, bytes) removed."""
    removed = 0
    freed = 0
    for path in _files():
        try:
            size = path.stat().st_size
            path.unlink()
            removed += 1
            freed += size
        except OSError:
            continue
    return removed, freed


def sft_records(
    records: Iterable[dict[str, Any]], *, with_reasoning: bool = True,
) -> Iterator[dict[str, Any]]:
    """Turn tap records into one-line supervised fine-tuning examples.

    The shape is deliberately plain — the prompt turns, then one
    assistant turn holding the answer — because every trainer accepts
    ``messages``; the reasoning rides beside it instead of being welded
    into the answer, so a pipeline that does not want it can drop the key
    and nothing else changes.
    """
    for item in records:
        messages = item.get("messages")
        if not messages:
            prompt = item.get("prompt")
            if prompt is None:
                continue
            messages = [{"role": "user", "content": prompt}]
        completion = item.get("completion") or ""
        if not completion:
            continue
        example: dict[str, Any] = {
            "messages": [
                *messages,
                {"role": "assistant", "content": completion},
            ],
        }
        reasoning = item.get("reasoning") or ""
        if with_reasoning and reasoning:
            example["reasoning"] = reasoning
        if item.get("effort"):
            example["effort"] = item["effort"]
        yield example
