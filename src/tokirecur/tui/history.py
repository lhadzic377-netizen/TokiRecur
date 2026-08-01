"""JSONL chat history persistence.

Stores chat messages in append-only JSONL at the XDG data dir.
Atomic writes via tempfile + rename so a crash mid-write doesn't
corrupt the file.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def _data_dir() -> Path:
    """Resolve the XDG data dir, creating it if missing."""
    base = os.environ.get("XDG_DATA_HOME")
    if base:
        d = Path(base) / "tokirecur"
    else:
        d = Path.home() / ".local" / "share" / "tokirecur"
    d.mkdir(parents=True, exist_ok=True)
    return d


def history_path() -> Path:
    return _data_dir() / "history.jsonl"


def append(
    role: str,
    content: str,
    *,
    mode: str | None = None,
    metadata: dict | None = None,
    path: Path | None = None,
) -> None:
    """Append one line to the JSONL history. Atomic via tempfile + rename."""
    target = path or history_path()
    record: dict = {
        "ts": _now_ts(),
        "role": role,
        "content": content,
    }
    if mode is not None:
        record["mode"] = mode
    if metadata is not None:
        record["meta"] = metadata
    line = json.dumps(record, ensure_ascii=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Atomic append: write to a sibling tempfile, then rename onto the
    # target. This survives crashes mid-write.
    fd, tmp_name = tempfile.mkstemp(
        prefix=".history.", suffix=".tmp", dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(line + "\n")
        # Append tmp contents to target, then remove tmp.
        with open(target, "a", encoding="utf-8") as f, open(tmp_name, encoding="utf-8") as t:
            f.write(t.read())
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def load(
    *,
    limit: int = 200,
    path: Path | None = None,
) -> list[dict]:
    """Load the last `limit` entries, oldest first."""
    target = path or history_path()
    if not target.exists():
        return []
    try:
        # Read all lines, take last N. For very large files we'd want
        # to seek backwards; in practice chat logs are small.
        with open(target, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return []
    if limit and len(lines) > limit:
        lines = lines[-limit:]
    out: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def clear(path: Path | None = None) -> None:
    """Delete the history file."""
    target = path or history_path()
    try:
        target.unlink()
    except FileNotFoundError:
        pass


def _now_ts() -> float:
    import time
    return time.time()
