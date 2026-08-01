"""KV cache blob storage.

Tensor blobs are stored in a single directory keyed by cache_id. The
blob format is a small header followed by the raw bytes the backend
gives us. For the llama-server HTTP backend the "bytes" we receive
today are an estimate (the server holds the KV in process memory); the
patched backend will give us real file paths.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import struct
from pathlib import Path

logger = logging.getLogger(__name__)

MAGIC = b"TRKV"
VERSION = 1
HEADER = struct.Struct(">4sI I I Q")  # magic, version, flags, reserved, payload_size


class BlobWriter:
    """Incremental writer for KV cache blobs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "wb")
        self._hasher = hashlib.sha256()
        self._bytes = 0

    def write_header(self, *, model_id: str, n_past: int, vocab_size: int) -> None:
        meta = json.dumps(
            {"model_id": model_id, "n_past": n_past, "vocab_size": vocab_size}
        ).encode("utf-8")
        meta_len = struct.Struct(">I")
        self._f.write(HEADER.pack(MAGIC, VERSION, 0, 0, len(meta) + 4))
        self._f.write(meta_len.pack(len(meta)))
        self._f.write(meta)
        self._hasher.update(MAGIC)
        self._hasher.update(HEADER.pack(MAGIC, VERSION, 0, 0, len(meta) + 4))
        self._hasher.update(meta_len.pack(len(meta)))
        self._hasher.update(meta)

    def write_payload(self, data: bytes) -> None:
        self._f.write(data)
        self._hasher.update(data)
        self._bytes += len(data)

    def finalize(self) -> str:
        self._f.write(self._hasher.digest())
        self._f.close()
        return self._bytes

    def __enter__(self) -> "BlobWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._f.closed:
            self._f.close()


def blob_path(cache_dir: Path, cache_id: str) -> Path:
    return cache_dir / f"{cache_id}.trkv"


def total_cache_bytes(cache_dir: Path) -> int:
    if not cache_dir.exists():
        return 0
    return sum(p.stat().st_size for p in cache_dir.glob("*.trkv"))


def evict_lru_files(cache_dir: Path, target_bytes: int) -> int:
    """Walk the cache dir in mtime order and delete until under target."""
    if not cache_dir.exists():
        return 0
    files = sorted(cache_dir.glob("*.trkv"), key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in files)
    n = 0
    for f in files:
        if total <= target_bytes:
            break
        size = f.stat().st_size
        try:
            f.unlink()
            total -= size
            n += 1
        except OSError:
            logger.warning("failed to evict %s", f)
    return n
