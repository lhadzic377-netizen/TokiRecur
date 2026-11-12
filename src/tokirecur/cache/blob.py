"""KV cache blob sidecar (``.trkv``) and reader.

The daemon's hot path persists KV state to ``<cache_id>.bin`` via
llama-server's ``?action=save`` endpoint. ``.trkv`` is a small
sidecar that records metadata + a SHA-256 digest of the ``.bin``
payload, so the tree-cache can do a fast file-existence check
without re-reading the whole KV and the backend can verify integrity
before issuing ``?action=restore``.

Format (all integers big-endian):

    +--------+---------+-----------+--------+--------+
    | MAGIC  | VERSION | meta_len  |  meta  | digest |
    | 4 B    | 4 B     | 4 B       | N B    | 32 B   |
    +--------+---------+-----------+--------+--------+

    MAGIC   = b"TRKV"
    VERSION = 1
    meta    = JSON {"model_id", "n_past", "vocab_size"}
    digest  = SHA-256 of the underlying ``.bin`` payload
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import struct
from pathlib import Path

logger = logging.getLogger(__name__)

# Public aliases (some callers reference these by name).
TRKV_MAGIC = b"TRKV"
TRKV_VERSION = 1
# Backwards-compat: the original BlobWriter exposed MAGIC/VERSION.
MAGIC = TRKV_MAGIC
VERSION = TRKV_VERSION

# cache_ids are content-addressed (SHA-256 hex prefixes) — nothing but
# hex belongs in a file name. This is also a path-traversal guard for
# the DELETE /cache/{cache_id} endpoint: any character outside the set
# is rejected before it can reach the filesystem.
_CACHE_ID_RE = re.compile(r"^[0-9a-f]{1,64}$")

# 4-byte magic + 4-byte version + 4-byte meta_len = 12-byte fixed header.
_HEADER = struct.Struct(">4s I I")
_META_LEN = struct.Struct(">I")
_SHA256 = hashlib.sha256
_SHA256_DIGEST_BYTES = 32


def compute_sha256(data: bytes) -> bytes:
    """Return the SHA-256 digest of ``data`` (32 raw bytes)."""
    return hashlib.sha256(data).digest()


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> bytes:
    """Stream ``path`` through SHA-256 in ``chunk_size`` blocks.

    The KV ``.bin`` can be multiple gigabytes; reading it whole to hash
    it would spike RSS by the full payload size. This keeps memory flat.
    """
    h = _SHA256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.digest()


class BlobWriter:
    """Incremental writer for a ``.trkv`` sidecar.

    Caller writes a header (model_id, n_past, vocab_size), then the
    underlying ``.bin`` payload bytes, then calls :meth:`finalize` to
    flush the trailing digest and close the file.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "wb")
        self._hasher = _SHA256()
        self._bytes = 0

    def write_header(self, *, model_id: str, n_past: int, vocab_size: int) -> None:
        meta = json.dumps(
            {"model_id": model_id, "n_past": n_past, "vocab_size": vocab_size}
        ).encode("utf-8")
        self._f.write(_HEADER.pack(TRKV_MAGIC, TRKV_VERSION, len(meta)))
        self._f.write(_META_LEN.pack(len(meta)))
        self._f.write(meta)

    def write_payload(self, data: bytes) -> None:
        self._f.write(data)
        self._hasher.update(data)
        self._bytes += len(data)

    def update_digest(self, data: bytes) -> None:
        """Feed ``data`` into the trailing digest WITHOUT embedding it.

        Used when the payload lives in an external file (llama-server's
        ``.bin``) and the sidecar records only its SHA-256, keeping the
        ``.trkv`` small instead of duplicating a multi-GB KV blob.
        """
        self._hasher.update(data)

    def finalize(self) -> int:
        """Flush the trailing SHA-256 digest and close the file.

        Returns the number of payload bytes written (NOT including the
        header, meta-length prefix, meta block, or trailing digest)."""
        self._f.write(self._hasher.digest())
        self._f.close()
        return self._bytes

    def __enter__(self) -> BlobWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._f.closed:
            self._f.close()


class BlobReader:
    """Read a ``.trkv`` sidecar written by :class:`BlobWriter`.

    Validates the magic + version on open. Exposes :attr:`digest`
    (the trailing SHA-256 stored in the file) and :meth:`read_payload`
    for the underlying bytes.
    """

    def __init__(self, path: Path) -> None:
        with open(path, "rb") as f:
            data = f.read()
        if len(data) < _HEADER.size + 4 + _SHA256_DIGEST_BYTES:
            raise ValueError(f"{path}: file too short to be a .trkv")
        magic, version, meta_len = _HEADER.unpack(data[: _HEADER.size])
        if magic != TRKV_MAGIC:
            raise ValueError(f"{path}: bad magic {magic!r}")
        if version != TRKV_VERSION:
            raise ValueError(f"{path}: unsupported version {version}")
        meta_end = _HEADER.size + 4 + meta_len
        if meta_end > len(data) - _SHA256_DIGEST_BYTES:
            raise ValueError(f"{path}: meta block overruns payload")
        self.meta = json.loads(
            data[_HEADER.size + 4 : meta_end].decode("utf-8")
        )
        self.payload_offset = meta_end
        self.digest = data[-_SHA256_DIGEST_BYTES:]
        # Keep the full file bytes for read_payload (it's the sidecar
        # itself, not the underlying .bin which can be huge).
        self._data = data

    def read_payload(self) -> bytes:
        """Return the payload bytes (everything between the meta block
        and the trailing 32-byte digest)."""
        return self._data[self.payload_offset : -_SHA256_DIGEST_BYTES]


def blob_path(cache_dir: Path, cache_id: str) -> Path:
    """Return the sidecar path for ``cache_id``.

    Raises :class:`ValueError` if ``cache_id`` is not a plain hex id —
    this is the path-traversal guard for the DELETE /cache/{cache_id}
    endpoint, which must not let a caller reach outside ``cache_dir``.
    """
    if not _CACHE_ID_RE.fullmatch(cache_id):
        raise ValueError(f"invalid cache_id: {cache_id!r}")
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
