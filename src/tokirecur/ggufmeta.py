"""Minimal dependency-free GGUF header reader.

Extracts just enough metadata to size the KV cache: embedding
dimension, block count, and the attention head counts (needed for the
exact llama.cpp KV math: ``head_dim = n_embd / n_heads``, KV per token
per layer = ``head_dim * n_head_kv * 2 * kv_bytes``). Reads only the
metadata key-value section — never the tensor data — so it is fast
even for multi-GB files. Returns ``None`` fields for anything
unreadable so callers can fall back gracefully.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

_MAGIC = b"GGUF"

# GGUF value types (v3): https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STRING, _ARRAY, _U64, _I64, _F64 = range(13)


@dataclass(frozen=True)
class GgufMeta:
    n_embd: int | None = None
    n_layer: int | None = None
    n_head: int | None = None
    n_head_kv: int | None = None
    context_length: int | None = None

    @property
    def complete(self) -> bool:
        return all(v is not None for v in (self.n_embd, self.n_layer, self.n_head, self.n_head_kv))


def _read_bytes(fh, fmt):
    size = struct.calcsize(fmt)
    raw = fh.read(size)
    if len(raw) != size:
        raise struct.error("truncated")
    return struct.unpack(fmt, raw)[0]


def _read_string(fh) -> str:
    length = _read_bytes(fh, "<Q")
    raw = fh.read(length)
    if len(raw) != length:
        raise struct.error("truncated string")
    return raw.decode("utf-8")


def _read_value(fh, vtype: int):
    if vtype == _U8:
        return _read_bytes(fh, "<B")
    if vtype == _I8:
        return _read_bytes(fh, "<b")
    if vtype == _U16:
        return _read_bytes(fh, "<H")
    if vtype == _I16:
        return _read_bytes(fh, "<h")
    if vtype == _U32:
        return _read_bytes(fh, "<I")
    if vtype == _I32:
        return _read_bytes(fh, "<i")
    if vtype == _F32:
        return _read_bytes(fh, "<f")
    if vtype == _BOOL:
        return _read_bytes(fh, "<B") != 0
    if vtype == _STRING:
        return _read_string(fh)
    if vtype == _U64:
        return _read_bytes(fh, "<Q")
    if vtype == _I64:
        return _read_bytes(fh, "<q")
    if vtype == _F64:
        return _read_bytes(fh, "<d")
    if vtype == _ARRAY:
        elem_type = _read_bytes(fh, "<I")
        count = _read_bytes(fh, "<Q")
        return [_read_value(fh, elem_type) for _ in range(count)]
    raise struct.error(f"unknown value type {vtype}")


def read_gguf_meta(path: Path | str) -> GgufMeta:
    """Read the metadata we need from a GGUF header, or all-``None``."""
    try:
        with open(path, "rb") as fh:
            if fh.read(4) != _MAGIC:
                return GgufMeta()
            version = _read_bytes(fh, "<I")
            tensor_fmt = "<I" if version < 3 else "<Q"
            _read_bytes(fh, tensor_fmt)  # tensor_count
            kv_count = _read_bytes(fh, tensor_fmt)
            arch: str | None = None
            meta: dict[str, object] = {}
            for _ in range(kv_count):
                key = _read_string(fh)
                vtype = _read_bytes(fh, "<I")
                value = _read_value(fh, vtype)
                if key == "general.architecture":
                    arch = str(value) if isinstance(value, str) else None
                meta[key] = value
            # Metadata keys are architecture-prefixed (e.g. `qwen2.*`,
            # `llama.*`, `qwen35.*`); fall back to `llama.*` when the
            # architecture key is missing.
            prefixes = [f"{arch}."] if arch else []
            prefixes.append("llama.")
            n_embd = n_layer = n_head = n_head_kv = context_length = None
            for pref in prefixes:
                if n_embd is None:
                    v = meta.get(f"{pref}embedding_length")
                    n_embd = int(v) if isinstance(v, (int, float)) else None
                if n_layer is None:
                    v = meta.get(f"{pref}block_count")
                    n_layer = int(v) if isinstance(v, (int, float)) else None
                if n_head is None:
                    v = meta.get(f"{pref}attention.head_count")
                    n_head = int(v) if isinstance(v, (int, float)) else None
                if n_head_kv is None:
                    v = meta.get(f"{pref}attention.head_count_kv")
                    n_head_kv = int(v) if isinstance(v, (int, float)) else None
                if context_length is None:
                    v = meta.get(f"{pref}context_length")
                    if not isinstance(v, (int, float)):
                        v = meta.get(f"{pref}max_position_embeddings")
                    context_length = int(v) if isinstance(v, (int, float)) else None
                if n_embd is not None and n_layer is not None:
                    break
            return GgufMeta(n_embd, n_layer, n_head, n_head_kv, context_length)
    except (OSError, struct.error, UnicodeDecodeError, ValueError):
        return GgufMeta()
