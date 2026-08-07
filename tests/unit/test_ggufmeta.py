"""Unit tests for the minimal GGUF header reader."""

import struct

import pytest

from tokirecur.ggufmeta import GgufMeta, read_gguf_meta

_I32, _F32, _STRING = 5, 7, 8


def _gguf_bytes(kv: dict[str, object]) -> bytes:
    """Build a minimal valid GGUF v3 header (metadata only, no tensors)."""
    out = bytearray(b"GGUF")
    out += struct.pack("<I", 3)  # version
    out += struct.pack("<Q", 0)  # tensor count
    out += struct.pack("<Q", len(kv))  # kv count
    for key, value in kv.items():
        b = key.encode()
        out += struct.pack("<Q", len(b)) + b
        if isinstance(value, int):
            out += struct.pack("<I", _I32) + struct.pack("<i", value)
        elif isinstance(value, float):
            out += struct.pack("<I", _F32) + struct.pack("<f", value)
        elif isinstance(value, str):
            sb = value.encode()
            out += struct.pack("<I", _STRING) + struct.pack("<Q", len(sb)) + sb
        else:
            raise AssertionError(f"unsupported test value {value!r}")
    return bytes(out)


def _write(tmp_path, kv):
    p = tmp_path / "model.gguf"
    p.write_bytes(_gguf_bytes(kv))
    return p


def test_reads_core_metadata(tmp_path):
    p = _write(tmp_path, {
        "general.architecture": "qwen2",
        "qwen2.embedding_length": 1024,
        "qwen2.block_count": 16,
        "qwen2.attention.head_count": 16,
        "qwen2.attention.head_count_kv": 8,
        "qwen2.context_length": 32768,
    })
    meta = read_gguf_meta(p)
    assert meta == GgufMeta(1024, 16, 16, 8, 32768)
    assert meta.complete


def test_reads_context_length(tmp_path):
    p = _write(tmp_path, {
        "general.architecture": "qwen2",
        "qwen2.embedding_length": 1024,
        "qwen2.block_count": 16,
        "qwen2.attention.head_count": 16,
        "qwen2.attention.head_count_kv": 8,
        "qwen2.context_length": 128000,
    })
    assert read_gguf_meta(p).context_length == 128000


def test_max_position_embeddings_fallback(tmp_path):
    """Older GGUFs use max_position_embeddings instead of context_length."""
    p = _write(tmp_path, {
        "general.architecture": "llama",
        "llama.embedding_length": 1024,
        "llama.block_count": 16,
        "llama.attention.head_count": 16,
        "llama.attention.head_count_kv": 8,
        "llama.max_position_embeddings": 64000,
    })
    assert read_gguf_meta(p).context_length == 64000


def test_unknown_architecture_falls_back_to_llama(tmp_path):
    p = _write(tmp_path, {
        "general.architecture": "unknownarch",
        "llama.embedding_length": 512,
        "llama.block_count": 8,
        "llama.attention.head_count": 8,
        "llama.attention.head_count_kv": 8,
        "llama.context_length": 4096,
    })
    meta = read_gguf_meta(p)
    assert meta == GgufMeta(512, 8, 8, 8, 4096)


def test_missing_context_length_is_none(tmp_path):
    p = _write(tmp_path, {
        "general.architecture": "llama",
        "llama.embedding_length": 512,
        "llama.block_count": 8,
        "llama.attention.head_count": 8,
        "llama.attention.head_count_kv": 8,
    })
    meta = read_gguf_meta(p)
    assert meta.context_length is None
    assert meta.complete  # head fields still fine without context


def test_garbage_file_returns_empty_meta(tmp_path):
    p = tmp_path / "junk.gguf"
    p.write_bytes(b"not a gguf at all")
    assert read_gguf_meta(p) == GgufMeta()


def test_missing_file_returns_empty_meta(tmp_path):
    assert read_gguf_meta(tmp_path / "absent.gguf") == GgufMeta()


def test_truncated_header_returns_empty_meta(tmp_path):
    p = tmp_path / "trunc.gguf"
    p.write_bytes(b"GGUF" + b"\x03" * 8)
    assert read_gguf_meta(p) == GgufMeta()


@pytest.mark.parametrize("head_kv,expected", [(8, 8), (0, 0)])
def test_zero_kv_heads_is_still_read(tmp_path, head_kv, expected):
    p = _write(tmp_path, {
        "general.architecture": "llama",
        "llama.embedding_length": 512,
        "llama.block_count": 8,
        "llama.attention.head_count": 8,
        "llama.attention.head_count_kv": head_kv,
    })
    assert read_gguf_meta(p).n_head_kv == expected
