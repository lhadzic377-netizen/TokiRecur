"""Unit tests for the split-part resolver and the lazy-load decision.

Two failures these guard:

1. A repository that publishes a split part under a different name leaves
   the model half-downloaded and unable to boot. The part is found by
   reading its own header (``split.no``), not by guessing from filenames.

2. A model with a tensor too big to hold resident has to keep it on disk.
   Loaded eagerly, a 110 GB table kills the server in the OOM killer
   before it can report anything.
"""

import struct

import pytest

from tokirecur.ctl import (
    _model_needs_lazy_load,
    _model_weight_gb,
    _resolve_split_gaps,
)
from tokirecur.ggufsplit import (
    LAZY_TENSOR_NAMES,
    lazy_tensor_bytes,
    read_split_info,
)

# ── GGUF header builders ───────────────────────────────────────────────

_MAGIC = b"GGUF"
_U16, _U32, _I32, _STRING = 2, 4, 5, 8


def _string(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, vtype: int, payload: bytes) -> bytes:
    return _string(key) + struct.pack("<I", vtype) + payload


def _header(entries: list[bytes], tensors: list[tuple[str, int]] | None = None) -> bytes:
    """A GGUF v3 header. ``tensors`` is (name, byte size) per tensor."""
    tensors = tensors or []
    out = bytearray(_MAGIC)
    out += struct.pack("<I", 3)
    out += struct.pack("<Q", len(tensors))
    out += struct.pack("<Q", len(entries))
    for e in entries:
        out += e
    # Tensor directory: name, n_dims, dims, type, offset. The offsets
    # are what read_gguf_tensor_sizes turns into sizes.
    offset = 0
    for name, size in tensors:
        out += _string(name)
        out += struct.pack("<I", 1)
        out += struct.pack("<Q", max(size, 1))
        out += struct.pack("<I", 20)  # IQ4_NL, size only matters via layout
        out += struct.pack("<Q", offset)
        offset += size
    return bytes(out)


def _split_entries(count: int, index: int, tensors: int = 1043) -> list[bytes]:
    return [
        _kv("split.no", _U16, struct.pack("<H", index)),
        _kv("split.count", _U16, struct.pack("<H", count)),
        _kv("split.tensors.count", _I32, struct.pack("<i", tensors)),
    ]


def _write_part(path, *, count: int, index: int, tensors=None, size: int = 0):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = _header(_split_entries(count, index), tensors)
    if size:
        body += b"\x00" * size
    path.write_bytes(body)


# ── the resolver ───────────────────────────────────────────────────────

def test_resolver_links_a_part_published_under_another_name(tmp_path, monkeypatch):
    """The Engram file is shard 6 under a different name; it gets linked."""
    local = tmp_path / "model"
    first = local / "IQ2_XXS" / "Model-IQ2_XXS-00001-of-00006.gguf"
    _write_part(first, count=6, index=0)
    for i in (2, 3, 4, 5):
        _write_part(local / "IQ2_XXS" / f"Model-IQ2_XXS-0000{i}-of-00006.gguf",
                    count=6, index=i - 1)
    # The missing part, published under its own name.
    engram = local / "Engram-IQ4_NL.gguf"
    _write_part(engram, count=6, index=5)

    calls = []

    def fake_header(repo_id, path, endpoint=None):
        calls.append(path)
        if path == "Engram-IQ4_NL.gguf":
            return read_split_info(engram)
        return None

    monkeypatch.setattr("tokirecur.ctl._remote_split_header", fake_header)

    def fake_download(repo_id, filename, local_dir=None, endpoint=None):
        return str(local / filename)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    linked = _resolve_split_gaps(
        "owner/repo",
        ["IQ2_XXS/Model-IQ2_XXS-00001-of-00006.gguf", "Engram-IQ4_NL.gguf"],
        "IQ2_XXS/Model-IQ2_XXS-00001-of-00006.gguf",
        local,
    )
    assert len(linked) == 1
    target = local / "IQ2_XXS" / "Model-IQ2_XXS-00006-of-00006.gguf"
    assert target.is_symlink()
    assert target.resolve() == engram.resolve()


def test_resolver_does_nothing_when_every_part_is_present(tmp_path, monkeypatch):
    local = tmp_path / "model"
    first = local / "M-00001-of-00003.gguf"
    _write_part(first, count=3, index=0)
    for i in (2, 3):
        _write_part(local / f"M-0000{i}-of-00003.gguf", count=3, index=i - 1)

    def fail(*a, **k):
        raise AssertionError("must not look for remote headers")

    monkeypatch.setattr("tokirecur.ctl._remote_split_header", fail)
    linked = _resolve_split_gaps(
        "owner/repo", ["M-00001-of-00003.gguf"],
        "M-00001-of-00003.gguf", local,
    )
    assert linked == []


def test_resolver_ignores_a_file_that_is_not_part_of_the_split(tmp_path, monkeypatch):
    """A different quantisation of the same model is not our shard 6."""
    local = tmp_path / "model"
    first = local / "M-00001-of-00003.gguf"
    _write_part(first, count=3, index=0)
    _write_part(local / "M-00002-of-00003.gguf", count=3, index=1)
    # A file whose header says it is part of a DIFFERENT split.
    other = local / "Other-IQ4_NL.gguf"
    _write_part(other, count=9, index=8)

    monkeypatch.setattr(
        "tokirecur.ctl._remote_split_header",
        lambda repo_id, path, endpoint=None: read_split_info(other),
    )
    linked = _resolve_split_gaps(
        "owner/repo", ["M-00001-of-00003.gguf", "Other-IQ4_NL.gguf"],
        "M-00001-of-00003.gguf", local,
    )
    assert linked == []


def test_resolver_ignores_a_model_that_is_not_a_split(tmp_path, monkeypatch):
    local = tmp_path / "model"
    plain = local / "plain.gguf"
    plain.parent.mkdir(parents=True)
    plain.write_bytes(_header([_kv("general.architecture", _STRING, _string("llama"))]))

    monkeypatch.setattr(
        "tokirecur.ctl._remote_split_header",
        lambda *a, **k: pytest.fail("must not probe remote headers"),
    )
    linked = _resolve_split_gaps(
        "owner/repo", ["plain.gguf"], "plain.gguf", local,
    )
    assert linked == []


# ── the lazy-load decision ─────────────────────────────────────────────

def _spec(gguf, name="m"):
    from tokirecur.ctl import ModelSpec

    raw = {
        "name": name, "label": name.capitalize(), "gguf": str(gguf),
        "llama_port": 8088, "daemon_port": 8787, "ctx_size": 4096,
        "parallel": 1, "kv": "f16", "max_tokens": 2048, "model_id": name,
    }
    return ModelSpec(raw=raw, **raw)


def test_lazy_tensor_bytes_finds_the_tensor_across_the_split(tmp_path):
    """The Engram table lives in the last shard; the others do not have it."""
    local = tmp_path
    first = local / "M-00001-of-00003.gguf"
    _write_part(first, count=3, index=0)
    _write_part(local / "M-00002-of-00003.gguf", count=3, index=1)
    big = 5 * (1 << 30)
    _write_part(
        local / "M-00003-of-00003.gguf", count=3, index=2,
        tensors=[(LAZY_TENSOR_NAMES[0], big)], size=0,
    )
    assert lazy_tensor_bytes(first) > 0


def test_lazy_tensor_bytes_is_zero_without_the_tensor(tmp_path):
    local = tmp_path
    first = local / "M-00001-of-00002.gguf"
    _write_part(first, count=2, index=0)
    _write_part(local / "M-00002-of-00002.gguf", count=2, index=1,
                tensors=[("blk.0.attn_q.weight", 1024)], size=1024)
    assert lazy_tensor_bytes(first) == 0


def test_lazy_tensor_bytes_is_zero_for_a_plain_model(tmp_path):
    plain = tmp_path / "plain.gguf"
    plain.write_bytes(_header([_kv("general.architecture", _STRING, _string("llama"))]))
    assert lazy_tensor_bytes(plain) == 0


def test_model_needs_lazy_load_is_false_without_a_lazy_tensor(tmp_path):
    gguf = tmp_path / "plain.gguf"
    gguf.write_bytes(_header([_kv("general.architecture", _STRING, _string("llama"))]))
    spec = _spec(gguf, "plain")
    assert _model_needs_lazy_load(spec) is False


def test_weight_gb_counts_every_part_of_the_split(tmp_path):
    """Sizing a split from its first shard alone reports 5.8 MB for 72 GB."""
    local = tmp_path
    first = local / "M-00001-of-00003.gguf"
    _write_part(first, count=3, index=0, size=8 << 20)
    _write_part(local / "M-00002-of-00003.gguf", count=3, index=1, size=32 << 20)
    _write_part(local / "M-00003-of-00003.gguf", count=3, index=2, size=16 << 20)
    spec = _spec(first, "split")
    weight = _model_weight_gb(spec)
    # 8 + 32 + 16 MiB of payload, and the metadata shards count too.
    assert weight > (56 << 20) / 1e9
    # The whole split, not the first shard alone.
    assert weight > 0.05
