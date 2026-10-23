"""Reading a GGUF header must not cost a minute per model.

The fit planner reads a model's header to work out its KV cache, and the
config writers ask for the fit of every model in the registry, once per
client. So a registry of thirty-odd models had its headers read hundreds
of times over, and `toki model use` — which the launcher calls before it
starts anything — spent two minutes doing it.

Two things fixed it: the header is parsed once and remembered, and the
parser stops as soon as it has what it came for.
"""

import struct
import time

import pytest

from tokirecur import ggufmeta

_I32 = 5
_STRING = 8
_ARRAY = 9


def _gguf(kv: list[tuple[str, object]]) -> bytes:
    """A GGUF v3 header. Values may be int, str, or a list of str."""
    out = bytearray(b"GGUF")
    out += struct.pack("<I", 3)     # version
    out += struct.pack("<Q", 0)     # tensor count
    out += struct.pack("<Q", len(kv))
    for key, value in kv:
        b = key.encode()
        out += struct.pack("<Q", len(b)) + b
        if isinstance(value, int):
            out += struct.pack("<I", _I32) + struct.pack("<i", value)
        elif isinstance(value, str):
            sb = value.encode()
            out += struct.pack("<I", _STRING) + struct.pack("<Q", len(sb)) + sb
        elif isinstance(value, list):
            out += struct.pack("<I", _ARRAY)
            out += struct.pack("<I", _STRING)   # element type
            out += struct.pack("<Q", len(value))
            for item in value:
                sb = item.encode()
                out += struct.pack("<Q", len(sb)) + sb
        else:
            raise AssertionError(f"unsupported {value!r}")
    return bytes(out)


@pytest.fixture(autouse=True)
def clear_caches():
    """Every test starts with an empty cache."""
    ggufmeta._META_CACHE.clear()
    ggufmeta._KIND_CACHE.clear()
    ggufmeta._TENSOR_CACHE.clear()
    yield
    ggufmeta._META_CACHE.clear()
    ggufmeta._KIND_CACHE.clear()
    ggufmeta._TENSOR_CACHE.clear()


def test_the_header_is_parsed_once_not_once_per_caller(tmp_path, monkeypatch):
    """The second reader gets the answer without touching the file."""
    path = tmp_path / "m.gguf"
    path.write_bytes(_gguf([
        ("general.architecture", "qwen35"),
        ("qwen35.embedding_length", 2048),
        ("qwen35.block_count", 40),
        ("qwen35.attention.head_count", 16),
        ("qwen35.attention.head_count_kv", 8),
    ]))

    calls = []
    real = ggufmeta._read_gguf_meta
    monkeypatch.setattr(
        ggufmeta, "_read_gguf_meta",
        lambda p: (calls.append(p), real(p))[1],
    )

    first = ggufmeta.read_gguf_meta(path)
    second = ggufmeta.read_gguf_meta(path)
    third = ggufmeta.read_gguf_meta(path)

    assert len(calls) == 1, "the header was read more than once"
    assert first == second == third
    assert first.n_embd == 2048
    assert first.n_layer == 40


def test_the_cache_follows_the_file(tmp_path):
    """A file that changes is read again; one that does not, is not."""
    path = tmp_path / "m.gguf"
    path.write_bytes(_gguf([("general.architecture", "qwen35")]))
    assert ggufmeta.read_gguf_kind(path).architecture == "qwen35"

    # A different file, same path: the cache must not answer for it.
    path.write_bytes(_gguf([("general.architecture", "llama")]))
    assert ggufmeta.read_gguf_kind(path).architecture == "llama"


def test_the_parser_stops_when_it_has_what_it_came_for(tmp_path):
    """A huge tokenizer must not be walked to look for a key that is not there.

    The architecture is the first key in the header, and the only other
    key the kind reader wants is the audiocpp spec — which only an audiocpp
    file can carry. So a normal model answers after its first key, and
    the tokenizer behind it is never touched.
    """
    path = tmp_path / "big.gguf"
    tokens = [f"token-{i:07d}" for i in range(120_000)]
    path.write_bytes(_gguf([
        ("general.architecture", "qwen35"),
        ("tokenizer.ggml.tokens", tokens),
    ]))

    start = time.monotonic()
    kind = ggufmeta.read_gguf_kind(path)
    elapsed = time.monotonic() - start

    assert kind.architecture == "qwen35"
    # Walking 120k tokens one length-prefix at a time is tens of
    # milliseconds; a second is a sign the walk is still going.
    assert elapsed < 1.0, f"the header walk took {elapsed:.2f}s"


def test_the_metadata_reader_skips_the_tokenizer(tmp_path):
    """Six numbers are wanted. A hundred thousand strings are not read."""
    path = tmp_path / "big.gguf"
    tokens = [f"token-{i:07d}" for i in range(120_000)]
    path.write_bytes(_gguf([
        ("general.architecture", "qwen35"),
        ("qwen35.embedding_length", 2048),
        ("qwen35.block_count", 40),
        ("qwen35.attention.head_count", 16),
        ("qwen35.attention.head_count_kv", 8),
        ("qwen35.context_length", 262144),
        ("tokenizer.ggml.tokens", tokens),
    ]))

    start = time.monotonic()
    meta = ggufmeta.read_gguf_meta(path)
    elapsed = time.monotonic() - start

    assert meta.n_embd == 2048
    assert meta.n_layer == 40
    assert meta.n_head == 16
    assert meta.n_head_kv == 8
    assert meta.context_length == 262144
    assert meta.complete
    assert elapsed < 1.5, f"reading six numbers took {elapsed:.2f}s"


def test_a_value_that_runs_past_the_end_of_the_file_is_rejected(tmp_path):
    """A header that lies about its own size must not be walked forever."""
    path = tmp_path / "liar.gguf"
    body = bytearray(b"GGUF")
    body += struct.pack("<I", 3)
    body += struct.pack("<Q", 0)
    body += struct.pack("<Q", 2)          # two keys
    key = b"tokenizer.ggml.tokens"
    body += struct.pack("<Q", len(key)) + key
    body += struct.pack("<I", _ARRAY)
    body += struct.pack("<I", _STRING)
    body += struct.pack("<Q", 2_000_000_000)   # a count the file cannot hold
    path.write_bytes(bytes(body))

    # It must come back as "no idea", and it must come back quickly.
    start = time.monotonic()
    kind = ggufmeta.read_gguf_kind(path)
    elapsed = time.monotonic() - start
    assert elapsed < 2.0, f"a lying header took {elapsed:.2f}s"
    assert kind.architecture is None
