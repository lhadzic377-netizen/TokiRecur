"""Unit tests for the GGUF split bookkeeping reader.

The bug this guards: the reader has to survive the metadata keys that
come *before* the split keys, including arrays of strings (the tokenizer
vocabulary) and arrays of ints. A reader that mis-parses the array
header — element type before count, not the other way round — reads
past the end of the vocabulary and then reports the split as absent,
which is how a model with a missing part looks like a model that is not
a split at all.
"""

import struct
from pathlib import Path

from tokirecur.ggufsplit import (
    SplitInfo,
    expected_part_names,
    read_split_info,
    read_split_info_bytes,
)

_U8, _I32, _F32, _BOOL, _STRING, _ARRAY, _U16, _U32, _U64 = (
    0, 5, 6, 7, 8, 9, 2, 4, 10,
)


def _string(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, vtype: int, payload: bytes) -> bytes:
    return _string(key) + struct.pack("<I", vtype) + payload


def _header(entries: list[bytes]) -> bytes:
    out = bytearray(b"GGUF")
    out += struct.pack("<I", 3)
    out += struct.pack("<Q", 0)  # tensor count
    out += struct.pack("<Q", len(entries))
    for e in entries:
        out += e
    return bytes(out)


def _split_header(*, count: int, index: int, tensors: int = 1043,
                  extra: list[bytes] | None = None) -> bytes:
    entries = list(extra or [])
    entries.append(_kv("split.no", _U16, struct.pack("<H", index)))
    entries.append(_kv("split.count", _U16, struct.pack("<H", count)))
    entries.append(_kv("split.tensors.count", _I32, struct.pack("<i", tensors)))
    return _header(entries)


def _array_of_strings(values: list[str]) -> bytes:
    payload = struct.pack("<I", _STRING) + struct.pack("<Q", len(values))
    for v in values:
        payload += _string(v)
    return _kv("tokenizer.ggml.tokens", _ARRAY, payload)


def _array_of_ints(values: list[int]) -> bytes:
    payload = struct.pack("<I", _I32) + struct.pack("<Q", len(values))
    for v in values:
        payload += struct.pack("<i", v)
    return _kv("some.ints", _ARRAY, payload)


def test_reads_split_count_and_index(tmp_path):
    p = tmp_path / "m-00001-of-00006.gguf"
    p.write_bytes(_split_header(count=6, index=0))
    info = read_split_info(p)
    assert info is not None
    assert info.count == 6
    assert info.index == 0
    assert info.tensors_count == 1043


def test_reads_the_last_part_by_its_own_header(tmp_path):
    """The Engram file is shard 6 under another name; its header says so."""
    p = tmp_path / "Engram-IQ4_NL.gguf"
    p.write_bytes(_split_header(count=6, index=5))
    info = read_split_info(p)
    assert info is not None
    assert (info.count, info.index) == (6, 5)


def test_survives_a_string_array_before_the_split_keys(tmp_path):
    """The tokenizer vocabulary precedes the split keys and is huge."""
    p = tmp_path / "m-00001-of-00003.gguf"
    p.write_bytes(_split_header(
        count=3, index=0,
        extra=[_array_of_strings([f"tok{i}" for i in range(500)])],
    ))
    info = read_split_info(p)
    assert info is not None
    assert info.count == 3


def test_survives_an_int_array_before_the_split_keys(tmp_path):
    p = tmp_path / "m-00001-of-00003.gguf"
    p.write_bytes(_split_header(
        count=3, index=1,
        extra=[_array_of_ints(list(range(200)))],
    ))
    info = read_split_info(p)
    assert info is not None
    assert (info.count, info.index) == (3, 1)


def test_survives_mixed_metadata_before_the_split_keys(tmp_path):
    p = tmp_path / "m-00001-of-00002.gguf"
    p.write_bytes(_split_header(
        count=2, index=0,
        extra=[
            _kv("general.architecture", _STRING, _string("deepseek41")),
            _kv("deepseek41.block_count", _I32, struct.pack("<i", 40)),
            _kv("deepseek41.rope.freq_base", _F32, struct.pack("<f", 1.0)),
            _kv("deepseek41.attention.kv_source_layers", _BOOL, b"\x01"),
            _array_of_ints([1, 2, 3]),
            _array_of_strings(["a", "b"]),
        ],
    ))
    info = read_split_info(p)
    assert info is not None
    assert info.count == 2


def test_a_file_that_is_not_a_split_reports_nothing(tmp_path):
    p = tmp_path / "plain.gguf"
    p.write_bytes(_header([
        _kv("general.architecture", _STRING, _string("llama")),
    ]))
    assert read_split_info(p) is None


def test_a_non_gguf_file_reports_nothing(tmp_path):
    p = tmp_path / "not.gguf"
    p.write_bytes(b"NOPE" + b"\x00" * 64)
    assert read_split_info(p) is None


def test_a_truncated_header_reports_nothing(tmp_path):
    p = tmp_path / "short.gguf"
    p.write_bytes(b"GGUF" + b"\x03\x00\x00\x00")
    assert read_split_info(p) is None


def test_read_split_info_bytes_matches_the_file_reader(tmp_path):
    p = tmp_path / "m-00001-of-00006.gguf"
    p.write_bytes(_split_header(count=6, index=2))
    from_bytes = read_split_info_bytes(p.read_bytes())
    from_file = read_split_info(p)
    assert from_bytes is not None and from_file is not None
    assert from_bytes == from_file


def test_expected_names_keep_the_digit_widths(tmp_path):
    """``-00001-of-00006`` is six parts of five-digit numbers."""
    p = tmp_path / "Model-IQ2_XXS-00001-of-00006.gguf"
    info = SplitInfo(count=6, index=0, tensors_count=1043)
    names = expected_part_names(p, info)
    assert len(names) == 6
    assert names[0].endswith("Model-IQ2_XXS-00001-of-00006.gguf")
    assert names[-1].endswith("Model-IQ2_XXS-00006-of-00006.gguf")


def test_expected_names_use_the_declared_count_not_the_filename(tmp_path):
    """llama.cpp takes the count from the header, not from the filename.

    It formats every part as ``%s-%05d-of-%05d.gguf`` with the count
    the header declares, so a repository that spells the total
    differently gets the names llama.cpp actually opens.
    """
    p = tmp_path / "Model-00001-of-00006.gguf"
    info = SplitInfo(count=5, index=0, tensors_count=10)
    names = expected_part_names(p, info)
    assert len(names) == 5
    assert names[-1].endswith("Model-00005-of-00005.gguf")


def test_expected_names_use_five_digits_like_llama_cpp(tmp_path):
    """llama.cpp always writes five digits, so the names must too."""
    p = tmp_path / "Model-00001-of-00006.gguf"
    info = SplitInfo(count=6, index=0, tensors_count=10)
    names = expected_part_names(p, info)
    assert names[-1].endswith("Model-00006-of-00006.gguf")


def test_expected_names_keep_the_folder(tmp_path):
    p = tmp_path / "IQ2_XXS" / "Model-00001-of-00006.gguf"
    p.parent.mkdir()
    p.write_bytes(_split_header(count=6, index=0))
    names = expected_part_names(p, SplitInfo(count=6, index=0, tensors_count=1))
    assert all(Path(n).parent.name == "IQ2_XXS" for n in names)
    assert Path(names[-1]).name == "Model-00006-of-00006.gguf"
