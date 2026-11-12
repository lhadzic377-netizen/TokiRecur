"""Split GGUF bookkeeping: which parts a model needs, and which are missing.

A multi-file GGUF split is a set of parts named
``<base>-<NNNN>-of-<NNNN>.gguf``. llama.cpp loads part 1 and then opens
parts 2..N by that exact name, so every part has to be on disk under
that name or the server refuses to start.

Some repos break that convention. DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder
publishes five parts named ``-of-00006`` and puts the sixth part — the
110 GB Engram lookup table — in the repository root under a completely
different name. The GGUF header still says ``split.count = 6``, so
llama.cpp looks for ``-00006-of-00006.gguf`` and finds nothing:

    failed to open GGUF file '...-00006-of-00006.gguf' (No such file or directory)

This module reads the split bookkeeping out of a GGUF header and works
out what is missing, so the pull can say what to fetch and link rather
than leaving a half-downloaded model that cannot boot.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

_MAGIC = b"GGUF"

# The llama.cpp split naming convention: <base>-<NNNNN>-of-<NNNNN>.gguf.
# llama.cpp formats both numbers with "%05d", so five digits is the
# width that matters; the regex accepts three or more so a repository
# that spelled them narrower is still recognised.
_PART_RE = re.compile(
    r"^(?P<base>.+)-(?P<idx>\d{3,})-of-(?P<total>\d{3,})\.gguf$"
)

# GGUF value types (subset: the ones a split header can hold).
_UINT8 = 0
_INT8 = 1
_UINT16 = 2
_INT16 = 3
_UINT32 = 4
_INT32 = 5
_FLOAT32 = 6
_BOOL = 7
_STRING = 8
_ARRAY = 9
_UINT64 = 10
_FLOAT64 = 11
_INT64 = 12

_ARRAY_FIXED = {
    _UINT8: 1, _INT8: 1, _UINT16: 2, _INT16: 2,
    _UINT32: 4, _INT32: 4, _FLOAT32: 4, _BOOL: 1,
    _UINT64: 8, _FLOAT64: 8, _INT64: 8,
}


class SplitInfoError(Exception):
    """The header could not be read as a GGUF split header."""


# The tensor llama.cpp reads on demand instead of keeping resident. The
# DeepSeek-V4.1 Engram lookup table lives here and is 110 GB on its
# own: loaded eagerly it needs more memory than the machine has, and
# the server dies in the OOM killer rather than refusing to start.
LAZY_TENSOR_NAMES = ("per_layer_token_embd.weight",)


@dataclass(frozen=True)
class SplitInfo:
    """What a GGUF says about its own place in a split.

    ``count`` is how many parts the split has in total, ``index`` is
    this file's 0-based position in it, and ``tensors_count`` is the
    number of tensors the whole split holds.
    """

    count: int
    index: int
    tensors_count: int


def _read_string(fh) -> str:
    raw = fh.read(8)
    if len(raw) != 8:
        raise SplitInfoError("truncated header")
    (length,) = struct.unpack("<Q", raw)
    if length > 1 << 20:
        raise SplitInfoError("implausible string length")
    data = fh.read(length)
    if len(data) != length:
        raise SplitInfoError("truncated string")
    return data.decode("utf-8", "replace")


def _skip_value(fh, vtype: int, size: int) -> None:
    """Advance past one value without materialising it."""
    if vtype == _STRING:
        _read_string(fh)
        return
    if vtype == _ARRAY:
        # The array header is the element type (u32) then the count (u64).
        raw = fh.read(4)
        if len(raw) != 4:
            raise SplitInfoError("truncated array type")
        (elem_type,) = struct.unpack("<I", raw)
        raw = fh.read(8)
        if len(raw) != 8:
            raise SplitInfoError("truncated array count")
        (count,) = struct.unpack("<Q", raw)
        if elem_type == _STRING:
            for _ in range(count):
                _read_string(fh)
            return
        width = _ARRAY_FIXED.get(elem_type)
        if width is None:
            raise SplitInfoError(f"unsupported array element type {elem_type}")
        # count is a u64: count*width can overflow the platform's seek
        # offset (OverflowError). An array claiming to run past the end of
        # the file is a corrupt header, so reject it before seeking.
        remaining = size - fh.tell()
        if count > remaining // width:
            raise SplitInfoError("array extends past the end of the file")
        fh.seek(count * width, 1)
        return
    width = _ARRAY_FIXED.get(vtype)
    if width is None:
        raise SplitInfoError(f"unsupported value type {vtype}")
    fh.seek(width, 1)


def _read_u16(fh) -> int:
    raw = fh.read(2)
    if len(raw) != 2:
        raise SplitInfoError("truncated value")
    (value,) = struct.unpack("<H", raw)
    return value


def read_split_info(path: Path | str) -> SplitInfo | None:
    """Read a GGUF's split bookkeeping, or None when it is not a split.

    Only the header is read, so this costs a few kilobytes however large
    the model file is.
    """
    try:
        with open(path, "rb") as fh:
            if fh.read(4) != _MAGIC:
                return None
            version = struct.unpack("<I", fh.read(4))[0]
            count_fmt = "<I" if version < 3 else "<Q"
            fh.read(struct.calcsize(count_fmt))  # tensor_count
            raw = fh.read(struct.calcsize(count_fmt))
            if len(raw) != struct.calcsize(count_fmt):
                return None
            (kv_count,) = struct.unpack(count_fmt, raw)
            size = fh.seek(0, 2)
            fh.seek(0)
            fh.read(4 + 4)
            fh.read(struct.calcsize(count_fmt) * 2)
            total = index = tensors = None
            for _ in range(kv_count):
                key = _read_string(fh)
                raw = fh.read(4)
                if len(raw) != 4:
                    return None
                (vtype,) = struct.unpack("<I", raw)
                if key == "split.count" and vtype == _UINT16:
                    total = _read_u16(fh)
                elif key == "split.no" and vtype == _UINT16:
                    index = _read_u16(fh)
                elif key == "split.tensors.count" and vtype in (_UINT32, _INT32):
                    raw = fh.read(4)
                    if len(raw) == 4:
                        tensors = struct.unpack("<i" if vtype == _INT32 else "<I", raw)[0]
                else:
                    _skip_value(fh, vtype, size)
            if total is None or index is None:
                return None
            return SplitInfo(
                count=total, index=index, tensors_count=tensors or 0,
            )
    except (OSError, struct.error, SplitInfoError):
        return None


def read_split_info_bytes(data: bytes) -> SplitInfo | None:
    """Read split bookkeeping out of a GGUF header held in memory.

    Used for a remote file: a range request for the first few tens of
    kilobytes is enough to see the header, and the header says which
    part of a split the file is — which is how a part published under a
    different name is still recognised as part of the split.
    """
    import io

    fh = io.BytesIO(data)
    try:
        if fh.read(4) != _MAGIC:
            return None
        version = struct.unpack("<I", fh.read(4))[0]
        count_fmt = "<I" if version < 3 else "<Q"
        fh.read(struct.calcsize(count_fmt))  # tensor_count
        raw = fh.read(struct.calcsize(count_fmt))
        if len(raw) != struct.calcsize(count_fmt):
            return None
        (kv_count,) = struct.unpack(count_fmt, raw)
        total = index = tensors = None
        for _ in range(kv_count):
            key = _read_string(fh)
            raw = fh.read(4)
            if len(raw) != 4:
                return None
            (vtype,) = struct.unpack("<I", raw)
            if key == "split.count" and vtype == _UINT16:
                total = _read_u16(fh)
            elif key == "split.no" and vtype == _UINT16:
                index = _read_u16(fh)
            elif key == "split.tensors.count" and vtype in (_UINT32, _INT32):
                raw = fh.read(4)
                if len(raw) == 4:
                    tensors = struct.unpack("<i" if vtype == _INT32 else "<I", raw)[0]
            else:
                _skip_value(fh, vtype, len(data))
        if total is None or index is None:
            return None
        return SplitInfo(count=total, index=index, tensors_count=tensors or 0)
    except (struct.error, SplitInfoError):
        return None


def expected_part_names(first_part: Path | str, info: SplitInfo) -> list[str]:
    """The names llama.cpp will look for, in order.

    This mirrors llama.cpp exactly: ``llama_split_prefix`` strips the
    trailing ``-<NNNNN>-of-<NNNNN>.gguf`` from the file it was given to
    form the prefix, and ``llama_split_path`` then formats every part
    as ``%s-%05d-of-%05d.gguf`` — five digits, always, with the count
    taken from the file's own header rather than from its name. A
    repository whose filename disagrees with its header therefore gets
    the names llama.cpp actually looks for, not the ones it spells.
    """
    path = Path(first_part)
    name = path.name
    m = _PART_RE.match(name)
    if not m:
        return [name]
    prefix = name[: m.start("idx") - 1]
    total = info.count
    return [
        str(path.with_name(f"{prefix}-{i:05d}-of-{total:05d}.gguf"))
        for i in range(1, total + 1)
    ]


def lazy_tensor_bytes(first_part: Path | str) -> int:
    """Bytes of the tensor llama.cpp reads on demand, across every shard.

    The tensor lives in whichever shard holds it, so every part of the
    split is checked. Only the header directories are read, so this
    costs a few kilobytes per shard however large the model is.

    Zero means the model has no such tensor — nothing to keep lazy.
    """
    from .ggufmeta import read_gguf_tensor_sizes

    path = Path(first_part)
    info = read_split_info(path)
    if info is None or info.count <= 1:
        return 0
    total = 0
    for name in expected_part_names(path, info):
        part = Path(name)
        if not part.exists():
            continue
        for tensor, size in read_gguf_tensor_sizes(part).items():
            if tensor in LAZY_TENSOR_NAMES:
                total += size
    return total
