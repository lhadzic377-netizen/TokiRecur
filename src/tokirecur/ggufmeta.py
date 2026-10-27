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

import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MAGIC = b"GGUF"

# GGUF value types (v3): https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STRING, _ARRAY, _U64, _I64, _F64 = range(13)


@dataclass(frozen=True)
class GgufRope:
    """The position-encoding configuration the model carries itself.

    A GGUF converted from a model trained with YaRN keeps the scaling in
    its metadata: the type, the factor the positions are stretched by,
    and the window the model was actually trained at. llama.cpp reads
    those and applies them by itself — no ``--rope-scaling`` on the
    command line — so a model with this metadata can address
    ``base × factor`` positions on its own.

    The same keys also appear on models that ship *linear* or *dynamic*
    scaling. Those stretch the positions too, but they are not YaRN: the
    distinction matters because YaRN is the one that also rescales the
    attention, and the one the rest of the system can reason about.
    """

    scaling_type: str | None = None
    factor: float | None = None
    original_context_length: int | None = None
    yarn_log_multiplier: float | None = None
    yarn_ext_factor: float | None = None
    yarn_attn_factor: float | None = None
    yarn_beta_fast: float | None = None
    yarn_beta_slow: float | None = None

    @property
    def is_yarn(self) -> bool:
        return (self.scaling_type or "").strip().lower() == "yarn"

    @property
    def stretches(self) -> bool:
        """True when the model's own metadata extends its own window."""
        return self.is_yarn and self.factor is not None and self.factor > 1.0

    def effective_context(self, base: int | None) -> int | None:
        """The window the model's own rope configuration gives it.

        ``base`` is the GGUF's own ``context_length`` — the window the
        weights were trained for. The factor stretches the positions by
        that much, so the model can address ``base × factor`` of them.
        """
        if not self.stretches or not base:
            return None
        return int(base * float(self.factor or 1.0))

    def describe(self) -> str:
        """One line for a launcher row: what the model does with its rope."""
        if not self.scaling_type:
            return ""
        if self.is_yarn and self.factor:
            return f"yarn {self.factor:g}×"
        return str(self.scaling_type)


@dataclass(frozen=True)
class GgufMeta:
    n_embd: int | None = None
    n_layer: int | None = None
    n_head: int | None = None
    n_head_kv: int | None = None
    context_length: int | None = None
    rope: GgufRope = GgufRope()

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


def _cache_key(path: Path | str) -> tuple[str, int, int] | None:
    """A key for a file's contents that changes when the file does.

    A GGUF's header is read over and over: the fit planner reads it for
    every model to work out the KV cache, the kind reader reads it to
    find the architecture, and the config writers ask for the fit of
    every model in the registry, once per client. None of that changes
    while the file does not.
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (str(path), st.st_size, st.st_mtime_ns)


#: Parsed header metadata, keyed by (path, size, mtime).
_META_CACHE: dict[tuple[str, int, int], GgufMeta] = {}
_KIND_CACHE: dict[tuple[str, int, int], GgufKind] = {}
_TENSOR_CACHE: dict[tuple[str, int, int], list[str]] = {}


def read_gguf_meta(path: Path | str) -> GgufMeta:
    """Read the metadata we need from a GGUF header, or all-``None``."""
    key = _cache_key(path)
    if key is not None:
        cached = _META_CACHE.get(key)
        if cached is not None:
            return cached
    result = _read_gguf_meta(path)
    if key is not None:
        _META_CACHE[key] = result
    return result


#: The metadata this module actually needs, matched on the tail of the
#: key so the architecture prefix need not be known in advance. Every
#: other key is skipped rather than read: the tokenizer alone is a
#: hundred thousand strings, and building them all to look at six
#: numbers is what made the first read of a model take half a second.
_META_SUFFIXES = (
    ".embedding_length",
    ".block_count",
    ".attention.head_count",
    ".attention.head_count_kv",
    ".context_length",
    ".max_position_embeddings",
    # The position encoding the model ships with. A model trained with
    # YaRN carries the factor and the window it was trained at, and
    # llama.cpp applies them without being told to — so reading them
    # is what lets a pulled model use the window it actually has.
    ".rope.scaling.type",
    ".rope.scaling.factor",
    ".rope.scaling.original_context_length",
    ".rope.scaling.yarn_log_multiplier",
    ".rope.scaling.yarn_ext_factor",
    ".rope.scaling.yarn_attn_factor",
    ".rope.scaling.yarn_beta_fast",
    ".rope.scaling.yarn_beta_slow",
)


#: GGUF rope keys -> the GgufRope field they fill. The GGUF names them
#: after llama.cpp's own key list (src/llama-arch.cpp), so a model
#: converted from a YaRN-trained one carries the factor and the trained
#: window and llama.cpp applies them with no flags.
_ROPE_KEYS: dict[str, str] = {
    "rope.scaling.type": "type",
    "rope.scaling.factor": "factor",
    "rope.scaling.original_context_length": "orig_ctx",
    "rope.scaling.yarn_log_multiplier": "log_mul",
    "rope.scaling.yarn_ext_factor": "ext_factor",
    "rope.scaling.yarn_attn_factor": "attn_factor",
    "rope.scaling.yarn_beta_fast": "beta_fast",
    "rope.scaling.yarn_beta_slow": "beta_slow",
}


def _read_gguf_meta(path: Path | str) -> GgufMeta:
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
            size = os.fstat(fh.fileno()).st_size
            for _ in range(kv_count):
                key = _read_string(fh)
                vtype = _read_bytes(fh, "<I")
                if key == "general.architecture":
                    value = _read_value(fh, vtype)
                    arch = str(value) if isinstance(value, str) else None
                elif key.endswith(_META_SUFFIXES):
                    value = _read_value(fh, vtype)
                    meta[key] = value
                else:
                    _skip_value(fh, vtype, size)
            # Metadata keys are architecture-prefixed (e.g. `qwen2.*`,
            # `llama.*`, `qwen35.*`); fall back to `llama.*` when the
            # architecture key is missing.
            prefixes = [f"{arch}."] if arch else []
            prefixes.append("llama.")
            n_embd = n_layer = n_head = n_head_kv = context_length = None
            rope: dict[str, Any] = {}
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
                if not rope:
                    for key, name in _ROPE_KEYS.items():
                        v = meta.get(f"{pref}{key}")
                        if v is not None:
                            rope[name] = v
                if n_embd is not None and n_layer is not None:
                    break
            return GgufMeta(
                n_embd, n_layer, n_head, n_head_kv, context_length,
                GgufRope(
                    scaling_type=(
                        str(rope["type"]) if rope.get("type") is not None else None
                    ),
                    factor=(
                        float(rope["factor"]) if isinstance(rope.get("factor"), (int, float))
                        else None
                    ),
                    original_context_length=(
                        int(rope["orig_ctx"])
                        if isinstance(rope.get("orig_ctx"), (int, float)) else None
                    ),
                    yarn_log_multiplier=(
                        float(rope["log_mul"])
                        if isinstance(rope.get("log_mul"), (int, float)) else None
                    ),
                    yarn_ext_factor=(
                        float(rope["ext_factor"])
                        if isinstance(rope.get("ext_factor"), (int, float)) else None
                    ),
                    yarn_attn_factor=(
                        float(rope["attn_factor"])
                        if isinstance(rope.get("attn_factor"), (int, float)) else None
                    ),
                    yarn_beta_fast=(
                        float(rope["beta_fast"])
                        if isinstance(rope.get("beta_fast"), (int, float)) else None
                    ),
                    yarn_beta_slow=(
                        float(rope["beta_slow"])
                        if isinstance(rope.get("beta_slow"), (int, float)) else None
                    ),
                ),
            )
    except (OSError, struct.error, UnicodeDecodeError, ValueError):
        return GgufMeta()


#: GGUF architectures that are speech recognizers, not chats. whisper.cpp
#: family files carry one of these in `general.architecture`.
STT_ARCHITECTURES = ("whisper", "parakeet", "nemo", "canary", "moonshine")

#: Diffusion image architectures, for GGUFs that name-match nothing.
IMAGE_ARCHITECTURES = ("stable-diffusion", "flux", "sd1", "sdxl", "qwen-image")


@dataclass(frozen=True)
class GgufKind:
    architecture: str | None = None
    audio_family: str | None = None
    audio_category: str | None = None
    audio_tasks: tuple = ()


_FIXED_SKIP = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _skip_value(fh, vtype: int, size: int | None = None) -> None:
    """Seek past one GGUF value without reading its payload.

    ``size`` is the real length of the file. A value that claims to
    extend past the end of it is a header this parser has lost its place
    in, and the walk stops there rather than running a count of
    billions against a file that has no more bytes.
    """
    if vtype in _FIXED_SKIP:
        fh.seek(_FIXED_SKIP[vtype], 1)
    elif vtype == 8:
        fh.seek(_read_bytes(fh, "<Q"), 1)
    elif vtype == 9:
        elem = _read_bytes(fh, "<I")
        count = _read_bytes(fh, "<Q")
        if elem in _FIXED_SKIP:
            fh.seek(_FIXED_SKIP[elem] * count, 1)
        else:
            for _ in range(count):
                _skip_value(fh, elem, size)
    else:
        raise struct.error(f"unknown value type {vtype}")
    if size is not None and fh.tell() > size:
        raise struct.error("value extends past the end of the file")


def read_gguf_kind(path: Path | str) -> GgufKind:
    """What a GGUF file IS, from its own header -- never from its filename.

    Reads `general.architecture` plus any `audiocpp.model_spec.*` keys
    (audio.cpp containers describe family/category/tasks inline). Header
    only, no tensor data; unreadable or missing files yield an empty kind
    so callers fall back to their other signals instead of failing.
    """
    key = _cache_key(path)
    if key is not None:
        cached = _KIND_CACHE.get(key)
        if cached is not None:
            return cached
    result = _read_gguf_kind(path)
    if key is not None:
        _KIND_CACHE[key] = result
    return result


def _read_gguf_kind(path: Path | str) -> GgufKind:
    try:
        with open(path, "rb") as fh:
            if fh.read(4) != _MAGIC:
                return GgufKind()
            version = _read_bytes(fh, "<I")
            tensor_fmt = "<I" if version < 3 else "<Q"
            _read_bytes(fh, tensor_fmt)
            kv_count = _read_bytes(fh, tensor_fmt)
            arch = None
            spec_json = None
            size = os.fstat(fh.fileno()).st_size
            for _ in range(kv_count):
                key = _read_string(fh)
                vtype = _read_bytes(fh, "<I")
                if key == "general.architecture":
                    value = _read_value(fh, vtype)
                    arch = str(value).lower() if isinstance(value, str) else None
                elif key == "audiocpp.model_spec.json":
                    value = _read_value(fh, vtype)
                    spec_json = str(value) if isinstance(value, str) else None
                else:
                    # Skip without reading: tokenizer arrays run to megabytes
                    # and this runs per model on every `list` call.
                    _skip_value(fh, vtype, size)
                # The only other key worth having is the audiocpp spec, and
                # only an audiocpp file can carry one. Everything else in a
                # header is a tokenizer array running to millions of
                # entries, and walking to the end of one to look for a key
                # that cannot be there is what made `toki model use` take
                # minutes instead of milliseconds.
                if arch is not None and arch != "audiocpp":
                    break
    except Exception:
        return GgufKind()
    family = category = None
    tasks: tuple = ()
    if spec_json:
        try:
            import json as _json

            spec = _json.loads(spec_json)
            family = str(spec.get("family", "")).lower() or None
            category = str(spec.get("category", "")).lower() or None
            raw_tasks = spec.get("tasks", [])
            tasks = tuple(str(x).lower() for x in raw_tasks if isinstance(x, str))
        except Exception:
            pass
    return GgufKind(arch, family, category, tasks)


def read_gguf_tensors(path: Path | str) -> list[str]:
    """The names of a GGUF's tensors, read from the header's directory.

    The directory is in the header, so this costs a few kilobytes of
    reading however large the model is.
    """
    key = _cache_key(path)
    if key is not None:
        cached = _TENSOR_CACHE.get(key)
        if cached is not None:
            return list(cached)
    result = _read_gguf_tensors(path)
    if key is not None:
        _TENSOR_CACHE[key] = result
    return list(result)


def _read_gguf_tensors(path: Path | str) -> list[str]:
    names: list[str] = []
    try:
        with open(path, "rb") as fh:
            if fh.read(4) != _MAGIC:
                return names
            version = _read_bytes(fh, "<I")
            count_fmt = "<I" if version < 3 else "<Q"
            tensor_count = _read_bytes(fh, count_fmt)
            kv_count = _read_bytes(fh, count_fmt)
            for _ in range(kv_count):
                _read_string(fh)
                vtype = _read_bytes(fh, "<I")
                _read_value(fh, vtype)
            # tensor info: name, n_dims, dimensions, type, offset
            dim_fmt = "<I" if version < 2 else "<Q"
            for _ in range(tensor_count):
                name = _read_string(fh)
                n_dims = _read_bytes(fh, "<I")
                for _ in range(n_dims):
                    _read_bytes(fh, dim_fmt)
                _read_bytes(fh, "<I")   # ggml type
                _read_bytes(fh, "<Q")   # offset
                names.append(name)
    except (OSError, struct.error, UnicodeDecodeError, ValueError):
        return names
    return names


def has_vision_encoder(path: Path | str) -> bool:
    """True when the model carries its own vision tower.

    Some models ship the vision encoder inside the weights; llama.cpp
    finds it there and needs no separate projector file. A model like that
    is a vision model even though the registry has no ``mmproj`` to point
    at - and a registry that only ever asks "is there a projector file?"
    calls it blind.

    The encoder's tensors are the ones llama.cpp names with a ``v.``
    prefix - the vision tower, as opposed to the language model.
    """
    return any(
        name.startswith("v.") or name.startswith("vision.") or ".v_proj." in name
        for name in read_gguf_tensors(path)
    )


def classify_kind(architecture: str | None, audio_category: str | None = None) -> str | None:
    """Map GGUF metadata to a registry kind, or None when it says nothing.

    Audio containers win outright: an audiocpp/fish file is TTS or STT no
    matter what its filename claims. Diffusion architectures read as image.
    Everything else returns None so the caller keeps its name/binary/mmproj
    signals -- this function only ever ADDS information, never overrules a
    stack that already knows what it is (sd-server stays image, mmproj
    stays vision).
    """
    arch = (architecture or "").lower()
    if arch == "audiocpp" or (audio_category in ("tts", "stt")):
        return audio_category if audio_category in ("tts", "stt") else "tts"
    if arch in STT_ARCHITECTURES:
        return "stt"
    if arch in IMAGE_ARCHITECTURES:
        return "image"
    return None
