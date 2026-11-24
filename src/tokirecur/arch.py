"""Which GGUF architectures an installed llama-server build can load.

llama.cpp gains a new architecture with the upstream merge that
introduces it. Inkling 2 Mini arrived as ggml-org/llama.cpp#32694
(merged 2026-11-20) and needed a new architecture name,
``inkling2_loop``, beside the plain ``inkling2``. A build from before
that merge refuses the file with

    llama_model_load: error loading model: unknown model architecture:
    'inkling2_loop'

which reads like a broken download rather than an out-of-date binary.

The architecture names live in the mapping table inside libllama, so the
build itself says which ones it knows: the exact name is either in the
library or it is not. The same trick the draft path uses for
``dspine_codebook``.
"""

from __future__ import annotations

from pathlib import Path

#: Architectures that need a named upstream merge, for the error message.
ARCH_HINTS: dict[str, str] = {
    "inkling2_loop": (
        "llama.cpp PR #32694 (Inkling 2 Mini, merged 2026-11-20)"
    ),
}


def _libraries_for(binary: Path) -> list[Path]:
    """The binary and the shared libraries beside it.

    The architecture table lives in libllama, so the thin llama-server
    wrapper on its own answers nothing. The libraries are found by name
    rather than by resolving symlinks: the install lays them out side by
    side.
    """
    out = [binary]
    folder = binary.parent
    patterns = ("libllama*", "llama*.dll", "llama*.so*", "llama*.dylib")
    try:
        for pattern in patterns:
            for entry in sorted(folder.glob(pattern)):
                if entry.is_file() and entry not in out:
                    out.append(entry)
    except OSError:
        pass
    return out


def _contains_bytes(path: Path, needle: bytes) -> bool:
    """Whether a file contains ``needle``, read in chunks.

    A llama.cpp library is tens of megabytes; slurping it whole to look
    for one name is needless. The overlap keeps a needle that straddles
    a chunk boundary from being missed.
    """
    chunk = 1 << 20
    overlap = len(needle) - 1
    tail = b""
    try:
        with open(path, "rb") as fh:
            while True:
                block = fh.read(chunk)
                if not block:
                    return False
                window = tail + block
                if needle in window:
                    return True
                tail = window[-overlap:] if overlap else b""
    except OSError:
        return False


def binary_supports_memt(binary: str | None) -> bool | None:
    """Whether the build can load MemT memory weights.

    Upstream llama.cpp has never heard of the MemT tensor type, so a
    build that cannot load one says so in its own binary: the branch's
    ggml carries the memory type name, upstream does not. The check is
    a search for that name in the binary and the libraries beside it -
    the same trick :func:`binary_supports_arch` uses for architectures.

    ``None`` means "cannot tell": the binary is missing or unreadable,
    and a refusal on a guess would be worse than letting the server try.
    """
    if not binary:
        return None
    path = Path(binary)
    if not path.exists():
        return None
    candidates = _libraries_for(path)
    # The branch's llama library names the memory in several places -
    # the attention's "attn_memt", the per-block weights
    # ("blk.%d.memt_f"), the graph input ("llm_graph_input_memt") and
    # the quantized form of the memory type ("memt_q8_0"). Any of them
    # says the build knows the memory; upstream says none of them.
    needles = (b"attn_memt", b"llm_graph_input_memt", b"memt_q8_0",
               b"memt_decay_bias")
    for candidate in candidates:
        if any(_contains_bytes(candidate, needle) for needle in needles):
            return True
    # No llama library beside the wrapper: the type may live in a
    # library this check cannot see, so there is no opinion to give.
    if not any(c.name.startswith("libllama") or c.name.startswith("llama")
               for c in candidates[1:]):
        return None
    # The libraries are there and none of them knows the memory type:
    # this is an upstream build.
    return False


def binary_supports_arch(binary: str | None, arch: str | None) -> bool | None:
    """Whether the build at ``binary`` knows the GGUF architecture ``arch``.

    ``None`` means "cannot tell" — the binary is missing, unreadable, or
    the GGUF does not name an architecture. Callers treat that as "no
    opinion" and let the model start.
    """
    if not binary or not arch:
        return None
    path = Path(binary)
    if not path.exists():
        return None
    needle = arch.encode("utf-8")
    candidates = _libraries_for(path)
    # The architecture table lives in libllama. Without a llama library
    # next to the wrapper there is nothing to read, so there is no
    # opinion to give - refusing the model on a guess would be worse
    # than letting the server try.
    if not any(c.name.startswith("libllama") or c.name.startswith("llama")
               for c in candidates[1:]):
        return None
    for candidate in candidates:
        if _contains_bytes(candidate, needle):
            return True
    # The libraries were there and none of them knows the architecture:
    # the build predates it.
    return False


def arch_support_hint(arch: str) -> str:
    """One line telling the user how to get a build that knows ``arch``."""
    hint = ARCH_HINTS.get(arch)
    if hint:
        return f"this architecture needs {hint}"
    return "this architecture needs a newer llama.cpp build"
