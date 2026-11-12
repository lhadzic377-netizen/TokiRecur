"""Speculative-decoding drafts: the second model that makes a target faster.

A draft model is not a model you can serve on its own. It is a second
GGUF that llama-server loads alongside a target and consults for
candidate tokens, then verifies against the target's own output. The
target produces the answer; the draft only makes the target's work
bigger per step.

That makes the pairing strict, and it is why the pairing is a first
class thing here rather than a flag someone types:

* the draft reads the target's hidden states, so it belongs to exactly
  one target (a DSpine draft for Qwen3.8-27B is useless with any other
  model);
* the draft needs its own VRAM, so a fit plan that ignores it asks for
  a context the machine cannot hold — the server then fails to allocate
  and falls back to host copies, or dies;
* the draft's own KV cache and the target's are separate, and the
  target keeps a copy of its recurrent state per drafted token, so the
  two together cost more than the target alone.

The upstream DSpine work (Tencent's DSpine, llama.cpp PR #31941) is
what this is built for, but nothing here is DSpine-specific: the
mechanism is llama.cpp's ``--spec-type draft-dflash``, which the
existing DFlash2 drafts also use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .ggufmeta import read_gguf_meta

# llama.cpp's speculative-decoding CLI surface. A draft is passed as
# --spec-draft-model, and the speculation family as --spec-type.
SPEC_TYPE_DRAFT_DFLASH = "draft-dflash"
SPEC_TYPE_DRAFT_MTP = "draft-mtp"

# How many tokens the drafter may propose in one block. The upstream
# DSpine card recommends 15 on CUDA, ROCm and SYCL, and 2-3 on Vulkan,
# where a long draft costs more to verify than it saves.
DRAFT_N_MAX_DEFAULT = 15

# The DSpine drafter's sliding window. Its own KV cache never grows
# past this, however long the target's context is.
DRAFT_KV_WINDOW = 8192

# Bytes per KV element in llama.cpp's cache formats. q8_0 is a 32-value
# block with a 2-byte scale (34 bytes); f16 is the drafter's default.
_KV_BYTES_PER_ELEMENT = {
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34.0 / 32.0,
    "q5_1": 24.0 / 32.0,
    "q5_0": 22.0 / 32.0,
    "q4_1": 20.0 / 32.0,
    "q4_0": 18.0 / 32.0,
    "iq4_nl": 18.0 / 32.0,
}

# Conservative fallback when the draft's metadata cannot be read: the
# DSpine b16 layout (5 layers, 8 KV heads, 160-wide heads) at f16.
_DRAFT_KV_BYTES_PER_TOKEN_FALLBACK = 2 * 5 * 8 * 160 * 2.0


@dataclass(frozen=True)
class DraftSpec:
    """A speculative-decoding draft attached to one target model.

    ``gguf`` is the draft's weights on disk. ``spec_type`` names the
    speculation family llama.cpp should use. ``n_max`` is how many
    tokens the drafter may propose per block.
    """

    gguf: str
    spec_type: str = SPEC_TYPE_DRAFT_DFLASH
    n_max: int = DRAFT_N_MAX_DEFAULT
    # Where it came from, so a draft that was never downloaded can be
    # fetched again: the repository has it, and a blind draft makes the
    # target slower rather than faster.
    repo: str = ""
    file: str = ""
    # Extra llama.cpp flags for the drafter (its own KV precision, its
    # own GPU offload). Kept as a list of argv tokens because
    # llama-server takes each flag separately.
    extra_args: list[str] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return Path(self.gguf)

    @property
    def exists(self) -> bool:
        return bool(self.gguf) and Path(self.gguf).exists()

    @property
    def label(self) -> str:
        """A short human name for the launcher row."""
        if not self.gguf:
            return ""
        return Path(self.gguf).stem

    def llama_args(self) -> list[str]:
        """The llama-server flags that attach this draft."""
        if not self.gguf:
            return []
        args = [
            "--spec-draft-model", str(self.gguf),
            "--spec-type", self.spec_type,
            "--spec-draft-n-max", str(self.n_max),
        ]
        args.extend(self.extra_args)
        return args

    def to_dict(self) -> dict:
        """The registry form, so a spec round-trips through YAML."""
        out: dict = {
            "gguf": self.gguf,
            "spec_type": self.spec_type,
            "n_max": self.n_max,
        }
        if self.repo:
            out["repo"] = self.repo
        if self.file:
            out["file"] = self.file
        if self.extra_args:
            out["extra_args"] = list(self.extra_args)
        return out

    @classmethod
    def from_dict(cls, raw: object) -> DraftSpec | None:
        """Read the registry form, tolerating an absent or empty block."""
        if not raw:
            return None
        if isinstance(raw, str):
            # A bare path: the shortest form, and the one a hand-written
            # registry entry reaches for.
            return cls(gguf=raw)
        if not isinstance(raw, dict):
            return None
        gguf = str(raw.get("gguf") or "").strip()
        if not gguf:
            return None
        n_max = raw.get("n_max", DRAFT_N_MAX_DEFAULT)
        try:
            n_max = int(n_max)
        except (TypeError, ValueError):
            n_max = DRAFT_N_MAX_DEFAULT
        extra = raw.get("extra_args") or []
        if isinstance(extra, str):
            extra = extra.split()
        return cls(
            gguf=gguf,
            spec_type=str(raw.get("spec_type") or SPEC_TYPE_DRAFT_DFLASH),
            n_max=n_max,
            repo=str(raw.get("repo") or ""),
            file=str(raw.get("file") or ""),
            extra_args=[str(a) for a in extra],
        )


def _draft_kv_type(draft: DraftSpec) -> str:
    """The drafter's KV precision, from the flags that will be passed.

    llama.cpp's default is f16. A registry entry that sets
    ``--spec-draft-type-k`` / ``--spec-draft-type-v`` gets the precision
    it asked for, so the fit plan is not estimating a cache the server
    will not build.
    """
    specified: list[str] = []
    args = draft.extra_args
    for flag in ("--spec-draft-type-k", "--spec-draft-type-v",
                 "-ctkd", "-ctvd"):
        try:
            value = args[args.index(flag) + 1]
        except (ValueError, IndexError):
            continue
        if value in _KV_BYTES_PER_ELEMENT:
            specified.append(value)
    if not specified:
        return "f16"
    # K and V can differ; size for the larger of the two.
    return max(specified, key=lambda v: _KV_BYTES_PER_ELEMENT[v])


def _kv_bytes_per_token(draft: DraftSpec) -> float:
    """The drafter's own KV cache cost per token, from its GGUF header.

    Read rather than guessed: the header carries the layer count, the
    KV head count and the embedding width, which are the whole
    calculation. The fallback is the DSpine b16 layout, used only when
    the header cannot be read.
    """
    per_element = _KV_BYTES_PER_ELEMENT[_draft_kv_type(draft)]
    try:
        meta = read_gguf_meta(Path(draft.gguf))
    except Exception:
        meta = None
    if meta is not None and meta.complete and meta.n_head:
        head_dim = meta.n_embd // meta.n_head
        if head_dim:
            elements = 2 * meta.n_layer * meta.n_head_kv * head_dim
            return elements * per_element
    return _DRAFT_KV_BYTES_PER_TOKEN_FALLBACK * (
        per_element / _KV_BYTES_PER_ELEMENT["f16"]
    )


def draft_vram_gb(draft: DraftSpec, ctx: int, parallel: int = 1) -> float:
    """VRAM the drafter adds, in GB.

    Three parts, all of them real and all of them paid at launch:

    * the drafter's weights;
    * its own KV cache — a sliding window of at most 8k tokens, however
      long the target's context is;
    * the target's recurrent-state copies. Qwen3.8 is a hybrid model,
      and while it verifies a draft it keeps a copy of its recurrent
      state for every drafted token. The DSpine card measures that at
      about 150 MB per copy, so a 16-token block costs about 2.4 GB —
      1.2 GB more than an 8-token DFlash2 block.
    """
    if not draft.gguf:
        return 0.0
    weights = 0.0
    try:
        weights = Path(draft.gguf).stat().st_size / 1e9
    except OSError:
        # A draft that is not on disk costs nothing at this point; the
        # launch gate refuses it before anything tries to load it.
        return 0.0
    window = min(max(0, ctx), DRAFT_KV_WINDOW)
    kv = _kv_bytes_per_token(draft) * window * max(1, parallel) / 1e9
    target_state = 0.15 * max(0, draft.n_max) * max(1, parallel)
    return weights * 1.2 + kv + target_state


def looks_like_dspine_adapter(filename: str) -> bool:
    """Whether a GGUF filename names a DSpine adapter, not a target.

    The adapters are published as ``<family>-DSpine-b16-<quant>.gguf``.
    They are not models: a DSpine adapter reads Qwen3.8-27B's hidden
    states and cannot answer a prompt on its own. Pulling one as a
    model produces a registry entry that can never serve.
    """
    return "dspine" in Path(filename).name.lower()


def dspine_adapter_hint(filename: str) -> str:
    """Why a DSpine adapter cannot be pulled as a model, and what to do."""
    return (
        f"'{Path(filename).name}' is DSpine adapter weights, not a "
        f"model. The adapter attaches to its Qwen3.8-27B target and "
        f"cannot be served on its own. Pull the target GGUF (for "
        f"example Qwen3.8-27B-Q4_K_M.gguf) and the adapter is fetched "
        f"and attached automatically when the llama-server build "
        f"supports DSpine."
    )


def draft_needs_dspine(draft: DraftSpec) -> bool:
    """Whether this draft's GGUF carries DSpine's ``dspine_*`` tensors.

    A stock llama.cpp can run a DFlash2 draft; only a DSpine conversion
    needs the added tensor mapping. Read from the draft's own header,
    so a hand-written registry entry with an ordinary draft is not
    refused because the build lacks DSpine.
    """
    if not draft.exists:
        return False
    try:
        from .ggufmeta import read_gguf_tensor_sizes

        names = read_gguf_tensor_sizes(draft.gguf)
    except Exception:
        return False
    return any("dspine_" in name for name in names)


def draft_supported_by(binary: str) -> bool:
    """True when this llama-server build can load a DSpine draft.

    Stock llama.cpp refuses the files: it has no mapping for the
    ``dspine_*`` tensors they carry and throws
    ``missing tensor info mapping for ...``. The support is one commit
    (llama.cpp PR #31941) that adds those names.

    The tell is the tensor-name string itself. Every CLI flag the
    speculation path uses (``--spec-draft-model``, ``--spec-type``,
    ``--spec-draft-type-k``) is already present in a stock build, so a
    flag check would call every build capable. The ``dspine_codebook``
    name is compiled into the library that does the mapping, and it
    exists only in a build that carries the DSpine commit — so that is
    what is looked for, in the binary and in the shared libraries
    beside it (the mapping lives in libllama, not in the thin
    llama-server wrapper).
    """
    if not binary:
        return False
    path = Path(binary)
    if not path.exists():
        return False
    for candidate in _libraries_for(path):
        if _contains_bytes(candidate, b"dspine_codebook"):
            return True
    return False


def _contains_bytes(path: Path, needle: bytes) -> bool:
    """Whether a file contains ``needle``, read in chunks.

    A llama.cpp library is tens of megabytes; slurping it whole just to
    search for one name is needless. The overlap keeps a needle that
    straddles a chunk boundary from being missed.
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


def _libraries_for(binary: Path) -> list[Path]:
    """The binary and the shared libraries beside it.

    A llama-server build puts the tensor mapping in libllama, so the
    wrapper on its own answers nothing. The libraries are found by
    name rather than by resolving every symlink: the install lays them
    out side by side.
    """
    out = [binary]
    folder = binary.parent
    # Linux/macOS call it libllama.so / libllama.dylib; the Windows
    # release ships llama.dll. Both carry the tensor mapping.
    patterns = ("libllama*", "llama*.dll", "llama*.so*", "llama*.dylib")
    try:
        for pattern in patterns:
            for entry in sorted(folder.glob(pattern)):
                if entry.is_file() and entry not in out:
                    out.append(entry)
    except OSError:
        pass
    return out


def draft_support_hint() -> str:
    """What to tell someone whose build cannot load the draft."""
    return (
        "this llama-server build predates DSpine support; it refuses the "
        "draft's dspine_* tensors. Build the `dspine` branch of "
        "https://github.com/Ajay9o9/llama.cpp (upstream master plus PR "
        "#31941, \"spec : add DSpine draft support\") or a release that "
        "includes it, and point the model's `binary:` at that build."
    )
