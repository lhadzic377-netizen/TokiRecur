"""MemT native memory: the weights, the build that loads them, the API.

A MemT model carries *memory weights* - tensors named ``blk.<n>.memt_*``
and metadata keys ``qwen35moe.memt.*`` - beside the backbone. Upstream
llama.cpp does not know that tensor type and refuses the file, so a
MemT model needs the ``memt-native-memory`` branch of Brishen/llama.cpp:
current upstream plus one commit that adds the memory (CPU and CUDA),
the converter, and the llama-server API.

What the branch's server adds:

* three request fields, on ``/v1/chat/completions`` and ``/completion``:
  ``memory_id`` (which memory state to read), ``memory_commit`` (write
  this request's turns into the memory when the reply ends normally) and
  ``memory_documents`` (documents to write into the memory before
  generating);
* the memory states themselves, kept in RAM by id: ``GET /memory`` lists
  them, and ``POST /memory/{id}?action=save|restore|erase`` saves,
  restores or erases one - save and restore need ``--slot-save-path``.

Without a ``memory_id`` the model behaves as the plain backbone, so the
fields are optional everywhere and a client that never sends them is
unaffected.

This module holds the knowledge the rest of toki needs: which weights
are MemT weights, whether an installed build can load them, and how to
get one.
"""

from __future__ import annotations

import struct
from pathlib import Path

from .ggufmeta import read_gguf_tensors

#: The fork and branch that carry the MemT memory.
MEMT_FORK = "Brishen/llama.cpp"
MEMT_BRANCH = "memt-native-memory"

#: Where the branch's build is installed, beside the other per-build
#: llama-server copies (``bin-inkling`` and friends).
MEMT_BIN_DIR = "bin-memt"

#: A MemT memory tensor is named after the block it belongs to:
#: ``blk.0.memt_0.weight`` and friends.
MEMT_TENSOR_MARK = ".memt_"

#: The metadata keys the branch's converter writes.
MEMT_KEY_MARK = ".memt."

#: The request fields the branch's server understands.
MEMORY_REQUEST_FIELDS = ("memory_id", "memory_commit", "memory_documents")

#: The memory actions the branch's server understands.
MEMORY_ACTIONS = ("save", "restore", "erase")


def memt_tensor_names(path: Path | str) -> list[str]:
    """The names of a GGUF's MemT memory tensors."""
    try:
        tensors = read_gguf_tensors(path)
    except (OSError, ValueError, struct.error):
        return []
    return [name for name in tensors if MEMT_TENSOR_MARK in name]


def has_memt_weights(path: Path | str) -> bool:
    """Whether a GGUF carries MemT memory weights.

    Read from the header's tensor directory, which costs a few kilobytes
    however large the model is.
    """
    return bool(memt_tensor_names(path))


def build_hint() -> str:
    """How to get a build that can load MemT weights."""
    return (
        "build the MemT branch of llama.cpp:\n"
        f"  toki model install-llama --source --fork {MEMT_FORK} "
        f"--branch {MEMT_BRANCH}"
    )


def cmake_flags(backend: str) -> list[str]:
    """The cmake flags for a backend, as the branch's README names them."""
    backend = (backend or "cpu").lower()
    if backend == "cuda":
        return ["-DGGML_CUDA=ON"]
    if backend == "rocm":
        return ["-DGGML_HIP=ON"]
    if backend == "vulkan":
        return ["-DGGML_VULKAN=ON"]
    return []


def build_steps(
    source_dir: Path,
    *,
    fork: str = MEMT_FORK,
    branch: str = MEMT_BRANCH,
    backend: str = "cpu",
    jobs: int | None = None,
) -> list[list[str]]:
    """The commands that build the MemT-capable llama-server.

    Kept here so the CLI, the docs and the error messages all say the
    same thing, and so the steps can be checked without running them.
    """
    # The build directory lives inside the source tree: a relative
    # "build" would land in whatever directory the command was run from,
    # and the next run would not find it.
    build_dir = source_dir / "build"
    steps: list[list[str]] = [
        [
            "git", "clone", "-b", branch,
            f"https://github.com/{fork}.git",
            str(source_dir),
        ],
        [
            "cmake", "-B", str(build_dir),
            *cmake_flags(backend),
            str(source_dir),
        ],
    ]
    build = ["cmake", "--build", str(build_dir)]
    if jobs:
        build += ["-j", str(jobs)]
    build += ["--target", "llama-server", "llama-quantize"]
    steps.append(build)
    return steps


def convert_steps(
    converter: Path,
    quantizer: Path,
    checkpoint: Path,
    out_file: Path,
    *,
    python: str = "python",
    quant: str = "Q4_K_M",
    memt_type: str = "q8_0",
) -> list[list[str]]:
    """The commands that turn a MemT checkpoint into a GGUF.

    The checkpoint is converted at bf16 (the memory weights have no
    cheaper home), then quantized - with the memory tensors pinned to
    Q8_0, because quantizing them to 4-bit throws away the thing they
    are for. The converter is the branch's own
    ``convert_hf_to_gguf.py``: it is the one that knows the memory
    tensors, and upstream's converter does not.
    """
    bf16 = out_file.with_suffix(".bf16.gguf")
    return [
        [
            python, str(converter), str(checkpoint),
            "--outfile", str(bf16),
            "--outtype", "bf16",
        ],
        [
            str(quantizer),
            "--tensor-type", f"memt_={memt_type}",
            str(bf16), str(out_file), quant,
        ],
    ]


def memt_build_dir(root: Path | None = None) -> Path:
    """Where the MemT branch's build is installed.

    Beside the other per-build llama-server copies (``bin-inkling`` and
    friends), not inside the shared ``bin`` dir - a build that only some
    models can use does not belong next to the one every model uses.
    """
    from .ctl import STATE_ROOT

    return (root or STATE_ROOT) / MEMT_BIN_DIR
