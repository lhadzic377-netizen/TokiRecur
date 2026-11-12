"""Finding the speculative-decoding draft that belongs to a target model.

A draft model is published separately from its target and has to be
matched to it: a DSpine draft for Qwen3.8-27B reads that model's
hidden states and is useless with anything else. The pairing is
therefore resolved from the target's own name, and the resolution is
recorded so the next pull of the same target finds the same draft
instead of guessing again.

The GGUFs are published by the model authors, not by this project.
Nothing is re-hosted here: the pull points at the repository that
already holds the file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Where the community publishes the DSpine conversions. The naming
# convention is "<target-family>-DSpine-b16-GGUF", and the target
# family is the target model's name without its quantisation suffix.
_DSPINE_REPO_TEMPLATE = "{owner}/{family}-DSpine-b16-GGUF"

# Repositories that hold the DSpine GGUF conversions, in the order
# they are tried. The first one is the author of the model card.
_DSPINE_OWNERS = ("aj9o9", "generalinference")

# The quantisation to fetch by default. Q4_K_M is the card's
# recommendation: same acceptance as Q8_0, less VRAM, and the code
# bridge tensors stay in BF16 either way.
_DSPINE_DEFAULT_QUANT = "Q4_K_M"

# The file name the card uses for that quantisation.
_DSPINE_QUANT_FILES = {
    "Q4_K_M": "Q4_K_M",
    "Q8_0": "Q8_0",
}

# A split part's suffix: llama.cpp names them "name-00001-of-00006.gguf".
_SPLIT_SUFFIX_RE = re.compile(r"-\d{5}-of-\d{5}$")

# A quantisation suffix at the end of a GGUF's stem. "Qwen3.8-27B-Q4_K_M"
# is the family "Qwen3.8-27B" with the quant "Q4_K_M".
_QUANT_SUFFIX_RE = re.compile(
    r"[-_](?:IQ\d(?:_[A-Z]+)*|Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4|NVFP4)$"
)


@dataclass(frozen=True)
class DraftCandidate:
    """A draft model found for a target, with where it comes from."""

    repo: str
    file: str
    family: str
    quant: str


def target_family(filename: str) -> str:
    """The target's model family, without its quantisation suffix.

    "Qwen3.8-27B-Q4_K_M.gguf" -> "Qwen3.8-27B". A name without a
    quantisation suffix is returned unchanged.
    """
    stem = filename.rsplit("/", 1)[-1]
    if stem.lower().endswith(".gguf"):
        stem = stem[:-5]
    # A split part names the same model as its siblings; the part
    # number is not part of the family.
    stem = _SPLIT_SUFFIX_RE.sub("", stem)
    # Only strip a suffix that is preceded by something, so a name that
    # is entirely a quantisation ("Q4_K_M.gguf") is left alone.
    stripped = _QUANT_SUFFIX_RE.sub("", stem)
    return stripped or stem


def adapter_target_family(repo_or_file: str) -> str:
    """The base-model family an adapter repository belongs to.

    "aj9o9/Qwen3.8-27B-DSpine-b16-GGUF" -> "Qwen3.8-27B". Adapter
    repositories name the target first and the adapter second, so the
    same family that :func:`target_family` finds on a model file can be
    recovered from the adapter's repository name.
    """
    segment = repo_or_file.rstrip("/").rsplit("/", 1)[-1]
    if segment.lower().endswith(".gguf"):
        stem = target_family(segment)
    else:
        stem = re.sub(r"(?i)-dspine(?:-b\d+)?-gguf$", "", segment)
        stem = re.sub(r"(?i)-gguf$", "", stem)
    stem = re.sub(r"(?i)-dspine(?:-b\d+)?$", "", stem)
    return stem or segment


def _candidate_repos(family: str) -> list[str]:
    return [_DSPINE_REPO_TEMPLATE.format(owner=o, family=family)
            for o in _DSPINE_OWNERS]


def find_draft(
    target_file: str,
    *,
    quant: str = _DSPINE_DEFAULT_QUANT,
    api: object | None = None,
) -> DraftCandidate | None:
    """The DSpine draft for a target model, or None when there is none.

    Resolved by name: the draft's repository is the target's family
    plus ``-DSpine-b16-GGUF``. The file is looked up in that
    repository, so a repository that publishes the draft under a
    different file name is still found.
    """
    family = target_family(target_file)
    if not family:
        return None
    if api is None:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
        except Exception:
            return None
    for repo in _candidate_repos(family):
        try:
            files = list(api.list_repo_files(repo))
        except Exception:
            continue
        wanted = _DSPINE_QUANT_FILES.get(quant, quant)
        for f in files:
            if not f.endswith(".gguf"):
                continue
            name = f.rsplit("/", 1)[-1]
            if wanted in name:
                return DraftCandidate(repo=repo, file=f, family=family,
                                      quant=quant)
    return None
