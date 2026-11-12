"""Selectable adapter modes for a base model.

A base model is one GGUF. An adapter is a second GGUF that changes how
that base runs: a DSpine/DFlash2 draft makes it speculate, a LoRA
changes its weights. Neither can be served alone — the adapter reads
the base's hidden states or patches its tensors — so an adapter is not
a model entry. It is a *mode* of one model.

The registry keeps a catalog per model (``adapters:``) and which catalog
entry is active (``adapter_mode:``). The active entry is projected onto
the older flat ``draft*`` fields, so every launch, fit and gate path has
one thing to read and a hand-written registry without a catalog still
works.

Plug and play is the point: installing an adapter beside its base makes
it show up in the launcher's settings dropdown. Selecting it writes the
flat fields and restarts the stack; selecting the base clears them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .draft import (
    DRAFT_KV_WINDOW,
    DRAFT_N_MAX_DEFAULT,
    SPEC_TYPE_DRAFT_DFLASH,
    DraftSpec,
    draft_vram_gb,
    looks_like_dspine_adapter,
)
from .draftmatch import target_family

KIND_DRAFT = "draft"
KIND_LORA = "lora"
ADAPTER_KINDS = (KIND_DRAFT, KIND_LORA)

# The dropdown's value for "run the base model, no adapter". A real
# registry stores "" for this; the sentinel exists because a Select
# widget treats an empty value as blank.
BASE_MODE_ID = "__base__"
BASE_LABEL = "base model (no adapter)"

_SPLIT_SUFFIX_RE = re.compile(r"-\d{5}-of-\d{5}$")
_QUANT_RE = re.compile(
    r"(IQ\d(?:_[A-Z0-9]+)*|Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4|NVFP4)"
)

# A LoRA adapter is small. Anything larger is far more likely to be a
# model or a draft, and reading every large header just to rule it out
# would make the settings pane slow on a machine with many models.
_LORA_SCAN_LIMIT = 4 << 30


def _slug(text: str) -> str:
    out = re.sub(r"[^a-z0-9._-]+", "-", text.lower()).strip("-._")
    return out


def mode_id_for(filename: str) -> str:
    """A stable id for an adapter file, independent of its directory."""
    stem = Path(filename).name
    if stem.lower().endswith(".gguf"):
        stem = stem[:-5]
    stem = _SPLIT_SUFFIX_RE.sub("", stem)
    return _slug(stem) or "adapter"


def quant_of(filename: str) -> str:
    """The quantisation suffix a filename advertises, or ''."""
    stem = Path(filename).stem.upper()
    matches = _QUANT_RE.findall(stem)
    return matches[-1] if matches else ""


def _label_for(path: str, kind: str) -> str:
    if kind == KIND_LORA:
        return "LoRA " + Path(path).stem
    if looks_like_dspine_adapter(path):
        quant = quant_of(path)
        return "DSpine" + (f" {quant}" if quant else "")
    return Path(path).stem


def kind_for_path(path: Path) -> str | None:
    """How a GGUF beside a model attaches, or None when it is not an
    adapter.

    The filename is enough for the published DSpine layout. A LoRA has
    no naming convention, so its own header decides: llama.cpp's LoRA
    conversion writes ``lora_a``/``lora_b`` tensors and an ``adapter``
    architecture.
    """
    if looks_like_dspine_adapter(path.name):
        return KIND_DRAFT
    try:
        if path.stat().st_size > _LORA_SCAN_LIMIT:
            return None
    except OSError:
        return None
    try:
        from .ggufmeta import read_gguf_kind, read_gguf_tensors

        if (read_gguf_kind(path).architecture or "").lower() == "adapter":
            return KIND_LORA
        return KIND_LORA if any(
            "lora" in name.lower() for name in read_gguf_tensors(path)
        ) else None
    except Exception:
        return None


@dataclass(frozen=True)
class AdapterMode:
    """One way to run a base model.

    ``id`` is what the registry stores in ``adapter_mode``. ``kind``
    says how it attaches; today a ``draft`` is passed as
    ``--spec-draft-model`` and a ``lora`` as ``--lora``. ``extra_args``
    carries tuning the adapter needs (its own KV precision, a scale).
    """

    id: str
    label: str = ""
    kind: str = KIND_DRAFT
    gguf: str = ""
    spec_type: str = SPEC_TYPE_DRAFT_DFLASH
    n_max: int = DRAFT_N_MAX_DEFAULT
    repo: str = ""
    file: str = ""
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    @property
    def path(self) -> Path:
        return Path(self.gguf)

    @property
    def exists(self) -> bool:
        return bool(self.gguf) and Path(self.gguf).is_file()

    @property
    def choice_label(self) -> str:
        return self.label or self.id

    @property
    def badge(self) -> str:
        """A short marker for the model row, like ``DSpine Q4_K_M``."""
        if self.kind == KIND_LORA:
            return "LoRA"
        if looks_like_dspine_adapter(self.gguf):
            quant = quant_of(self.gguf)
            return "DSpine" + (f" {quant}" if quant else "")
        return "draft"

    def draft_spec(self) -> DraftSpec | None:
        if self.kind != KIND_DRAFT or not self.gguf:
            return None
        return DraftSpec(
            gguf=self.gguf,
            spec_type=self.spec_type or SPEC_TYPE_DRAFT_DFLASH,
            n_max=self.n_max or DRAFT_N_MAX_DEFAULT,
            repo=self.repo,
            file=self.file,
            extra_args=list(self.extra_args),
        )

    def launch_args(self) -> list[str]:
        """The flags that attach this mode to llama-server."""
        if self.kind == KIND_DRAFT:
            draft = self.draft_spec()
            return draft.llama_args() if draft is not None else []
        if self.kind == KIND_LORA:
            if not self.gguf:
                return []
            if any(a in ("--lora", "--lora-scaled") for a in self.extra_args):
                return list(self.extra_args)
            return ["--lora", self.gguf, *self.extra_args]
        raise ValueError(
            f"adapter mode '{self.id}' has kind '{self.kind}', which this "
            f"build cannot attach (known kinds: {', '.join(ADAPTER_KINDS)})"
        )

    def to_dict(self) -> dict:
        out: dict = {
            "id": self.id,
            "label": self.choice_label,
            "kind": self.kind,
            "gguf": self.gguf,
        }
        if self.kind == KIND_DRAFT:
            out["spec_type"] = self.spec_type or SPEC_TYPE_DRAFT_DFLASH
            out["n_max"] = self.n_max or DRAFT_N_MAX_DEFAULT
        if self.repo:
            out["repo"] = self.repo
        if self.file:
            out["file"] = self.file
        if self.extra_args:
            out["extra_args"] = list(self.extra_args)
        return out

    @classmethod
    def from_dict(cls, raw: object) -> AdapterMode | None:
        if not raw:
            return None
        if isinstance(raw, str):
            raw = {"gguf": raw}
        if not isinstance(raw, dict):
            return None
        gguf = str(raw.get("gguf") or "").strip()
        if not gguf:
            return None
        kind = str(raw.get("kind") or KIND_DRAFT).strip().lower()
        mode_id = str(raw.get("id") or "").strip() or mode_id_for(gguf)
        try:
            n_max = int(raw.get("n_max", DRAFT_N_MAX_DEFAULT))
        except (TypeError, ValueError):
            n_max = DRAFT_N_MAX_DEFAULT
        extra = raw.get("extra_args") or []
        if isinstance(extra, str):
            extra = extra.split()
        return cls(
            id=mode_id,
            label=str(raw.get("label") or "") or _label_for(gguf, kind),
            kind=kind,
            gguf=gguf,
            spec_type=str(raw.get("spec_type") or SPEC_TYPE_DRAFT_DFLASH),
            n_max=n_max,
            repo=str(raw.get("repo") or ""),
            file=str(raw.get("file") or ""),
            extra_args=tuple(str(a) for a in extra),
        )


def adapter_modes(spec: object) -> list[AdapterMode]:
    """The registered catalog, in registry order, without duplicates."""
    out: list[AdapterMode] = []
    seen: set[str] = set()
    for raw in getattr(spec, "adapters", None) or []:
        mode = AdapterMode.from_dict(raw)
        if mode is None or mode.id in seen:
            continue
        seen.add(mode.id)
        out.append(mode)
    return out


def _legacy_mode(spec: object) -> AdapterMode | None:
    """A registry that predates the catalog: its flat draft is a mode."""
    gguf = str(getattr(spec, "draft", "") or "").strip()
    if not gguf:
        return None
    return AdapterMode(
        id=mode_id_for(gguf),
        label=_label_for(gguf, KIND_DRAFT),
        kind=KIND_DRAFT,
        gguf=gguf,
        spec_type=str(getattr(spec, "draft_spec_type", "") or "")
        or SPEC_TYPE_DRAFT_DFLASH,
        n_max=int(getattr(spec, "draft_n_max", 0) or 0)
        or DRAFT_N_MAX_DEFAULT,
        repo=str(getattr(spec, "draft_repo", "") or ""),
        file=str(getattr(spec, "draft_file", "") or ""),
        extra_args=tuple(getattr(spec, "draft_extra_args", None) or ()),
    )


def available_modes(spec: object) -> list[AdapterMode]:
    """The catalog, or the legacy flat draft when there is no catalog."""
    catalog = adapter_modes(spec)
    if catalog:
        return catalog
    legacy = _legacy_mode(spec)
    return [legacy] if legacy is not None else []


def active_mode_id(spec: object) -> str:
    return str(getattr(spec, "adapter_mode", "") or "").strip()


def active_mode(spec: object) -> AdapterMode | None:
    """The mode the stack will launch with, or None for the base model.

    The flat ``draft*`` fields are the projection of the active mode.
    A registry that sets them without an ``adapter_mode`` id is either
    the pre-catalog layout or a hand-written switch, and in both the
    flat draft is what runs. A registry with a catalog and no flat
    draft is explicitly on the base model.
    """
    wanted = active_mode_id(spec)
    if wanted:
        return next(
            (m for m in available_modes(spec) if m.id == wanted), None
        )
    flat = _legacy_mode(spec)
    if flat is not None:
        return next(
            (m for m in adapter_modes(spec)
             if Path(m.gguf) == Path(flat.gguf)),
            flat,
        )
    return None


def resolve_draft(spec: object) -> DraftSpec | None:
    """The active adapter as a draft, when it is a draft."""
    mode = active_mode(spec)
    return mode.draft_spec() if mode is not None else None


def adapter_launch_args(spec: object) -> list[str]:
    mode = active_mode(spec)
    return mode.launch_args() if mode is not None else []


def attached_vram_gb(spec: object, ctx: int, parallel: int = 1) -> float:
    """VRAM the active adapter adds to the base model, in GB."""
    mode = active_mode(spec)
    if mode is None:
        return 0.0
    if mode.kind == KIND_DRAFT:
        draft = mode.draft_spec()
        return draft_vram_gb(draft, ctx, parallel) if draft else 0.0
    if mode.kind == KIND_LORA:
        try:
            return Path(mode.gguf).stat().st_size / 1e9 * 1.2
        except OSError:
            return 0.0
    return 0.0


def describe_active(spec: object) -> str:
    """What the settings row shows: the mode's label, or the base."""
    wanted = active_mode_id(spec)
    mode = active_mode(spec)
    if mode is not None:
        return mode.choice_label + ("" if mode.exists else " (missing)")
    if wanted:
        return f"{wanted} (not installed)"
    return BASE_LABEL


def mode_fields(spec: object, mode_id: str) -> dict:
    """The registry fields that select ``mode_id`` on ``spec``.

    Returns the whole projection, not one field: the catalog, the
    active id and the flat ``draft*`` fields have to change together or
    the next load reads a half-switched model. ``BASE_MODE_ID`` (or an
    empty string) clears the projection and keeps the catalog.

    Selecting a mode found beside the model but not yet registered adds
    it to the catalog, which is how a dropped-in file becomes a mode.
    """
    raw_id = (mode_id or "").strip()
    catalog = adapter_modes(spec)
    if raw_id in ("", BASE_MODE_ID):
        return {
            "adapter_mode": "",
            "adapters": [m.to_dict() for m in catalog],
            "draft": "",
            "draft_spec_type": "",
            "draft_n_max": 0,
            "draft_repo": "",
            "draft_file": "",
            "draft_extra_args": [],
        }
    mode = next((m for m in catalog if m.id == raw_id), None)
    if mode is None:
        # A registry that predates the catalog has one active mode in
        # the flat fields; picking it again must not say "not
        # installed" and must migrate it into the catalog.
        mode = next(
            (m for m in available_modes(spec) if m.id == raw_id), None
        )
    if mode is None:
        mode = next(
            (m for m in discover_adapters(spec) if m.id == raw_id), None
        )
    if mode is not None and not any(m.id == mode.id for m in catalog):
        catalog = [*catalog, mode]
    if mode is None:
        raise KeyError(
            f"adapter mode '{raw_id}' is not installed for "
            f"'{getattr(spec, 'name', 'this model')}'"
        )
    out = {
        "adapter_mode": mode.id,
        "adapters": [m.to_dict() for m in catalog],
        "draft": "",
        "draft_spec_type": "",
        "draft_n_max": 0,
        "draft_repo": "",
        "draft_file": "",
        "draft_extra_args": [],
    }
    if mode.kind == KIND_DRAFT:
        out.update({
            "draft": mode.gguf,
            "draft_spec_type": mode.spec_type or SPEC_TYPE_DRAFT_DFLASH,
            "draft_n_max": mode.n_max or DRAFT_N_MAX_DEFAULT,
            "draft_repo": mode.repo,
            "draft_file": mode.file,
            "draft_extra_args": list(mode.extra_args),
        })
    return out


def _target_path(spec: object) -> Path | None:
    gguf = str(getattr(spec, "gguf", "") or "").strip()
    if not gguf:
        return None
    path = Path(gguf)
    if path.is_absolute():
        return path if path.is_file() else None
    beside = Path.home() / "models" / str(getattr(spec, "name", "")) / path.name
    return beside if beside.is_file() else None


def _scan_adapter_folder(
    folder: Path, known_paths: set[str], seen_ids: set[str],
) -> list[AdapterMode]:
    out: list[AdapterMode] = []
    try:
        entries = sorted(folder.glob("*.gguf"))
    except OSError:
        return out
    for path in entries:
        name = path.name.lower()
        if name.startswith("mmproj") or name.startswith("vae"):
            continue
        if _SPLIT_SUFFIX_RE.search(path.stem):
            continue
        try:
            resolved = str(path.resolve())
        except OSError:
            continue
        if resolved in known_paths:
            continue
        kind = kind_for_path(path)
        if kind is None:
            continue
        mode_id = mode_id_for(path.name)
        if mode_id in seen_ids:
            mode_id = f"{mode_id}-{len(seen_ids) + 1}"
        seen_ids.add(mode_id)
        known_paths.add(resolved)
        out.append(AdapterMode(
            id=mode_id,
            label=_label_for(str(path), kind),
            kind=kind,
            gguf=str(path),
        ))
    return out


def _family_slugs(name: str) -> set[str]:
    """The family names a sibling adapter directory may be prefixed by."""
    stem = re.sub(r"(?i)-gguf$", "", Path(name).name)
    return {
        key for key in (_slug(target_family(Path(name).name)), _slug(stem))
        if key
    }


def discover_adapters(spec: object) -> list[AdapterMode]:
    """Adapter GGUFs sitting beside the model but not in the catalog.

    This is the "someone already has the base model" path: the adapter
    is dropped into the model's own directory (where a pull puts it) or
    lives in its own sibling directory named after the target family
    (``qwen3.8-27b-dspine-b16-gguf``), and the settings dropdown offers
    it without a registry edit. It is read-only; selecting the mode
    registers it through :func:`mode_fields`.
    """
    target = _target_path(spec)
    if target is None:
        return []
    folder = target.parent
    known_paths = {
        str(Path(m.gguf).resolve()) for m in available_modes(spec) if m.gguf
    }
    known_paths.add(str(target.resolve()))
    seen_ids = {m.id for m in available_modes(spec)}

    out = _scan_adapter_folder(folder, known_paths, seen_ids)
    # The published DSpine layout puts the adapter in a repository (and
    # often a folder) of its own. Scan only sibling folders that name
    # themselves as adapters and carry this target's family, so an
    # unrelated model's directory is never read.
    families = _family_slugs(
        str(getattr(spec, "gguf", "") or "") or str(getattr(spec, "name", ""))
    )
    families |= _family_slugs(str(getattr(spec, "name", "")))
    try:
        siblings = sorted(p for p in folder.parent.iterdir() if p.is_dir())
    except OSError:
        return out
    for sibling in siblings:
        if sibling == folder:
            continue
        lowered = sibling.name.lower()
        if "dspine" not in lowered and "lora" not in lowered:
            continue
        slug = _slug(sibling.name)
        if not any(family in slug for family in families):
            continue
        out.extend(_scan_adapter_folder(sibling, known_paths, seen_ids))
    return out


def choice_map(spec: object) -> dict[str, str]:
    """Dropdown choices: base first, then registered, then found."""
    choices: dict[str, str] = {BASE_LABEL: BASE_MODE_ID}
    for mode in available_modes(spec):
        label = mode.choice_label
        if label in choices:
            label = f"{label} ({mode.id})"
        choices[label] = mode.id
    for mode in discover_adapters(spec):
        label = f"{mode.choice_label} (found on disk)"
        while label in choices:
            label += " "
        choices[label] = mode.id
    return choices


def adapter_kv_hint(mode: AdapterMode) -> str:
    """The one-line reason a mode cannot run on the current build, if any."""
    if not mode.exists:
        return f"the adapter file is missing: {mode.gguf}"
    if mode.kind not in ADAPTER_KINDS:
        return (
            f"kind '{mode.kind}' is not supported (known kinds: "
            f"{', '.join(ADAPTER_KINDS)})"
        )
    return ""


__all__ = [
    "ADAPTER_KINDS",
    "BASE_LABEL",
    "BASE_MODE_ID",
    "DRAFT_KV_WINDOW",
    "KIND_DRAFT",
    "KIND_LORA",
    "AdapterMode",
    "active_mode",
    "active_mode_id",
    "adapter_kv_hint",
    "adapter_launch_args",
    "adapter_modes",
    "attached_vram_gb",
    "available_modes",
    "choice_map",
    "describe_active",
    "discover_adapters",
    "kind_for_path",
    "mode_fields",
    "mode_id_for",
    "quant_of",
    "resolve_draft",
]
