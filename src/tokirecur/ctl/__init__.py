"""Multi-model plug-and-play management.

Each model in the registry owns a full stack: a llama-server and a
tokirecur daemon on dedicated ports. ``toki model use <name>`` brings
up the chosen model (the 9B "small" model stays on at all times),
stops any other big-model stack to free RAM, and rewrites the default
model in opencode, openclaw and hermes configs so the same model is
selected everywhere.

Registry:    ~/.config/tokirecur/models.yaml
State:       ~/.local/state/tokirecur/<name>/  (pid files + logs)
Active:      ~/.config/tokirecur/active-model
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

from ..ggufmeta import read_gguf_meta

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY = Path(
    os.environ.get("TOKI_MODEL_REGISTRY", str(Path.home() / ".config/tokirecur/models.yaml"))
)
STATE_ROOT = Path(
    os.environ.get("TOKI_STATE_DIR", str(Path.home() / ".local/state/tokirecur"))
)
ACTIVE_MARKER = Path(
    os.environ.get("TOKI_ACTIVE_MODEL", str(Path.home() / ".config/tokirecur/active-model"))
)

DEFAULT_AUTH = os.environ.get("TOKI_AUTH_TOKEN", "demo-token")

# Reference defaults: registry entries matching these names (source ==
# "default") are treated as user-configured and never rewritten by
# backfill_binary_env. They are NOT preloaded — a fresh install starts
# with an empty registry.
DEFAULT_MODELS: list[dict[str, Any]] = [
    {
        "name": "9b",
        "label": "Qwen3.5 9B (Q4_K_M)",
        "gguf": str(Path.home() / "Downloads/Qwen3.5-9B-Q4_K_M.gguf"),
        "llama_port": 8088,
        "daemon_port": 8787,
        "ctx_size": 200000,
        "parallel": 2,
        "kv": "f16",
        "max_tokens": 100000,
        "model_id": "qwen3.5-9b",
        "always_on": False,
        "boot": False,
        "source": "default",
    },
    {
        "name": "35b",
        "label": "Qwen3.6 35B-A3B (UD-Q4_K_M)",
        "gguf": str(Path.home() / "models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
        "llama_port": 8089,
        "daemon_port": 8788,
        "ctx_size": 200000,
        "parallel": 2,
        "kv": "f16",
        "max_tokens": 100000,
        "model_id": "qwen3.6-35b-a3b",
        "always_on": False,
        "source": "default",
    },
    {
        "name": "laguna",
        "label": "Laguna S 2.1 (Q4_K_M, 64k ctx)",
        "gguf": str(Path.home() / "models/laguna/laguna-s-2.1-Q4_K_M.gguf"),
        "llama_port": 8090,
        "daemon_port": 8789,
        "ctx_size": 131072,
        "parallel": 2,
        "kv": "q8_0",
        "max_tokens": 64000,
        "model_id": "laguna-s-2.1",
        "always_on": False,
        "boot": False,
        "binary": str(Path.home() / "llama.cpp/build/bin/llama-server"),
        "source": "default",
    },
]


@dataclass
class ModelSpec:
    name: str
    label: str
    gguf: str
    llama_port: int
    daemon_port: int
    ctx_size: int
    parallel: int
    kv: str
    max_tokens: int
    model_id: str
    always_on: bool = False
    boot: bool = False
    binary: str = "llama-server"
    extra_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    # "auto" (default): the daemon streams reasoning and reserves token
    # headroom so thinking models always answer. "off": pass
    # `--reasoning off` to llama-server for templates that honor it
    # (stock Qwen3.5); ignored by templates that hard-code thinking.
    reasoning: str = "auto"
    # Tracks how the spec entered the registry. "default" for the
    # built-in DEFAULT_MODELS (always treated as configured, even when
    # they lack an explicit `binary`); "pulled" for entries created by
    # `pull_model`. Used by `backfill_binary_env` to skip configured
    # defaults that intentionally rely on PATH defaults.
    source: str = "default"

    @property
    def llama_url(self) -> str:
        return f"http://127.0.0.1:{self.llama_port}"

    @property
    def daemon_url(self) -> str:
        return f"http://127.0.0.1:{self.daemon_port}"

    def state_dir(self, root: Path | None = None) -> Path:
        return (root or STATE_ROOT) / "model" / self.name


def _ensure_registry() -> Path:
    DEFAULT_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    if not DEFAULT_REGISTRY.exists():
        # Fresh installs start with zero preloaded models — the user
        # explicitly installs one via `toki model pull` / `toki up`.
        _atomic_write_text(
            DEFAULT_REGISTRY,
            yaml.safe_dump({"models": []}, sort_keys=False),
        )
    return DEFAULT_REGISTRY


def _estimate_need(spec: ModelSpec) -> float:
    """Rough resident footprint (GB): weights + KV + workspace headroom."""
    weight_gb = Path(spec.gguf).stat().st_size / 1e9 if Path(spec.gguf).exists() else 0.0
    return weight_gb * 1.2 + 4.0


def _available_gb() -> float:
    """Best-effort available-RAM estimate in GB; ``inf`` when unknown.

    Linux reads MemAvailable from /proc/meminfo. macOS queries
    hw.memsize via sysctl (total RAM — there is no cheap per-process
    "available" figure, and total is the useful bound for the UMA
    guard). Windows uses GlobalMemoryStatusEx. Anything else falls
    back to ``inf`` so the RAM guard never blocks a machine we
    cannot measure.
    """
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576.0
    except OSError:
        pass
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            return int(out.stdout.strip()) / 1e9
        except Exception:
            return float("inf")
    if sys.platform.startswith("win"):
        try:
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            ms = _MEMORYSTATUSEX()
            ms.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return ms.ullAvailPhys / 1e9
        except Exception:
            return float("inf")
    return float("inf")


_KV_BYTES_PER_ELEM = {"f16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625,
                      "q4_1": 0.6875, "q5_0": 0.6875, "q5_1": 0.8125}
_CTX_FLOOR = 512


def _kv_bytes_per_token(spec: ModelSpec) -> float | None:
    """llama.cpp KV cache bytes per token across all layers/slots.

    Exact math: per layer, k+v each hold ``head_dim * n_head_kv``
    values, with ``head_dim = n_embd / n_head``. Requires the model's
    GGUF metadata; ``None`` when unreadable (caller skips clamping).
    """
    meta = read_gguf_meta(spec.gguf)
    if not meta.complete or meta.n_head == 0:
        return None
    head_dim = meta.n_embd / meta.n_head
    per_layer = head_dim * meta.n_head_kv * 2.0 * _KV_BYTES_PER_ELEM.get(spec.kv, 2.0)
    return per_layer * meta.n_layer * spec.parallel


def _clamp_ctx(spec: ModelSpec, avail_gb: float | None = None,
               weight_gb: float | None = None) -> int:
    """Largest context (>= ``_CTX_FLOOR``) whose KV cache fits in RAM.

    Uses the same headroom budget as ``_estimate_need``: weights * 1.2
    + 4 GB. When the GGUF metadata is unavailable (or the machine's
    RAM can't be measured) the configured ``ctx_size`` is returned
    untouched and llama-server's own auto-reduction applies.
    """
    kv_per_token = _kv_bytes_per_token(spec)
    if kv_per_token is None:
        return spec.ctx_size
    if avail_gb is None:
        avail_gb = _available_gb()
    if weight_gb is None:
        weight_gb = Path(spec.gguf).stat().st_size / 1e9 if Path(spec.gguf).exists() else 0.0
    budget = avail_gb - weight_gb * 1.2 - 4.0
    if budget <= 0 or not math.isfinite(budget):
        return spec.ctx_size  # unknown RAM or weights alone don't fit
    max_fit = int(budget * 1e9 / kv_per_token)
    return max(_CTX_FLOOR, min(spec.ctx_size, max_fit))


def _effective_ctx(spec: ModelSpec) -> int:
    """The context llama-server actually runs with (clamped, then
    corrected to the real slot size after start). Falls back to the
    configured ``ctx_size`` before the first start."""
    state = spec.state_dir() / "effective_ctx.txt"
    try:
        return int(state.read_text().strip())
    except (OSError, ValueError):
        return spec.ctx_size


def _check_ram(spec: ModelSpec) -> None:
    """Refuse to start when the model cannot fit in available RAM."""
    need = _estimate_need(spec)
    avail = _available_gb()
    if avail < need:
        raise RuntimeError(
            f"not enough free RAM to start '{spec.name}': need ~{need:.0f} GB "
            f"(model + KV + headroom), only {avail:.0f} GB available. "
            f"Stop other big-model stacks first (they are stopped automatically, "
            f"or pass --keep-others only if RAM allows)."
        )


def load_registry(path: Path | None = None) -> list[ModelSpec]:
    """Load the model registry, creating the default file on first run.

    Caches the parsed result per-registry-path for the lifetime of the
    process. Call :func:`invalidate_registry_cache` after any mutation
    (the writers do this) so subsequent reads see the change.
    """
    reg = _ensure_registry() if path is None else path
    cache_key = str(reg)
    cached = _REGISTRY_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)  # hand back a copy so callers can't mutate
    data = _load_registry_yaml(reg)
    specs: list[ModelSpec] = []
    for m in data.get("models", []):
        specs.append(_build_spec(m))
    _REGISTRY_CACHE[cache_key] = list(specs)
    return specs


def _load_registry_yaml(reg: Path) -> dict[str, Any]:
    """Read + parse the registry YAML, falling back to defaults if missing."""
    if not reg.exists():
        reg = _ensure_registry()
    return yaml.safe_load(reg.read_text()) or {}


def _build_spec(m: dict[str, Any]) -> ModelSpec:
    """Build a ModelSpec from a raw registry entry, filling defaults."""
    m = dict(m)
    raw = dict(m)
    m.setdefault("label", m["name"])
    m.setdefault("kv", "f16")
    m.setdefault("parallel", 2)
    m.setdefault("always_on", False)
    m.setdefault("boot", False)
    m.setdefault("model_id", m["name"])
    m.setdefault("binary", "llama-server")
    m.setdefault("extra_args", [])
    m.setdefault("source", "default")
    return ModelSpec(raw=raw, **m)


# Process-wide registry cache: {registry_path_str -> [ModelSpec]}.
# Invalidated by save_registry / pull_model / update_spec_field /
# backfill_binary_env. Skipped when a custom `path` is supplied.
_REGISTRY_CACHE: dict[str, list[ModelSpec]] = {}


def invalidate_registry_cache(path: Path | None = None) -> None:
    """Drop cached registry entries. Call after any write."""
    if path is None:
        _REGISTRY_CACHE.clear()
    else:
        _REGISTRY_CACHE.pop(str(path), None)


def get_model(name: str, registry: Path | None = None) -> ModelSpec:
    for spec in load_registry(registry):
        if spec.name == name:
            return spec
    if not load_registry(registry):
        raise KeyError(
            "no models installed — pull one first (`toki model pull <hf-repo>`)"
        )
    raise KeyError(f"unknown model '{name}' (see `toki model list`)")


def read_active(registry: Path | None = None) -> str | None:
    if ACTIVE_MARKER.exists():
        name = ACTIVE_MARKER.read_text().strip()
        try:
            get_model(name, registry)
            return name
        except KeyError:
            pass  # fall through to always-on default
    for spec in load_registry(registry):
        if spec.always_on:
            return spec.name
    return None


def set_active(name: str, registry: Path | None = None) -> None:
    get_model(name, registry)  # raises KeyError on bad name
    ACTIVE_MARKER.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_MARKER.write_text(name + "\n")

# ── Registry save / pull ──────────────────────────────────────────────


def _atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` atomically.

    A non-atomic ``write_text`` truncates the target before writing the
    new bytes, so a crash mid-write leaves a half-empty YAML file that
    fails to load (and silently wipes the user's registry). Writing to
    a sibling temp file in the same directory and ``os.replace``-ing
    it onto the target makes the swap atomic on POSIX filesystems.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


# POSIX-only advisory lock for read-modify-write on the registry.
# Atomic rename alone prevents torn files but not lost updates: two
# processes can both read the file, both edit in-memory, both rename
# back, and the later writer silently clobbers the earlier change.
# ``fcntl.flock`` is BSD/Linux/macOS — Windows would need a different
# mechanism; the safe-on-other-platforms path is just no locking.
try:
    import fcntl  # type: ignore[import-not-found]
    _HAS_FLOCK = True
except ImportError:
    fcntl = None  # type: ignore[assignment]
    _HAS_FLOCK = False


class _flock:
    """Context manager that acquires an exclusive ``fcntl`` lock on a
    lockfile adjacent to ``path``.

    No-op when ``fcntl`` is unavailable (Windows). The lockfile lives
    in the same directory as the target so concurrent processes
    editing the same registry contend on a single inode.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: int | None = None

    def __enter__(self) -> _flock:
        if not _HAS_FLOCK:
            return self
        lock_path = self.path.with_name(self.path.name + ".lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX)
        except OSError:
            # Best-effort: if we can't acquire the lock, don't block
            # the registry edit. The atomic-rename still prevents torn
            # files; we just lose the cross-process coordination.
            os.close(self._fd)
            self._fd = None
        return self

    def __exit__(self, *exc) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(self._fd)
            self._fd = None


def save_registry(specs: list[ModelSpec], path: Path | None = None) -> Path:
    """Persist the model registry, preserving every original key.

    Holds an exclusive ``fcntl`` lock on a sibling lockfile so two
    concurrent writers can't lose each other's changes. The lock is
    a no-op on platforms without ``fcntl`` (Windows); atomic-rename
    still prevents torn files in that case.
    """
    reg = path or DEFAULT_REGISTRY
    with _flock(reg):
        _atomic_write_text(
            reg,
            yaml.safe_dump({"models": [s.raw for s in specs]}, sort_keys=False),
        )
    invalidate_registry_cache(reg)
    return reg


def set_binary(name: str, binary: str, path: Path | None = None,
               env: dict[str, str] | None = None) -> ModelSpec:
    """Point an existing registry model at a different llama-server binary.

    Replaces ``binary`` in place, keeps everything else (env, ports,
    extra_args) untouched, and rewrites the registry atomically. When
    ``env`` is given, its entries are merged into the model's env
    (existing keys win on conflict — callers supply only what they
    want to add or override).
    """
    specs = load_registry(path)
    match = next((s for s in specs if s.name == name), None)
    if match is None:
        if not specs:
            raise KeyError(
                f"no models installed — pull one first "
                f"(`toki model pull <hf-repo> --name {name}`)"
            )
        raise KeyError(f"model '{name}' not in registry")
    raw = dict(match.raw)
    raw["binary"] = binary
    if env:
        merged = dict(raw.get("env") or {})
        merged.update(env)
        raw["env"] = merged
    updated = ModelSpec(raw=raw, **raw)
    save_registry([updated if s.name == name else s for s in specs], path)
    return updated


# Fields editable from the launcher settings pane. Each maps to a
# validator: the value comes in as a string and is converted/checked
# before being written back. Anything not in this list is read-only
# from the settings UI.
def _truthy(v: str) -> bool:
    """Parse yes/no/true/false/1/0 → bool; anything else is rejected."""
    s = v.strip().lower()
    if s in ("true", "yes", "1", "on"):
        return True
    if s in ("false", "no", "0", "off"):
        return False
    raise ValueError(f"expected true/false/yes/no/1/0, got {v!r}")


_EDITABLE_FIELDS: dict[str, Callable[[str], Any]] = {
    "gguf": str,
    "llama_port": int,
    "daemon_port": int,
    "ctx_size": int,
    "kv": str,
    "max_tokens": int,
    "parallel": int,
    "model_id": str,
    "binary": str,
    "reasoning": str,
    "always_on": _truthy,
    "boot": _truthy,
}


def update_spec_field(
    name: str,
    field: str,
    value: str,
    registry: Path | None = None,
) -> ModelSpec:
    """Update one registry field on a named spec, validate, persist.

    Returns the updated spec so the caller can refresh its view.
    Raises ``ValueError`` for unknown fields or invalid values.
    """
    if field not in _EDITABLE_FIELDS:
        raise ValueError(
            f"field '{field}' is not editable (choose from "
            + ", ".join(sorted(_EDITABLE_FIELDS))
            + ")"
        )
    try:
        parsed = _EDITABLE_FIELDS[field](value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"bad value for '{field}': {e}") from e
    # Lock spans the whole read-modify-write so two concurrent
    # callers can't both read the same version, both edit, and both
    # write — the second writer would clobber the first.
    reg = registry or DEFAULT_REGISTRY
    with _flock(reg):
        specs = load_registry(reg)
        idx = next((i for i, s in enumerate(specs) if s.name == name), None)
        if idx is None:
            raise KeyError(f"unknown model '{name}'")
        spec = specs[idx]
        new_raw = dict(spec.raw)
        new_raw[field] = parsed
        # Mirror ModelSpec defaults so the dataclass accepts the new raw.
        new_raw.setdefault("label", name)
        new_raw.setdefault("kv", "f16")
        new_raw.setdefault("parallel", 2)
        new_raw.setdefault("always_on", False)
        new_raw.setdefault("boot", False)
        new_raw.setdefault("model_id", name)
        new_raw.setdefault("binary", "llama-server")
        new_raw.setdefault("extra_args", [])
        new_raw.setdefault("env", {})
        new_raw.setdefault("reasoning", "auto")
        new_raw.setdefault("source", "default")
        # Always re-declare key so the YAML reflects the change.
        new_spec = ModelSpec(raw=new_raw, **new_raw)
        specs[idx] = new_spec
        _atomic_write_text(
            reg,
            yaml.safe_dump({"models": [s.raw for s in specs]}, sort_keys=False),
        )
        invalidate_registry_cache(reg)
    return new_spec


def next_free_ports(specs: list[ModelSpec]) -> tuple[int, int]:
    """Pick the next unused llama/daemon port pair (max + 1)."""
    llama = max((s.llama_port for s in specs), default=8087) + 1
    daemon = max((s.daemon_port for s in specs), default=8786) + 1
    return llama, daemon


def _slugify_name(name: str) -> str:
    """Lowercase, replace anything not [a-z0-9._-] with a dash."""
    return re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-._")


def _reference_spec(specs: list[ModelSpec]) -> ModelSpec | None:
    """Pick a 'good' existing spec to inherit binary/env from (preferring
    a custom ROCm build over the Debian default)."""
    custom = [s for s in specs if s.binary and s.binary != "llama-server"]
    return custom[0] if custom else None


def pull_defaults(specs: list[ModelSpec]) -> tuple[str, dict[str, str]]:
    """Resolve binary + env defaults for a new model.

    Order: TOKI_LLAMA_SERVER env → an existing spec's binary + env (the
    machine's tuned build, e.g. ROCm b10008) → PATH's llama-server.
    """
    env_override = os.environ.get("TOKI_LLAMA_SERVER")
    if env_override:
        ref = _reference_spec(specs)
        return env_override, dict(ref.env) if ref else {}
    ref = _reference_spec(specs)
    if ref:
        return ref.binary, dict(ref.env)
    return "llama-server", {}


_PREFERRED_QUANTS = (
    "Q4_K_M", "Q5_K_M", "Q4_0", "Q8_0", "Q6_K",
    "Q4_K_S", "Q3_K_M", "Q2_K", "IQ4_XS", "IQ3_XXS", "BF16", "F16",
)


def recommend_quant(files: list[str]) -> str:
    """Pick the most sensible quant from GGUF filenames (first hit wins)."""
    for quant in _PREFERRED_QUANTS:
        for f in files:
            if quant in f:
                return f
    return files[0]


def _gguf_sizes(api: Any, repo_id: str, files: list[str]) -> dict[str, int]:
    """Real file sizes from the hub (best-effort; {} on any failure)."""
    try:
        infos = api.get_paths_info(repo_id, paths=files)
    except Exception:
        return {}
    return {i.path: i.size for i in infos if getattr(i, "size", None)}


def pull_model(
    repo_id: str,
    name: str | None = None,
    *,
    file: str | None = None,
    ctx_size: int = 200000,
    kv: str = "f16",
    endpoint: str | None = None,
    reasoning: str = "auto",
    registry: Path | None = None,
    choose_file: Callable[[list[tuple[str, int | None]]], str] | None = None,
) -> ModelSpec:
    """Download a GGUF from Hugging Face (mirror-friendly) and register it.

    ``name`` defaults to a lowercase slug of the repo's last segment;
    if that slug is already taken, a ``-2`` suffix is added. ``endpoint``
    sets the HF mirror base URL (e.g. ``http://localhost:8090``) for this
    call only; otherwise the standard ``HF_ENDPOINT`` env var is honored
    by huggingface_hub. When the repo holds several GGUF files, ``file``
    selects one; otherwise ``choose_file`` is called with the real
    candidates as ``(filename, size_bytes)`` pairs (sizes fetched from
    the hub, best-effort); without either, an error listing the
    candidates is raised. Binary and env inherit from an existing spec's
    custom build (e.g. the machine's tuned ROCm llama.cpp) when
    available; ``reasoning`` selects the thinking-model handling.
    """
    if not name:
        name = repo_id.rstrip("/").rsplit("/", 1)[-1]
    name = _slugify_name(name)
    if not name:
        raise ValueError(
            f"cannot derive a model name from '{repo_id}' — pass --name"
        )
    specs = load_registry(registry)
    if any(s.name == name for s in specs):
        raise ValueError(f"model '{name}' already exists in the registry")

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is required for `toki model pull` "
            "(pip install huggingface_hub)"
        ) from None

    api = HfApi(endpoint=endpoint)
    try:
        files = api.list_repo_files(repo_id)
    except Exception as e:
        raise RuntimeError(f"cannot reach '{repo_id}' on Hugging Face: {e}") from e
    ggufs = [f for f in files if f.endswith(".gguf")]
    if file is not None:
        if file not in ggufs:
            raise RuntimeError(
                f"no file '{file}' in '{repo_id}'; candidates: "
                + (", ".join(ggufs) if ggufs else "none")
            )
        filename = file
    elif len(ggufs) == 1:
        filename = ggufs[0]
    elif not ggufs:
        raise RuntimeError(
            f"'{repo_id}' has no GGUF files — toki needs GGUF, not safetensors. "
            "Look for a *-GGUF repo (e.g. a bartowski or owao build of the "
            "same model) and pull that instead."
        )
    else:
        candidates = sorted(ggufs)
        if choose_file is not None:
            sizes = _gguf_sizes(api, repo_id, candidates)
            filename = choose_file([(f, sizes.get(f)) for f in candidates])
            if filename not in candidates:
                raise RuntimeError(
                    f"file chooser returned unknown file '{filename}'"
                )
        else:
            raise RuntimeError(
                f"'{repo_id}' has several GGUF files; pick one with --file: "
                + ", ".join(candidates[:8])
            )

    local_dir = Path.home() / "models" / name
    local_dir.mkdir(parents=True, exist_ok=True)
    old_endpoint = os.environ.get("HF_ENDPOINT")
    try:
        if endpoint:
            os.environ["HF_ENDPOINT"] = endpoint
        snapshot_download(
            repo_id,
            allow_patterns=[f"*{filename}"],
            local_dir=str(local_dir),
            resume_download=True,
        )
    except Exception as e:
        raise RuntimeError(f"download failed: {e}") from e
    finally:
        if old_endpoint is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = old_endpoint

    llama_port, daemon_port = next_free_ports(specs)
    binary, env = pull_defaults(specs)
    raw = {
        "name": name,
        "label": Path(filename).stem,
        "gguf": str(local_dir / filename),
        "llama_port": llama_port,
        "daemon_port": daemon_port,
        "ctx_size": ctx_size,
        "parallel": 2,
        "kv": kv,
        "max_tokens": ctx_size // 2,
        "model_id": repo_id,
        "always_on": False,
        "boot": False,
        "binary": binary,
        "env": env,
        "reasoning": reasoning,
        "source": "pulled",
    }
    spec = ModelSpec(raw=raw, **raw)
    save_registry([*specs, spec], registry)
    return spec


# ── Process helpers ────────────────────────────────────────────────────


def _is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _wait_for(url: str, timeout: float = 240.0) -> bool:
    started = time.time()
    while time.time() - started < timeout:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def llama_reachable(spec: ModelSpec) -> bool:
    try:
        return httpx.get(f"{spec.llama_url}/health", timeout=2.0).status_code == 200
    except Exception:
        return False


def daemon_reachable(spec: ModelSpec) -> bool:
    try:
        return httpx.get(f"{spec.daemon_url}/healthz", timeout=2.0).status_code == 200
    except Exception:
        return False


def slot_context(spec: ModelSpec) -> int | None:
    """Per-slot context (from /slots) of the model's llama-server."""
    try:
        r = httpx.get(f"{spec.llama_url}/slots", timeout=3.0)
        if r.status_code != 200:
            return None
        slots = r.json()
        return slots[0]["n_ctx"] if slots else None
    except Exception:
        return None


def llama_pid(spec: ModelSpec) -> int | None:
    return _read_pid(spec.state_dir() / "llama.pid")


def daemon_pid(spec: ModelSpec) -> int | None:
    return _read_pid(spec.state_dir() / "daemon.pid")


# Cache of (pid, alive?) for llama/daemon PIDs read from pid files.
# Reads are sequential per poll, so the cache is invalidated when the
# file's mtime changes (cheap stat). The previous version fell back to
# `subprocess.run ss -tlnp` on every poll whenever a pid file was
# missing — a multi-second shellout for nothing.
_PID_FILE_CACHE: dict[Path, tuple[float, int | None]] = {}


def _read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    cached = _PID_FILE_CACHE.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        pid = int(path.read_text().strip())
    except (ValueError, OSError):
        pid = None
    _PID_FILE_CACHE[path] = (mtime, pid)
    return pid


# Cache of (pid, alive?) for ports probed via `ss`. Keyed by port.
# TTL is short (1 s) so we still detect freshly-started processes
# promptly when a pid file is missing — without shelling out every poll.
_PORT_PID_CACHE: dict[int, tuple[float, int | None]] = {}


def _pid_on_port(port: int) -> int | None:
    """Find the pid listening on `port` via `ss` (Linux).

    Cached for ~1 s so a busy poll loop doesn't shell out per model per
    tick. The TTL is short enough that newly-spawned processes are
    picked up promptly while the daemon is still probing.
    """
    now = time.time()
    cached = _PORT_PID_CACHE.get(port)
    if cached and now - cached[0] < 1.0:
        return cached[1]
    try:
        out = subprocess.run(
            ["ss", "-tlnp"], capture_output=True, text=True, timeout=5.0
        ).stdout
    except Exception:
        return None
    pid = None
    for line in out.splitlines():
        if f":{port} " not in line and not line.endswith(f":{port}"):
            continue
        m = re.search(r"pid=(\d+)", line)
        if m:
            pid = int(m.group(1))
            break
    _PORT_PID_CACHE[port] = (now, pid)
    return pid


def model_status(spec: ModelSpec) -> dict[str, Any]:
    lp, dp = llama_pid(spec), daemon_pid(spec)
    llama_up = llama_reachable(spec)
    daemon_up = daemon_reachable(spec)
    return {
        "name": spec.name,
        "label": spec.label,
        "llama": {
            "pid": lp if lp and _is_alive(lp) else (_pid_on_port(spec.llama_port) if llama_up else None),
            "reachable": llama_up,
            "slot_ctx": slot_context(spec),
        },
        "daemon": {
            "pid": dp if dp and _is_alive(dp) else (_pid_on_port(spec.daemon_port) if daemon_up else None),
            "reachable": daemon_up,
        },
        "gguf": spec.gguf,
    }


# ── Start / stop ───────────────────────────────────────────────────────


def _llama_args(spec: ModelSpec, ctx_override: int | None = None) -> list[str]:
    if not Path(spec.gguf).exists():
        raise FileNotFoundError(f"model file not found: {spec.gguf}")
    ctx = spec.ctx_size if ctx_override is None else ctx_override
    args = [
        spec.binary,
        "-m", spec.gguf,
        "--host", "127.0.0.1",
        "--port", str(spec.llama_port),
        "-c", str(ctx),
        "--parallel", str(spec.parallel),
        "-lv", "3",
    ]
    if spec.kv != "f16":
        args += ["--cache-type-k", spec.kv, "--cache-type-v", spec.kv]
    if spec.reasoning == "off":
        args += ["--reasoning", "off"]
    elif spec.reasoning == "on":
        args += ["--reasoning", "on"]
    args += spec.extra_args
    return args


# systemd sets INVOCATION_ID in the environment of every service process;
# use it to decide whether to detach the spawned llama-server / daemon
# into a new session. Detaching is the right call for `toki up` (the
# launcher should be able to exit without killing the stack) but wrong
# under systemd (the wrapper is the cgroup owner; detached children
# survive cgroup kills and become orphaned zombies when the wrapper
# finally reaps them).
_UNDER_SYSTEMD = bool(os.environ.get("INVOCATION_ID"))
_DETACH = not _UNDER_SYSTEMD


def _log_tail(path: Path, n: int = 40) -> str:
    """Last ``n`` lines of a log file, for surfacing start failures."""
    try:
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return "(log unavailable)"


def _wait_healthy(url: str, proc: subprocess.Popen | None, log_path: Path,
                  timeout: float, service: str) -> None:
    """Wait for a service's health endpoint, failing fast if the child
    process dies instead of spinning until ``timeout``.

    Raises RuntimeError with the log tail on child death, so "starting
    model" can never hang silently behind a crashed llama-server.
    """
    started = time.time()
    while time.time() - started < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"{service} exited during startup (exit code {proc.returncode}); "
                f"last lines of {log_path}:\n{_log_tail(log_path)}"
            )
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code < 500:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(
        f"{service} did not become healthy within {timeout:.0f}s; "
        f"last lines of {log_path}:\n{_log_tail(log_path)}"
    )


def _real_slot_ctx(url: str) -> int | None:
    """The context the running llama-server actually gave its slots.

    llama.cpp auto-reduces the requested ``-c`` when the KV cache
    can't fit memory, so the configured ctx_size can lie. Reading the
    live slot size gives providers (opencode configs, daemon budget)
    the single source of truth.
    """
    try:
        r = httpx.get(f"{url}/slots", timeout=3.0)
        r.raise_for_status()
        ctxs = [int(s.get("n_ctx", 0)) for s in r.json() if s.get("n_ctx")]
        return min(ctxs) if ctxs else None
    except Exception:
        return None


def start_model(spec: ModelSpec, wait: bool = True) -> None:
    """Bring up the model's llama-server + daemon (idempotent)."""
    state = spec.state_dir()
    state.mkdir(parents=True, exist_ok=True)

    if not llama_reachable(spec):
        if not Path(spec.gguf).exists():
            raise FileNotFoundError(f"model file not found: {spec.gguf}")
        _check_ram(spec)
        ctx = _clamp_ctx(spec)
        if ctx < spec.ctx_size:
            logger.warning(
                "%s: context clamped %d -> %d (KV cache must fit in RAM)",
                spec.name, spec.ctx_size, ctx,
            )
        log = open(state / "llama.log", "ab")
        proc = subprocess.Popen(
            _llama_args(spec, ctx_override=ctx),
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=_DETACH,
            env={**os.environ, **spec.env} if spec.env else None,
        )
        (state / "llama.pid").write_text(str(proc.pid))
        logger.info("%s: llama-server started (pid=%d)", spec.name, proc.pid)
        if wait:
            _wait_healthy(
                f"{spec.llama_url}/health", proc, state / "llama.log",
                timeout=1800.0, service=f"llama-server ({spec.name})",
            )
        # llama.cpp may have reduced the requested context further to
        # fit memory; the real slot size is the truth clients must use.
        real_ctx = _real_slot_ctx(spec.llama_url)
        effective = min(real_ctx, ctx) if real_ctx else ctx
        (state / "effective_ctx.txt").write_text(str(max(_CTX_FLOOR, effective)))

    if not daemon_reachable(spec):
        effective = _effective_ctx(spec)
        log = open(state / "daemon.log", "ab")
        env = {
            **os.environ,
            "TOKI_BACKEND_URL": spec.llama_url,
            "TOKI_MAX_CONTEXT_TOKENS": str(effective),
            "TOKI_REASONING": spec.reasoning,
        }
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "tokirecur.api.app:create_app_from_env",
                "--factory",
                "--host", "127.0.0.1",
                "--port", str(spec.daemon_port),
                "--log-level", "info",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=_DETACH,
        )
        (state / "daemon.pid").write_text(str(proc.pid))
        logger.info("%s: daemon started (pid=%d)", spec.name, proc.pid)
        if wait:
            _wait_healthy(
                f"{spec.daemon_url}/healthz", proc, state / "daemon.log",
                timeout=60.0, service=f"toki daemon ({spec.name})",
            )


def _stop_pidfile(path: Path, grace: float = 8.0) -> bool:
    pid = _read_pid(path)
    if pid is None or not _is_alive(pid):
        path.unlink(missing_ok=True)
        return False
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return False
    deadline = time.time() + grace
    while time.time() < deadline:
        if not _is_alive(pid):
            path.unlink(missing_ok=True)
            return True
        time.sleep(0.2)
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except OSError:
        pass
    path.unlink(missing_ok=True)
    return True


def stop_model(spec: ModelSpec, force: bool = False) -> None:
    """Stop the model's stack. If the stack is up but unmanaged (no pid
    file), only stop it when `force` is set (kills whatever listens on
    the ports)."""
    state = spec.state_dir()
    _stop_pidfile(state / "llama.pid")
    _stop_pidfile(state / "daemon.pid")
    if force:
        for port in (spec.llama_port, spec.daemon_port):
            pid = _pid_on_port(port)
            if pid:
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass
    elif llama_reachable(spec) or daemon_reachable(spec):
        raise RuntimeError(
            f"stack for '{spec.name}' is running but was not started by `toki model`; "
            f"pass --force to stop whatever listens on :{spec.llama_port}/:{spec.daemon_port}"
        )


def backfill_binary_env(specs: list[ModelSpec] | None = None) -> list[str]:
    """Rewrite any non-default registry entry missing `binary`/`env`
    to inherit the machine's tuned build (ROCm build + HSA vars)
    from the first reference spec that already has them.

    Configured defaults (names matching ``DEFAULT_MODELS`` — 9b, 35b,
    laguna) are NEVER touched: they intentionally use the PATH
    ``llama-server`` and rewriting them would silently change their
    behavior on a machine where they've been working fine.

    Legacy pulled entries (before ``pull_defaults`` propagated
    binary/env) ship with the bare PATH default and no env — this
    rewrites them in place. Idempotent: no entry is reported as
    updated if its binary/env don't actually change.

    A side effect: legacy entries without ``source`` in their raw
    dict are tagged ``source="pulled"`` so the next ``pull_model``
    doesn't collide on them and future repairs stay idempotent.
    """
    specs = specs if specs is not None else load_registry()
    binary, env = pull_defaults(specs)
    if binary == "llama-server" and not env:
        return []  # no reference spec to inherit from
    default_names = {m["name"] for m in DEFAULT_MODELS}
    updated: list[str] = []
    new_specs: list[ModelSpec] = []
    for s in specs:
        if s.name in default_names:
            new_specs.append(s)
            continue
        needs_binary = not s.binary or s.binary == "llama-server"
        needs_env = not s.env
        if needs_binary or needs_env:
            new_raw = dict(s.raw)
            new_raw["binary"] = binary
            new_raw["env"] = dict(env)
            new_raw.setdefault("source", "pulled")
            # Mirror load_registry's defaults so ModelSpec(**raw) accepts it.
            new_raw.setdefault("label", new_raw["name"])
            new_raw.setdefault("kv", "f16")
            new_raw.setdefault("parallel", 2)
            new_raw.setdefault("always_on", False)
            new_raw.setdefault("boot", False)
            new_raw.setdefault("model_id", new_raw["name"])
            new_raw.setdefault("extra_args", [])
            new_s = ModelSpec(raw=new_raw, **new_raw)
            if new_s.binary != s.binary or dict(new_s.env) != dict(s.env):
                updated.append(s.name)
            new_specs.append(new_s)
        else:
            new_specs.append(s)
    if updated:
        save_registry(new_specs)
    return updated


def use_model(name: str, *, stop_others: bool = True, registry: Path | None = None) -> ModelSpec:
    """Ensure the chosen model is up, selected, and visible to every CLI.

    The always-on small model (9b) is kept running; other big-model
    stacks are stopped (unless ``stop_others=False``) to free RAM.
    On success the opencode / openclaw / hermes configs are rewritten
    so they have a provider entry for every registry spec — callers
    (``toki model use``, ``toki model pull --use``, the launcher)
    don't have to do this themselves.
    """
    target = get_model(name, registry)
    all_models = load_registry(registry)

    # Self-heal any registry entries written before `pull_defaults`
    # existed (no machine-tuned binary/env). Best-effort: only runs when
    # there's a reference spec to inherit from, otherwise no-ops.
    healed = backfill_binary_env(all_models)
    if healed:
        all_models = load_registry(registry)
        target = get_model(name, registry)
        logger.info("repaired registry entries: %s", ", ".join(healed))

    # Free RAM FIRST: stop competing big-model stacks before loading the
    # target (a 64 GB model cannot share memory with another big model).
    if stop_others:
        for other in all_models:
            if other.name != target.name and not other.always_on:
                try:
                    if llama_reachable(other) or daemon_reachable(other):
                        stop_model(other)
                        logger.info("stopped '%s' (switching to '%s')", other.name, target.name)
                except RuntimeError as e:
                    logger.warning("could not stop '%s': %s", other.name, e)

    start_model(target)

    # Always-on small model for titles / small requests. Best-effort:
    # never fail the switch because a helper model does not fit in RAM
    # or has a broken spec (missing GGUF raises FileNotFoundError, which
    # is an OSError — not a RuntimeError).
    for spec in all_models:
        if spec.always_on and spec.name != target.name:
            try:
                start_model(spec)
            except (RuntimeError, OSError) as e:
                logger.warning("could not start always-on model '%s': %s", spec.name, e)

    set_active(target.name, registry)

    # Wire opencode / openclaw / hermes so every spec (defaults + pulled)
    # has a provider entry and the active model line points at `target`.
    # We never fail `use_model` because of a sync hiccup — a broken
    # client config is recoverable; an unstarted stack isn't.
    try:
        sync_cli_configs(target)
    except Exception as e:
        logger.warning("could not sync CLI configs: %s", e)

    return target


# ── CLI config writers ─────────────────────────────────────────────────


def _backup(path: Path) -> None:
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak-model"))


def _provider_entry(spec: ModelSpec) -> dict[str, Any]:
    # Advertise the context the model ACTUALLY runs with (clamped to
    # fit RAM, corrected to the live slot size after start) — never the
    # registry max_tokens. Advertising a bigger window than the slot
    # exists makes opencode overfill the context and hit llama-server's
    # 400 "not enough tokens".
    ctx = _effective_ctx(spec)
    return {
        "name": f"Toki Recur ({spec.label})",
        "npm": "@ai-sdk/openai-compatible",
        "options": {
            "baseURL": f"{spec.daemon_url}/v1",
            "apiKey": DEFAULT_AUTH,
        },
        "models": {
            spec.model_id: {
                "name": f"{spec.label} (Toki Recur)",
                "contextWindow": ctx,
                "limit": {"context": ctx, "output": min(8192, max(1, ctx // 2))},
            }
        },
    }


def _small_model_ref(specs: list[ModelSpec]) -> str:
    """Legacy opencode `small_model` field — `tokirecur/<short_name>` form.

    Prefers an always-on spec (always available for cheap tasks like
    titles); falls back to the first spec.
    """
    if specs:
        primary = next((s for s in specs if s.always_on), specs[0])
        return f"tokirecur/{primary.name}"
    return "tokirecur/qwen3.5-9b"


def write_opencode_config(
    spec: ModelSpec, specs: list[ModelSpec] | None = None,
    path: Path | None = None,
) -> Path:
    """Write a provider entry for every registry spec + set the active.

    `specs` may be passed in to avoid re-reading the registry; if it's
    None the registry is loaded once and threaded through the call.
    """
    cfg_path = path or Path(os.environ.get(
        "OPENCODE_CONFIG", str(Path.home() / ".config/opencode/opencode.json")))
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg: dict[str, Any] = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    providers = cfg.setdefault("provider", {})
    all_specs = specs if specs is not None else load_registry()
    for s in all_specs:
        providers[f"tokirecur-{s.name}"] = _provider_entry(s)

    small = _small_model_ref(all_specs)
    cfg["small_model"] = small
    cfg["model"] = f"tokirecur-{spec.name}/{spec.model_id}"

    _backup(cfg_path)
    _atomic_write_text(cfg_path, json.dumps(cfg, indent=2) + "\n")
    return cfg_path


def write_openclaw_config(
    spec: ModelSpec, specs: list[ModelSpec] | None = None,
    path: Path | None = None,
) -> Path:
    """Write a provider entry for every registry spec + set the active."""
    cfg_path = path or Path(os.environ.get(
        "OPENCLAW_CONFIG", str(Path.home() / ".openclaw/openclaw.json")))
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg: dict[str, Any] = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    models = cfg.setdefault("models", {})
    models.setdefault("mode", "merge")
    providers = models.setdefault("providers", {})
    all_specs = specs if specs is not None else load_registry()
    for s in all_specs:
        providers[f"tokirecur-{s.name}"] = {
            "api": "openai-completions",
            "baseUrl": f"{s.daemon_url}/v1",
            "apiKey": DEFAULT_AUTH,
            "timeoutSeconds": 3600,
            "models": [
                {
                    "id": s.model_id,
                    "name": f"{s.label} (Toki Recur)",
                    "contextWindow": s.max_tokens,
                    "cost": {"cacheRead": 0, "cacheWrite": 0, "input": 0, "output": 0},
                }
            ],
        }

    agents = cfg.setdefault("agents", {})
    defaults = agents.setdefault("defaults", {})
    defaults["model"] = {"primary": f"tokirecur-{spec.name}/{spec.model_id}"}
    defaults["timeoutSeconds"] = 3600

    _backup(cfg_path)
    _atomic_write_text(cfg_path, json.dumps(cfg, indent=2) + "\n")
    return cfg_path


def write_hermes_config(
    spec: ModelSpec, specs: list[ModelSpec] | None = None,
    path: Path | None = None,
) -> Path:
    """Write a provider entry for every registry spec + set the active."""
    cfg_path = path or Path(os.environ.get(
        "HERMES_CONFIG", str(Path.home() / ".hermes/config.yaml")))
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg: dict[str, Any] = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}

    providers = cfg.setdefault("providers", {})
    all_specs = specs if specs is not None else load_registry()
    model_ids = [s.model_id for s in all_specs]
    for s in all_specs:
        providers[f"tokirecur-{s.name}"] = {
            "api": f"{s.daemon_url}/v1",
            "api_key": DEFAULT_AUTH,
            "default_model": s.model_id,
            "models": list(model_ids),
            "name": f"Toki Recur ({s.label})",
        }

    cfg.setdefault("model", {}).update(
        {
            "api_key": DEFAULT_AUTH,
            "base_url": f"{spec.daemon_url}/v1",
            "default": spec.model_id,
            "provider": f"tokirecur-{spec.name}",
        }
    )

    _backup(cfg_path)
    _atomic_write_text(cfg_path, yaml.safe_dump(cfg, sort_keys=False))
    return cfg_path


def sync_cli_configs(spec: ModelSpec) -> dict[str, Path]:
    """Write the active model into all three CLI configs.

    Reads the registry once and threads the loaded specs through each
    writer — the previous version re-parsed the YAML four times per
    switch (once per writer plus a second `load_registry` inside
    `_small_model_ref`).
    """
    specs = load_registry()
    return {
        "opencode": write_opencode_config(spec, specs),
        "openclaw": write_openclaw_config(spec, specs),
        "hermes": write_hermes_config(spec, specs),
    }
