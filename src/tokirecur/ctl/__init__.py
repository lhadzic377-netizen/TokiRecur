"""Multi-model plug-and-play management.

Each model in the registry owns a full stack: a llama-server and a
tokirecur daemon on dedicated ports. ``toki model use <name>`` brings
up the chosen model (the 9B "small" model stays on at all times),
stops any other big-model stack to free RAM, and rewrites the default
model in every model-capable CLI config so the same model is selected
wherever the client supports a local OpenAI-compatible endpoint.

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
import secrets
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import httpx
import yaml

try:  # Python 3.11+
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

import tomli_w

from ..ggufmeta import read_gguf_meta

logger = logging.getLogger(__name__)

def _writable_dir(parent: Path, name: str) -> Path:
    """Return ``parent/name`` if it (or a creatable parent) is writable
    by the current user; otherwise return a macOS-native fallback.

    ``~/.config`` on macOS is often owned by a different account or
    created with restrictive perms by other tools, which would block
    ``mkdir`` at import time and crash the launcher. The macOS
    fallback (``~/Library/Application Support/<name>``) is owned by
    the current user by default.
    """
    import sys as _sys

    def _try(p: Path) -> Path | None:
        try:
            p.mkdir(parents=True, exist_ok=True)
            # Probe write access — mkdir(exist_ok=True) is a no-op when
            # the dir already exists, which silently masks perm errors.
            probe = p / ".toki-write-probe"
            probe.write_text("")
            probe.unlink()
            return p
        except OSError:
            return None

    chosen = _try(parent / name)
    if chosen is not None:
        return chosen
    if _sys.platform == "darwin":
        mac = Path.home() / "Library" / "Application Support" / name
        chosen = _try(mac)
        if chosen is not None:
            return chosen
    # Last resort: temp dir, but warn so the user notices.
    import tempfile as _tempfile
    fallback = Path(_tempfile.gettempdir()) / name
    return _try(fallback) or fallback


def _default_paths() -> tuple[Path, Path, Path]:
    """Resolve the writable registry, state, and active-marker paths.

    Honors ``TOKI_MODEL_REGISTRY``/``TOKI_STATE_DIR``/``TOKI_ACTIVE_MODEL``
    env vars first; then probes the XDG default. On macOS, falls back
    to ``~/Library/Application Support/tokirecur`` when the XDG path
    is not writable.
    """
    home = Path.home()
    if os.environ.get("TOKI_MODEL_REGISTRY"):
        reg = Path(os.environ["TOKI_MODEL_REGISTRY"])
    else:
        reg = _writable_dir(home / ".config", "tokirecur") / "models.yaml"
    if os.environ.get("TOKI_STATE_DIR"):
        state = Path(os.environ["TOKI_STATE_DIR"])
    else:
        state = _writable_dir(home / ".local", "state/tokirecur")
    if os.environ.get("TOKI_ACTIVE_MODEL"):
        marker = Path(os.environ["TOKI_ACTIVE_MODEL"])
    else:
        marker = reg.parent / "active-model"
    return reg, state, marker


DEFAULT_REGISTRY = Path(
    os.environ.get("TOKI_MODEL_REGISTRY", str(Path.home() / ".config/tokirecur/models.yaml"))
)
STATE_ROOT = Path(
    os.environ.get("TOKI_STATE_DIR", str(Path.home() / ".local/state/tokirecur"))
)
ACTIVE_MARKER = Path(
    os.environ.get("TOKI_ACTIVE_MODEL", str(Path.home() / ".config/tokirecur/active-model"))
)

# Resolve to a writable location at import time: if the XDG default
# isn't writable (common on macOS where ~/.config is owned by a
# different account), fall back to the macOS-native Application
# Support dir, then a temp dir. The env vars above always win.
try:
    _resolved = _default_paths()
    DEFAULT_REGISTRY = _resolved[0]
    STATE_ROOT = _resolved[1]
    ACTIVE_MARKER = _resolved[2]
except Exception:  # noqa: BLE001 — never let path resolution crash import
    pass

def _configured_auth_token() -> str | None:
    """Read the daemon token without manufacturing a fallback value."""
    return configured_auth_token()


#: Values that are not secrets.
#:
#: The README used to show ``"apiKey": "demo-token"`` in a copy-and-paste
#: config, and a machine that followed it ended up with a daemon whose
#: "token" was written down in this project's own documentation — so
#: anything that could reach the port could use the model. A placeholder is
#: treated as no token at all, and replaced with a real one.
_KNOWN_PLACEHOLDERS = frozenset({
    "demo-token", "demo_token", "changeme", "change-me", "your-token-here",
    "your_token_here", "token", "placeholder", "sk-demo", "none", "null",
    "todo", "replace-me",
})


def _is_placeholder(value: str) -> bool:
    return value.strip().lower() in _KNOWN_PLACEHOLDERS


def _new_auth_token(path: Path) -> str:
    """Write a real token to the auth file and return it."""
    token = secrets.token_urlsafe(32)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(token + "\n")
        os.chmod(path, 0o600)
    except OSError:
        pass
    return token


def configured_auth_token() -> str | None:
    """The token configured on this machine, or None.

    The env var wins, then the auth file. A file holding a well-known
    placeholder is replaced with a real token: a placeholder protects
    nothing, and leaving it in place is how a machine ends up serving the
    model to anything that can reach the port.
    """
    env = os.environ.get("TOKI_AUTH_TOKEN")
    if env and env.strip() and not _is_placeholder(env):
        return env.strip()
    for path in (
        Path.home() / ".config" / "tokirecur" / "auth",
        Path("/etc/toki-recur/auth"),
    ):
        try:
            if not path.is_file():
                continue
            content = path.read_text().strip()
            if not content:
                continue
            if _is_placeholder(content):
                return _new_auth_token(path)
            return content
        except OSError:
            pass
    return None


def auth_token() -> str:
    """The token the daemon will actually accept.

    Order has to match api.app._get_auth_token exactly - env, then the auth
    file - or client configs and the daemon disagree and every request is a
    401. The previous version skipped the auth file entirely and invented a
    fresh random token instead, which meant that with no env var set:

      * two calls returned two different tokens, and
      * neither matched the token the daemon was using

    so a dsh web profile came up, displayed the model selector, and then
    failed its first turn with "API key is invalid". Symmetry with the daemon
    is the whole fix; do not reorder one side without the other.
    """
    return configured_auth_token() or secrets.token_urlsafe(32)


# Back-compat alias: historically a module-level constant frozen at
# import. Call auth_token() to get a value that honors env changes made
# after import (e.g. under tests).
DEFAULT_AUTH = auth_token()

# Sampling defaults for the daemon's OpenAI-compatible endpoints.
# Greedy decode (temperature 0.0) with no repetition penalty is the
# loop engine behind agent harnesses: identical context reproduces
# identical tokens → the same tool call forever. These values break
# that lock without touching the KV cache (sampling is decode-time).
# Iterate here or per-model via the registry `sampling:` block.
# Client-supplied values always win over these defaults.
DEFAULT_SAMPLING: dict[str, Any] = {
    "temperature": 0.4,
    "repeat_penalty": 1.15,
    "top_k": 40,
    "min_p": 0.05,
}

# Behavioral guardrail injected into the existing CLI harness configs
# (opencode, mimo, openclaw, hermes, codex) so agentic models don't lock
# into repeating
# the same tool call / message. Never written if the user already has
# their own instructions/system prompt in that harness's config.
#
# v2 style guide (learned from v1 feedback — small local models got
# confused and over-thought): NO identity claim (the harness already
# states who the model is — claiming "you are an autonomous agent"
# conflicts with it), NO abstract meta-language ("tool call", "identical
# call", "follow-ups unless asked"), positive-first concrete rules, and
# nothing that can be read as "keep going" ("do not skip steps" fought
# with "stop when done" and pushed the model into long self-monitoring
# loops). Keep it short: each extra rule costs attention.
#
# v3 (post-35-page research review): ZERO negations (v2's "Don't repeat"
# phrasing still invited the model to self-monitor — "am I repeating?"),
# every rule is one primitive idea (Toki Pona-style decomposition), the
# failure rule carries a concrete tactic ("change something") AND a retry
# cap ("once more") to kill the infinite same-request loop, and rule 4 is
# a positive output contract that doubles as the ask-when-stuck fallback.
# Guarded by a unit test (rules/words/negation budget) so it can't grow
# back into "contextual overload" territory.
ANTI_LOOP_SYSTEM_PROMPT = (
    "Working rules\n\n"
    "1. Do one step at a time.\n"
    "2. When a step works, continue with the next one.\n"
    "3. When a step fails, change something and try once more.\n"
    "4. End each turn with one new thing: a result, a question, or done.\n"
    "5. When the task is finished, stop and answer briefly.\n"
)

# Reference defaults: registry entries matching these names (source ==
# "default") are treated as user-configured and never rewritten by
# backfill_binary_env. They are NOT preloaded — a fresh install starts
# with an empty registry.
#
# Portable by construction: `gguf` is a bare FILENAME (never an absolute
# path) resolved at use time against `~/models/<name>/` — the directory
# `toki model pull` and the installers' starter block download into. No
# `binary` is pinned here (the local build is not everyone's build); the
# installed binary resolves via `_resolve_binary`/`_installed_binary`.
DEFAULT_MODELS: list[dict[str, Any]] = [
    {
        "name": "9b",
        "label": "Qwen3.5 9B (Q4_K_M)",
        "gguf": "Qwen3.5-9B-Q4_K_M.gguf",
        "llama_port": 8088,
        "daemon_port": 8787,
        "ctx_size": 200000,
        "parallel": 2,
        "kv": "f16",
        "max_tokens": 100000,
        "max_output_tokens": 16384,
        "model_id": "qwen3.5-9b",
        "always_on": False,
        "boot": False,
        "source": "default",
    },
    {
        "name": "35b",
        "label": "Qwen3.6 35B-A3B (UD-Q4_K_M)",
        "gguf": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        "llama_port": 8089,
        "daemon_port": 8788,
        "ctx_size": 200000,
        "parallel": 2,
        "kv": "f16",
        "max_tokens": 100000,
        "max_output_tokens": 16384,
        "model_id": "qwen3.6-35b-a3b",
        "always_on": False,
        "source": "default",
    },
    {
        "name": "laguna",
        "label": "Laguna S 2.1 (Q4_K_M, 64k ctx)",
        "gguf": "laguna-s-2.1-Q4_K_M.gguf",
        "llama_port": 8090,
        "daemon_port": 8789,
        "ctx_size": 131072,
        "parallel": 2,
        "kv": "q8_0",
        "max_tokens": 64000,
        "model_id": "laguna-s-2.1",
        "always_on": False,
        "boot": False,
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
    # Where the model was pulled from, so a projector that was never
    # downloaded can still be found: the repository has it, and a blind
    # model answers every picture as though it had been shown none.
    repo: str = ""
    # Path to a multimodal projector (mmproj) GGUF for vision models. Held as
    # its own field rather than an `--mmproj <path>` entry in `extra_args`
    # because each extra_arg is passed to llama-server as a single argv token:
    # a space-joined "--mmproj /path/to.gguf" dies at startup with
    # `invalid argument: --mmproj /path/to.gguf`.
    mmproj: str = ""
    # Image-generation stacks (binary "sd-server"): the diffusion GGUF
    # rides in `gguf` like any model, but the pipeline needs two more
    # weights beside it. Held as their own fields for the same reason as
    # mmproj — paths, not argv flags.
    vae: str = ""
    llm: str = ""
    env: dict[str, str] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    # "auto" (default): the daemon streams reasoning and reserves token
    # headroom so thinking models always answer. "off": pass
    # `--reasoning off` to llama-server for templates that honor it
    # (stock Qwen3.5); ignored by templates that hard-code thinking.
    reasoning: str = "auto"
    # Per-model sampling overrides merged over ctl.DEFAULT_SAMPLING by
    # the daemon (threaded as TOKI_SAMPLING at spawn). Keys:
    # temperature / repeat_penalty / top_k / top_p / min_p. Empty
    # (default) means the global defaults apply unchanged.
    sampling: dict[str, Any] = field(default_factory=dict)
    # Hard per-turn output budget in tokens, enforced by the daemon
    # (threaded as TOKI_MAX_OUTPUT_TOKENS at spawn). Guards against
    # runaway single generations: registry `max_tokens` is advertising
    # only (the context window), and client-side caps (e.g. opencode's
    # limit.output) are not always sent as max_tokens. None = no cap.
    max_output_tokens: int | None = None
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


def resolve_gguf_path(spec: ModelSpec) -> Path:
    """The on-disk location of a spec's model weights.

    Absolute paths (user registry entries, pulled models) are used as-is
    when they exist. Anything else — including bare filenames from the
    portable DEFAULT_MODELS and absolute paths whose file has moved —
    resolves against ``~/models/<name>/``, the directory `toki model
    pull` and the installers' starter block download into. When the
    directory holds GGUFs but none match, the first one is used (best
    effort). Raises FileNotFoundError with a one-command hint.
    """
    gguf_text = getattr(spec, "gguf", "") or ""
    gguf = Path(gguf_text) if gguf_text else None
    candidates: list[Path] = []
    if gguf is not None and gguf.is_absolute():
        if gguf.exists():
            return gguf
        candidates.append(Path.home() / "models" / spec.name / gguf.name)
    elif gguf is not None:
        candidates.append(Path.home() / "models" / spec.name / gguf)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    model_dir = Path.home() / "models" / spec.name
    if model_dir.is_dir():
        # A projector is a GGUF too, and it lives in the same directory as
        # the model it belongs to. When the model file has been deleted, the
        # "first gguf here" fallback used to find the projector and hand it
        # back as the weights - a model that loads a vision tower and calls
        # it a language model.
        fallback = [p for p in sorted(model_dir.glob("*.gguf")) if not _is_projector(p)]
        if fallback:
            return fallback[0]
    raise FileNotFoundError(
        f"model file not found: {gguf_text or spec.name} — run `toki model install "
        "starter` (or `toki model pull <hf-repo>`), then retry"
    )


def _is_projector(path: Path) -> bool:
    """Whether a GGUF is a vision projector rather than a model."""
    if path.name.lower().startswith("mmproj"):
        return True
    try:
        from ..ggufmeta import read_gguf_kind

        return read_gguf_kind(path).architecture == "clip"
    except Exception:
        return False


def model_file_missing(spec: ModelSpec) -> bool:
    """Whether this model's weights are on disk at all.

    The registered path is not the only place they can be: a portable
    entry names a bare file, and a model that has been moved is found by
    name under ``~/models/<name>/``. So the question is asked of the
    resolver, not of the string in the registry - and a model whose file
    has merely moved is not a model that is missing.
    """
    try:
        resolve_gguf_path(spec)
    except (FileNotFoundError, OSError):
        return True
    return False


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
    """Rough resident footprint (GB): weights + KV + workspace headroom.

    Image stacks count every weight they load — the denoiser plus its VAE
    and text encoder — because the launch gate (_check_ram) must refuse
    honestly when the full pipeline will not fit, not just the GGUF.
    """
    try:
        weight_gb = resolve_gguf_path(spec).stat().st_size / 1e9
    except FileNotFoundError:
        weight_gb = 0.0
    extra = 0.0
    if _spec_is_image(spec):
        for companion in (spec.vae, spec.llm):
            try:
                if companion and Path(companion).is_file():
                    extra += Path(companion).stat().st_size / 1e9
            except OSError:
                pass
    return (weight_gb + extra) * 1.2 + 4.0


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
        # Prefer available over total — UMA boxes need the real budget,
        # not the marketing number. vm_stat is cheap and present on
        # every macOS.
        try:
            out = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, timeout=5,
            ).stdout
            ps = 4096
            free = inactive = speculative = 0
            for line in out.splitlines():
                if "page size of" in line.lower():
                    try:
                        ps = int(line.split()[-2])
                    except ValueError:
                        pass
                elif "Pages free:" in line:
                    free = int(line.split(":")[1].strip().split()[0].replace(".", ""))
                elif "Pages inactive:" in line:
                    inactive = int(line.split(":")[1].strip().split()[0].replace(".", ""))
                elif "Pages speculative:" in line:
                    speculative = int(line.split(":")[1].strip().split()[0].replace(".", ""))
            avail = (free + inactive + speculative) * ps / 1e9
            if avail > 1:
                return avail
        except Exception:
            pass
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            # Total * 0.6 as a conservative available estimate when
            # vm_stat failed.
            return int(out.stdout.strip()) / 1e9 * 0.6
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


def _kv_bytes_per_token(spec: ModelSpec, kv_type: str | None = None,
                        parallel: int | None = None) -> float | None:
    """llama.cpp KV cache bytes per token across all layers.

    Exact math: per layer, k+v each hold ``head_dim * n_head_kv``
    values, with ``head_dim = n_embd / n_head``. Requires the model's
    GGUF metadata; ``None`` when unreadable (caller skips clamping).

    Modern llama.cpp treats ``-c`` as the *total* context: with the
    unified KV cache (``-kvu``, which Toki always passes) every slot
    can use the full window against one shared ``n_ctx``-cell pool, so
    per-token cost does not scale with ``parallel``. The ``kv_type`` /
    ``parallel`` overrides exist for the fit planner to evaluate
    alternatives; ``parallel`` only gates ``None`` metadata fallbacks
    and no longer multiplies the estimate.
    """
    gguf = getattr(spec, "gguf", None)
    if not gguf:
        return None
    meta = read_gguf_meta(gguf)
    if not meta.complete or meta.n_head == 0:
        return None
    head_dim = meta.n_embd / meta.n_head
    kv_bytes = _KV_BYTES_PER_ELEM.get(kv_type or spec.kv, 2.0)
    per_layer = head_dim * meta.n_head_kv * 2.0 * kv_bytes
    return per_layer * meta.n_layer


def _gguf_context_length(spec: ModelSpec) -> int | None:
    """The model's trained context window (GGUF ``context_length``)."""
    gguf = getattr(spec, "gguf", None)
    if not gguf:
        return None
    meta = read_gguf_meta(gguf)
    if meta and meta.context_length:
        return meta.context_length
    return None


def _rope_arg_value(extra_args: list[str], flag: str) -> str | None:
    """Value of ``flag`` inside an ``extra_args`` list.

    Accepts both ``--flag value`` and ``--flag=value`` — llama.cpp's parser
    takes either, so the registry may use whichever reads better.
    """
    for i, arg in enumerate(extra_args):
        if arg == flag:
            return extra_args[i + 1] if i + 1 < len(extra_args) else None
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def _rope_scaled_max_ctx(spec: ModelSpec, trained_ctx: int | None = None) -> int | None:
    """Context ceiling implied by a YaRN rope-scaling config, else ``None``.

    YaRN exists to run a model *past* its trained window. llama.cpp
    re-derives the model's train context as
    ``n_ctx_train = n_ctx_orig_yarn / rope_freq_scale``
    (``src/llama-context.cpp``), and ``--rope-scale N`` sets
    ``rope_freq_scale = 1/N`` (``common/arg.cpp``) — so a registry entry
    with ``--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144``
    raises the ceiling to 262144 * 4 = 1048576.

    Without this the fit planner caps the launch at the GGUF's own
    ``context_length`` and the YaRN flags buy nothing: the registry would
    ask for 1M, llama-server would be told 262144, and both the daemon
    and the harness would advertise a window the model never had.
    """
    extra = list(getattr(spec, "extra_args", None) or [])
    if not extra:
        return None
    scaling = (_rope_arg_value(extra, "--rope-scaling") or "").strip().lower()
    if scaling != "yarn":
        return None
    factor_raw = _rope_arg_value(extra, "--rope-scale")
    if not factor_raw:
        return None
    try:
        factor = float(factor_raw)
    except ValueError:
        return None
    if factor <= 0:
        return None
    orig_raw = _rope_arg_value(extra, "--yarn-orig-ctx")
    if orig_raw:
        try:
            orig = int(orig_raw)
        except ValueError:
            return None
    else:
        # llama.cpp falls back to the model's training context when the
        # flag is absent — mirror that so the ceiling matches the server.
        orig = trained_ctx if trained_ctx is not None else _gguf_context_length(spec)
    if not orig or orig <= 0:
        return None
    return int(orig * factor)


def _model_max_ctx(spec: ModelSpec) -> int | None:
    """The model's own max context, or ``None``.

    A YaRN rope-scaling config raises the ceiling past the GGUF's trained
    window — that is the whole point of the flag — so it wins over the
    metadata. Without it, the cap is the GGUF's ``context_length``.
    """
    trained = _gguf_context_length(spec)
    scaled = _rope_scaled_max_ctx(spec, trained)
    if scaled:
        return scaled
    return trained


def _clamp_ctx(spec: ModelSpec, avail_gb: float | None = None,
               weight_gb: float | None = None) -> int:
    """Largest context (>= ``_CTX_FLOOR``) whose KV cache fits in RAM.

    Uses the same headroom budget as ``_estimate_need``: weights * 1.2
    + 4 GB. Also caps at the model's own max context (a 350M model is
    never going to use 200k even when RAM would allow). When the GGUF
    metadata is unavailable (or the machine's RAM can't be measured)
    the configured ``ctx_size`` is returned untouched and llama-server's
    own auto-reduction applies.

    The KV estimate covers the full ``n_ctx`` pool (unified KV cache,
    ``-kvu``), so the configured window is kept whenever the measured
    RAM holds it — the window only scales down when RAM genuinely
    can't fit the model plus the full-window KV cache.
    """
    cap = spec.ctx_size
    model_max = _model_max_ctx(spec)
    if model_max:
        cap = min(cap, model_max)
    kv_per_token = _kv_bytes_per_token(spec)
    if kv_per_token is None:
        return cap
    if avail_gb is None:
        avail_gb = _available_gb()
    if weight_gb is None:
        try:
            weight_gb = resolve_gguf_path(spec).stat().st_size / 1e9
        except (AttributeError, FileNotFoundError, OSError):
            weight_gb = 0.0
    budget = avail_gb - weight_gb * 1.2 - 4.0
    if not math.isfinite(budget):
        # RAM can't be measured (inf) or the measurement is garbage:
        # keep the configured window and let llama-server's own
        # auto-reduction apply — a silent shrink, but not a lie.
        return cap
    if budget <= 0:
        # Weights + headroom alone don't fit the measured RAM. The old
        # behavior passed the full ctx to llama.cpp, which then silently
        # auto-reduced the KV cache to whatever fit — every client was
        # told a window that never existed. Return the smallest honest
        # value instead; `_check_ram` raises before launch anyway.
        return _CTX_FLOOR
    max_fit = int(budget * 1e9 / kv_per_token)
    return max(_CTX_FLOOR, min(cap, max_fit))


# KV precision ladder for the fit planner: quality order, best first.
_KV_QUALITY_ORDER = ("f16", "q8_0", "q4_1", "q4_0")


def _fit_plan(spec: ModelSpec, avail_gb: float | None = None,
              weight_gb: float | None = None) -> tuple[int, str, int]:
    """The launch config ``(ctx, kv, parallel)`` that best fits RAM.

    Computes the largest context each KV precision can hold under the
    same budget as ``_clamp_ctx`` (weights * 1.2 + 4 GB headroom,
    capped at the model's own max ctx) and picks the biggest window.
    Ties prefer the spec's own kv/parallel (no quality surprise on
    machines with plenty of RAM), then higher-quality KV.

    Modern llama.cpp is launched with the unified KV cache (``-kvu``),
    where ``-c`` is the total context shared across all slots — each
    slot can use the full window against one ``n_ctx``-cell pool. The
    slot count no longer consumes extra KV memory, so ``parallel`` is
    kept at the spec's value and only KV precision trades memory for
    window. On the 16 GB class this is the difference between a
    ~45k window (f16) and a ~90k+ window (q4_1) for a 9B — the memory
    is identical, only the advertised window changes.
    """
    cap = spec.ctx_size
    model_max = _model_max_ctx(spec)
    if model_max:
        cap = min(cap, model_max)
    if avail_gb is None:
        avail_gb = _available_gb()
    if weight_gb is None:
        try:
            weight_gb = resolve_gguf_path(spec).stat().st_size / 1e9
        except (AttributeError, FileNotFoundError, OSError):
            weight_gb = 0.0
    budget = avail_gb - weight_gb * 1.2 - 4.0
    if _kv_bytes_per_token(spec) is None or not math.isfinite(budget) or budget <= 0:
        # Unknown metadata/RAM (or weights alone don't fit — `_check_ram`
        # raises before launch): keep the spec's own config untouched.
        return (cap, getattr(spec, "kv", "f16"), getattr(spec, "parallel", 1))
    candidates = []
    for kv in _KV_QUALITY_ORDER:
        per = _kv_bytes_per_token(spec, kv_type=kv)
        if per is None:
            continue
        ctx = max(_CTX_FLOOR, min(cap, int(budget * 1e9 / per)))
        candidates.append((ctx, kv, getattr(spec, "parallel", 1)))
    return max(
        candidates,
        key=lambda c: (
            c[0],                       # biggest window wins
            c[1] == spec.kv,            # …then the spec's own kv
            -_KV_QUALITY_ORDER.index(c[1]),  # …then higher KV quality
            c[2],                       # …then more slots
        ),
    )


def _effective_ctx(spec: ModelSpec) -> int:
    """The context llama-server actually runs with (clamped, then
    corrected to the real slot size after start). Falls back to the
    RAM-clamped value before the first start — never the raw registry
    ``ctx_size``, which clients would overfill against a window the
    server can't provide (the 0.2.16 "not enough context" class)."""
    sd = getattr(spec, "state_dir", None)
    if callable(sd):
        try:
            p = sd() / "effective_ctx.txt"
            # Stale file: registry ctx changed without a restart (user
            # edited ctx_size but didn't force_restart). The file's
            # mtime predates the registry's — discard and recompute
            # via _fit_plan so clients don't overfill a 131k file
            # when the registry now asks for 200k.
            val = int(p.read_text().strip())
            try:
                reg_mtime = DEFAULT_REGISTRY.stat().st_mtime
                if p.stat().st_mtime < reg_mtime:
                    raise ValueError("stale effective_ctx")
            except (OSError, ValueError):
                if "stale" in str(sys.exc_info()[1] or ""):
                    raise
                pass
            return val
        except (OSError, ValueError):
            pass
    # The pre-start fallback must be the plan's actual launch window —
    # NOT _clamp_ctx, which computes the worst case (the spec's own
    # kv/parallel). A 16 GB box selecting 65k for a 9B fits ~31k as
    # f16@2 slots but the full 65k as q8_0@1 slot; _fit_plan launches
    # the latter, so advertising the former makes clients (and status
    # readouts) show a context that never matches the server.
    plan_ctx = _fit_plan(spec)[0]
    return plan_ctx if plan_ctx < spec.ctx_size else spec.ctx_size


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
    """Read + parse the registry YAML, falling back to defaults if missing.

    A corrupt YAML (hand-edit, disk full, editor crash) previously
    bubbled ``YAMLError`` and left the launcher in a crash loop until
    the user hand-deleted the file. Now the corrupt file is backed up
    to ``.bak-corrupt.<ts>`` and an empty registry is returned so the
    next ``ensure`` path rebuilds ``DEFAULT_MODELS``.
    """
    if not reg.exists():
        reg = _ensure_registry()
    try:
        return yaml.safe_load(reg.read_text()) or {}
    except (yaml.YAMLError, UnicodeDecodeError, OSError) as e:
        try:
            bak = reg.with_name(f"{reg.name}.bak-corrupt.{int(time.time())}")
            shutil.copy2(reg, bak)
            logger.warning("registry %s corrupt (%s) — backed up to %s", reg, e, bak)
        except OSError:
            pass
        try:
            reg.unlink()
        except OSError:
            pass
        reg = _ensure_registry()
        try:
            return yaml.safe_load(reg.read_text()) or {}
        except (yaml.YAMLError, UnicodeDecodeError, OSError):
            return {}


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


def merged_sampling(spec: ModelSpec) -> dict[str, Any]:
    """Effective sampling profile for a spec: global defaults overlaid
    with the spec's registry ``sampling:`` block. Passed to the daemon
    as ``TOKI_SAMPLING`` (JSON) at spawn so per-request resolution in
    the API layer always has the active model's profile."""
    merged = dict(DEFAULT_SAMPLING)
    merged.update(spec.sampling or {})
    return merged


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
            "no models installed — run `toki model install starter` "
            "(or `toki model pull <hf-repo>`) to set up your first model"
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


def _atomic_write_text(path: Path, text: str, *, mode: int | None = None) -> None:
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
    if mode is not None:
        tmp.chmod(mode)
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

    Contention is resolved by waiting (LOCK_EX blocks) up to
    ``timeout`` seconds; on timeout the acquisition FAILS LOUDLY
    (``RuntimeError``) instead of silently continuing — two writers
    proceeding unlocked would lose one edit to the atomic-rename
    race, and silence hides exactly the kind of corruption this
    lock exists to prevent.
    """

    def __init__(self, path: Path, timeout: float = 10.0) -> None:
        self.path = path
        self.timeout = timeout
        self._fd: int | None = None

    def __enter__(self) -> _flock:
        if not _HAS_FLOCK:
            return self
        import time as _time
        lock_path = self.path.with_name(self.path.name + ".lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        deadline = _time.monotonic() + self.timeout
        while True:
            try:
                # Non-blocking acquire inside a retry loop so we can
                # bound total wait and surface contention clearly.
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError:
                if _time.monotonic() >= deadline:
                    os.close(self._fd)
                    self._fd = None
                    raise RuntimeError(
                        f"registry {self.path} is locked by another "
                        f"toki process (waited {self.timeout:.0f}s). "
                        f"If no other toki is running, remove "
                        f"{lock_path} and retry."
                    ) from None
                _time.sleep(0.1)

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


# Fields whose change requires a llama-server restart (the spec's
# launch window changes; the running server cannot honour the new value).
_RESTART_REQUIRED_FIELDS = frozenset({
    "ctx_size", "kv", "parallel", "binary", "gguf",
    "llama_port", "daemon_port",
})

# Fields whose change only affects CLI client configs (opencode.json
# advertises contextWindow/limit.output). No daemon restart needed.
_SYNC_ONLY_FIELDS = frozenset({
    "max_tokens", "model_id", "reasoning",
})


def update_spec_field(
    name: str,
    field: str,
    value: str,
    registry: Path | None = None,
    apply: bool = False,
) -> ModelSpec:
    """Update one registry field on a named spec, validate, persist.

    Returns the updated spec so the caller can refresh its view.
    Raises ``ValueError`` for unknown fields or invalid values.

    When ``apply=True`` the function also drives the right downstream
    effect so the edit actually takes effect on the running stack:

      * ``ctx_size``, ``kv``, ``parallel``, ``binary``, ``gguf``,
        ``llama_port``, ``daemon_port`` → ``use_model(name)`` (stops
        the running daemon if any, restarts with the new settings,
        rewrites the model-capable CLI configs).

      * ``max_tokens``, ``model_id``, ``reasoning`` → ``sync_cli_configs``
        only (no daemon restart).
      * ``always_on``, ``boot`` → registry-only (no restart, no sync).

    ``apply=False`` (the default) preserves the legacy behaviour: write
    the registry and return, leave the running stack alone. Existing
    CLI / installer callers keep working unchanged; only the TUI
    settings panel passes ``apply=True``.
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

    # Downstream orchestration. Only runs when the caller asked for it
    # (the TUI passes apply=True; existing CLI/installer callers stay
    # on the legacy "registry only" path).
    if apply:
        if field in _RESTART_REQUIRED_FIELDS:
            # use_model() restarts the model's stack (stops the running
            # daemon if any, restarts with the new settings, and
             # rewrites the model-capable CLI configs). We swallow errors so the
             # caller's UX still shows the registry write succeeded even

            # if the restart hiccups — the user can retry from the
            # same screen.
            try:
                use_model(name, stop_others=False, registry=reg,
                          force_restart=True)
            except (RuntimeError, OSError) as exc:
                logger.warning(
                    "could not restart '%s' after %s change: %s",
                    name, field, exc,
                )
                raise
        elif field in _SYNC_ONLY_FIELDS:
            try:
                sync_cli_configs(new_spec)
            except Exception as exc:
                logger.warning(
                    "could not sync CLI configs after %s.%s change: %s",
                    name, field, exc,
                )
        # always_on / boot: registry-only, nothing to do.

    return new_spec


def _port_free(port: int) -> bool:
    """True when ``port`` can be bound on 127.0.0.1 (no listener)."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def next_free_ports(specs: list[ModelSpec]) -> tuple[int, int]:
    """Pick the next unused llama/daemon port pair.

    Starts at max+1 from the registry, then probes the OS so a port
    already taken by an unrelated service (Jenkins on 8080, etc.)
    is skipped instead of hanging ``_wait_healthy`` for 30 min.
    Also rejects a pair where both ports would be identical.
    """
    base_llama = max((s.llama_port for s in specs), default=8087) + 1
    base_daemon = max((s.daemon_port for s in specs), default=8786) + 1
    for _ in range(100):
        if base_llama == base_daemon:
            base_daemon += 1
        if base_llama < 1024:
            base_llama = 1024
        if base_daemon < 1024:
            base_daemon = 1024
        if _port_free(base_llama) and _port_free(base_daemon):
            return base_llama, base_daemon
        # Bump both and try again — keeps the daemon port consecutive
        # with the llama port for readability.
        base_llama += 1
        base_daemon += 1
    # Fallback: at least return a non-colliding registry pair
    return base_llama, base_daemon


def _slugify_name(name: str) -> str:
    """Lowercase, replace anything not [a-z0-9._-] with a dash."""
    return re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-._")


def _reference_spec(specs: list[ModelSpec]) -> ModelSpec | None:
    """Pick a 'good' existing spec to inherit binary/env from (preferring
    a custom ROCm build over the Debian default).

    A spec whose ``binary`` is an absolute path that no longer exists on
    this machine (e.g. a registry shipped from another box) is skipped —
    inheriting it would hand every new pull a binary that Popen can't
    start. Bare ``llama-server`` names are kept (they resolve via PATH /
    ``_installed_binary`` at launch).
    """
    for s in specs:
        if not s.binary or s.binary == "llama-server":
            continue
        pinned = Path(s.binary)
        if pinned.is_absolute() and not pinned.exists():
            continue
        return s
    return None


def _installed_binary() -> str | None:
    """The llama-server build installed by ``toki model install-llama``.

    ``install-llama`` unpacks the pinned release into ``STATE_ROOT/bin``,
    which is not on PATH — on Windows a bare ``llama-server`` name then
    fails ``subprocess.Popen`` with WinError 2. Return the concrete
    installed path when present so pulls/backfills never fall back to
    the PATH default on a machine that has a real build.
    """
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    candidate = STATE_ROOT / "bin" / name
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    # File exists but not executable (e.g. permission skew after restore)
    # — try to repair the bit and re-check so the next pull doesn't
    # hand every new model a bare "llama-server" that fails on Windows.
    if candidate.exists():
        try:
            candidate.chmod(candidate.stat().st_mode | 0o111)
            if os.access(candidate, os.X_OK):
                return str(candidate)
        except OSError:
            pass
    return None


def pull_defaults(specs: list[ModelSpec]) -> tuple[str, dict[str, str]]:
    """Resolve binary + env defaults for a new model.

    Order: TOKI_LLAMA_SERVER env → an existing spec's binary + env (the
    machine's tuned build, e.g. ROCm b10008) → the build installed by
    ``toki model install-llama`` → PATH's llama-server.
    """
    env_override = os.environ.get("TOKI_LLAMA_SERVER")
    if env_override:
        ref = _reference_spec(specs)
        return env_override, dict(ref.env) if ref else {}
    ref = _reference_spec(specs)
    if ref:
        return ref.binary, dict(ref.env)
    installed = _installed_binary()
    if installed:
        return installed, {}
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


_GGUF_SPLIT_RE = re.compile(
    r"^(?P<base>.+)-(?P<idx>\d{3,5})-of-(?P<total>\d{3,5})\.gguf$"
)


def _gguf_split_siblings(filename: str) -> list[str] | None:
    """All sibling parts of a multi-file GGUF split, or None if not split.

    e.g. ``UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf``
    returns all three parts in numeric order; non-split paths return None.
    The split pattern is the upstream llama.cpp convention:
    ``<name>-<NNNN>-of-<NNNN>.gguf`` (3-5 digit indices, preserved).
    """
    m = _GGUF_SPLIT_RE.match(filename)
    if not m:
        return None
    base = m.group("base")
    idx = m.group("idx")
    total = m.group("total")
    idx_w = len(idx)
    total_w = len(total)
    n = int(total)
    total_int = int(total)
    return [
        f"{base}-{i:0{idx_w}d}-of-{total_int:0{total_w}d}.gguf"
        for i in range(1, n + 1)
    ]


# Preferred multimodal projector types, best first. F16 is what llama.cpp's
# own conversion emits and it is what the vision model was trained against;
# Q8_0 halves the projector for anyone who cares more about RAM than exactness.
_MMPROJ_PREFERENCE = ("f16", "bf16", "f32", "q8_0")


def _is_mmproj(filename: str) -> bool:
    """True for a multimodal projector file, by llama.cpp's naming convention.

    Repos spell it ``mmproj-F16.gguf``, ``mmproj-Qwen3VL-4B-Instruct-F16.gguf``
    or ``mmproj-model-f16.gguf`` - but they also spell it after the model it
    belongs to, as ``gemma-4-E2B.mmproj-Q8_0.gguf`` or
    ``model-mmproj-f16.gguf``. A rule that only looked at the start of the
    name found every projector except those, and the models they belonged
    to stayed blind with no sign of why.
    """
    stem = Path(filename).name.lower()
    if not stem.endswith(".gguf"):
        return False
    return stem.startswith("mmproj") or ".mmproj" in stem or "-mmproj" in stem


def _pick_mmproj(files: list[str]) -> str | None:
    """Choose which projector to fetch alongside a model GGUF.

    A vision model without its projector does not fail loudly: llama-server
    starts, accepts a chat request, and answers as if it had seen no image at
    all. That silent blindness is why ``pull_model`` downloads the projector
    instead of leaving it to the person who ran the command.
    """
    candidates = [f for f in files if _is_mmproj(f)]

    def rank(name: str) -> tuple[int, str]:
        stem = Path(name).name.lower()
        # Compare the quant token exactly, not as a substring: "bf16" contains
        # "f16", so a substring test hands every BF16 projector first place and
        # the F16 file never gets picked.
        token = stem.rsplit("-", 1)[-1].removesuffix(".gguf")
        for i, want in enumerate(_MMPROJ_PREFERENCE):
            if token == want:
                return (i, name.lower())
        return (len(_MMPROJ_PREFERENCE), name.lower())

    return sorted(candidates, key=rank)[0] if candidates else None


def _expand_gguf_patterns(filename: str, mmproj: str | None = None) -> list[str]:
    """``allow_patterns`` for huggingface_hub covering ``filename`` and, if
    it is part of a multi-file GGUF split, every sibling part as well.

    Hugging Face splits (the ``-NNNN-of-NNNN.gguf`` convention used by
    llama.cpp) require *every* part on disk: llama-server loads part 1,
    dereferences parts 2..N, and fails with
    ``failed to open GGUF file ... (No such file or directory)`` if any
    is missing. We therefore download all siblings, not just the one the
    user picked.

    ``mmproj`` is the multimodal projector for a vision model and is fetched
    in the same pass for the reason in :func:`_pick_mmproj`.
    """
    patterns: list[str] = []
    siblings = _gguf_split_siblings(filename)
    if siblings is None:
        patterns.append(f"*{filename}")
    else:
        patterns.extend(f"*{p}" for p in siblings)
    if mmproj:
        patterns.append(f"*{mmproj}")
    return patterns


def _auto_ctx_size(gguf: Path, *, parallel: int, kv: str) -> int:
    """Pick a context for a fresh pull that fits this machine.

    Uses the same RAM budget as ``_clamp_ctx`` (weights * 1.2 + 4 GB
    headroom), capped at the model's own max context and a sane 64k
    ceiling — a small model on a modest machine should never request
    a KV cache it cannot serve, and 64k is already far beyond chat
    use. When the GGUF metadata is unreadable the KV cannot be sized,
    so the conservative 16k default applies instead of the old flat
    200k (still capped at the model max when that IS known).
    """
    meta = read_gguf_meta(gguf)
    model_max = meta.context_length if (meta and meta.context_length) else None
    if not (meta and meta.complete):
        return max(_CTX_FLOOR, min(16384, model_max or 16384))
    placeholder = ModelSpec(
        name="_auto", label="_auto", gguf=str(gguf), llama_port=0,
        daemon_port=0, ctx_size=200000, parallel=parallel, kv=kv,
        max_tokens=1, model_id="_auto",
    )
    return max(_CTX_FLOOR, min(_clamp_ctx(placeholder), 65536))


def pull_model(
    repo_id: str,
    name: str | None = None,
    *,
    file: str | None = None,
    ctx_size: int | None = None,
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
    candidates is raised. ``ctx_size`` defaults to an auto-picked value
    that fits this machine and the model (RAM budget + model max ctx);
    pass an explicit value to override. Binary and env inherit from an
    existing spec's custom build (e.g. the machine's tuned ROCm
    llama.cpp) when available; ``reasoning`` selects the thinking-model
    handling.
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

    # A projector is not a model. Caught before the hub round-trip so the
    # error is the same offline as online.
    if file is not None and _is_mmproj(file):
        raise RuntimeError(
            f"'{file}' is a multimodal projector, not a model — pull the "
            "model's own GGUF and the projector comes with it automatically."
        )

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
    mmproj_file = _pick_mmproj(files)
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
        candidates = sorted(f for f in ggufs if not _is_mmproj(f))
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
            allow_patterns=_expand_gguf_patterns(filename, mmproj_file),
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
    is_image = any(k in f"{name} {filename}".lower() for k in IMAGE_KEYS)
    if is_image:
        # A diffusion denoiser: no llama defaults, no ctx, no env pins.
        binary, env = "sd-server", {}
    if ctx_size is None:
        try:
            ctx_size = 0 if is_image else _auto_ctx_size(local_dir / filename, parallel=2, kv=kv)
        except Exception:
            if is_image:
                ctx_size = 0
            else:
                raise
    raw = {
        "name": name,
        "label": Path(filename).stem,
        "gguf": str(local_dir / filename),
        "llama_port": llama_port,
        "daemon_port": daemon_port,
        "ctx_size": ctx_size,
        "parallel": 2,
        "kv": kv,
        "max_tokens": max(1, ctx_size // 2),
        # Use a short, stable identifier derived from the GGUF filename
        # stem (e.g. "Qwen3.5-0.8B-Q4_K_M.gguf" -> "qwen3.5-0.8b-q4_k_m").
        # The full repo id (e.g. "unsloth/Qwen3.5-0.8B-GGUF") is too long
        # and case-sensitive for hermes's model name matching.
        "model_id": _slugify_name(Path(filename).stem),
        "always_on": False,
        "boot": False,
        "binary": binary,
        "env": env,
        "reasoning": reasoning,
        "source": "pulled",
        # Where it came from. A projector that was not downloaded can be
        # fetched again later only if the repository is still known - and
        # "the model is blind" is not a state anybody should have to
        # discover by hand.
        "repo": repo_id,
        # Vision models only accept images when llama-server is given the
        # projector; without this field the stack starts and answers blind.
        "mmproj": str(local_dir / mmproj_file) if mmproj_file else "",
    }
    spec = ModelSpec(raw=raw, **raw)
    save_registry([*specs, spec], registry)
    if not mmproj_file:
        # The repo did not carry a tower. It may still be on this machine,
        # belonging to another model of the same architecture - and a model
        # that is blind is a model that answers as though it had seen
        # nothing, which is the worst kind of wrong. So the family tower
        # is adopted now, while the person is still here to hear about it.
        healed = backfill_mmproj(registry=registry)
        if healed:
            logger.info(
                "attached a projector already on this machine: %s", ", ".join(healed)
            )
    return spec


def normalize_model_ids(registry: Path | None = None) -> list[tuple[str, str, str]]:
    """Rewrite any spec whose ``model_id`` is a Hugging Face repo path
    (e.g. ``unsloth/Qwen3.5-0.8B-GGUF``) to a short, stable identifier
    derived from the GGUF filename stem.

    The repo path is the historical default from `pull_model` (before
    0.2.15) and is too long / case-sensitive for hermes's model-name
    matching, which surfaces as a "model 'X' not found" 404 from
    llama-server. The rewrite makes hermes/openclaw/opencode/codex see a
    short, consistent model name.

    Returns a list of ``(name, old_model_id, new_model_id)`` tuples so
    the caller can show what changed. Existing registry entries that
    already have a short model_id are left alone.
    """
    specs = load_registry(registry)
    if not specs:
        return []
    changes: list[tuple[str, str, str]] = []
    for s in specs:
        mid = s.model_id or ""
        # Treat anything containing a "/" as a repo path; short ids
        # like "qwen3.5-0.8b" or "qwen3.5-0.8b-q4_k_m" stay.
        if "/" not in mid:
            continue
        gguf = Path(s.gguf).name if s.gguf else ""
        new_mid = _slugify_name(Path(gguf).stem) if gguf else _slugify_name(mid.rsplit("/", 1)[-1])
        if not new_mid or new_mid == mid:
            continue
        changes.append((s.name, mid, new_mid))
        update_spec_field(s.name, "model_id", new_mid, registry=registry)
        logger.info("%s: model_id %r -> %r", s.name, mid, new_mid)
    return changes


# ── Process helpers ────────────────────────────────────────────────────


def _is_alive(pid: int) -> bool:
    """Is the pid still running?

    On Windows, ``os.kill(pid, 0)`` can return with an exception set
    (CPython wraps the underlying error as ``SystemError: <built-in
    function kill> returned a result with an exception set``), which
    the launcher catches in ``_header()`` and crashes the menu. Probe
    via ``OpenProcess``/``GetExitCodeProcess`` instead.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            OpenProcess = kernel32.OpenProcess
            OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            OpenProcess.restype = wintypes.HANDLE
            GetExitCodeProcess = kernel32.GetExitCodeProcess
            GetExitCodeProcess.argtypes = [wintypes.HANDLE,
                                           ctypes.POINTER(wintypes.DWORD)]
            GetExitCodeProcess.restype = wintypes.BOOL
            CloseHandle = kernel32.CloseHandle
            CloseHandle.argtypes = [wintypes.HANDLE]
            CloseHandle.restype = wintypes.BOOL
            handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return False
            try:
                code = wintypes.DWORD()
                if not GetExitCodeProcess(handle, ctypes.byref(code)):
                    return False
                return code.value == STILL_ACTIVE
            finally:
                CloseHandle(handle)
        except (OSError, SystemError):
            return False
    try:
        os.kill(pid, 0)
    except (OSError, SystemError):
        return False
    # PID may have been recycled to an unrelated process — verify
    # the cmdline still looks like llama-server/daemon.
    if os.name == "posix":
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
            if cmd and "llama-server" not in cmd and "uvicorn" not in cmd and "tokirecur" not in cmd:
                return False
        except OSError:
            pass
    return True


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


#: Repo/file name fragments that mark a diffusion-model GGUF. These are
#: denoisers (text-to-image), not captioners: a "qwen-image" GGUF cannot
#: load in llama-server any more than a chat GGUF can paint.
IMAGE_KEYS = ("qwen-image", "qwen_image", "flux", "sdxl", "stable-diffusion",
              "sd-turbo", "sd3", "hidream", "wan2")


def _spec_is_image(spec: ModelSpec) -> bool:
    if (spec.binary or "").endswith("sd-server"):
        return True
    hay = f"{spec.name} {spec.model_id} {Path(spec.gguf).name.lower()}".lower()
    return any(k in hay for k in IMAGE_KEYS)


def _sd_binary() -> str:
    """The sd-server this install built, mirroring llama-server's
    state-bin convention; PATH only as a fallback."""
    state_bin = Path.home() / ".local/state/tokirecur/bin/sd-server"
    if state_bin.is_file():
        return str(state_bin)
    return "sd-server"


def _sd_args(spec: ModelSpec) -> list[str]:
    """sd-server invocation for an image spec.

    Turbo distilled checkpoints (qwen-image-*-turbo) run 8 steps at CFG 1
    with euler — their shipped schedule. Non-turbo image families default
    to sd.cpp's own resolution-dependent flow (CFG 6), and any of it can
    be overridden per request through the API.
    """
    args = [
        _sd_binary(),
        "--diffusion-model", str(resolve_gguf_path(spec)),
        "--vae", str(_companion_path(spec, "vae")),
        "--llm", str(_companion_path(spec, "llm")),
        "--listen-ip", "127.0.0.1",
        "--listen-port", str(spec.llama_port),
        "--sampling-method", "euler",
    ]
    if "turbo" in spec.name.lower():
        args += ["--cfg-scale", "1.0", "--steps", "8"]
    else:
        args += ["--cfg-scale", "6.0"]
    args += spec.extra_args
    return args


def _companion_path(spec: ModelSpec, which: str) -> Path:
    """On-disk location of the VAE / text encoder for this image stack."""
    return Path(spec.gguf).parent / "companions" / (
        "vae.safetensors" if which == "vae" else "text-encoder.gguf"
    )


def _ensure_image_companions(spec: ModelSpec, registry: Path | None = None) -> None:
    """Heal an image stack's VAE + text encoder, once, like backfill_mmproj.

    The denoiser GGUF alone cannot generate: sd.cpp needs the family VAE
    and the LLM text encoder beside it. Known recipes only — an unknown
    image family fails loudly with the two Hugging Face ids it needs,
    rather than downloading something plausible and wrong.
    """
    if not _spec_is_image(spec) or (spec.vae and spec.llm):
        return
    if "qwen-image-2.1" in spec.name.lower() or "qwen_image_2.1" in spec.name.lower():
        vae_file = "vae/qwen_image_2.1_vae_bf16.safetensors"
        vae_repo = "unsloth/Qwen-Image-2.1-FP8"
        llm_repo, llm_file = (
            "unsloth/Qwen3-VL-8B-Instruct-GGUF",
            "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf",
        )
    else:
        raise RuntimeError(
            f"unknown image model family '{spec.name}' — pull the VAE and "
            "text-encoder companions, then set `vae:` and `llm:` in "
            "~/.config/tokirecur/models.yaml"
        )
    dest = _companion_path(spec, "vae").parent
    dest.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    vae_path = dest / "vae.safetensors"
    llm_path = dest / "text-encoder.gguf"
    if not vae_path.is_file():
        got = hf_hub_download(vae_repo, vae_file, local_dir=dest)
        if Path(got).resolve() != vae_path.resolve():
            vae_path.write_bytes(Path(got).read_bytes())
    if not llm_path.is_file():
        got = hf_hub_download(llm_repo, llm_file, local_dir=dest)
        if Path(got).resolve() != llm_path.resolve():
            llm_path.write_bytes(Path(got).read_bytes())
    reg = registry or DEFAULT_REGISTRY
    with _flock(reg):
        specs = load_registry(reg)
        for row in specs:
            if row.name == spec.name:
                row.raw["binary"] = "sd-server"
                row.raw["vae"] = str(vae_path)
                row.raw["llm"] = str(llm_path)
                row.vae, row.llm = str(vae_path), str(llm_path)
        _atomic_write_text(
            reg, yaml.safe_dump({"models": [s.raw for s in specs]}, sort_keys=False)
        )
        invalidate_registry_cache(reg)
    spec.vae, spec.llm = str(vae_path), str(llm_path)
    logger.info("attached image companions for '%s'", spec.name)


def backfill_image_stacks(all_models, registry: Path | None = None) -> list[str]:
    """Registry self-heal for image rows, run at every sync point.

    Fixes rows registered before image support (binary still
    llama-server, a nonsense ctx from the denoiser's metadata) and marks
    them as sd-server stacks so `use` starts the right thing.
    """
    fixed: list[str] = []
    changed = False
    for spec in all_models:
        if not _spec_is_image(spec) or (spec.binary or "").endswith("sd-server"):
            if _spec_is_image(spec) and spec.ctx_size:
                # already an image row; nothing to heal
                pass
            continue
        if _spec_is_image(spec):
            spec.binary = "sd-server"
            spec.raw["binary"] = "sd-server"
            if spec.ctx_size:
                spec.ctx_size = 0
                spec.raw["ctx_size"] = 0
            fixed.append(spec.name)
            changed = True
    if changed:
        reg = registry or DEFAULT_REGISTRY
        _atomic_write_text(
            reg, yaml.safe_dump({"models": [s.raw for s in all_models]}, sort_keys=False)
        )
        invalidate_registry_cache(reg)
    return fixed


def _normalised_model_name(name: str) -> str:
    """A model name reduced to one shape, for comparing a file to a server.

    llama-server reports the name of the model it loaded: no path, no
    extension. `Path(...).stem` cannot be used on it - a name with a dot
    in it ("Qwen3.5-122B-A10B") is treated as a file with an extension,
    and the stem comes out as "Qwen3". So the extension is taken off by
    hand, and only when it is one.

    Sharded models are named `...-00001-of-00003` on disk and reported
    without the shard suffix, and the registry and the server can differ
    in case and separators, so both sides are folded to one shape.
    """
    stem = name.strip().rsplit("/", 1)[-1]
    if stem.lower().endswith(".gguf"):
        stem = stem[: -len(".gguf")]
    stem = stem.lower().replace("_", "-").replace(".", "-")
    shard = stem.rfind("-00001-of-")
    if shard > 0:
        stem = stem[:shard]
    return stem


def _served_model_ids(base: str) -> set[str]:
    """The model ids a server on ``base`` says it is serving."""
    try:
        data = httpx.get(f"{base}/v1/models", timeout=2.0).json()
    except Exception:
        return set()
    ids: set[str] = set()
    if not isinstance(data, dict):
        return ids
    for row in data.get("data") or []:
        if isinstance(row, dict) and row.get("id"):
            ids.add(_normalised_model_name(str(row["id"])))
    for row in data.get("models") or []:
        if isinstance(row, dict):
            for key in ("id", "name", "model"):
                if row.get(key):
                    ids.add(_normalised_model_name(str(row[key])))
    return ids


def llama_reachable(spec: ModelSpec) -> bool:
    """Is THIS model's llama-server on its port?

    A server answering /health is not enough. Something else on the machine
    can hold the port - the desktop app's voice server sat on one of the
    registry's llama ports and answered /health with 200 - and a model that
    is not running then reads as running. The consequences are real: the
    status line lies, and `use_model` tries to free the RAM of a stack that
    was never up, fails to, and warns about a port it cannot stop.

    So the server is asked what it is serving, and the answer has to be
    this model. A server that will not say is given the benefit of the
    doubt: the point is to catch the impostor, not to start a second copy
    of a model that is genuinely up.
    """
    if _spec_is_image(spec):
        # sd-server has no /health; its OpenAI model listing is the
        # liveliness answer.
        try:
            return httpx.get(f"{spec.llama_url}/v1/models", timeout=2.0).status_code == 200
        except Exception:
            return False
    try:
        if httpx.get(f"{spec.llama_url}/health", timeout=2.0).status_code != 200:
            return False
    except Exception:
        return False
    served = _served_model_ids(spec.llama_url)
    if not served:
        return True
    gguf = getattr(spec, "gguf", "") or ""
    if not gguf:
        # Nothing to compare against: a model with no file cannot be the
        # thing on the port.
        return False
    return _normalised_model_name(gguf) in served


def daemon_reachable(spec: ModelSpec) -> bool:
    """The model's daemon is up AND wired to THIS model's llama-server.

    A daemon that answers /healthz but proxies a dead backend — or was
    started for a different backend (e.g. a leftover ``toki serve`` from
    an old install still bound to the model's daemon port) — is NOT the
    model's daemon: clients get connection-refused through it. Only
    count it as reachable when its own upstream probe reports the
    backend live AND that backend is the model's llama_url.
    """
    headers = None
    token = _configured_auth_token()
    if token:
        headers = {"Authorization": f"Bearer {token}"}
    try:
        if httpx.get(
            f"{spec.daemon_url}/healthz", timeout=2.0, headers=headers
        ).status_code != 200:
            return False
        r = httpx.get(
            f"{spec.daemon_url}/v1/health/llama", timeout=2.0, headers=headers
        )
        if r.status_code != 200:
            return False
        data = r.json()
        if data.get("status") != "ok":
            return False
        backend = data.get("backend")
        # A daemon answering ok but for a DIFFERENT llama — or one that
        # cannot state its backend at all (stale install, old venv) — is
        # not the model's daemon. Exact match required.
        return backend == spec.llama_url
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
    """Find the pid listening on `port` (best effort, per platform).

    Linux uses ``ss -tlnp``, macOS ``lsof -iTCP``, Windows
    ``netstat -ano -p TCP``; returns None when the tool is missing or
    nothing is listening. Cached for ~1 s so a busy poll loop doesn't
    shell out per model per tick. The TTL is short enough that
    newly-spawned processes are picked up promptly while the daemon is
    still probing.
    """
    now = time.time()
    cached = _PORT_PID_CACHE.get(port)
    if cached and now - cached[0] < 1.0:
        return cached[1]
    try:
        if sys.platform.startswith("linux"):
            out = subprocess.run(
                ["ss", "-tlnp"], capture_output=True, text=True, timeout=5.0
            ).stdout
            pid = None
            for line in out.splitlines():
                if f":{port} " not in line and not line.endswith(f":{port}"):
                    continue
                m = re.search(r"pid=(\d+)", line)
                if m:
                    pid = int(m.group(1))
                    break
        elif sys.platform == "darwin":
            out = subprocess.run(
                ["lsof", "-nP", "-iTCP", f":{port}", "-sTCP:LISTEN"],
                capture_output=True, text=True, timeout=5.0,
            ).stdout
            pid = None
            for line in out.splitlines()[1:]:
                parts = line.split()
                if len(parts) >= 2 and parts[0] != "COMMAND":
                    try:
                        pid = int(parts[1])
                        break
                    except ValueError:
                        continue
        elif sys.platform.startswith("win"):
            out = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True, text=True, timeout=5.0,
            ).stdout
            pid = None
            for line in out.splitlines():
                if "LISTENING" not in line:
                    continue
                parts = line.split()
                if len(parts) >= 5 and parts[1].endswith(f":{port}"):
                    try:
                        pid = int(parts[-1])
                        break
                    except ValueError:
                        continue
        else:
            return None
    except Exception:
        # Fallback for minimal containers without ss/lsof/netstat:
        # try fuser, then /proc/net/tcp scan.
        try:
            out = subprocess.run(
                ["fuser", f"{port}/tcp"], capture_output=True, text=True, timeout=5.0
            ).stdout
            m = re.search(r"(\d+)", out)
            if m:
                pid = int(m.group(1))
                _PORT_PID_CACHE[port] = (now, pid)
                return pid
        except Exception:
            pass
        return None
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


def gpu_state(spec: ModelSpec) -> dict[str, str | None]:
    """GPU evidence from the model's llama.log.

    Returns ``{"backend", "device", "layers"}``; every value is None
    when the log is missing or shows no GPU. Matches both the classic
    ``offloaded 34/37 layers to GPU`` line and the modern backend
    device lines (``- CUDA0 : NVIDIA GeForce GTX 1650 ...``), so the
    check survives llama.cpp log format churn.
    """
    empty: dict[str, str | None] = {"backend": None, "device": None,
                                    "layers": None}
    # MLX has no llama.cpp log format. If the registry points at the
    # mlx_lm.server entry point, report the backend from the binary
    # build instead of parsing the log.
    try:
        from .. import diagnose
        if spec.binary and "mlx_lm" in spec.binary:
            kind = diagnose.binary_backend(spec.binary)
            if kind:
                # mlx_lm.server doesn't print a device name; pick a
                # sensible default for Apple Silicon.
                device = "Apple GPU" if sys.platform == "darwin" else None
                return {"backend": "MLX", "device": device, "layers": None}
    except Exception:  # noqa: BLE001
        pass

    log = spec.state_dir() / "llama.log"
    if not log.exists():
        return empty
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return empty
    result = dict(empty)
    for line in text.splitlines():
        m = re.search(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU", line)
        if m:
            result["layers"] = f"{m.group(1)}/{m.group(2)}"
        # "- CUDA0 : NVIDIA GeForce GTX 1650 (sm_75, 4096 MiB)" etc.
        # Older builds wrote "CUDA0 : ..." without the leading dash.
        # system_info lines also mention backends but only as flags
        # ("CUDA0 : NO_VMM = 1") — never a device; skip them.
        if "system_info" in line:
            continue
        m = re.search(
            r"(?:-\s+)?(CUDA\d+|ROCm\d+|Vulkan\d+|Metal)\s*:\s*"
            r"([^()\n]+?)\s*(?:\([^)]*\))?\s*$",
            line,
        )
        if m:
            name = m.group(1)
            result["backend"] = name.rstrip("0123456789")
            result["device"] = m.group(2).strip()
    return result


# ── Start / stop ───────────────────────────────────────────────────────


def _resolve_binary(spec: ModelSpec) -> str:
    """Concrete llama-server path for launching.

    Entries whose ``binary`` is still the bare PATH default (legacy
    entries written before backfill, or pulls from a registry without
    a tuned reference spec) resolve to the build installed by
    ``toki model install-llama`` when one exists — on Windows the bare
    name is not on PATH and Popen would fail with WinError 2.
    """
    if spec.binary and spec.binary != "llama-server":
        pinned = Path(spec.binary)
        if pinned.is_absolute() and not pinned.exists():
            pass  # dead weight — fall through to installed build
        else:
            return spec.binary
    installed = _installed_binary()
    if installed:
        return installed
    return "llama-server"


def _effective_ngl(spec: ModelSpec) -> int:
    """Resolve the -ngl value that ``_llama_args`` will actually pass.

    Same precedence as ``_llama_args``: ``extra_args`` pinning wins
    (the user explicitly chose it via registry or model config), then
    ``TOKI_NGL`` env override, then the hardware-profile default. When
    ``extra_args`` pins -ngl 0 the function still returns 0 so the
    fail-fast check below doesn't misfire on a legitimate CPU run.
    """
    for i, a in enumerate(spec.extra_args):
        if a in ("-ngl", "--n-gpu-layers") and i + 1 < len(spec.extra_args):
            try:
                return int(spec.extra_args[i + 1])
            except ValueError:
                return 0
    return _default_ngl()


def _default_ngl() -> int:
    """Offload layers for llama-server. ``TOKI_NGL`` wins; otherwise the
    hardware profile decides; if no profile is available, CPU wins.

    The hardware profile is the authority — it was written by
    ``toki install-llama`` at install time, when the actual GPUs on
    the box were enumerated. A GPU with ``use_for_llm=True`` means
    discrete, ≥ 4 GB VRAM, not shared RAM; that's the right signal
    for offload. A box with only an Intel iGPU / AMD APU / shared-RAM
    part has no GPU with ``use_for_llm=True`` and the CPU backend
    (``libggml-cpu-alderlake.so`` on Meteor Lake, etc.) handles every
    layer.

    The runtime probes (nvidia-smi, hipconfig, vulkaninfo, Linux ICD
    probe) are gone: those heuristics wrongly picked lavapipe / iGPUs
    and made CPU-only boxes slower than no offload at all. The profile
    is what ``install-llama`` already trusts to pick the build, so
    it's the right thing to trust here too.
    """
    raw = os.environ.get("TOKI_NGL")
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            pass
    try:
        from ..hardware import profile_box
        prof = profile_box()
        if prof is not None and any(g.use_for_llm for g in prof.gpus):
            return 99
    except Exception:
        pass
    return 0


def _check_gpu_offload_binary_match(
    binary: str, ngl: int | None, *, strict: bool = False,
) -> bool:
    """Return True when ``ngl > 0`` against a CPU binary would degrade
    throughput silently. Callers in ``strict`` mode (``start_model``)
    must abort with a clear error; non-strict callers (the launcher
    settings preview) just emit the warning.

    The classic footgun the CPU/Vulkan build mix produced on Intel
    iGPU boxes: -ngl 99 with a CPU build silently falls back to CPU
    per-layer, or worse, the Vulkan build initialises the iGPU and
    syncs on every token (0.78 t/s measured). Once install-llama can
    land the wrong backend (e.g. a Linux ROCm box with no release
    asset ships a CPU build), the model still launches, the user
    gets degraded perf, and the only feedback is a stderr warning
    that's easy to miss. Refusing to start forces the issue.
    """
    if not ngl or ngl <= 0:
        return True
    try:
        from ..diagnose import binary_backend
        kind = binary_backend(binary)
    except Exception:
        return True  # can't inspect; trust the user's explicit ngl
    if not kind or kind.lower() in ("cuda", "rocm", "vulkan", "metal"):
        return True
    msg = (
        f"GPU offload requested (-ngl {ngl}) but the installed "
        f"llama-server build is {kind}; expected throughput is CPU "
        f"only. Install the matching GPU build with "
        f"`toki model install-llama --backend auto` (or pass "
        f"TOKI_NGL=0 to force CPU mode for this launch)."
    )
    if strict:
        raise RuntimeError(msg)
    click.echo(f"warning: {msg}", err=True)
    return False


def _warn_gpu_offload_without_gpu_binary(binary: str, ngl: int) -> None:
    """Backward-compatible warning wrapper. New code should call
    ``_check_gpu_offload_binary_match`` directly with the right
    ``strict`` flag."""
    _check_gpu_offload_binary_match(binary, ngl, strict=False)


def _intel_igpu_only() -> bool:
    """True when the only Vulkan device on this Windows box is an
    integrated Intel GPU (HD/UHD/Iris).

    Those iGPUs are far slower than the CPU for llama.cpp and their old
    Vulkan drivers make a full offload thrash at 100% util without
    generating tokens. Offload is skipped unless a discrete/Arc GPU is
    present; ``TOKI_NGL`` still wins if set explicitly.
    """
    if os.name != "nt":
        return False
    if not shutil.which("vulkaninfo"):
        return False
    try:
        out = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout or ""
    except Exception:
        return False
    names = [
        line.split("=", 1)[-1].strip()
        for line in out.splitlines()
        if "deviceName" in line
    ]
    if not names:
        return False
    intel_igpu = "Intel" in names[0] and "Arc" not in names[0]
    return intel_igpu and len(names) == 1


_SOFTWARE_VULKAN_MARKERS = ("lvp", "lavapipe", "llvmpipe", "swiftshader")


def _linux_vulkan_works() -> bool:
    """True when this Linux box has at least one loadable *real* Vulkan ICD.

    ``libvulkan.so.1`` is the loader; without an ICD JSON in
    ``/usr/share/vulkan/icd.d/`` pointing at an installed driver
    library, every Vulkan call returns VK_ERROR_INCOMPATIBLE_DRIVER
    and llama-server's Vulkan backend can probe/init forever without
    producing tokens. Many Ubuntu installs have the loader (Steam,
    OBS, browsers) without any usable ICD, so checking for the loader
    alone produces a vulkan-on-CPU build that hangs at runtime.

    Software Vulkan ICDs — Mesa lavapipe/llvmpipe (``libvulkan_lvp.so``,
    the only Vulkan on many headless CPU-only Ubuntu boxes) and
    SwiftShader — are excluded: they execute on the CPU and are far
    slower than llama.cpp's native CPU kernels, so offloading to them
    (``-ngl 99``) makes a CPU box *slower*, not faster.
    """
    import json

    loader_paths = (
        Path("/usr/lib/x86_64-linux-gnu/libvulkan.so.1"),
        Path("/usr/lib/aarch64-linux-gnu/libvulkan.so.1"),
        Path("/usr/lib/libvulkan.so.1"),
    )
    if not any(p.exists() for p in loader_paths):
        return False
    icd_dir = Path("/usr/share/vulkan/icd.d")
    if not icd_dir.is_dir():
        return False
    for icd_json in icd_dir.glob("*.json"):
        try:
            data = json.loads(icd_json.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        lib_path = (data.get("ICD") or {}).get("library_path") or ""
        if not lib_path:
            continue
        low = Path(lib_path).name.lower()
        if any(marker in low for marker in _SOFTWARE_VULKAN_MARKERS):
            continue
        # ICD library_path can be absolute or relative. The loader
        # checks a few standard locations in order.
        candidates = [
            Path(lib_path),
            Path("/usr/lib") / lib_path,
            Path("/usr/lib/x86_64-linux-gnu") / lib_path,
            Path("/usr/lib/aarch64-linux-gnu") / lib_path,
        ]
        if any(c.is_file() for c in candidates):
            return True
    return False


def _intel_igpu_only_linux() -> bool:
    """True when this Linux box's only GPU is an Intel iGPU.

    Reads PCI vendor IDs from ``/sys/class/drm`` (no vulkaninfo
    required, no subprocess) and looks for the 0x8086 Intel vendor.
    Treats the box as iGPU-only when Intel is present and no
    NVIDIA (0x10de) or AMD/ATI (0x1002) discrete GPU shows up. Mirrors
    the Windows check so the same Vulkan-driver-bug class of iGPUs
    (Skylake/Kaby Lake/Coffee Lake UHD) get CPU builds on Linux too.
    """
    drm_root = Path("/sys/class/drm")
    if not drm_root.is_dir():
        return False
    vendors: set[str] = set()
    for card in drm_root.glob("card*"):
        # Skip render nodes (card0 renders as renderD128) — those are
        # for offscreen rendering, not a separate GPU.
        if not (card / "device").is_dir():
            continue
        try:
            vendors.add((card / "device" / "vendor").read_text().strip())
        except OSError:
            continue
    if "0x8086" not in vendors:
        return False
    discrete = vendors & {"0x10de", "0x1002"}
    return not discrete


_BINARY_FLAGS: dict[str, set[str]] = {}

def _binary_supports(binary: str, flag: str) -> bool:
    """True if the llama-server binary knows a CLI flag. Asks the binary
    itself (`--help`, cached per path) — scanning embedded strings fails on
    thin wrapper binaries whose logic lives in sidecar .so files. Fails
    closed: an unanswerable binary gets old behavior, never a fatal flag."""
    if binary in _BINARY_FLAGS:
        return flag in _BINARY_FLAGS[binary]
    found: set[str] = set()
    try:
        import subprocess

        out = subprocess.run(
            [binary, "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        text = (out.stdout or "") + (out.stderr or "")
        for probe in ("--slot-save-path", "--alias", "--reasoning"):
            if probe in text:
                found.add(probe)
    except Exception:
        pass
    _BINARY_FLAGS[binary] = found
    return flag in found


def model_aliases(spec: ModelSpec) -> list[str]:
    """Every model name the stack can hand to a client or receive back.

    The union covers: the registry model id, the GGUF filename stem
    (what a freshly-pulled model gets before normalization), ``default``
    (toki's own chat client), the opencode provider model name
    (``tokirecur-<name>/<model_id>``), and the legacy short form
    (``tokirecur/<name>``). Aliases are registered on llama-server so a
    validating build accepts any of them instead of returning the
    ``model '<name>' not found`` 404.
    """
    aliases = {spec.model_id} if spec.model_id else set()
    if spec.gguf:
        aliases.add(Path(spec.gguf).name.removesuffix(".gguf"))
    aliases.update(("default", f"tokirecur/{spec.name}"))
    if spec.model_id:
        aliases.add(f"tokirecur-{spec.name}/{spec.model_id}")
    return sorted(a for a in aliases if a)


#: The private name, for callers inside the package.
_model_aliases = model_aliases


def _llama_spawn_env(spec: ModelSpec) -> dict[str, str] | None:
    """Spawn env for the model's llama-server.

    Self-heals builds whose RUNPATH no longer resolves (e.g. a binary
    rebuilt on another box whose rpath points at a build tree that does
    not exist here): the loader finds the sidecar libs (libllama-server
    -impl.so, libggml*.so) by LD_LIBRARY_PATH instead, so the model
    starts without any manual fix. Prepend the binary's own directory
    and the ROCm lib dir when present; keep everything else the registry
    env set.
    """
    base = {**os.environ, **spec.env} if spec.env else dict(os.environ)
    binary = _resolve_binary(spec)
    if binary and binary != "llama-server":
        parts = [os.path.dirname(os.path.abspath(binary))]
        rocm_lib = "/opt/rocm/lib"
        if os.path.isdir(rocm_lib):
            parts.append(rocm_lib)
        existing = base.get("LD_LIBRARY_PATH")
        if existing:
            parts.append(existing)
        base["LD_LIBRARY_PATH"] = ":".join(parts)
    return base


def _llama_args(spec: ModelSpec, ctx_override: int | None = None,
                kv_override: str | None = None,
                parallel_override: int | None = None) -> list[str]:
    # MLX has a different invocation: mlx_lm.server takes a model
    # directory and exposes an OpenAI-compatible API on a port.
    binary = _resolve_binary(spec)
    gguf = resolve_gguf_path(spec) if not (binary == "mlx_lm.server" or binary.endswith("/mlx_lm.server")) else Path(spec.gguf)
    if binary == "mlx_lm.server" or binary.endswith("/mlx_lm.server"):
        if not Path(spec.gguf).exists():
            raise FileNotFoundError(f"model directory not found: {spec.gguf}")
        return [
            sys.executable, "-m", "mlx_lm.server",
            "--model", str(spec.gguf),
            "--host", "127.0.0.1",
            "--port", str(spec.llama_port),
        ]
    ctx = spec.ctx_size if ctx_override is None else ctx_override
    kv = spec.kv if kv_override is None else kv_override
    parallel = spec.parallel if parallel_override is None else parallel_override
    # TOKI_CTX_SIZE / TOKI_THREADS: hard user overrides. They are
    # appended AFTER extra_args so llama.cpp's last-flag-wins parser
    # honours them even over a registry pin.
    try:
        toki_ctx = int(os.environ.get("TOKI_CTX_SIZE", ""))
    except ValueError:
        toki_ctx = None
    try:
        toki_threads = int(os.environ.get("TOKI_THREADS", ""))
    except ValueError:
        toki_threads = None
    args = [
        binary,
        "-m", str(gguf),
        "--host", "127.0.0.1",
        "--port", str(spec.llama_port),
        "-c", str(ctx),
        "--parallel", str(parallel),
        "-kvu",
        "-lv", "3",
    ]
    # Offload to the GPU when one is present. Registry `extra_args` may
    # already pin `-ngl` (or the long form); never duplicate it.
    ngl_pinned = any(a in ("-ngl", "--n-gpu-layers") for a in spec.extra_args)
    ngl = _default_ngl() if not ngl_pinned else None
    if ngl is not None:
        _warn_gpu_offload_without_gpu_binary(binary, ngl)
        args += ["-ngl", str(ngl)]

    # CPU-mode tuning: when no GPU is in play (ngl == 0), use every
    # physical core, plus mlock and batch sizes. None of these are
    # useful on GPU paths — the GPU does the matmul, not the CPU.
    # Thread count is recomputed fresh (never read from a cached or
    # hand-edited profile) so a stale hardware.json cannot silently
    # cap CPU throughput. ``effective_cpu_threads`` drops the slowest
    # tier on hybrid chips (e.g. LPE on Core Ultra 155U) so the
    # llama.cpp thread pool isn't gated by 1.5 GHz cores.
    threads_added = False
    if (ngl == 0) and not any(a in ("-t", "--threads") for a in spec.extra_args):
        threads = None
        # TOKI_THREADS wins when set; otherwise auto-detect.
        if toki_threads:
            threads = toki_threads
        else:
            try:
                from ..hardware import cpu_topology, effective_cpu_threads
                threads = effective_cpu_threads(cpu_topology())
            except Exception:
                threads = None
        if threads is None:
            try:
                from ..hardware import profile_box
                threads = int(profile_box().recommended.get("threads", 4))
            except Exception:
                threads = 4
        args += ["-t", str(max(2, threads))]
        threads_added = True
        args += ["--mlock"]
        # Batch sizes — only set if the registry didn't already.
        if not any(a in ("-b", "--batch-size") for a in spec.extra_args):
            args += ["-b", "2048"]
        if not any(a in ("-ub", "--ubatch-size") for a in spec.extra_args):
            args += ["-ub", "512"]
    if kv != "f16":
        args += ["--cache-type-k", kv, "--cache-type-v", kv]
    if spec.reasoning == "off":
        args += ["--reasoning", "off"]
    elif spec.reasoning == "on":
        args += ["--reasoning", "on"]
    # A vision model needs its projector or it answers blind. Emitted as two
    # separate argv tokens, and skipped when the registry already pins
    # --mmproj in extra_args so the pin still wins.
    mmproj = getattr(spec, "mmproj", "") or ""
    if mmproj and not any(a in ("-mm", "--mmproj") for a in spec.extra_args):
        if not Path(mmproj).exists():
            raise FileNotFoundError(
                f"model '{spec.name}' is a vision model but its projector is "
                f"missing: {mmproj} — re-pull it, or drop the mmproj "
                f"field to run it text-only"
            )
        args += ["--mmproj", mmproj]
    # Register every model name the stack can emit as an alias so a
    # validating (router-mode) llama-server can never 404 a client that
    # talks to it directly with a stale/legacy name. One comma-separated
    # flag: repeated --alias is deprecated upstream and warns on every boot.
    aliases = [a for a in _model_aliases(spec) if a and "," not in a]
    if aliases:
        args += ["--alias", ",".join(aliases)]
    # KV persistence needs a server-side directory (--slot-save-path),
    # which only exists in recent llama.cpp. Gated on the binary so stacks
    # keep booting on the old build until the new one lands.
    if _binary_supports(binary, "--slot-save-path"):
        slot_dir = Path.home() / ".local" / "state" / "tokirecur" / "slot-save"
        try:
            slot_dir.mkdir(parents=True, exist_ok=True)
            args += ["--slot-save-path", str(slot_dir)]
        except OSError:
            pass
    args += spec.extra_args
    # Hard user overrides, appended LAST so llama.cpp's last-flag-wins
    # parser honours them even over registry extra_args pins.
    if toki_ctx:
        args += ["-c", str(toki_ctx)]
    if toki_threads and not threads_added:
        args += ["-t", str(toki_threads)]
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


def _detach_kwargs() -> dict[str, Any]:
    """Popen kwargs that detach a spawned service from the launcher.

    ``start_new_session`` (POSIX) breaks the child out of the process
    group so Ctrl+C in the launching console can't kill the model
    stack. On Windows it is a silent no-op: the child stays in the
    console's process group and receives CTRL_C_EVENT with the
    launcher — which shut the daemon down cleanly while the launch
    thread still waited on it (reported as "exited during startup").
    CREATE_NEW_PROCESS_GROUP is the Windows equivalent: the child no
    longer receives console Ctrl+C events (stopping still works via
    the pid, which is a hard TerminateProcess on Windows).
    """
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": _DETACH}


def _log_tail(path: Path, n: int = 40) -> str:
    """Last ``n`` lines of a log file, for surfacing start failures."""
    try:
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return "(log unavailable)"


def _healing_hint_for_log(log_path: Path, service: str) -> str:
    """Extra hint appended to a startup failure. Inspects the log tail
    for known self-healable patterns (tensor mismatch, OOM, port busy)
    and for GPU/backend mismatches via diagnose.hint."""
    tail = _log_tail(log_path).lower()
    hints: list[str] = []
    if "wrong number of tensors" in tail or "done_getting_tensors" in tail:
        hints.append(
            "hint: GGUF tensor count mismatch — file was quantized with a "
            "different llama.cpp version or is truncated. Re-pull the model: "
            "`toki model pull <repo>` or remove ~/models/<name> and re-pull"
        )
    if "out of memory" in tail or "failed to allocate" in tail or "cannot allocate" in tail:
        hints.append(
            "hint: OOM — try a smaller ctx_size/kv (e.g. 50000/q8_0) or free RAM by stopping other models"
        )
    if "address already in use" in tail or "eaddrinuse" in tail:
        hints.append("hint: port busy — run `toki model status` and `toki down --yes` to free the port, or change llama_port/daemon_port")
    # Generic GPU hint (only when service is llama)
    if "llama" in service.lower():
        try:
            from .. import diagnose as _diag
            if log_path.parent.name != "e2e-test":
                # Find the spec for this state dir by name
                name = log_path.parent.name
                try:
                    from ..ctl import get_model as _gm
                    spec = _gm(name)
                    hs = _diag.hint(_diag.gpu_hardware(), spec)
                    if hs:
                        hints.extend(hs)
                except Exception:
                    pass
        except Exception:
            pass
    return ("\n".join(hints) + "\n") if hints else ""


def _wait_healthy(url: str, proc: subprocess.Popen | None, log_path: Path,
                  timeout: float, service: str,
                  accept: "Callable[[httpx.Response], bool] | None" = None) -> None:
    """Wait for a service's health endpoint, failing fast if the child
    process dies instead of spinning until ``timeout``.

    Raises RuntimeError with the log tail on child death, so "starting
    model" can never hang silently behind a crashed llama-server.

    A 2xx/3xx from an endpoint that is NOT our service (a foreign
    webapp on the same port answering ``/health`` with its own body)
    fails fast after a short grace window instead of burning the full
    ``timeout`` — the child can't win a port that's already owned.
    """
    started = time.time()
    foreign_since: float | None = None
    while time.time() - started < timeout:
        if proc is not None and proc.poll() is not None:
            extra = _healing_hint_for_log(log_path, service)
            raise RuntimeError(
                f"{service} exited during startup (exit code {proc.returncode}); "
                f"last lines of {log_path}:\n{_log_tail(log_path)}\n"
                + (extra or "")
            )
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code < 500:
                # llama-server /health answers {"status":"ok"}; the
                # daemon healthz answers {"ok":true} or similar JSON.
                # Anything else (HTML, empty, other app's schema) that
                # persists past 5 s means a foreign service owns the
                # port — fail fast instead of waiting out `timeout`.
                if accept is not None and accept(r):
                    # Caller-defined liveness (sd-server: any 2xx from the
                    # model listing; a port conflict shows up as the child
                    # dying on bind, which poll() catches above).
                    return
                body = str(getattr(r, "text", "") or "")[:512].lower()
                looks_ours = (
                    "status" in body or "ok" in body
                    or not body.strip()
                )
                if looks_ours:
                    return
                # Foreign schema persisting past 5 s means another
                # service owns the port — our child can't win it (it
                # would bind-fail and be caught by poll() above).
                if foreign_since is None:
                    foreign_since = time.time()
                elif time.time() - foreign_since > 5.0:
                    raise RuntimeError(
                        f"port for {url} answers but not with a "
                        f"{service} response (got "
                        f"{r.status_code}, body starts: {body[:80]!r}). "
                        f"Another service likely owns this port — "
                        f"change llama_port/daemon_port in settings "
                        f"or stop the other service."
                    )
        except RuntimeError:
            raise
        except Exception:
            foreign_since = None
        time.sleep(0.5)
    extra = _healing_hint_for_log(log_path, service)
    raise RuntimeError(
        f"{service} did not become healthy within {timeout:.0f}s; "
        f"last lines of {log_path}:\n{_log_tail(log_path)}\n"
        + (extra or "")
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


def _real_slot_count(url: str) -> int | None:
    """How many slots the running llama-server actually exposes.

    Used to size the daemon's ``max_slots`` clamp when the server was
    launched with a reduced ``--parallel`` (fit planner) — the daemon
    must never round-robin onto a slot that doesn't exist (400s).
    """
    try:
        r = httpx.get(f"{url}/slots", timeout=3.0)
        r.raise_for_status()
        slots = r.json()
        return len(slots) if isinstance(slots, list) else None
    except Exception:
        return None


def _takeover_stale_daemon(spec: ModelSpec) -> None:
    """Free the model's daemon port from a stale daemon before spawning.

    A daemon that answers /healthz but proxies a dead backend (e.g. a
    leftover ``toki serve`` from an old install still bound to the model's
    daemon port) blocks the correct per-model daemon from binding. Stop
    it best-effort when it fails the strict reachability check — only
    processes that answer the daemon healthz are touched, never an
    unrelated listener on the port.
    """
    pid = _pid_on_port(spec.daemon_port)
    if pid is None or not _is_alive(pid):
        return
    our = daemon_pid(spec)
    if our and our == pid:
        # Our own managed daemon, but failing the strict reachability
        # check (old code without a backend report, or pointed at the
        # wrong llama). Replace it so the fresh spawn can bind.
        logger.warning(
            "%s: our daemon on :%d (pid %d) fails the strict reachability "
            "check — restarting it with current code",
            spec.name, spec.daemon_port, pid,
        )
        _stop_pidfile(spec.state_dir() / "daemon.pid")
        return
    try:
        r = httpx.get(f"{spec.daemon_url}/healthz", timeout=2.0)
        is_daemon = r.status_code == 200
    except Exception:
        is_daemon = False
    if not is_daemon:
        logger.warning(
            "%s: pid %d on :%d does not answer the daemon healthz — "
            "leaving it alone",
            spec.name, pid, spec.daemon_port,
        )
        return
    logger.warning(
        "%s: stale daemon on :%d (pid %d) with a dead backend — stopping it "
        "so the correct daemon can bind",
        spec.name, spec.daemon_port, pid,
    )
    _signal_proc(pid, signal.SIGTERM)
    deadline = time.time() + 8.0
    while time.time() < deadline and _is_alive(pid):
        time.sleep(0.2)
    if _is_alive(pid):
        _hard_kill(pid)
    if _is_alive(pid):
        raise RuntimeError(
            f"port :{spec.daemon_port} is held by pid {pid}, which does not "
            f"answer the daemon healthz and could not be stopped; it is not "
            f"this model's daemon (backend mismatch or stale install)"
        )


def start_model(spec: ModelSpec, wait: bool = True) -> None:
    """Bring up the model's llama-server + daemon (idempotent)."""
    state = spec.state_dir()
    state.mkdir(parents=True, exist_ok=True)

    # Image stacks are one sd-server, not llama+daemon: the daemon is the
    # text path and cannot carry image generation, and llama-server cannot
    # load a diffusion GGUF at all. The port scheme is reused (`llama_port`
    # is the image server's port; `daemon_port` stays unused).
    if _spec_is_image(spec):
        _ensure_image_companions(spec)
        if llama_reachable(spec):
            return
        args = _sd_args(spec)
        log = open(state / "llama.log", "ab")
        proc = subprocess.Popen(
            args,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            **_detach_kwargs(),
        )
        (state / "llama.pid").write_text(str(proc.pid))
        logger.info("%s: sd-server started (pid=%d)", spec.name, proc.pid)
        if wait:
            _wait_healthy(
                f"{spec.llama_url}/v1/models", proc, state / "llama.log",
                timeout=600.0, service=f"sd-server ({spec.name})",
                accept=lambda r: r.status_code == 200,
            )
        return

    # Keep the user's requested ctx_size in the registry even when
    # it exceeds the GGUF's declared max. The launch still clamps via
    # _fit_plan/_clamp_ctx (and shows reg:200000 · live:131072 in the
    # launcher), but the registry no longer snaps back to 131072
    # silently — “I want 200000 and it goes to 131072”.
    max_ctx = _model_max_ctx(spec)
    if max_ctx and spec.ctx_size > max_ctx:
        logger.warning(
            "%s: ctx %d > model max %d — keeping requested %d, "
            "launch will clamp to %d (reg · live will show the clamp)",
            spec.name, spec.ctx_size, max_ctx, spec.ctx_size, max_ctx,
        )

    if not llama_reachable(spec):
        if not spec.gguf or spec.gguf == "llama-server":
            raise FileNotFoundError(f"model file not found: {spec.gguf}")
        resolve_gguf_path(spec)  # raises a one-command hint when missing
        _check_ram(spec)
        ctx, kv, parallel = _fit_plan(spec)
        if ctx < spec.ctx_size or kv != spec.kv or parallel != spec.parallel:
            logger.warning(
                "%s: launch plan adjusted — ctx %d (spec %d), kv %s (spec %s), "
                "slots %d (spec %d) so the KV cache fits in RAM",
                spec.name, ctx, spec.ctx_size, kv, spec.kv, parallel, spec.parallel,
            )
        # Fail-fast on GPU/CPU mismatch before any process is spawned.
        # If the registry or TOKI_NGL asks for GPU offload but the
        # installed llama-server is a CPU build, refuse to start —
        # silent CPU fallback wastes RAM on a broken KV cache layout
        # and produces 0.5–1 t/s on boxes that should hit 40+.
        args = _llama_args(spec, ctx_override=ctx, kv_override=kv,
                           parallel_override=parallel)
        binary = _resolve_binary(spec)
        _check_gpu_offload_binary_match(
            binary,
            _effective_ngl(spec),
            strict=True,
        )
        log = open(state / "llama.log", "ab")
        proc = subprocess.Popen(
            args,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            **_detach_kwargs(),
            env=_llama_spawn_env(spec),
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
        _takeover_stale_daemon(spec)
        effective = _effective_ctx(spec)
        real_slots = _real_slot_count(spec.llama_url)
        slots = real_slots if real_slots else spec.parallel
        log = open(state / "daemon.log", "ab")
        env = {
            **os.environ,
            "TOKI_BACKEND_URL": spec.llama_url,
            "TOKI_MAX_CONTEXT_TOKENS": str(effective),
            "TOKI_MAX_SLOTS": str(slots),
            "TOKI_REASONING": spec.reasoning,
            "TOKI_SAMPLING": json.dumps(merged_sampling(spec)),
        }
        if spec.max_output_tokens:
            env["TOKI_MAX_OUTPUT_TOKENS"] = str(spec.max_output_tokens)
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
            **_detach_kwargs(),
        )
        (state / "daemon.pid").write_text(str(proc.pid))
        logger.info("%s: daemon started (pid=%d)", spec.name, proc.pid)
        if wait:
            _wait_healthy(
                f"{spec.daemon_url}/healthz", proc, state / "daemon.log",
                timeout=60.0, service=f"toki daemon ({spec.name})",
            )


def _signal_proc(pid: int, sig: int) -> None:
    """Send ``sig`` to the process — to its whole process group on POSIX,
    and to the single process on Windows (where ``os.killpg`` does not
    exist and the pid is the only handle we have)."""
    if hasattr(os, "killpg") and hasattr(os, "getpgid"):
        try:
            os.killpg(os.getpgid(pid), sig)
            return
        except OSError:
            pass
    try:
        os.kill(pid, sig)
    except OSError:
        pass


def _hard_kill(pid: int) -> None:
    """Best-effort hard kill: ``taskkill /F`` on Windows (where Python's
    ``signal`` module has no SIGKILL), ``os.kill(SIGKILL)`` on POSIX."""
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/pid", str(pid), "/f"],
                capture_output=True, text=True, timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        return
    _signal_proc(pid, signal.SIGKILL)


def _stop_pidfile(path: Path, grace: float = 8.0) -> bool:
    pid = _read_pid(path)
    if pid is None or not _is_alive(pid):
        path.unlink(missing_ok=True)
        return False
    _signal_proc(pid, signal.SIGTERM)
    deadline = time.time() + grace
    while time.time() < deadline:
        if not _is_alive(pid):
            path.unlink(missing_ok=True)
            return True
        time.sleep(0.2)
    _hard_kill(pid)
    path.unlink(missing_ok=True)
    return True


def stop_model(spec: ModelSpec, force: bool = False) -> None:
    """Stop the model's stack. If the stack is up but unmanaged (no pid
    file), only stop it when `force` is set (kills whatever listens on
    the ports)."""
    state = spec.state_dir()
    _stop_pidfile(state / "llama.pid")
    _stop_pidfile(state / "daemon.pid")
    (state / "effective_ctx.txt").unlink(missing_ok=True)
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


def remove_model(
    name: str,
    *,
    force: bool = False,
    purge: bool = False,
    registry: Path | None = None,
) -> dict[str, Any]:
    """Drop a model from the registry.

    Refuses to remove the active model unless ``force=True``; ``purge=True``
    additionally deletes the model weights (``~/models/<name>/``) and the
    runtime state directory (``~/.local/state/tokirecur/model/<name>/``)
    so re-adding the model requires re-downloading everything.

    Returns a summary ``{"removed": name, "purged_weights": bool,
    "purged_state": bool, "was_active": bool}``.
    """
    specs = load_registry(registry)
    try:
        spec = get_model(name, registry)
    except KeyError:
        raise KeyError(f"no model named '{name}' in the registry") from None

    was_active = read_active(registry) == name
    if was_active and not force:
        raise RuntimeError(
            f"'{name}' is the active model — `toki model use <other>` first, "
            "or pass --force"
        )

    # Stop the stack first so we don't leave a daemon pointing at a model
    # that is no longer registered. Force it if the user already opted in.
    try:
        stop_model(spec, force=force)
    except RuntimeError:
        if not force:
            raise
        # With --force the message above already stopped whatever is on
        # the ports; the unreachable llama-server is the user's problem.

    kept = [s for s in specs if s.name != name]
    save_registry(kept, registry)
    invalidate_registry_cache()

    purged_weights = False
    purged_state = False
    if purge:
        weights_dir = Path(spec.gguf).parent.resolve()
        # Don't follow a symlink that points outside the user's models dir.
        try:
            weights_dir.relative_to(Path.home() / "models")
            shutil.rmtree(weights_dir, ignore_errors=True)
            purged_weights = True
        except ValueError:
            # gguf path lives outside ~/models (legacy custom paths) —
            # leave it alone, the user opted in to that layout.
            purged_weights = False

        state_dir = spec.state_dir()
        shutil.rmtree(state_dir, ignore_errors=True)
        purged_state = True

        # The slot-save directory is shared across all models when the
        # registry default `--slot-save-path` is used; only clean it up
        # if it's now empty, otherwise we'd nuke another stack's state.
        slot_save = Path.home() / ".local" / "state" / "tokirecur" / "slot-save"
        try:
            slot_save.rmdir()  # rmdir only succeeds on empty dirs
        except OSError:
            pass

    if was_active and ACTIVE_MARKER.exists():
        try:
            ACTIVE_MARKER.unlink()
        except OSError:
            pass

    return {
        "removed": name,
        "purged_weights": purged_weights,
        "purged_state": purged_state,
        "was_active": was_active,
    }


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
            # Built-in defaults are reference shapes, but a default whose
            # binary is an absolute path missing on this machine (registry
            # shipped from another box) must be repaired — `_resolve_binary`
            # handles it at launch, and we persist the fix so pulled models
            # stop inheriting the dead path via `_reference_spec`.
            dead = s.binary and Path(s.binary).is_absolute() and not Path(s.binary).exists()
            if dead:
                new_raw = dict(s.raw)
                new_raw["binary"] = binary
                new_raw["env"] = dict(env)
                new_s = ModelSpec(raw=new_raw, **new_raw)
                if new_s.binary != s.binary or dict(new_s.env) != dict(s.env):
                    updated.append(s.name)
                new_specs.append(new_s)
            else:
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


def _repo_of(spec: ModelSpec) -> str:
    """Where a model came from, if that is still knowable.

    A pull records it. A model pulled before it was recorded does not
    have it, but the download cache remembers: the repository is named
    after the folder the model was pulled into, so the cache can be
    matched to it.
    """
    recorded = getattr(spec, "repo", "") or ""
    if recorded:
        return str(recorded)
    gguf = Path(spec.gguf) if spec.gguf else None
    if gguf is None:
        return ""
    folder = gguf.parent.name.lower()
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache.is_dir():
        return ""
    for entry in cache.glob("models--*"):
        repo = entry.name[len("models--"):].replace("--", "/")
        if repo.split("/")[-1].lower() == folder:
            return repo
    return ""


def active_spec() -> ModelSpec | None:
    """The active model from the registry, or None.

    The one answer to "which model is this machine running". The
    launcher, `toki up` and the interactive shell all asked for it
    separately and got it differently: two of them had a hard-coded
    path to a starter download, and the third looked at a fixed port
    that nothing was on.
    """
    try:
        active = read_active()
        if not active:
            return None
        return get_model(active)
    except Exception:
        return None


def _gguf_architecture(path: Path) -> str | None:
    """The architecture a GGUF declares, e.g. ``qwen35moe``.

    Memoised on (path, size, mtime): the architecture of a file on disk
    does not change, and the same file used to be asked about once per
    model in the registry. Finding a model's family projector asks about
    every other model, so a registry of thirty-odd models was reading the
    same header a thousand times over - and `toki model use` (and so the
    launcher, and so every harness it starts) waited for all of it.
    """
    try:
        stat = os.stat(path)
    except OSError:
        return None
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key in _ARCH_CACHE:
        return _ARCH_CACHE[key]
    arch: str | None = None
    try:
        from ..ggufmeta import read_gguf_kind

        arch = read_gguf_kind(path).architecture
    except Exception:
        arch = None
    _ARCH_CACHE[key] = arch
    return arch


#: Architecture lookups, keyed by (path, size, mtime). A file on disk does
#: not change its architecture, and the registry asks about the same file
#: once per model in it.
_ARCH_CACHE: dict[tuple[str, int, int], str | None] = {}


def _family_projector(spec: ModelSpec, specs: list[ModelSpec]) -> str:
    """A projector belonging to the same family, if the machine has one.

    A fine-tune is published without its vision tower. The tower is not in
    the weights at all - llama.cpp loads it from a separate file - so a
    fine-tune of a vision model sees with its family's projector, exactly
    as the base model does. The two usually live in different directories,
    because they were downloaded from different repositories, and nothing
    connects them: the fine-tune's own folder has no projector, so the
    model answers every image as though it had seen nothing, and the only
    sign is that it is quietly blind.

    So when a model has no projector of its own, look for one that belongs
    to a model of the same architecture. The architecture is what decides
    the tower: two models of the same arch share a vision encoder, and two
    models of different arches never do.
    """
    gguf = Path(spec.gguf) if spec.gguf else None
    if gguf is None or not gguf.exists():
        return ""
    arch = _gguf_architecture(gguf)
    if not arch:
        return ""
    for other in specs:
        if other.name == spec.name:
            continue
        if not other.mmproj or not Path(other.mmproj).exists():
            continue
        other_gguf = Path(other.gguf) if other.gguf else None
        if other_gguf is None or not other_gguf.exists():
            continue
        if _gguf_architecture(other_gguf) != arch:
            continue
        return str(other.mmproj)
    return ""


def _fetch_projector(spec: ModelSpec, repo_id: str) -> str:
    """Download the projector a model's repository ships, and wire it up.

    A model pulled before the puller knew about projectors has no tower on
    disk and never had one, so looking for it beside the model finds
    nothing. The repository still has it. A model that is blind answers
    every picture as though it had been shown none, and that is the kind
    of failure nobody notices until something depends on it.

    Returns the path to the projector, or an empty string when the
    repository has none, cannot be reached, or the download fails.
    """
    if not repo_id or not spec.gguf:
        return ""
    gguf = Path(spec.gguf)
    try:
        from huggingface_hub import HfApi, hf_hub_download

        files = HfApi().list_repo_files(repo_id)
        mmproj = _pick_mmproj(files)
        if not mmproj:
            return ""
        local_dir = gguf.parent
        local_dir.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(
            repo_id,
            filename=mmproj,
            local_dir=str(local_dir),
        )
        return str(path)
    except Exception as error:
        logger.warning(
            "could not fetch the projector for %s from %s: %s",
            spec.name, repo_id, error,
        )
        return ""


def backfill_mmproj(specs: list[ModelSpec] | None = None,
                    registry: Path | None = None) -> list[str]:
    """Point entries at a multimodal projector they already have on disk.

    Vision models pulled before ``pull_model`` learned about projectors
    registered with no ``mmproj`` field, so llama-server started and answered
    every image request as though it had seen nothing. When the projector is
    sitting next to the GGUF, adopt it; this needs no network and makes the
    repair idempotent.

    Only entries that actually have a projector are touched - beside the
    model, or belonging to another model of the same architecture - so
    text-only models are left exactly as they are.
    """
    specs = specs if specs is not None else load_registry(registry)
    updated: list[str] = []
    new_specs: list[ModelSpec] = []
    for s in specs:
        if s.mmproj and Path(s.mmproj).exists():
            new_specs.append(s)
            continue
        gguf = Path(s.gguf) if s.gguf else None
        found = ""
        spec_repo = _repo_of(s)
        if gguf is not None and gguf.parent.is_dir():
            nearby = sorted(
                p for p in gguf.parent.glob("*.gguf")
                if _is_mmproj(p.name) and p != gguf
            )
            if nearby:
                # Same preference order as a fresh pull.
                best = _pick_mmproj([p.name for p in nearby])
                found = str(gguf.parent / best) if best else ""
        if not found:
            # No tower of its own: its family's, if the machine has one.
            found = _family_projector(s, specs)
        if not found:
            # Not on the machine at all. If we know where the model came
            # from, the repository still has the tower, and a blind model
            # is one that answers every picture as though it had seen
            # none. So fetch it.
            found = _fetch_projector(s, spec_repo)
        if not found:
            new_specs.append(s)
            continue
        new_raw = dict(s.raw)
        new_raw["mmproj"] = found
        new_specs.append(ModelSpec(raw=new_raw, **new_raw))
        updated.append(s.name)
    if updated:
        save_registry(new_specs, registry)
    return updated


def use_model(
    name: str, *, stop_others: bool = True, registry: Path | None = None,
    sync_only: bool = False, force_restart: bool = False,
) -> ModelSpec:
    """Ensure the chosen model is up, selected, and visible to every CLI.

    The always-on small model (9b) is kept running; other big-model
    stacks are stopped (unless ``stop_others=False``) to free RAM.
    On success the model-capable CLI configs are rewritten so they point
    at the active registry model —
    callers (``toki model use``, ``toki model pull --use``, the launcher)
    don't have to do this themselves.

    ``force_restart=True`` (used by settings edits) stops the target's
    own running stack first when it's already up, so launch-affecting
    changes (``-c``, ``--parallel``, ``-kvu``, model file) actually
    take effect instead of being silently skipped because the server
    was reachable.

    With ``sync_only=True`` nothing is started or stopped: the registry
    is healed (legacy model ids normalized, missing binary/env backfilled),
    the active marker is set, and all model-capable CLI configs are rewritten.
    This is the plug-and-play repair path used by installers and daemon
    startup so agent configs are always correct without user commands.
    """
    target = get_model(name, registry)
    all_models = load_registry(registry)

    # Legacy repo-path model ids (pre-0.2.15) surface as "model 'X' not
    # found" from validating llama-server builds. Normalize at EVERY
    # sync point so configs can never carry a repo path again.
    changed = normalize_model_ids(registry)
    if changed:
        all_models = load_registry(registry)
        target = get_model(name, registry)
        logger.info("normalized legacy model ids: %s", ", ".join(f"{a}->{b}" for _, a, b in changed))

    # Self-heal any registry entries written before `pull_defaults`
    # existed (no machine-tuned binary/env). Best-effort: only runs when
    # there's a reference spec to inherit from, otherwise no-ops.
    healed = backfill_binary_env(all_models)
    if healed:
        all_models = load_registry(registry)
        target = get_model(name, registry)
        logger.info("repaired registry entries: %s", ", ".join(healed))

    # Adopt a projector that was downloaded next to the model but never
    # registered, so a vision model pulled before this fix stops answering
    # blind the next time it is selected.
    mended = backfill_mmproj(all_models, registry)
    if mended:
        all_models = load_registry(registry)
        target = get_model(name, registry)
        logger.info("attached multimodal projectors: %s", ", ".join(mended))

    # Image rows registered before sd-server support (a pull from the old
    # code path) heal here: llama-server binary and a bogus ctx become an
    # sd-server stack with no context.
    healed = backfill_image_stacks(all_models, registry)
    if healed:
        all_models = load_registry(registry)
        target = get_model(name, registry)
        logger.info("healed image stacks: %s", ", ".join(healed))

    if sync_only:
        # Config-only pass: the stack is not touched (installers and
        # daemon startup use this to heal drift without booting models).
        # Image stacks are NOT chat models: no active marker, no client
        # config sync — opencode must keep pointing at the chat daemon.
        if _spec_is_image(target):
            return target
        set_active(target.name, registry)
        try:
            sync_cli_configs(target)
        except Exception as e:
            logger.warning("could not sync CLI configs: %s", e)
        return target

    # Image stacks: start (or keep) the sd-server and stop there. No
    # active marker, no client-config sync, no always-on companion —
    # those are all chat-model semantics and would point hermes at a
    # port that cannot answer completions.
    if _spec_is_image(target):
        if stop_others:
            for other in all_models:
                if other.name != target.name and not other.always_on and not _spec_is_image(other):
                    try:
                        if llama_reachable(other) or daemon_reachable(other):
                            stop_model(other)
                            logger.info("stopped '%s' (switching to '%s')", other.name, target.name)
                    except RuntimeError as e:
                        logger.warning("could not stop '%s': %s", other.name, e)
        start_model(target)
        return target

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

    # Settings edits: the target's own stack is already running with the
    # OLD settings. start_model() is idempotent (skips when the server is
    # reachable), so without this stop the new -c/-kvu/--parallel never
    # takes effect until a manual restart.
    if force_restart:
        if llama_reachable(target) or daemon_reachable(target):
            try:
                stop_model(target)
                logger.info("restarted '%s' after settings change", target.name)
            except RuntimeError as e:
                logger.warning("could not stop '%s' for restart: %s", target.name, e)

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

    # Wire model-capable CLI clients so the active model line points at
    # `target` and every registry spec remains available where supported.
    # We never fail `use_model` because of a sync hiccup — a broken
    # client config is recoverable; an unstarted stack isn't.
    try:
        sync_cli_configs(target)
    except Exception as e:
        logger.warning("could not sync CLI configs: %s", e)

    return target


# ── CLI config writers ─────────────────────────────────────────────────


def _client_endpoint(spec: ModelSpec, mode: str = "managed") -> str:
    base = spec.llama_url if mode == "raw" else spec.daemon_url
    return f"{base}/v1"


def _client_output_limit(spec: ModelSpec) -> int:
    ctx = max(1, _effective_ctx(spec))
    configured = getattr(spec, "max_output_tokens", None)
    return max(1, min(int(configured or 8192), max(1, ctx // 2)))


def _client_reasoning(spec: ModelSpec) -> bool:
    return getattr(spec, "reasoning", "auto") != "off"


def _qwen_config_path() -> Path:
    explicit = os.environ.get("QWEN_CONFIG")
    if explicit:
        path = Path(explicit).expanduser()
        return path / "settings.json" if path.is_dir() else path
    home = os.environ.get("QWEN_HOME")
    if home:
        return Path(home).expanduser() / "settings.json"
    return Path.home() / ".qwen" / "settings.json"


def _factory_config_path() -> Path:
    explicit = os.environ.get("FACTORY_CONFIG")
    if explicit:
        return Path(explicit).expanduser()
    home = os.environ.get("FACTORY_HOME")
    return (Path(home).expanduser() if home else Path.home() / ".factory") / "settings.json"


def _cline_config_dir() -> Path:
    explicit = os.environ.get("CLINE_TOKIRECUR_CONFIG")
    if explicit:
        return Path(explicit).expanduser()
    return Path.home() / ".cline" / "tokirecur"


def cline_config_dir() -> Path:
    return _cline_config_dir()


def _cline_config_path() -> Path:
    return cline_config_dir() / "data" / "settings" / "providers.json"


def _claude_settings_path() -> Path:
    explicit = os.environ.get("TOKI_CLAUDE_SETTINGS")
    if explicit:
        return Path(explicit).expanduser()
    return Path.home() / ".config" / "tokirecur" / "claude" / "settings.json"


def _dsh_config_path() -> Path:
    home = os.environ.get("DSH_HOME")
    return (Path(home).expanduser() if home else Path.home() / ".dsh") / "settings.yaml"


def _dsh_profiles_root() -> Path:
    home = os.environ.get("DSH_HOME")
    return (Path(home).expanduser() if home else Path.home() / ".dsh") / "profiles"


def _dsh_profile_patch_path(profile: str) -> Path:
    """A dsh profile's own patch layer, which outranks every bundle layer."""
    return _dsh_profiles_root() / profile / "cordis.patch.yml"


def _existing_dsh_profiles() -> list[str]:
    """Every profile directory that actually exists.

    All of them need the binding, not just the TUI one. A profile ignores
    ~/.dsh/settings.yaml, so an unbound `web` profile boots, serves its UI,
    and then fails the first message with a MISSING_CREDENTIAL for
    deepseek-official - which looks exactly like a broken web UI.
    """
    root = _dsh_profiles_root()
    if not root.is_dir():
        return []
    return sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and (entry / "cordis.patch.yml").is_file()
    )


_DSH_PATCH_HEADER = """# Managed by toki. Do not edit by hand - `toki model use` rewrites this.
#
# ~/.dsh/settings.yaml is the config for `dsh --profile headless`. It is NOT
# the user override layer for a named profile: `dsh --profile tui
# --dump-config` and `--dump-default-config` come back identical, so a profile
# ignores it entirely and falls through to its bundle default of
# deepseek-official/deepseek-flash, which wants a DEEPSEEK_API_KEY this machine
# does not have. These two entries are what bind the profile to the local
# daemon, and they have to live here.
"""


def _write_dsh_profile_patch(
    spec: ModelSpec, profile: str, mode: str | None = None,
) -> Path | None:
    """Bind one dsh profile to the bound model.

    Returns None when the profile does not exist. A missing profile is not an
    error: it means that profile was never created, and inventing one would
    install a plugin tree nobody asked for.
    """
    path = _dsh_profile_patch_path(profile)
    if not path.parent.is_dir():
        return None
    model = {
        "id": spec.model_id,
        "name": spec.label,
        "contextWindow": _effective_ctx(spec),
        "maxTokens": _client_output_limit(spec),
        "input": ["text"],
    }
    patch = [
        {"id": "agent-default-model",
         "config": {"provider": "tokirecur", "model": spec.model_id}},
        {"id": "llm-pi-ai",
         "config": {"providers": {"tokirecur": {
             "apiKeyEnv": "TOKI_AUTH_TOKEN",
             "api": "openai-completions",
             "baseURL": _client_endpoint(spec, _resolve_mode("dsh", mode)),
             "models": [model],
         }}}},
    ]
    _write_private_config(path, _DSH_PATCH_HEADER + yaml.safe_dump(
        patch, sort_keys=False, default_flow_style=False))
    return path


def _write_all_dsh_profile_patches(
    spec: ModelSpec, mode: str | None = None,
) -> list[Path]:
    return [
        written
        for profile in _existing_dsh_profiles()
        if (written := _write_dsh_profile_patch(spec, profile, mode=mode)) is not None
    ]


def _grok_config_path() -> Path:
    home = os.environ.get("GROK_HOME")
    return (Path(home).expanduser() if home else Path.home() / ".grok") / "config.toml"


def _kilo_config_path() -> Path:
    explicit = os.environ.get("KILO_CONFIG")
    if explicit:
        return Path(explicit).expanduser()
    config_dir = os.environ.get("KILO_CONFIG_DIR")
    if config_dir:
        base = Path(config_dir).expanduser()
        for name in ("kilo.jsonc", "kilo.json"):
            candidate = base / name
            if candidate.exists():
                return candidate
        return base / "kilo.jsonc"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "kilo" / "kilo.jsonc"


def _pi_config_dir() -> Path:
    explicit = os.environ.get("PI_CODING_AGENT_DIR")
    return Path(explicit).expanduser() if explicit else Path.home() / ".pi" / "agent"


def _kimi_config_path() -> Path:
    home = os.environ.get("KIMI_CODE_HOME")
    return (Path(home).expanduser() if home else Path.home() / ".kimi-code") / "config.toml"


def _hermes_config_path() -> Path:
    explicit = os.environ.get("HERMES_CONFIG")
    if explicit:
        return Path(explicit).expanduser()
    home = os.environ.get("HERMES_HOME")
    if home:
        return Path(home).expanduser() / "config.yaml"
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local).expanduser() if local else Path.home() / "AppData" / "Local"
        return base / "hermes" / "config.yaml"
    return Path.home() / ".hermes" / "config.yaml"


def _omp_config_dir() -> Path:
    root_value = os.environ.get("PI_CONFIG_DIR") or os.environ.get("OMP_CONFIG_DIR")
    root = Path(root_value).expanduser() if root_value else Path.home() / ".omp"
    if not root.is_absolute():
        root = Path.home() / root
    profile = os.environ.get("OMP_PROFILE")
    if profile is None:
        profile = os.environ.get("PI_PROFILE")
    profile = profile.strip() if profile else None
    if profile and profile.lower() == "default":
        profile = None
    reserved = re.fullmatch(r"(con|prn|aux|nul|com[0-9]|lpt[0-9])(?:\..*)?", profile or "")
    if profile and not reserved and re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", profile):
        return root / "profiles" / profile / "agent"
    if profile:
        raise ValueError(f"invalid OMP profile: {profile!r}")
    explicit = os.environ.get("PI_CODING_AGENT_DIR")
    if explicit:
        return Path(explicit).expanduser()
    return root / "agent"


def _omp_config_path() -> Path:
    config_dir = _omp_config_dir()
    for name in ("models.yml", "models.yaml"):
        candidate = config_dir / name
        if candidate.exists():
            return candidate
    legacy = config_dir / "models.json"
    if legacy.exists():
        return legacy
    return config_dir / "models.yml"


def _strip_jsonc_trailing_commas(text: str) -> str:
    out: list[str] = []
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        out.append(ch)
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == ",":
            next_index = i + 1
            while next_index < len(text) and text[next_index].isspace():
                next_index += 1
            if next_index < len(text) and text[next_index] in "}]":
                out.pop()
    return "".join(out)


def _load_jsonc(path: Path) -> dict[str, Any]:
    text = path.read_text() if path.exists() else "{}"
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        cleaned: list[str] = []
        in_string = False
        escaped = False
        i = 0
        while i < len(text):
            ch = text[i]
            if in_string:
                cleaned.append(ch)
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                i += 1
                continue
            if ch == '"':
                in_string = True
                cleaned.append(ch)
                i += 1
                continue
            if ch == "/" and i + 1 < len(text) and text[i + 1] == "/":
                i += 2
                while i < len(text) and text[i] not in "\r\n":
                    i += 1
                continue
            if ch == "/" and i + 1 < len(text) and text[i + 1] == "*":
                i += 2
                while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                    i += 1
                i = min(len(text), i + 2)
                continue
            cleaned.append(ch)
            i += 1
        value = json.loads(_strip_jsonc_trailing_commas("".join(cleaned)))
        return value if isinstance(value, dict) else {}


def write_qwen_config(
    spec: ModelSpec, specs: list[ModelSpec] | None = None,
    path: Path | None = None, mode: str | None = None,
) -> Path:
    """Write Qwen Code's local OpenAI-compatible route and default model."""
    cfg_path = path or _qwen_config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    if not isinstance(cfg, dict):
        cfg = {}
    providers = cfg.get("modelProviders")
    if not isinstance(providers, dict):
        providers = {}
        cfg["modelProviders"] = providers
    entries = providers.get("openai")
    if not isinstance(entries, list):
        entries = []
    entries = [
        entry for entry in entries
        if not (isinstance(entry, dict)
                and str(entry.get("name", "")).startswith("Toki Recur ("))
    ]
    endpoint = _client_endpoint(spec, _resolve_mode("qwen", mode))
    output = _client_output_limit(spec)
    entries.append({
        "id": spec.model_id,
        "name": f"Toki Recur ({spec.label})",
        "envKey": "TOKI_AUTH_TOKEN",
        "baseUrl": endpoint,
        "wireApi": "chat-completions",
        "generationConfig": {
            "contextWindowSize": _effective_ctx(spec),
            "samplingParams": {"max_tokens": output},
        },
    })
    providers["openai"] = entries
    cfg["model"] = {"name": spec.model_id}
    security = cfg.get("security")
    if not isinstance(security, dict):
        security = {}
        cfg["security"] = security
    auth = security.get("auth")
    if not isinstance(auth, dict):
        auth = {}
        security["auth"] = auth
    auth["baseUrl"] = endpoint
    auth["selectedType"] = "openai"
    _write_private_config(cfg_path, json.dumps(cfg, indent=2) + "\n")
    return cfg_path


def write_droid_config(
    spec: ModelSpec, specs: list[ModelSpec] | None = None,
    path: Path | None = None, mode: str | None = None,
) -> Path:
    """Write Factory Droid's BYOK custom model entry."""
    cfg_path = path or _factory_config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    if not isinstance(cfg, dict):
        cfg = {}
    models = cfg.get("customModels")
    if not isinstance(models, list):
        models = []
    models = [
        model for model in models
        if not (isinstance(model, dict)
                and str(model.get("displayName", "")).startswith("Toki Recur ("))
    ]
    endpoint = _client_endpoint(spec, _resolve_mode("droid", mode))
    models.append({
        "model": spec.model_id,
        "displayName": f"Toki Recur ({spec.label})",
        "baseUrl": endpoint,
        "apiKey": "${TOKI_AUTH_TOKEN}",
        "provider": "generic-chat-completion-api",
        "maxOutputTokens": _client_output_limit(spec),
    })
    cfg["customModels"] = models
    cfg["model"] = f"custom:{spec.model_id}"
    _write_private_config(cfg_path, json.dumps(cfg, indent=2) + "\n")
    return cfg_path


def write_cline_config(
    spec: ModelSpec, specs: list[ModelSpec] | None = None,
    path: Path | None = None, mode: str | None = None,
) -> Path:
    """Write Cline's isolated OpenAI-compatible provider settings."""
    cfg_path = path or _cline_config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    if not isinstance(cfg, dict):
        cfg = {}
    cfg["version"] = cfg.get("version", 1)
    providers = cfg.get("providers")
    if not isinstance(providers, dict):
        providers = {}
        cfg["providers"] = providers
    provider = providers.get("openai-compatible")
    if not isinstance(provider, dict):
        provider = {}
        providers["openai-compatible"] = provider
    settings = provider.get("settings")
    if not isinstance(settings, dict):
        settings = {}
        provider["settings"] = settings
    settings.update({
        "provider": "openai-compatible",
        "apiKey": _configured_auth_token() or auth_token(),
        "model": spec.model_id,
        "baseUrl": _client_endpoint(spec, _resolve_mode("cline", mode)),
        "contextWindow": _effective_ctx(spec),
        "maxTokens": _client_output_limit(spec),
    })
    provider["updatedAt"] = datetime.now(timezone.utc).isoformat(
        timespec="milliseconds",
    ).replace("+00:00", "Z")
    provider["tokenSource"] = "manual"
    cfg["lastUsedProvider"] = "openai-compatible"
    _backup(cfg_path, mode=0o600)
    _atomic_write_text(cfg_path, json.dumps(cfg, indent=2) + "\n", mode=0o600)
    return cfg_path


def write_claude_config(
    spec: ModelSpec, specs: list[ModelSpec] | None = None,
    path: Path | None = None, mode: str | None = None,
) -> Path:
    """Write a private Claude Code override for the local Anthropic route."""
    del specs, mode
    cfg_path = path or _claude_settings_path()
    token = _configured_auth_token() or auth_token()
    model_id = spec.model_id
    cfg = {
        "env": {
            "ANTHROPIC_BASE_URL": f"{spec.daemon_url}/anthropic",
            "ANTHROPIC_API_KEY": token,
            "ANTHROPIC_MODEL": model_id,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model_id,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model_id,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model_id,
        },
    }
    _write_private_config(cfg_path, json.dumps(cfg, indent=2) + "\n")
    return cfg_path


def write_dsh_config(
    spec: ModelSpec, specs: list[ModelSpec] | None = None,
    path: Path | None = None, mode: str | None = None,
) -> Path:
    """Write DeepSeek Harness's llm-pi-ai route and default model."""
    cfg_path = path or _dsh_config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    if not isinstance(cfg, dict):
        cfg = {}
    llm = cfg.get("llm-pi-ai")
    if not isinstance(llm, dict):
        llm = {}
        cfg["llm-pi-ai"] = llm
    providers = llm.get("providers")
    if not isinstance(providers, dict):
        providers = {}
        llm["providers"] = providers
    providers.pop("tokirecur", None)
    reasoning_efforts = False if not _client_reasoning(spec) else {
        "off": None,
        "minimal": "minimal",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "xhigh",
        "max": "max",
    }
    providers["tokirecur"] = {
        "apiKeyEnv": "TOKI_AUTH_TOKEN",
        "api": "openai-completions",
        "baseURL": _client_endpoint(spec, _resolve_mode("dsh", mode)),
        "models": [{
            "id": spec.model_id,
            "name": spec.label,
            "contextWindow": _effective_ctx(spec),
            "maxTokens": _client_output_limit(spec),
            "input": ["text"],
            "reasoningEfforts": reasoning_efforts,
        }],
    }
    cfg["agent-default-model"] = {
        "provider": "tokirecur",
        "model": spec.model_id,
    }
    _write_private_config(cfg_path, yaml.safe_dump(cfg, sort_keys=False))
    # The profile layers are what tui and web actually read. Written here so
    # they can never drift from the headless settings above.
    try:
        _write_all_dsh_profile_patches(spec, mode=mode)
    except Exception as e:  # never let a profile write break the settings write
        logger.warning("could not write dsh profile patch: %s", e)
    return cfg_path


def write_grok_config(
    spec: ModelSpec, specs: list[ModelSpec] | None = None,
    path: Path | None = None, mode: str | None = None,
) -> Path:
    """Write Grok Build's custom OpenAI-compatible model route."""
    cfg_path = path or _grok_config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("rb") as fh:
            loaded = tomllib.load(fh)
        if isinstance(loaded, dict):
            cfg = loaded
    models = cfg.get("models")
    if not isinstance(models, dict):
        models = {}
        cfg["models"] = models
    model_table = cfg.get("model")
    if not isinstance(model_table, dict):
        model_table = {}
        cfg["model"] = model_table
    alias = f"tokirecur-{spec.name}"
    for key in [key for key in model_table if str(key).startswith("tokirecur-")]:
        model_table.pop(key, None)
    model_table[alias] = {
        "model": spec.model_id,
        "name": f"Toki Recur ({spec.label})",
        "base_url": _client_endpoint(spec, _resolve_mode("grok", mode)),
        "env_key": "TOKI_AUTH_TOKEN",
        "api_backend": "chat_completions",
        "context_window": _effective_ctx(spec),
        "max_completion_tokens": _client_output_limit(spec),
    }
    models["default"] = alias
    _write_private_config(cfg_path, tomli_w.dumps(cfg))
    return cfg_path


def write_kilo_config(
    spec: ModelSpec, specs: list[ModelSpec] | None = None,
    path: Path | None = None, mode: str | None = None,
) -> Path:
    """Write Kilo Code's custom OpenAI-compatible provider."""
    cfg_path = path or _kilo_config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = _load_jsonc(cfg_path)
    providers = cfg.get("provider")
    if not isinstance(providers, dict):
        providers = {}
        cfg["provider"] = providers
    provider = providers.get("openai-compatible")
    if not isinstance(provider, dict):
        provider = {}
        providers["openai-compatible"] = provider
    options = provider.get("options")
    if not isinstance(options, dict):
        options = {}
        provider["options"] = options
    options.update({
        "apiKey": "{env:TOKI_AUTH_TOKEN}",
        "baseURL": _client_endpoint(spec, _resolve_mode("kilo", mode)),
    })
    models = provider.get("models")
    if not isinstance(models, dict):
        models = {}
        provider["models"] = models
    models = {
        key: value for key, value in models.items()
        if not (isinstance(value, dict)
                and str(value.get("name", "")).startswith("Toki Recur ("))
    }
    models[spec.model_id] = {
        "name": f"Toki Recur ({spec.label})",
        "tool_call": True,
        "reasoning": _client_reasoning(spec),
        "limit": {
            "context": _effective_ctx(spec),
            "output": _client_output_limit(spec),
        },
        "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
    }
    provider["models"] = models
    cfg["model"] = f"openai-compatible/{spec.model_id}"
    _write_private_config(cfg_path, json.dumps(cfg, indent=2) + "\n")
    return cfg_path


def write_pi_config(
    spec: ModelSpec, specs: list[ModelSpec] | None = None,
    path: Path | None = None, mode: str | None = None,
) -> Path:
    """Write Pi's compatible endpoint and startup defaults."""
    agent_dir = path.parent if path is not None else _pi_config_dir()
    cfg_path = path or agent_dir / "models.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    if not isinstance(cfg, dict):
        cfg = {}
    providers = cfg.get("providers")
    if not isinstance(providers, dict):
        providers = {}
        cfg["providers"] = providers
    providers["tokirecur"] = {
        "baseUrl": _client_endpoint(spec, _resolve_mode("pi", mode)),
        "api": "openai-completions",
        "apiKey": "${TOKI_AUTH_TOKEN}",
        "models": [{
            "id": spec.model_id,
            "name": f"Toki Recur ({spec.label})",
            "contextWindow": _effective_ctx(spec),
            "maxTokens": _client_output_limit(spec),
            "input": ["text"],
            "reasoning": _client_reasoning(spec),
        }],
    }
    settings_path = agent_dir / "settings.json"
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    if not isinstance(settings, dict):
        settings = {}
    settings["defaultProvider"] = "tokirecur"
    settings["defaultModel"] = spec.model_id
    _write_private_config(cfg_path, json.dumps(cfg, indent=2) + "\n")
    _write_private_config(settings_path, json.dumps(settings, indent=2) + "\n")
    return cfg_path


def write_kimi_config(
    spec: ModelSpec, specs: list[ModelSpec] | None = None,
    path: Path | None = None, mode: str | None = None,
) -> Path:
    """Write Kimi Code's OpenAI-compatible provider and model alias."""
    cfg_path = path or _kimi_config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("rb") as fh:
            loaded = tomllib.load(fh)
        if isinstance(loaded, dict):
            cfg = loaded
    providers = cfg.get("providers")
    if not isinstance(providers, dict):
        providers = {}
        cfg["providers"] = providers
    models = cfg.get("models")
    if not isinstance(models, dict):
        models = {}
        cfg["models"] = models
    providers["tokirecur"] = {
        "type": "openai",
        "base_url": _client_endpoint(spec, _resolve_mode("kimi", mode)),
        "api_key_env": "TOKI_AUTH_TOKEN",
    }
    for key, value in list(models.items()):
        if isinstance(value, dict) and value.get("provider") == "tokirecur":
            models.pop(key, None)
    alias = f"tokirecur/{spec.model_id}"
    models[alias] = {
        "provider": "tokirecur",
        "model": spec.model_id,
        "max_context_size": _effective_ctx(spec),
        "display_name": f"Toki Recur ({spec.label})",
        "capabilities": ["thinking", "tool_use"] if _client_reasoning(spec)
        else ["tool_use"],
    }
    cfg["default_model"] = alias
    _write_private_config(cfg_path, tomli_w.dumps(cfg))
    return cfg_path


def write_omp_config(
    spec: ModelSpec, specs: list[ModelSpec] | None = None,
    path: Path | None = None, mode: str | None = None,
) -> Path:
    """Write OMP's custom OpenAI-compatible provider."""
    cfg_path = path or _omp_config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg_path.suffix.lower() == ".json":
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    else:
        cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    if not isinstance(cfg, dict):
        cfg = {}
    providers = cfg.get("providers")
    if not isinstance(providers, dict):
        providers = {}
        cfg["providers"] = providers
    providers["tokirecur"] = {
        "baseUrl": _client_endpoint(spec, _resolve_mode("omp", mode)),
        "api": "openai-completions",
        "apiKey": "TOKI_AUTH_TOKEN",
        "auth": "apiKey",
        "models": [{
            "id": spec.model_id,
            "name": f"Toki Recur ({spec.label})",
            "contextWindow": _effective_ctx(spec),
            "maxTokens": _client_output_limit(spec),
            "input": ["text"],
            "reasoning": _client_reasoning(spec),
        }],
    }
    if cfg_path.suffix.lower() == ".json":
        text = json.dumps(cfg, indent=2) + "\n"
    else:
        text = yaml.safe_dump(cfg, sort_keys=False)
    _write_private_config(cfg_path, text)
    return cfg_path


def _client_modes_path(root: Path | None = None) -> Path:
    return (root or STATE_ROOT) / "client-modes.json"


def read_client_modes(root: Path | None = None) -> dict[str, str]:
    """Per-client inference mode (``raw`` | ``managed``), persisted so
    the launcher and ``toki model use`` agree. Unknown clients default
    to ``managed`` — the mode where the daemon owns context truth."""
    try:
        path = _client_modes_path(root)
        data = json.loads(path.read_text()) if path.exists() else {}
        modes = {k: v for k, v in data.items() if v in ("raw", "managed")}
        return modes
    except (OSError, ValueError):
        return {}


def write_client_mode(client: str, mode: str,
                      root: Path | None = None) -> dict[str, str]:
    modes = read_client_modes(root)
    if mode not in ("raw", "managed"):
        raise ValueError(f"invalid mode {mode!r}: must be raw or managed")
    modes[client] = mode
    path = _client_modes_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, json.dumps(modes, indent=2) + "\n")
    return modes


def client_mode(client: str, root: Path | None = None) -> str:
    return read_client_modes(root).get(client, "managed")


def _resolve_mode(client: str, mode: str | None,
                  root: Path | None = None) -> str:
    """Explicit ``--mode`` wins; otherwise the persisted per-client mode."""
    return mode if mode in ("raw", "managed") else client_mode(client, root)


def _backup(path: Path, *, mode: int | None = None) -> None:
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak-model")
        shutil.copy2(path, backup)
        if mode is not None:
            backup.chmod(mode)


def _write_private_config(path: Path, text: str) -> None:
    _backup(path, mode=0o600)
    _atomic_write_text(path, text, mode=0o600)


# Marker used to spot a toki-owned anti-loop block inside a user's
# AGENTS.md-style bootstrap file (openclaw workspace, codex global).
# Bump _ANTI_LOOP_VERSION when the prompt text changes so re-syncs
# REPLACE a stale toki block instead of leaving old text behind.
_ANTI_LOOP_VERSION = 3
_ANTI_LOOP_MARKER = f"<!-- toki-recur: anti-loop rules v{_ANTI_LOOP_VERSION} -->"
# Matches any toki-owned block from ANY version (v1 wrote an unversioned
# marker) up to the next HTML comment or the end of the file.
_ANTI_LOOP_BLOCK_RE = re.compile(
    r"\n?<!-- toki-recur: anti-loop rules(?: v\d+)? -->.*?(?=\n<!--|\Z)",
    re.DOTALL,
)


def _inject_anti_loop_prompt(cfg: dict[str, Any], path: str) -> bool:
    """Set the anti-loop system prompt on ``cfg`` if the user hasn't.

    Returns True when the prompt was injected, False when the user
    already provides their own instructions/system prompt at ``path``
    (dotted config path, e.g. ``agent.system_prompt``). Never clobbers.
    """
    keys = path.split(".")
    node: Any = cfg
    for k in keys[:-1]:
        if not isinstance(node, dict) or k not in node:
            node.setdefault(k, {})
        node = node[k]
    leaf = keys[-1]
    existing = node.get(leaf) if isinstance(node, dict) else None
    if existing:
        return False
    if not isinstance(node, dict):
        return False
    node[leaf] = ANTI_LOOP_SYSTEM_PROMPT
    return True


def _inject_anti_loop_md_file(
    cfg_dir: Path, instructions_key: str = "instructions",
) -> str | None:
    """Write ``tokirecur-anti-loop.md`` beside the config and return the
    path to reference in ``instructions`` (opencode-style file lists).

    Returns None when the target already exists and isn't toki-owned, so
    the caller skips wiring rather than overwriting a user's file.
    """
    md = cfg_dir / "tokirecur-anti-loop.md"
    owned = md.exists() and bool(_ANTI_LOOP_BLOCK_RE.search(md.read_text()))
    if md.exists() and not owned:
        return None
    md.write_text(
        f"{_ANTI_LOOP_MARKER}\n\n{ANTI_LOOP_SYSTEM_PROMPT}\n"
    )
    return str(md)


def _inject_anti_loop_bootstrap(
    md_path: Path, marker: str = _ANTI_LOOP_MARKER,
) -> bool:
    """Append a marked anti-loop block to a user AGENTS.md-style file.

    Idempotent and self-upgrading: any toki-owned block (any version) is
    replaced with the current text, and the block is (re)appended at the
    end when absent. User content around the block is never touched.
    Returns True when the file now carries the block.
    """
    if md_path.exists():
        text = md_path.read_text()
        block = f"{marker}\n\n{ANTI_LOOP_SYSTEM_PROMPT}\n"
        body = _ANTI_LOOP_BLOCK_RE.sub("", text).strip("\n")
        if body:
            body += "\n\n"
        md_path.write_text(body + block)
        return True
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("a") as fh:
        fh.write(f"{marker}\n\n{ANTI_LOOP_SYSTEM_PROMPT}\n\n")
    return True


def _known_daemon_ports(specs: list[ModelSpec]) -> set[int]:
    """The daemon ports every registry spec can be serving on.

    Used to purge stale ``tokirecur-*`` providers whose baseUrl points
    at a dead daemon port — the 0.2.16 "openclaw default model on port
    18002 with contextWindow 90" class, where an old sync (or a
    hand-setup wizard) left every client pointing at a server that no
    longer exists.
    """
    return {s.daemon_port for s in specs}


def _purge_stale_tokirecur_providers(
    providers: dict[str, Any], specs: list[ModelSpec],
) -> int:
    """Drop ``tokirecur-*`` provider keys whose baseUrl points at a port
    no registry spec serves. Returns the number of purged entries."""
    live = _known_daemon_ports(specs)
    purged = 0
    for key in [k for k in providers if k.startswith("tokirecur-")]:
        entry = providers[key]
        urls = []
        if isinstance(entry, dict):
            if isinstance(entry.get("options"), dict):
                urls.append(entry["options"].get("baseURL"))
            for url_key in ("baseUrl", "base_url"):
                if isinstance(entry.get(url_key), str):
                    urls.append(entry[url_key])
        base = next((u for u in urls if isinstance(u, str)), "")
        port = None
        try:
            port = int(base.rsplit(":", 1)[1].split("/", 1)[0])
        except (ValueError, IndexError):
            port = None
        if port is not None and port not in live:
            providers.pop(key, None)
            purged += 1
    return purged


def _provider_entry(spec: ModelSpec, mode: str = "managed") -> dict[str, Any]:
    # Advertise the context the model ACTUALLY runs with (clamped to
    # fit RAM, corrected to the live slot size after start) — never the
    # registry max_tokens. Advertising a bigger window than the slot
    # exists makes opencode overfill the context and hit llama-server's
    # 400 "not enough tokens".
    ctx = _effective_ctx(spec)
    base = spec.llama_url if mode == "raw" else spec.daemon_url
    return {
        "name": f"Toki Recur ({spec.label})",
        "npm": "@ai-sdk/openai-compatible",
        "options": {
            "baseURL": f"{base}/v1",
            "apiKey": _configured_auth_token() or auth_token(),
        },
        "models": {
            spec.model_id: {
                "name": f"{spec.label} (Toki Recur)",
                "contextWindow": ctx,
                "limit": {"context": ctx, "output": min(8192, max(1, ctx // 2))},
            }
        },
    }


def _advertised_specs(
    specs: list[ModelSpec], active: ModelSpec, mode: str,
) -> list[ModelSpec]:
    """Registry specs a client config may point at: the active one plus
    any spec whose serving endpoint is genuinely up.

    Only the active model's daemon runs, so a config that advertises
    every registry spec is full of dead ports — the harness then routes
    title-gen / small-model / model-switch requests into connection
    refused, and opencode falls back to using the first assistant
    response as the session title. Writing only reachable endpoints
    fixes that for every model generically: whichever spec the user
    activates, its daemon comes up and the next sync advertises it.
    """
    probe = llama_reachable if mode == "raw" else daemon_reachable
    out = [s for s in specs if s.name == active.name]
    for s in specs:
        if s.name != active.name and probe(s):
            out.append(s)
    return out


def _small_model_ref(
    specs: list[ModelSpec], active: ModelSpec | None = None,
) -> str:
    """Legacy opencode `small_model` field — `tokirecur/<short_name>` form.

    Picks the cheapest spec with a live endpoint for cheap tasks like
    titles: an always-on spec first, otherwise the first advertised
    spec that isn't the active one; falls back to the active spec
    (whose daemon the caller just brought up). `specs` must already be
    reachability-filtered, so a dead daemon is never chosen.
    """
    if specs:
        primary = next((s for s in specs if s.always_on), None)
        if primary is None:
            primary = next(
                (s for s in specs if active is None or s.name != active.name),
                None,
            )
        if primary is None:
            primary = specs[0]
        return f"tokirecur/{primary.name}"
    return "tokirecur/qwen3.5-9b"


MIMO_SCHEMA = "https://mimo.xiaomi.com/mimocode/config.json"


def _mimo_config_file(base: Path) -> Path:
    for name in ("mimocode.jsonc", "mimocode.json"):
        candidate = base / name
        if candidate.exists():
            return candidate
    return base / "mimocode.json"


def _mimo_config_path() -> Path:
    explicit = os.environ.get("MIMOCODE_CONFIG")
    if explicit:
        return Path(explicit).expanduser()
    config_dir = os.environ.get("MIMOCODE_CONFIG_DIR")
    if config_dir:
        return _mimo_config_file(Path(config_dir).expanduser())
    profile = os.environ.get("MIMOCODE_HOME")
    if profile:
        return _mimo_config_file(Path(profile).expanduser() / "config")
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local).expanduser() if local else Path.home() / "AppData" / "Local"
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return _mimo_config_file(base / "mimocode")


def _mimo_model_ref(spec: ModelSpec) -> str:
    return f"tokirecur-{spec.name}/{spec.model_id}"


def _mimo_small_model_ref(
    specs: list[ModelSpec], active: ModelSpec | None = None,
) -> str:
    if not specs:
        return "tokirecur/qwen3.5-9b"
    primary = next((s for s in specs if s.always_on), None)
    if primary is None:
        primary = next(
            (s for s in specs if active is None or s.name != active.name),
            None,
        )
    return _mimo_model_ref(primary or specs[0])


def _mimo_provider_entry(spec: ModelSpec, mode: str = "managed") -> dict[str, Any]:
    ctx = _effective_ctx(spec)
    base = spec.llama_url if mode == "raw" else spec.daemon_url
    return {
        "name": f"Toki Recur ({spec.label})",
        "npm": "@ai-sdk/openai-compatible",
        "only_configured_models": True,
        "options": {
            "baseURL": f"{base}/v1",
            "apiKey": _configured_auth_token() or auth_token(),
        },
        "models": {
            spec.model_id: {
                "name": f"{spec.label} (Toki Recur)",
                "limit": {
                    "context": ctx,
                    "output": _client_output_limit(spec),
                },
            }
        },
    }


def write_mimo_config(
    spec: ModelSpec, specs: list[ModelSpec] | None = None,
    path: Path | None = None, mode: str | None = None,
) -> Path:
    """Write MiMo Code's OpenAI-compatible provider configuration."""
    cfg_path = path or _mimo_config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg: dict[str, Any] = _load_jsonc(cfg_path) if cfg_path.exists() else {}

    providers = cfg.get("provider")
    if not isinstance(providers, dict):
        providers = {}
        cfg["provider"] = providers
    for stale in [k for k in providers if k.startswith("tokirecur-")]:
        providers.pop(stale, None)

    all_specs = specs if specs is not None else load_registry()
    mode = _resolve_mode("mimo", mode)
    advertised = _advertised_specs(all_specs, spec, mode)
    for advertised_spec in advertised:
        providers[f"tokirecur-{advertised_spec.name}"] = _mimo_provider_entry(
            advertised_spec, mode=mode,
        )

    cfg["$schema"] = MIMO_SCHEMA
    cfg["model"] = _mimo_model_ref(spec)
    cfg["small_model"] = _mimo_small_model_ref(advertised, active=spec)

    existing = cfg.get("instructions")
    if not existing:
        md = _inject_anti_loop_md_file(cfg_path.parent)
        if md is not None:
            cfg["instructions"] = [md]

    _backup(cfg_path, mode=0o600)
    _atomic_write_text(cfg_path, json.dumps(cfg, indent=2) + "\n", mode=0o600)
    return cfg_path


def write_opencode_config(
    spec: ModelSpec, specs: list[ModelSpec] | None = None,
    path: Path | None = None, mode: str | None = None,
) -> Path:
    """Write a provider entry for every registry spec + set the active.

    `specs` may be passed in to avoid re-reading the registry; if it's
    None the registry is loaded once and threaded through the call.
    `mode` selects the inference path: ``managed`` (default) points at
    the toki daemon (context truth, KV recycling, /v1/models); ``raw``
    points straight at llama-server.
    """
    cfg_path = path or Path(os.environ.get(
        "OPENCODE_CONFIG", str(Path.home() / ".config/opencode/opencode.json")))
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg: dict[str, Any] = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    # A vanilla opencode config carries `"provider": "<name>"` (a string).
    # Replace it with the provider dict we manage — never crash the sync
    # over a foreign config shape.
    providers = cfg.get("provider")
    if not isinstance(providers, dict):
        providers = {}
        cfg["provider"] = providers
    # Purge stale tokirecur-* entries from a previous toki version so
    # a renamed or removed model doesn't leave a dangling key.
    for stale in [k for k in providers if k.startswith("tokirecur-")]:
        providers.pop(stale, None)
    all_specs = specs if specs is not None else load_registry()
    mode = _resolve_mode("opencode", mode)
    advertised = _advertised_specs(all_specs, spec, mode)
    for s in advertised:
        providers[f"tokirecur-{s.name}"] = _provider_entry(s, mode=mode)

    small = _small_model_ref(advertised, active=spec)
    cfg["small_model"] = small
    cfg["model"] = f"tokirecur-{spec.name}/{spec.model_id}"

    # Anti-loop guardrail: opencode's `instructions` takes FILE PATHS,
    # so we write a toki-owned markdown file and reference it. Never
    # clobber a user instruction list.
    existing = cfg.get("instructions")
    if not existing:
        md = _inject_anti_loop_md_file(cfg_path.parent)
        if md is not None:
            cfg["instructions"] = [md]

    _backup(cfg_path, mode=0o600)
    _atomic_write_text(cfg_path, json.dumps(cfg, indent=2) + "\n", mode=0o600)
    return cfg_path


def write_openclaw_config(
    spec: ModelSpec, specs: list[ModelSpec] | None = None,
    path: Path | None = None, mode: str | None = None,
) -> Path:
    """Write a provider entry for every registry spec + set the active.

    ``mode=raw`` points the providers at llama-server directly (no
    daemon); ``managed`` (default) routes through the daemon. Either
    way ``contextWindow`` is the model's effective context.
    """
    cfg_path = path or Path(os.environ.get(
        "OPENCLAW_CONFIG", str(Path.home() / ".openclaw/openclaw.json")))
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg: dict[str, Any] = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    # Guard against a foreign config shape (e.g. `models` as a list).
    models = cfg.get("models")
    if not isinstance(models, dict):
        models = {}
        cfg["models"] = models
    models.setdefault("mode", "merge")
    providers = models.setdefault("providers", {})
    if not isinstance(providers, dict):
        providers = {}
        models["providers"] = providers
    # Purge stale tokirecur-* entries from a previous toki version so
    # a renamed or removed model doesn't leave a dangling key.
    for stale in [k for k in providers if k.startswith("tokirecur-")]:
        providers.pop(stale, None)
    all_specs = specs if specs is not None else load_registry()
    mode = _resolve_mode("openclaw", mode)
    advertised = _advertised_specs(all_specs, spec, mode)
    for s in advertised:
        base = s.llama_url if mode == "raw" else s.daemon_url
        providers[f"tokirecur-{s.name}"] = {
            "api": "openai-completions",
            "baseUrl": f"{base}/v1",
            "apiKey": _configured_auth_token() or auth_token(),
            "timeoutSeconds": 3600,
            "models": [
                {
                    "id": s.model_id,
                    "name": f"{s.label} (Toki Recur)",
                    # Advertise the context the model ACTUALLY runs with —
                    # never max_tokens (a lie that made openclaw overfill
                    # the slot and hang on "request sent, no response").
                    "contextWindow": _effective_ctx(s),
                    "cost": {"cacheRead": 0, "cacheWrite": 0, "input": 0, "output": 0},
                }
            ],
        }

    agents = cfg.setdefault("agents", {})
    defaults = agents.setdefault("defaults", {})
    defaults["model"] = {"primary": f"tokirecur-{spec.name}/{spec.model_id}"}
    defaults["timeoutSeconds"] = 3600

    # Anti-loop guardrail: openclaw has no config-injectable system
    # prompt yet (feature request #50103 — the strict agent-defaults
    # schema rejects unknown keys), so the rules go into the workspace
    # AGENTS.md bootstrap file, which openclaw injects into every run.
    workspace = Path(os.environ.get(
        "OPENCLAW_WORKSPACE_DIR",
        str(Path.home() / ".openclaw/workspace")))
    if isinstance(defaults.get("workspace"), str):
        workspace = Path(defaults["workspace"])
    _inject_anti_loop_bootstrap(workspace / "AGENTS.md")

    _backup(cfg_path, mode=0o600)
    _atomic_write_text(cfg_path, json.dumps(cfg, indent=2) + "\n", mode=0o600)
    return cfg_path


def write_hermes_config(
    spec: ModelSpec, specs: list[ModelSpec] | None = None,
    path: Path | None = None, mode: str | None = None,
) -> Path:
    """Write a single ``llamacpp`` custom provider pointing at the active
    model, and purge stale ``tokirecur-*`` entries from a previous toki
    version that hermes can no longer resolve.

    hermes's ``resolve_provider`` only accepts built-in provider IDs
    (``openrouter``, ``custom``) and the aliases it ships (``llamacpp``,
    ``ollama``, ``vllm``, ...). A custom key like ``tokirecur-<name>``
    raises "Unknown provider" and the chat fails. The legacy
    ``custom_providers`` list is read natively, so we register one
    ``llamacpp`` entry that points at the active model's daemon and
    list every registry model_id under ``models`` so hermes can pick
    any of them via ``--model``.

    hermes resolves a requested provider by scanning the newer
    ``providers:`` dict FIRST, then falling back to the legacy
    ``custom_providers:`` list. A stale ``providers.llamacpp`` entry
    (hermes's own setup wizard, or a pre-0.2.16 toki, can write one
    pointing straight at llama-server) therefore shadows the daemon
    entry below, which surfaces as llama.cpp's "model 'X' not found"
    404. We purge and re-take the ``llamacpp`` key in BOTH formats so
    hermes always routes through the toki daemon.
    """
    cfg_path = path or _hermes_config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg: dict[str, Any] = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    if not isinstance(cfg, dict):
        cfg = {}

    # Drop any legacy tokirecur-* provider keys left by a previous toki
    # version, plus any non-toki "llamacpp" key (see docstring). The user
    # may keep their own (ollama, anthropic, ...) keys.
    providers = cfg.get("providers")
    if not isinstance(providers, dict):
        providers = {}
        cfg["providers"] = providers
    for stale in [k for k in providers
                  if k.startswith("tokirecur-") or k == "llamacpp"]:
        providers.pop(stale, None)

    all_specs = specs if specs is not None else load_registry()
    mode = _resolve_mode("hermes", mode)
    base = spec.llama_url if mode == "raw" else spec.daemon_url

    # Write/overwrite a single "llamacpp" custom_providers entry. Any
    # user-defined custom_providers entries (e.g. an ollama-local) are
    # preserved; only the toki-owned "llamacpp" entry is replaced.
    custom = cfg.setdefault("custom_providers", [])
    if not isinstance(custom, list):
        custom = []
        cfg["custom_providers"] = custom
    # Per-model `context_length` in dict form is hermes's authoritative
    # context source (get_custom_provider_context_length — resolution
    # step 0, before any probe). Without it hermes's probe of the daemon
    # fails (or reports < 64k, hermes's probe minimum) and it defaults
    # to 256k — the root cause of the "request sent, never answered"
    # class on small-RAM machines.
    models_with_ctx: dict[str, dict[str, Any]] = {
        s.model_id: {"context_length": _effective_ctx(s)}
        for s in all_specs
    }
    entry = {
        "name": "llamacpp",
        "base_url": f"{base}/v1",
        "api_key": _configured_auth_token() or auth_token(),
        "models": models_with_ctx,
        "default_model": spec.model_id,
        "_toki": True,  # marker so a future re-write can find it
    }
    has_entry = False
    for i, c in enumerate(custom):
        if isinstance(c, dict) and c.get("name") == "llamacpp" and c.get("_toki"):
            custom[i] = entry
            has_entry = True
            break
    if not has_entry:
        custom.append(entry)

    # Same provider in the newer ``providers:`` dict schema, which hermes
    # scans before ``custom_providers``. Without this a stale user/legacy
    # "llamacpp" key would keep pointing requests at llama-server.
    providers["llamacpp"] = {
        "api": f"{base}/v1",
        "api_key": _configured_auth_token() or auth_token(),
        "models": models_with_ctx,
        "default_model": spec.model_id,
        "name": "Toki Recur (llamacpp)",
        "_toki": True,
    }

    # Old hermes configs carry `model: <name>` as a string; replace it
    # with the dict form rather than crashing the sync.
    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict):
        model_cfg = {}
        cfg["model"] = model_cfg
    model_cfg.update(
        {
            "api_key": _configured_auth_token() or auth_token(),
            "base_url": f"{base}/v1",
            "default": spec.model_id,
            "provider": "llamacpp",
            # Explicit context override for the active model (resolution
            # step 0) — the only channel that beats hermes's 64k probe
            # minimum on small-RAM machines.
            "context_length": _effective_ctx(spec),
        }
    )

    # Anti-loop guardrail: hermes's manual system-prompt field. Skipped
    # when the user already set their own agent.system_prompt.
    _inject_anti_loop_prompt(cfg, "agent.system_prompt")

    _purge_stale_hermes_context_cache(all_specs)

    _backup(cfg_path, mode=0o600)
    _atomic_write_text(cfg_path, yaml.safe_dump(cfg, sort_keys=False), mode=0o600)
    return cfg_path


def _purge_stale_hermes_context_cache(specs: list[ModelSpec]) -> None:
    """Drop hermes ``context_length_cache.yaml`` entries for toki daemons.

    Entries are keyed ``model@base_url``. A stale cache (a 256k default
    cached before the daemon advertised /v1/models, or an entry for a
    daemon port that no longer exists) would keep hermes believing a
    window that isn't there, even with the config override in place
    (display paths and /model switches read the cache). Only toki-owned
    base URLs are removed; user caches (ollama, ...) survive.
    """
    try:
        path = _hermes_config_path().parent / "context_length_cache.yaml"
        if not path.exists():
            return
        data = yaml.safe_load(path.read_text()) or {}
        cache = data.get("context_lengths") or {}
        if not isinstance(cache, dict):
            return
        daemon_urls = {f"{s.daemon_url}/v1" for s in specs}
        stale = {k for k in cache if k.rsplit("@", 1)[-1] in daemon_urls}
        if not stale:
            return
        for k in stale:
            cache.pop(k, None)
        logger.info("purged %d stale hermes context-length cache entries", len(stale))
        _atomic_write_text(path, yaml.safe_dump(
            {"context_lengths": cache}, sort_keys=False))
    except Exception as e:  # noqa: BLE001 — cache purge is best-effort
        logger.warning("could not purge hermes context cache: %s", e)


def write_codex_config(
    spec: ModelSpec, specs: list[ModelSpec] | None = None,
    path: Path | None = None, mode: str | None = None,
) -> Path:
    """Write a provider entry for every registry spec + set the active.

    Codex CLI reads ``~/.codex/config.toml`` (or ``$CODEX_HOME``):
    ``model_provider`` + ``model`` select the stack at the top level,
    and each registry spec gets a ``[model_providers.tokirecur-<name>]``
    entry pointing at the toki daemon's OpenAI-compatible API.
    Existing keys are preserved (scalar keys are re-emitted before the
    table sections, as TOML requires).

    ``mode`` is accepted for interface parity but codex ALWAYS routes
    through the daemon: it speaks the Responses API (``wire_api =
    "responses"``), which llama-server does not serve — a ``raw`` codex
    provider would fail every request.
    """
    cfg_path = path or Path(os.environ.get(
        "CODEX_HOME", str(Path.home() / ".codex"))) / "config.toml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("rb") as fh:
            cfg = tomllib.load(fh)

    top = {k: v for k, v in cfg.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in cfg.items() if isinstance(v, dict)}
    providers = tables.setdefault("model_providers", {})
    all_specs = specs if specs is not None else load_registry()
    # Drop tokirecur-* entries pointing at daemon ports that no longer
    # exist (stale syncs leave codex on a dead 18002-style server — the
    # "request sent, never answered" class for codex too).
    if _purge_stale_tokirecur_providers(providers, all_specs):
        logger.info("purged stale tokirecur-* providers from codex config")
    # Codex always routes through the daemon (wire_api = "responses" is
    # daemon-only), so reachability is the managed probe.
    advertised = _advertised_specs(all_specs, spec, "managed")
    for s in advertised:
        providers[f"tokirecur-{s.name}"] = {
            "name": f"Toki Recur ({s.label})",
            "base_url": f"{s.daemon_url}/v1",
            # Codex ≥ v0.146 removed `wire_api = "chat"`; the daemon's
            # /v1/responses shim maps the Responses API onto the chat
            # pipeline.
            "wire_api": "responses",
            # env_key tells codex which environment variable carries the
            # bearer key. The launcher injects the real value into the
            # child environment; it is never persisted in this config.
            "env_key": "TOKI_AUTH_TOKEN",
        }
    # A provider for a model that is not advertised points at a port
    # nothing is serving. The purge above only drops ports the registry
    # no longer knows about, so a model that is still registered but
    # whose stack is down kept its section — and codex was left offering
    # models whose every request goes nowhere.
    keep = {f"tokirecur-{s.name}" for s in advertised}
    for key in [k for k in providers if k.startswith("tokirecur-") and k not in keep]:
        providers.pop(key, None)

    top["model_provider"] = f"tokirecur-{spec.name}"
    top["model"] = spec.model_id

    # Anti-loop guardrail: codex has NO `system_prompt` config key (the
    # top-level ConfigToml schema doesn't define one — an unknown key is
    # silently ignored, so v1's write was a no-op). The documented layered
    # channel is the global instructions file ~/.codex/AGENTS.md (read as
    # user context instructions, never replacing the built-in prompt).
    # A user's own global AGENTS.md content is preserved around the block.
    # Drop the dead v1 scalar (only if it's exactly toki's old text).
    if top.get("system_prompt") and top["system_prompt"].startswith(
        "You are an autonomous agent completing a task."
    ):
        top.pop("system_prompt")
    _inject_anti_loop_bootstrap(cfg_path.parent / "AGENTS.md")

    _backup(cfg_path, mode=0o600)
    _atomic_write_text(cfg_path, tomli_w.dumps({**top, **tables}), mode=0o600)
    return cfg_path


def sync_cli_configs(spec: ModelSpec, mode: str | None = None) -> dict[str, Path]:
    """Write the active model into every model-capable CLI config.

    Reads the registry once and threads the loaded specs through each
    writer — the previous version re-parsed the YAML four times per
    switch (once per writer plus a second `load_registry` inside
    `_small_model_ref`).

    `mode` is the inference path for the sync: ``raw`` (straight to
    llama-server), ``managed`` (through the daemon), or ``None`` to
    use each client's persisted mode. Codex always stays managed (its
    Responses-API shim only exists on the daemon).

    Each writer is fault-isolated: a corrupt/foreign config for one CLI
    must never block the other three (that failure mode silently left
    hermes pointing at a stale endpoint for the whole 0.2.16 era).

    Failures are tracked per tool in ``_SYNC_FAILURES``; when the same
    tool fails ``_SYNC_FAILURE_ESCALATE`` times in a row, the warning
    escalates to ``logger.error`` with an actionable fix — a
    persistently read-only config dir should not stay invisible.
    """
    specs = load_registry()
    out: dict[str, Path] = {}
    for tool, writer in (
        ("opencode", write_opencode_config),
        ("mimo", write_mimo_config),
        ("openclaw", write_openclaw_config),
        ("hermes", write_hermes_config),
        ("codex", write_codex_config),
        ("qwen", write_qwen_config),
        ("droid", write_droid_config),
        ("cline", write_cline_config),
        ("claude", write_claude_config),
        ("dsh", write_dsh_config),
        ("grok", write_grok_config),
        ("kilo", write_kilo_config),
        ("pi", write_pi_config),
        ("kimi", write_kimi_config),
        ("omp", write_omp_config),
    ):
        try:
            tool_mode = _resolve_mode(tool, mode)
            out[tool] = writer(spec, specs, mode=tool_mode)
            _SYNC_FAILURES.pop(tool, None)
        except Exception as e:  # defensive: one bad config never blocks the rest
            count = _SYNC_FAILURES.get(tool, 0) + 1
            _SYNC_FAILURES[tool] = count
            msg = "could not write %s config (attempt %d): %s"
            if count >= _SYNC_FAILURE_ESCALATE:
                logger.error(
                    msg + "\n  fix: check the path is writable "
                    "(disk full? read-only home?), then re-run "
                    "`toki model use` — this failure has repeated %d "
                    "times and the %s config is stale",
                    tool, count, e, count, tool,
                )
            else:
                logger.warning(msg, tool, count, e)
    return out


# Consecutive-failure counters for sync_cli_configs, keyed by tool.
_SYNC_FAILURES: dict[str, int] = {}
# After this many consecutive failures for one tool, escalate from
# warning to error with an actionable hint.
_SYNC_FAILURE_ESCALATE = 3


def heal_cli_configs(registry: Path | None = None) -> dict[str, Path] | None:
    """Best-effort self-heal of model-capable CLI configs.

    Called at every stack startup (``toki serve``, ``toki up``, daemon
    spawn) so the agent configs are correct even when the user never
    runs ``toki model use``: legacy repo-path model ids are normalized
    and the model-capable CLI configs are rewritten to point at the active
    model's daemon. This is the plug-and-play repair point that kills
    the "model '<name>' not found" 404 class from stale configs.

    Never raises (a broken config is recoverable; a failed stack start
    is not). Returns the written paths, or ``None`` when no model is
    active.
    """
    try:
        changed = normalize_model_ids(registry)
        if changed:
            logger.info("normalized legacy model ids: %s",
                        ", ".join(f"{a}->{b}" for _, a, b in changed))
        healed = backfill_binary_env(load_registry(registry))
        if healed:
            logger.info("repaired registry entries: %s", ", ".join(healed))
        active = read_active(registry)
        if not active:
            return None
        return sync_cli_configs(get_model(active, registry))
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("agent config heal skipped: %s", e)
        return None
