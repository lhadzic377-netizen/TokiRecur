"""GPU / llama-build diagnostics shared by ``toki doctor`` and the
Windows CLI launcher.

Two questions get answered, with evidence, on any machine:

1. Is there a GPU here? (nvidia-smi, ROCm tools, Vulkan, Metal)
2. Is the model's llama-server actually using it? (llama.log backend
   device lines / ``offloaded N/M layers to GPU``)

When a GPU exists but the running build is CPU-only, ``hint()`` names
the exact commands to fix it — no guessing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import ctl, winproc

# Windows NVIDIA driver floors for llama.cpp CUDA builds. CUDA 12.4
# needs 551.61+. CUDA 13.x needs 580.65 at minimum (minor-version
# compatibility: a 13.4 build loads on the 13.0 floor); below that,
# install-llama prefers the 12.4 asset so the build actually loads.
CUDA_12_WINDOWS_DRIVER = (551, 61)
CUDA_13_WINDOWS_DRIVER = (580, 65)


def _run(argv: list[str], timeout: float = 5.0) -> str | None:
    try:
        out = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def driver_version() -> tuple[int, ...] | None:
    """Installed NVIDIA driver as a comparable tuple (e.g. (551, 61))."""
    nvidia = shutil.which("nvidia-smi")
    if not nvidia:
        return None
    out = _run([nvidia, "--query-gpu=driver_version", "--format=csv,noheader"])
    if not out:
        return None
    parts: list[int] = []
    for chunk in out.splitlines()[0].strip().split("."):
        try:
            parts.append(int(chunk))
        except ValueError:
            break
    return tuple(parts) if parts else None


def cuda_driver_verdict(driver: tuple[int, ...] | None) -> tuple[bool, bool]:
    """(too_old_for_any_cuda_build, prefer_cuda12_asset) for a driver tuple.

    ``too_old`` means the driver is below the CUDA 12.4 floor, so no
    CUDA llama.cpp build will load at all. ``prefer_cuda12`` means the
    driver can run the 12.4 build but not newer (12.8/13.x) builds.
    """
    if driver is None:
        return False, False
    return driver < CUDA_12_WINDOWS_DRIVER, driver < CUDA_13_WINDOWS_DRIVER


def gpu_hardware() -> dict[str, str | None]:
    """Detected GPU hardware: {"backend", "name", "vram", "driver"}."""
    if sys.platform == "darwin":
        return {"backend": "Metal", "name": "Apple GPU", "vram": None,
                "driver": None}
    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        out = _run([nvidia, "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader"])
        if out:
            parts = [p.strip() for p in out.splitlines()[0].split(",")]
            return {
                "backend": "CUDA",
                "name": parts[0] if parts else None,
                "vram": parts[1] if len(parts) > 1 else None,
                "driver": parts[2] if len(parts) > 2 else None,
            }
    if shutil.which("hipconfig") or os.path.exists("/opt/rocm") or os.path.exists(
        "C:/Program Files/AMD/ROCm"
    ):
        name = None
        out = _run(["rocminfo"]) or ""
        for line in out.splitlines():
            if "Marketing Name" in line:
                name = line.split(":", 1)[-1].strip()
                break
        return {"backend": "ROCm", "name": name, "vram": None, "driver": None}
    if shutil.which("vulkaninfo"):
        out = _run(["vulkaninfo", "--summary"]) or ""
        name = None
        for line in out.splitlines():
            if "deviceName" in line:
                name = line.split("=", 1)[-1].strip()
                break
        return {"backend": "Vulkan", "name": name, "vram": None, "driver": None}
    return {"backend": None, "name": None, "vram": None, "driver": None}


def installed_build() -> dict[str, str | None]:
    """The llama-server builds on disk: state-bin + per-model registry.

    Returns {"state_bin", "registry"} with resolved paths (or None).
    """
    from .ctl import _installed_binary, load_registry

    state_bin = _installed_binary()
    registry: list[str] = []
    try:
        for spec in load_registry():
            registry.append(ctl._resolve_binary(spec))
    except Exception:
        pass
    return {"state_bin": state_bin, "registry": sorted(set(registry)) or None}


def cuda_runtime_dlls(directory: str | os.PathLike[str]) -> list[str]:
    """cudart/cublas DLLs next to a Windows llama-server, sorted.

    ``ggml-cuda.dll`` links against these at load time. Without them the
    backend fails to load and llama.cpp silently falls back to CPU, so
    this is the difference between "CUDA build" and "CUDA works".
    """
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    return sorted(
        n for n in names
        if n.lower().startswith(("cudart64_", "cublas64_", "cublaslt64_"))
    )


def _cudart_on_path() -> bool:
    """cudart DLLs supplied by a CUDA toolkit install on PATH.

    A locally-built llama-server links against the toolkit's cudart
    rather than a copy next to the exe; Windows resolves it through
    the DLL search path.
    """
    if not winproc.IS_WINDOWS:
        return False
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        entry = entry.strip('"')
        if not entry:
            continue
        try:
            if any(Path(entry).glob("cudart64_*.dll")):
                return True
        except OSError:
            continue
    return False


def cuda_runtime_ready(binary: str | None) -> bool:
    """Can a CUDA build find cudart at load time?"""
    if not binary:
        return False
    if cuda_runtime_dlls(os.path.dirname(binary) or "."):
        return True
    return _cudart_on_path()


def binary_backend(binary: str | None) -> str | None:
    """Infer a llama.cpp build's backend from its linked libraries.

    Deterministic per binary, unlike log parsing which newer builds
    suppress at default verbosity. Windows: CUDA/ROCm/Vulkan DLLs sit
    next to llama-server.exe in the release zip. POSIX: `ldd`.

    A bare ``mlx_lm.server`` (the Apple MLX inference server) is
    reported as ``MLX`` — it's a Python module, not a shared object.
    """
    if not binary:
        return None
    if binary.endswith("mlx_lm.server") or binary == "mlx_lm.server":
        return "MLX"
    if not os.path.exists(binary):
        return None
    if winproc.IS_WINDOWS:
        try:
            names = {
                n.lower()
                for n in os.listdir(os.path.dirname(binary) or ".")
            }
        except OSError:
            return None
        # The backend module is the authoritative signal: current
        # release builds ship ggml-<backend>.dll (the Windows ROCm
        # build is ggml-hip.dll). Runtime DLLs alone are ambiguous — a
        # stale cudart64_*.dll next to a Vulkan build must not read as
        # CUDA, or -ngl 99 passes on a binary that can never use it.
        if "ggml-cuda.dll" in names:
            return "CUDA"
        if "ggml-hip.dll" in names or "ggml-rocm.dll" in names:
            return "ROCm"
        if "ggml-vulkan.dll" in names:
            return "Vulkan"
        if "ggml-metal.dll" in names:
            return "Metal"
        # Pre-dynamic-backend builds: no ggml-<backend>.dll at all.
        if any(n.startswith(("cudart", "cublas", "nvrtc")) for n in names):
            return "CUDA"
        if any(n.startswith(("amdhip", "hipblas", "rocblas")) for n in names):
            return "ROCm"
        if "vulkan-1.dll" in names:
            return "Vulkan"
        return "CPU"
    # Ask the binary what it can see before trusting its link table: a
    # GPU build loads its backend as a separate libggml-*.so at runtime,
    # so `ldd` lists only the CPU core and a working Vulkan/CUDA/ROCm
    # build reads as CPU. That wrong answer made install-llama reject a
    # correct download ("expected backend vulkan but this build is cpu")
    # while the same binary happily enumerated Vulkan0.
    out = _run([binary, "--list-devices"])
    if out:
        low = out.lower()
        if "vulkan" in low:
            return "Vulkan"
        if "cuda" in low:
            return "CUDA"
        if "rocm" in low or "hip" in low:
            return "ROCm"
    out = _run(["ldd", binary])
    if not out:
        return None
    low = out.lower()
    if any(x in low for x in ("librocm", "amdhip64", "hipblas")):
        return "ROCm"
    if any(x in low for x in ("libcudart", "libcublas")):
        return "CUDA"
    if "vulkan" in low:
        return "Vulkan"
    return "CPU"


def _active_spec():
    try:
        active = ctl.read_active()
        if active:
            return ctl.get_model(active)
    except Exception:
        return None
    return None


def describe_gpu(spec=None) -> str:
    """One-line GPU status for the launcher header / status rows."""
    spec = spec or _active_spec()
    if spec is None:
        return "no model selected"
    st = ctl.gpu_state(spec)
    if not st["backend"]:
        return "CPU only (no GPU backend in llama.log)"
    parts = [st["backend"]]
    if st["device"]:
        parts.append(st["device"])
    if st["layers"]:
        parts.append(f"offloaded {st['layers']} layers")
    return " · ".join(parts)


def run() -> str:
    """Full diagnostic report (plain text, one line per row)."""
    lines: list[str] = []
    hw = gpu_hardware()
    if hw["backend"]:
        bits = [hw["backend"], hw["name"] or "unknown device"]
        if hw["vram"]:
            bits.append(hw["vram"])
        if hw["driver"]:
            bits.append(f"driver {hw['driver']}")
        lines.append("gpu: " + " · ".join(bits))
    else:
        lines.append("gpu: none detected (CPU-only machine?)")
    if hw["backend"] == "CUDA":
        too_old, _ = cuda_driver_verdict(driver_version())
        if too_old:
            lines.append(
                "gpu note: NVIDIA driver is older than the "
                f"{CUDA_12_WINDOWS_DRIVER[0]}.{CUDA_12_WINDOWS_DRIVER[1]} floor "
                "for CUDA 12.4+ llama.cpp builds — update the NVIDIA driver, "
                "then run: toki model install-llama --backend cuda --name <model>"
            )

    builds = installed_build()
    state = builds["state_bin"]
    if state:
        kind = binary_backend(state)
        note = f"  (build: {kind})" if kind else ""
        if winproc.IS_WINDOWS and kind == "CUDA" and not cuda_runtime_ready(state):
            note += "  [missing cudart — will run on CPU]"
        lines.append(f"llama-server (state bin): {state}{note}")
    else:
        lines.append("llama-server (state bin): not installed")
    if builds["registry"]:
        for b in builds["registry"]:
            backend = binary_backend(b)
            suffix = f"  (build: {backend})" if backend else ""
            if winproc.IS_WINDOWS and backend == "CUDA" and not cuda_runtime_ready(b):
                suffix += "  [missing cudart — will run on CPU]"
            lines.append(f"llama-server (registry): {b}{suffix}")
    else:
        lines.append("llama-server (registry): no models registered")

    spec = _active_spec()
    if spec is not None:
        lines.append(f"model: {spec.name} -> {spec.gguf}")
        lines.append(f"model gpu: {describe_gpu(spec)}")
        st = ctl.model_status(spec)
        lines.append(
            f"model stack: llama={'up' if st['llama']['reachable'] else 'down'} "
            f"daemon={'up' if st['daemon']['reachable'] else 'down'}"
        )
    else:
        lines.append("model: none active — run `toki model use <name>`")

    lines.append("")
    lines.extend(hint(hw))
    return "\n".join(lines)


def hint(hw: dict[str, str | None] | None = None) -> list[str]:
    """Actionable fix lines when the GPU is unused, else []."""
    hw = hw or gpu_hardware()
    if not hw["backend"]:
        return []
    spec = _active_spec()
    if spec is None:
        return []
    st = ctl.gpu_state(spec)
    if st["backend"]:
        return []
    build = ctl._resolve_binary(spec)
    build_kind = binary_backend(build)
    lines = [
        f"note: a GPU ({hw['backend']}) is present but '{spec.name}' shows no "
        "GPU offload in llama.log.",
    ]
    reason = gpu_fallback_reason(spec)
    if winproc.IS_WINDOWS and build_kind == "CUDA" and not cuda_runtime_ready(build):
        lines.append(
            f"  its CUDA build is missing its runtime DLLs "
            f"(cudart64_*.dll) next to {build}."
        )
        lines.append(
            f"  fix: toki model install-llama --backend cuda --name {spec.name}"
        )
        lines.append(f"  then: toki model use {spec.name}")
    elif build_kind in (None, "CPU"):
        lines.append(
            f"  its llama-server ({build}) is a CPU build (or unknown)."
        )
        lines.append(
            f"  fix: toki model install-llama --backend "
            f"{'cuda' if hw['backend'] == 'CUDA' else hw['backend'].lower()} "
            f"--name {spec.name}"
        )
        lines.append(f"  then: toki model use {spec.name}")
    else:
        if reason:
            # The log knows why. A person reading "the build is ROCm but no
            # offload is visible" would go looking for a wrong build; the
            # actual answer is usually in the log, and it is a different
            # fix every time.
            lines.append(f"  {reason}.")
            lines.append(
                f"  fix: free the card (another model may be on it), then "
                f"toki model use {spec.name}"
            )
        else:
            lines.append(
                f"  its build is {build_kind} but no offload is visible — the log "
                "may just be terse, or -ngl was forced to 0."
            )
            lines.append(
                f"  check: TOKI_NGL={os.environ.get('TOKI_NGL', 'unset -> auto')}; "
                "registry extra_args must not pin -ngl 0"
            )
    return lines


def gpu_fallback_reason(spec) -> str | None:
    """Why the model came up without the GPU, when the log says.

    The log is appended to across restarts, and the reason is usually
    in it: the card was full, or the layers were pinned. Without this the
    doctor says "no offload is visible" and the person goes looking for a
    wrong build, when the build is right and the card is busy.
    """
    log = spec.state_dir() / "llama.log"
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return None
    for line in reversed(text.splitlines()):
        low = line.lower()
        if "out of memory" in low and ("alloc" in low or "buffer" in low):
            return "the GPU could not hold the model (out of memory)"
        if "n_gpu_layers already set by user" in low:
            return "the model was pinned to a GPU layer count the card could not hold"
    return None


def active_stack_ports() -> tuple[int, int]:
    """The (llama, daemon) ports the active model is served on.

    The defaults — 8080 and 8787 — are for a fresh install. Every model
    in the registry gets its own pair, so probing the defaults reported a
    stack that was up as down, on any machine whose models had been
    given their own ports.
    """
    spec = _active_spec()
    if spec is None:
        return 8080, 8787
    return spec.llama_port, spec.daemon_port


def backend_wedged(base_url: str, *, gap: float = 60.0) -> str | None:
    """Why llama-server can serve nothing, or None if it can.

    llama-server keeps a cancelled task's slot flagged busy: the client
    disconnected, the server cancelled the task and never released the
    slot. Two such cancellations wedge the whole server — ``/health``
    still answers 200 while every completion queues forever.

    The stall has to be total to call it: every slot busy, none of them
    having produced a single token or prompt block, and nothing changing
    across the whole window. A server prefilling a long prompt reports
    no progress either — its prompt counter moves in 2048-token blocks,
    which on a slow CPU is minutes apart — so a short sample would call
    a working backend wedged.
    """
    try:
        import httpx
    except Exception:
        return None

    def sample() -> list[tuple] | None:
        try:
            r = httpx.get(f"{base_url.rstrip('/')}/slots", timeout=2.0)
            r.raise_for_status()
            slots = r.json()
        except Exception:
            return None
        if not slots or not isinstance(slots, list):
            return None
        out = []
        for s in slots:
            if not isinstance(s, dict):
                continue
            nxt = s.get("next_token") or []
            decoded = 0
            if isinstance(nxt, list) and nxt and isinstance(nxt[0], dict):
                decoded = int(nxt[0].get("n_decoded") or 0)
            out.append((
                s.get("id"),
                bool(s.get("is_processing")),
                s.get("id_task"),
                int(s.get("n_prompt_tokens_processed") or 0),
                decoded,
            ))
        return out or None

    first = sample()
    if not first or not all(row[1] for row in first):
        return None
    time.sleep(gap)
    second = sample()
    if not second or first != second:
        return None
    if not all(row[1] for row in second):
        return None
    # Nothing produced, by any slot: a live server has decoded tokens or
    # processed prompt blocks by now.
    if any(row[3] or row[4] for row in second):
        return None
    tasks = ", ".join(str(row[2]) for row in second)
    return (
        f"wedged: {len(second)}/{len(second)} slots busy with no progress "
        f"for {int(gap)}s (task ids {tasks})"
    )
