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

from . import ctl

# Windows NVIDIA driver floors for llama.cpp CUDA builds: 12.4 needs
# 551.61+, and 12.8/13.x-era builds need 570+ (13.x officially wants
# ~590-610, but NVIDIA stopped publishing Windows minimums for 13.0+,
# so 570 is the conservative cut for preferring the 12.4 asset).
CUDA_12_WINDOWS_DRIVER = (551, 61)
CUDA_13_WINDOWS_DRIVER = (570, 0)


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
    if os.name == "nt":
        try:
            dlls = " ".join(os.listdir(os.path.dirname(binary) or ".")).lower()
        except OSError:
            return None
        if any(x in dlls for x in ("cudart", "cublas", "nvrtc")):
            return "CUDA"
        if "amdhip" in dlls or "hipblas" in dlls:
            return "ROCm"
        if "vulkan-1" in dlls:
            return "Vulkan"
        return "CPU"
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
    lines.append(f"llama-server (state bin): {builds['state_bin'] or 'not installed'}")
    if builds["registry"]:
        for b in builds["registry"]:
            backend = binary_backend(b)
            suffix = f"  (build: {backend})" if backend else ""
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
    if build_kind in (None, "CPU"):
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
