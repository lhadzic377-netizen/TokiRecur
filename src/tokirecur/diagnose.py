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
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import ctl, winproc
from .claude_compat import api_key_status as claude_api_key_status

# NVIDIA driver floors for llama.cpp CUDA builds. Windows ships CUDA
# 12.4 and 13.x; Linux ships CUDA 12.8 and 13.x. CUDA 13.x needs
# 580.65 at minimum (minor-version compatibility: a 13.4 build loads on
# the 13.0 floor); below that, install-llama prefers the CUDA 12.x asset
# so the build actually loads.
CUDA_12_WINDOWS_DRIVER = (551, 61)
CUDA_12_8_LINUX_DRIVER = (570, 26)
CUDA_12_8_WINDOWS_DRIVER = (570, 65)
CUDA_13_DRIVER = (580, 65)
# Backwards-compatible alias used by older callers/tests.
CUDA_13_WINDOWS_DRIVER = CUDA_13_DRIVER


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


def cuda_driver_floor(os_name: str = "windows") -> tuple[int, int]:
    """Lowest NVIDIA driver that can load the oldest CUDA asset we ship."""
    if os_name in ("ubuntu", "linux"):
        return CUDA_12_8_LINUX_DRIVER
    return CUDA_12_WINDOWS_DRIVER


def cuda_asset_family(os_name: str = "windows") -> str:
    """Oldest CUDA release family with a llama.cpp asset on this OS."""
    return "12.8" if os_name in ("ubuntu", "linux") else "12.4"


def cuda_driver_verdict(
    driver: tuple[int, ...] | None, os_name: str = "windows",
) -> tuple[bool, bool]:
    """(too_old_for_any_cuda_build, prefer_cuda12_asset) for a driver tuple.

    ``too_old`` means the driver is below the floor of the oldest CUDA
    asset llama.cpp ships for this OS (12.4 on Windows, 12.8 on Linux),
    so no CUDA build will load at all. ``prefer_cuda12`` means the
    driver can run the 12.x build but not a CUDA 13.x build.
    """
    if driver is None:
        return False, False
    return driver < cuda_driver_floor(os_name), driver < CUDA_13_DRIVER


def _windows_gpu_hardware() -> dict[str, str | None]:
    """GPU hardware from Windows WMI, including ineligible adapters.

    The vendor tools are unreliable here: ``rocminfo`` is a Linux tool,
    ``hipconfig`` is absent unless the HIP SDK is installed, and
    ``vulkaninfo`` ships with the Vulkan SDK, not the GPU driver. WMI
    is on every Windows box. When the best adapter is an iGPU that the
    profile refuses to offload to, report its name with ``backend``
    None so the doctor can say what is present instead of claiming a
    CPU-only machine.
    """
    from .hardware import (
        _BACKEND_TABLE,
        _enumerate_gpus_windows,
        _gpu_eligible,
    )

    try:
        gpus = _enumerate_gpus_windows()
    except Exception:
        gpus = []
    empty = {"backend": None, "name": None, "vram": None, "driver": None}
    if not gpus:
        return empty
    eligible = [g for g in gpus if _gpu_eligible(g)]
    eligible.sort(key=lambda g: (g.shared_ram, -g.vram_gb))
    if not eligible:
        gpu = max(gpus, key=lambda g: g.vram_gb)
        return {
            "backend": None,
            "name": gpu.name,
            "vram": f"{gpu.vram_gb:.1f} GB" if gpu.vram_gb else None,
            "driver": None,
        }
    gpu = eligible[0]
    backend = _BACKEND_TABLE.get(gpu.vendor, {}).get("windows", "vulkan")
    label = {
        "cuda": "CUDA", "rocm": "ROCm", "vulkan": "Vulkan",
        "metal": "Metal", "sycl": "SYCL", "openvino": "OpenVINO",
    }.get(backend, backend.upper())
    return {
        "backend": label,
        "name": gpu.name,
        "vram": f"{gpu.vram_gb:.1f} GB" if gpu.vram_gb else None,
        "driver": None,
    }


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
    if winproc.IS_WINDOWS:
        # WMI knows AMD/Intel even without any vendor tool installed,
        # and it agrees with the hardware profile (AMD → Vulkan, Intel
        # Arc → SYCL). Run it before the ROCm-directory and vulkaninfo
        # heuristics: the Vulkan SDK is usually absent, and an installed
        # HIP toolkit would otherwise make the doctor recommend ROCm
        # while install-llama's profile recommends Vulkan.
        win = _windows_gpu_hardware()
        if win["backend"] or win["name"]:
            return win
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


def cuda_runtime_libs(directory: str | os.PathLike[str]) -> list[str]:
    """cudart/cublas shared libraries next to a Linux llama-server."""
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    return sorted(
        n for n in names
        if n.lower().startswith(
            ("libcudart.so", "libcublas.so", "libcublaslt.so")
        )
    )


def _dll_on_path(*prefixes: str) -> bool:
    """Is a DLL matching one of ``prefixes`` on the Windows PATH?

    A locally-built GPU server links against the vendor toolkit's
    runtime rather than a copy next to the exe; Windows resolves it
    through the DLL search path.
    """
    if not winproc.IS_WINDOWS:
        return False
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        entry = entry.strip('"')
        if not entry:
            continue
        try:
            for prefix in prefixes:
                if next(Path(entry).glob(f"{prefix}*.dll"), None):
                    return True
        except OSError:
            continue
    return False


def _glob_any(dirs: list[str] | tuple[str, ...],
              prefixes: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    for entry in dirs:
        if not entry:
            continue
        try:
            for prefix in prefixes:
                found.extend(str(p) for p in Path(entry).glob(prefix + "*"))
        except OSError:
            continue
    return sorted(set(found))


def _posix_lib_dirs(*extra: str) -> list[str]:
    dirs: list[str] = []
    for part in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        part = part.strip()
        if part:
            dirs.append(part)
    dirs.extend(extra)
    return dirs


def _ldconfig_has(*names: str) -> bool:
    out = _run(["ldconfig", "-p"], timeout=10)
    if not out:
        return False
    low = out.lower()
    return any(name.lower() in low for name in names)


def _cuda_lib_dirs() -> list[str]:
    dirs = ["/usr/local/cuda/lib64", "/usr/local/cuda/lib"]
    for base in ("/usr/local/cuda/targets",):
        try:
            dirs.extend(str(p) for p in Path(base).glob("*/lib"))
            dirs.extend(str(p) for p in Path(base).glob("*/lib64"))
        except OSError:
            pass
    return dirs


def _sycl_lib_dirs() -> list[str]:
    dirs: list[str] = []
    for base, pattern in (
        ("/opt/intel/oneapi/compiler", "*/lib"),
        ("/opt/intel/oneapi/compiler", "*/lib/intel64"),
        ("/opt/intel/oneapi/mkl", "*/lib"),
        ("/opt/intel/oneapi/mkl", "*/lib/intel64"),
    ):
        try:
            dirs.extend(str(p) for p in Path(base).glob(pattern))
        except OSError:
            pass
    dirs.extend((
        "/opt/intel/oneapi/compiler/latest/lib",
        "/opt/intel/oneapi/compiler/latest/lib/intel64",
        "/opt/intel/oneapi/mkl/latest/lib",
        "/opt/intel/oneapi/mkl/latest/lib/intel64",
    ))
    return dirs


def _openvino_lib_dirs() -> list[str]:
    dirs = ["/opt/intel/openvino/runtime/lib/intel64"]
    try:
        dirs.extend(
            str(p) for p in Path("/opt/intel").glob(
                "openvino*/runtime/lib/intel64"
            )
        )
    except OSError:
        pass
    return dirs


def _posix_runtime_ready(
    binary: str | None,
    prefixes: tuple[str, ...],
    extra_dirs: tuple[str, ...] = (),
    *,
    use_ldconfig: bool = True,
) -> bool:
    """True when every required shared-library prefix resolves.

    Looks next to the binary first (release archives carry their
    runtime there), then LD_LIBRARY_PATH + vendor dirs, then the
    dynamic linker cache. A locally-built server may link against a
    system toolkit instead of a sibling copy.
    """
    dirs: list[str] = []
    if binary:
        dirs.append(os.path.dirname(binary) or ".")
    dirs.extend(_posix_lib_dirs(*extra_dirs))
    for prefix in prefixes:
        if _glob_any(dirs, (prefix,)):
            continue
        if use_ldconfig and _ldconfig_has(prefix):
            continue
        return False
    return True


def _cudart_on_path() -> bool:
    return _dll_on_path("cudart64_")


def cuda_runtime_ready(binary: str | None) -> bool:
    """Can a CUDA build find cudart at load time?"""
    if winproc.IS_WINDOWS:
        if not binary:
            return False
        if cuda_runtime_dlls(os.path.dirname(binary) or "."):
            return True
        return _cudart_on_path()
    return _posix_runtime_ready(binary, ("libcudart.so",), tuple(_cuda_lib_dirs()))


def rocm_runtime_dlls(directory: str | os.PathLike[str]) -> list[str]:
    """HIP runtime DLLs shipped next to a Windows ROCm llama-server.

    The win-rocm release carries ``ggml-hip.dll`` plus ``amdhip64_7.dll``,
    ``amd_comgr.dll`` and ``rocm_kpack.dll``. Without them the HIP
    backend fails to load and llama.cpp silently runs on CPU.
    """
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    return sorted(
        n for n in names
        if n.lower().endswith(".dll")
        and n.lower().startswith(("amdhip64", "amd_comgr", "rocm_kpack"))
    )


def rocm_runtime_ready(binary: str | None) -> bool:
    """Can a ROCm build load its HIP runtime?"""
    if winproc.IS_WINDOWS:
        if not binary:
            return False
        runtime = rocm_runtime_dlls(os.path.dirname(binary) or ".")
        if any(n.lower().startswith("amdhip64") for n in runtime):
            return True
        return _dll_on_path("amdhip64")
    return _posix_runtime_ready(
        binary, ("libamdhip64.so",), ("/opt/rocm/lib", "/opt/rocm/lib64"),
    )


def vulkan_runtime_ready(binary: str | None) -> bool:
    """Can a Vulkan build find the Vulkan loader?

    llama.cpp's Vulkan archive does not bundle the loader; it comes
    from the GPU driver (``vulkan-1.dll`` on Windows, ``libvulkan.so.1``
    on Linux). If it is missing, the backend fails to load and the
    server silently runs on CPU.
    """
    if winproc.IS_WINDOWS:
        if not binary:
            return False
        directory = os.path.dirname(binary) or "."
        if os.path.exists(os.path.join(directory, "vulkan-1.dll")):
            return True
        root = os.environ.get("SystemRoot", r"C:\Windows")
        return any(
            os.path.exists(os.path.join(root, sub, "vulkan-1.dll"))
            for sub in ("System32", "SysWOW64")
        )
    if sys.platform == "darwin":
        return False
    return _posix_runtime_ready(
        binary, ("libvulkan.so",),
        ("/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu",
         "/usr/lib64", "/usr/lib"),
    )


def sycl_runtime_dlls(directory: str | os.PathLike[str]) -> list[str]:
    """SYCL runtime DLLs shipped next to a Windows SYCL llama-server.

    The win-sycl zip bundles ``sycl9.dll`` plus the oneMKL/UR/TBB
    runtimes; a local oneAPI build may instead find them on PATH.
    """
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    return sorted(
        n for n in names
        if n.lower().startswith("sycl") and n.lower().endswith(".dll")
    )


def sycl_runtime_ready(binary: str | None) -> bool:
    """Can a SYCL build load its runtime?

    Windows release zips are self-contained. Linux release zips are
    not: they link against the Intel oneAPI DPC++ runtime, oneMKL and
    the Level Zero loader, so the user must have those installed.
    """
    if winproc.IS_WINDOWS:
        if not binary:
            return False
        if sycl_runtime_dlls(os.path.dirname(binary) or "."):
            return True
        return _dll_on_path("sycl")
    return _posix_runtime_ready(
        binary,
        ("libsycl.so", "libmkl_sycl", "libze_loader.so"),
        tuple(_sycl_lib_dirs()),
    )


def sycl_runtime_missing(binary: str | None) -> list[str]:
    """Which Linux SYCL runtime pieces are absent, for a useful error."""
    if winproc.IS_WINDOWS:
        return [] if sycl_runtime_ready(binary) else ["sycl*.dll"]
    dirs: list[str] = []
    if binary:
        dirs.append(os.path.dirname(binary) or ".")
    dirs.extend(_posix_lib_dirs(*_sycl_lib_dirs()))
    missing: list[str] = []
    for prefix, label in (
        ("libsycl.so", "libsycl.so (oneAPI DPC++ runtime)"),
        ("libmkl_sycl", "libmkl_sycl.so (oneMKL)"),
        ("libze_loader.so", "libze_loader.so (Level Zero loader)"),
    ):
        if not _glob_any(dirs, (prefix,)) and not _ldconfig_has(prefix):
            missing.append(label)
    return missing


def openvino_runtime_ready(binary: str | None) -> bool:
    """Can an OpenVINO build load the OpenVINO core runtime?"""
    if winproc.IS_WINDOWS:
        if not binary:
            return False
        directory = os.path.dirname(binary) or "."
        if os.path.exists(os.path.join(directory, "openvino.dll")):
            return True
        return _dll_on_path("openvino")
    return _posix_runtime_ready(
        binary, ("libopenvino.so",), tuple(_openvino_lib_dirs()),
    )


def openvino_gpu_plugin_ready(binary: str | None) -> bool:
    """Is the OpenVINO Intel GPU plugin present?

    Without it the OpenVINO backend enumerates the CPU plugin and the
    server silently runs on CPU, which is not the GPU path the user
    asked for.
    """
    if winproc.IS_WINDOWS:
        if not binary:
            return False
        directory = os.path.dirname(binary) or "."
        if os.path.exists(os.path.join(
            directory, "openvino_intel_gpu_plugin.dll",
        )):
            return True
        return _dll_on_path("openvino_intel_gpu_plugin")
    return _posix_runtime_ready(
        binary, ("libopenvino_intel_gpu_plugin.so",),
        tuple(_openvino_lib_dirs()),
    )


def metal_device_ready(binary: str | None) -> bool:
    """Can this macOS Metal build enumerate an MTL device?"""
    if not binary:
        return False
    out = _run([binary, "--list-devices"])
    return bool(out and re.search(r"\bMTL\d+\b", out, re.IGNORECASE))


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
        if "ggml-sycl.dll" in names:
            return "SYCL"
        if "ggml-openvino.dll" in names:
            return "OpenVINO"
        # Pre-dynamic-backend builds: no ggml-<backend>.dll at all.
        if any(n.startswith(("cudart", "cublas", "nvrtc")) for n in names):
            return "CUDA"
        if any(n.startswith(("amdhip", "hipblas", "rocblas")) for n in names):
            return "ROCm"
        if "vulkan-1.dll" in names:
            return "Vulkan"
        return "CPU"
    # On Linux the release archive ships the backend as a sibling
    # libggml-<backend>.so. Check those first: --list-devices prints
    # "(none)" when the GPU runtime or driver is missing, and that is a
    # runtime failure, not a CPU build. install-llama can then name the
    # missing runtime instead of claiming the wrong asset landed.
    if sys.platform != "darwin":
        try:
            names = {
                n.lower()
                for n in os.listdir(os.path.dirname(binary) or ".")
            }
        except OSError:
            names = set()
        if any(n.startswith("libggml-cuda") for n in names):
            return "CUDA"
        if any(n.startswith(("libggml-hip", "libggml-rocm")) for n in names):
            return "ROCm"
        if any(n.startswith("libggml-vulkan") for n in names):
            return "Vulkan"
        if any(n.startswith("libggml-sycl") for n in names):
            return "SYCL"
        if any(n.startswith("libggml-openvino") for n in names):
            return "OpenVINO"
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
        if "sycl" in low:
            return "SYCL"
        if "openvino" in low:
            return "OpenVINO"
        if "metal" in low or re.search(r"\bmtl\d+\b", low):
            return "Metal"
        # --list-devices prints "(none)" when only the CPU backend is
        # registered (or when a GPU backend failed to load). Either way
        # the server will run on CPU.
        if "(none)" in low:
            return "CPU"
    out = _run(["ldd", binary])
    if not out and sys.platform == "darwin":
        # macOS has no ldd. The release tarball ships the Metal backend
        # as libggml-metal*.dylib next to llama-server; a CPU-only build
        # has libggml-cpu*.dylib and no Metal module. otool is the last
        # resort because the Xcode command line tools are not always
        # installed on a user's Mac.
        try:
            names = {
                n.lower()
                for n in os.listdir(os.path.dirname(binary) or ".")
            }
        except OSError:
            names = set()
        if any(
            n.startswith(("libggml-metal", "ggml-metal"))
            and n.endswith(".dylib")
            for n in names
        ):
            return "Metal"
        out = _run(["otool", "-L", binary])
        if out:
            low = out.lower()
            if "metal" in low:
                return "Metal"
            return "CPU"
        if any(
            n.startswith(("libggml-cpu", "ggml-cpu"))
            and n.endswith(".dylib")
            for n in names
        ):
            return "CPU"
        return None
    if not out:
        return None
    low = out.lower()
    if any(x in low for x in (
        "libggml-hip", "libggml-rocm", "librocm", "amdhip64", "hipblas",
    )):
        return "ROCm"
    if any(x in low for x in (
        "libggml-cuda", "libcudart", "libcublas",
    )):
        return "CUDA"
    if any(x in low for x in ("libggml-vulkan", "libvulkan")):
        return "Vulkan"
    if any(x in low for x in ("libggml-sycl", "libsycl")):
        return "SYCL"
    if any(x in low for x in ("libggml-openvino", "libopenvino")):
        return "OpenVINO"
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


def _runtime_note(kind: str | None, build: str | None) -> str:
    """Doctor suffix when a GPU build cannot load its runtime."""
    if not kind:
        return ""
    win = winproc.IS_WINDOWS
    if kind == "CUDA" and not cuda_runtime_ready(build):
        lib = "cudart64_*.dll" if win else "libcudart.so"
        return f"  [missing {lib} — will run on CPU]"
    if kind == "ROCm" and not rocm_runtime_ready(build):
        lib = "amdhip64*.dll" if win else "libamdhip64.so"
        return f"  [missing {lib} — will run on CPU]"
    if kind == "Vulkan" and not vulkan_runtime_ready(build):
        lib = "vulkan-1.dll" if win else "libvulkan.so.1"
        return f"  [missing {lib} — will run on CPU]"
    if kind == "SYCL" and not sycl_runtime_ready(build):
        return "  [missing oneAPI SYCL runtime — will run on CPU]"
    if kind == "OpenVINO" and not openvino_runtime_ready(build):
        lib = "openvino.dll" if win else "libopenvino.so"
        return f"  [missing {lib} — will run on CPU]"
    if kind == "OpenVINO" and not openvino_gpu_plugin_ready(build):
        return "  [missing OpenVINO GPU plugin — will run on CPU]"
    return ""


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
    elif hw.get("name"):
        lines.append(
            f"gpu: {hw['name']} (present, not used for LLM offload)"
        )
    else:
        lines.append("gpu: none detected (CPU-only machine?)")
    if hw["backend"] == "CUDA":
        os_name = "ubuntu" if sys.platform.startswith("linux") else "windows"
        too_old, _ = cuda_driver_verdict(driver_version(), os_name)
        if too_old:
            floor = cuda_driver_floor(os_name)
            family = cuda_asset_family(os_name)
            lines.append(
                "gpu note: NVIDIA driver is older than the "
                f"{floor[0]}.{floor[1]} floor for CUDA {family}+ "
                "llama.cpp builds — update the NVIDIA driver, "
                "then run: toki model install-llama --backend cuda --name <model>"
            )

    builds = installed_build()
    state = builds["state_bin"]
    if state:
        kind = binary_backend(state)
        note = f"  (build: {kind})" if kind else ""
        note += _runtime_note(kind, state)
        lines.append(f"llama-server (state bin): {state}{note}")
    else:
        lines.append("llama-server (state bin): not installed")
    if builds["registry"]:
        for b in builds["registry"]:
            backend = binary_backend(b)
            suffix = f"  (build: {backend})" if backend else ""
            suffix += _runtime_note(backend, b)
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

    # Claude Code sends ANTHROPIC_API_KEY only when it has approved that
    # key, and falls back to another credential when it has not. A key it
    # declined is a stale state worth naming: the fallback credential is
    # then whatever else the machine has.
    claude_status = claude_api_key_status(ctl.configured_auth_token() or "")
    if claude_status == "rejected":
        lines.append(
            "claude code: the local API key was rejected by Claude Code, "
            "so it is never sent — fix: toki model use <name>"
        )

    lines.append("")
    lines.extend(hint(hw))
    return "\n".join(lines)


def _cpu_mode_requested(spec) -> bool:
    """Is CPU-only the requested mode, not an accident?

    ``TOKI_NGL=0`` and a registry ``-ngl 0`` pin are explicit choices;
    warning "a GPU is present but unused" there sends people to install
    a GPU build they deliberately turned off.
    """
    if os.environ.get("TOKI_NGL") == "0":
        return True
    extra_args = getattr(spec, "extra_args", ()) or ()
    for i, arg in enumerate(extra_args):
        if arg in ("-ngl", "--n-gpu-layers") and i + 1 < len(extra_args):
            try:
                return int(extra_args[i + 1]) <= 0
            except ValueError:
                return False
    return False


def hint(hw: dict[str, str | None] | None = None) -> list[str]:
    """Actionable fix lines when the GPU is unused, else []."""
    hw = hw or gpu_hardware()
    if not hw["backend"]:
        return []
    spec = _active_spec()
    if spec is None:
        return []
    if _cpu_mode_requested(spec):
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
    win = winproc.IS_WINDOWS
    if build_kind == "CUDA" and not cuda_runtime_ready(build):
        runtime = "cudart64_*.dll" if win else "libcudart.so"
        lines.append(
            f"  its CUDA build is missing its runtime "
            f"({runtime}) next to {build}."
        )
        lines.append(
            f"  fix: toki model install-llama --backend cuda --name {spec.name}"
        )
        lines.append(f"  then: toki model use {spec.name}")
    elif build_kind == "ROCm" and not rocm_runtime_ready(build):
        runtime = ("amdhip64*.dll" if win
                   else "libamdhip64.so (install ROCm or use Vulkan)")
        lines.append(
            f"  its ROCm build is missing its HIP runtime "
            f"({runtime}) next to {build}."
        )
        lines.append(
            f"  fix: toki model install-llama --backend rocm --name {spec.name}"
        )
        lines.append(f"  then: toki model use {spec.name}")
    elif build_kind == "Vulkan" and not vulkan_runtime_ready(build):
        loader = "vulkan-1.dll" if win else "libvulkan.so.1"
        lines.append(
            f"  its Vulkan build cannot load: {loader} is missing "
            "(the GPU driver provides the loader)."
        )
        lines.append(
            "  fix: install or repair the AMD/Intel GPU driver, then "
            f"toki model use {spec.name}"
        )
    elif build_kind == "SYCL" and not sycl_runtime_ready(build):
        if win:
            lines.append(
                f"  its SYCL build is missing its runtime DLLs "
                f"(sycl*.dll) next to {build}."
            )
            lines.append(
                f"  fix: toki model install-llama --backend sycl --name {spec.name}"
            )
        else:
            missing = sycl_runtime_missing(build)
            lines.append(
                "  its SYCL build cannot load the Intel oneAPI runtime"
                + (f"; missing: {', '.join(missing)}" if missing else "")
                + "."
            )
            lines.append(
                "  fix: install the Intel oneAPI DPC++ runtime + oneMKL "
                "+ Level Zero, or use --backend vulkan"
            )
        lines.append(f"  then: toki model use {spec.name}")
    elif (build_kind == "OpenVINO"
          and not openvino_runtime_ready(build)):
        runtime = "openvino.dll" if win else "libopenvino.so"
        lines.append(
            f"  its OpenVINO build is missing {runtime} next to {build}."
        )
        lines.append(
            f"  fix: reinstall it, or use the SYCL build: "
            f"toki model install-llama --backend sycl --name {spec.name}"
        )
    elif (build_kind == "OpenVINO"
          and not openvino_gpu_plugin_ready(build)):
        lines.append(
            "  its OpenVINO build is missing the Intel GPU plugin — it "
            "would enumerate the CPU plugin and run on CPU."
        )
        lines.append(
            f"  fix: reinstall the OpenVINO build, or use the SYCL build: "
            f"toki model install-llama --backend sycl --name {spec.name}"
        )
    elif (sys.platform == "darwin" and build_kind == "Metal"
          and not metal_device_ready(build)):
        lines.append(
            "  its Metal build cannot enumerate an MTL device — macOS is "
            "too old or Metal is unavailable."
        )
        lines.append("  fix: update macOS, or run CPU-only with TOKI_NGL=0.")
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
