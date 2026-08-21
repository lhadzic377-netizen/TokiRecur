"""Hardware profiling + adaptive defaults.

TokiRecur ships without curated model lists; users run whatever GGUF
they pull. That means install defaults have to adapt to wildly
different machines — Intel Core Ultra 155U (2 P-cores + 8 E-cores,
Meteor Lake), MacBook Air M2 (8 perf + 2 efficiency), Threadripper
(64 cores, no GPU), Steam Deck (Zen 2 + RDNA 2 iGPU), a server with
4× H100. One rule doesn't fit all.

This module profiles the actual box once and persists a structured
``HardwareProfile`` to ``~/.config/tokirecur/hardware.json``. The
installer, the daemon launcher, and ``toki doctor --profile`` all
read that file instead of running vendor-name heuristics.

The profile is measured-not-guessed where possible (SIMD capabilities
via ``/proc/cpuinfo``, P-cores vs E-cores via per-CPU max frequency,
GPU vendors via ``/sys/class/drm`` + Vulkan ICD JSONs, llama-server
build flags by inspecting the unpacked release asset). It is *not* a
real-world matmul benchmark at install time — that requires a model,
and install-llama runs before any model exists. A synthetic benchmark
is shipped as an opt-in via ``toki doctor --profile --bench`` for
users who want to validate the recommendation against their actual
workload.

The profile is cached for 30 days. After that, the next install or
doctor run re-profiles.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

PROFILE_VERSION = 3
# Cache the profile for 30 days; after that, re-profile.
# How long a cached hardware profile is trusted before re-profiling.
# TOKI_PROFILE_TTL (seconds) overrides. 30 days is long because the
# profile is also re-saved on every `toki doctor`; the TTL is only
# about stale data surviving a reboot after a GPU swap.
PROFILE_TTL_SECONDS = int(
    os.environ.get("TOKI_PROFILE_TTL", "2592000")   # 30 * 24 * 3600
)
# Fallback when /proc/cpuinfo isn't readable: assume 4 logical cores.
_FALLBACK_LOGICAL = 4

# Known performance-core CPU IDs on Intel hybrid CPUs. The KVM-style
# /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq read is the
# canonical way to detect P vs E on Linux; the constants below cover
# platforms where that sysfs path is missing (containers, WSL1).
_INTEL_P_CORE_FAMILIES = {6}  # Core/Core-Xeon
_INTEL_E_CORE_FAMILIES = {6}  # same family; disambiguated by max freq

# Intel iGPU device-ID prefixes. Anything NOT in this list is treated
# as a discrete GPU (Arc Alchemist, Battlemage, etc.) — discrete Arc
# has dedicated VRAM and is usable for LLM, unlike iGPUs.
#
# Coverage:
#   0x9a..   Intel Xe / Iris Xe / Arc iGPU on Tiger Lake / Alder Lake /
#             Raptor Lake / Meteor Lake / Arrow Lake (family 6).
#   0x46..   Older Skylake GT2 / GT4 / Iris Pro (family 6).
#   0x4c..   Atom-class integrated graphics (Cherry Trail, Goldmont).
#   0x3e..   Kaby Lake / Coffee Lake UHD 620/630.
_INTEL_IGPU_PREFIXES = ("0x9a", "0x46", "0x4c", "0x3e")


# ── CPU ────────────────────────────────────────────────────────────


@dataclass
class CPUProfile:
    vendor: str          # "GenuineIntel", "AuthenticAMD", "Apple"
    name: str            # "Intel(R) Core(TM) Ultra 7 155U"
    family: int
    model: int
    physical_cores: int
    logical_cores: int
    p_core_ids: list[int]   # CPU IDs to pin matmul threads to
    e_core_ids: list[int]
    max_boost_mhz: float
    simd: list[str]       # ["sse4_2", "avx", "avx2", "avx_vnni", ...]
    # Per-core max frequency in MHz, indexed by CPU ID. Empty when the
    # platform can't report it (macOS, Windows, containers without
    # cpufreq); ``effective_cpu_threads`` then falls back to
    # ``physical_cores`` rather than guessing tiers it can't measure.
    core_max_freqs_mhz: list[float] = field(default_factory=list)


def _read_proc_cpuinfo() -> dict[str, str]:
    """Parse /proc/cpuinfo into a flat dict keyed by field name."""
    out: dict[str, str] = {}
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return out
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        out[k.strip()] = v.strip()
    return out


def _count_logical_cpus() -> int:
    """How many logical CPUs does this machine report?

    Counts ``processor N:`` blocks in /proc/cpuinfo on Linux, falls back
    to os.cpu_count() elsewhere. Factored out so tests can stub the
    Linux path without re-implementing the regex.
    """
    if sys.platform.startswith("linux"):
        try:
            text = Path("/proc/cpuinfo").read_text()
            return len(re.findall(r"^processor\s*:\s*\d+", text, re.M))
        except OSError:
            pass
    return os.cpu_count() or _FALLBACK_LOGICAL


def _cpu_max_freq_path(cpu_id: int) -> Path | None:
    """Where the kernel exposes this CPU's max frequency, in kHz."""
    p = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/cpufreq/cpuinfo_max_freq")
    return p if p.exists() else None


def _total_ram_gb() -> float:
    """Total system RAM in GB.

    Linux: ``/proc/meminfo`` MemTotal. macOS: ``sysctl hw.memsize``
    (bytes). Falls back to 0.0 on other platforms or when unreadable.
    """
    if sys.platform.startswith("linux"):
        try:
            text = Path("/proc/meminfo").read_text()
        except OSError:
            return 0.0
        for line in text.splitlines():
            if line.startswith("MemTotal:"):
                try:
                    return int(line.split()[1]) / 1048576.0
                except (ValueError, IndexError):
                    return 0.0
    elif sys.platform == "darwin":
        try:
            r = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                return int(r.stdout.strip()) / 1024**3
        except (OSError, ValueError):
            return 0.0
    return 0.0


def _cpu_capacity_path(cpu_id: int) -> Path | None:
    """Sysfs capacity value (1024 = baseline). Not always present."""
    p = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/cpu_capacity")
    return p if p.exists() else None


def _detect_simd_linux() -> list[str]:
    """Parse the ``flags`` line from /proc/cpuinfo for the first CPU.
    Order doesn't matter; consumers pick what they need."""
    info = _read_proc_cpuinfo()
    flags = info.get("flags", "")
    out = []
    for k, nice in (
        ("sse4_2", "sse4_2"),
        ("avx", "avx"),
        ("avx2", "avx2"),
        ("avx512f", "avx512f"),
        ("avx512_vnni", "avx512_vnni"),
        ("avx_vnni", "avx_vnni"),
        ("amx_tile", "amx_tile"),
        ("amx_bf16", "amx_bf16"),
    ):
        if k in flags.split():
            out.append(nice)
    return out


def _detect_simd_macos() -> list[str]:
    """Use sysctl hw.optional for Apple Silicon + Intel Macs."""
    out = []
    pairs = (
        ("hw.optional.avx2", "avx2"),
        ("hw.optional.avx512f", "avx512f"),
        ("hw.optional.amx_tile", "amx_tile"),
        ("hw.optional.neon", "neon"),
        ("hw.optional.neon_hpfp", "neon_hpfp"),
        ("hw.optional.neon_fp16", "neon_fp16"),
    )
    for key, nice in pairs:
        try:
            r = subprocess.run(
                ["sysctl", "-n", key], capture_output=True, text=True, timeout=2,
            )
            if r.returncode == 0 and r.stdout.strip() == "1":
                out.append(nice)
        except Exception:
            pass
    return out


def _detect_simd_windows() -> list[str]:
    """No reliable user-mode SIMD query on Windows without admin. Use the
    registry's processor brand string and assume AVX2 on any x86 CPU
    made after 2015. Real detection would need WMI/cpuid; deferred."""
    info = _read_proc_cpuinfo() if sys.platform != "win32" else {}
    if not info:
        return ["avx2"]   # safe default for any modern Windows CPU
    return _detect_simd_linux()


def cpu_topology() -> CPUProfile:
    """Probe this box's CPU and return a structured profile.

    Detection is OS-specific but the output shape is identical. P-cores
    are detected via per-CPU max frequency on Linux (the kernel exposes
    it for any modern Intel hybrid CPU) and via ``hw.perflevel`` on
    macOS. When neither path works, all cores are treated as P-cores
    — better to over-thread than to under-utilise.
    """
    if sys.platform.startswith("linux"):
        return _cpu_topology_linux()
    if sys.platform == "darwin":
        return _cpu_topology_macos()
    if sys.platform.startswith("win"):
        return _cpu_topology_windows()
    return CPUProfile(
        vendor="unknown", name="unknown", family=0, model=0,
        physical_cores=max(1, os.cpu_count() or 1),
        logical_cores=max(1, os.cpu_count() or 1),
        p_core_ids=[], e_core_ids=[],
        max_boost_mhz=0.0, core_max_freqs_mhz=[],
        simd=[],
    )


def _cpu_topology_linux() -> CPUProfile:
    info = _read_proc_cpuinfo()
    vendor = info.get("vendor_id", info.get("Hardware", "unknown"))
    name = info.get("model name", "unknown")
    try:
        family = int(info.get("cpu family", "0"))
        model = int(info.get("model", "0"))
    except ValueError:
        family = model = 0

    # /proc/cpuinfo is one big block per CPU. cpu cores is the
    # number of physical cores per package; siblings is the total
    # number of logical CPUs in the same package. Use cpu cores
    # directly to get the physical-core count.
    cores_field = info.get("cpu cores", "1")
    try:
        physical_cores = max(1, int(cores_field))
    except ValueError:
        physical_cores = max(1, _count_logical_cpus() // 2)
    cpu_count = _count_logical_cpus()

    # P-core vs E-core: per-CPU max frequency. P-cores boost higher.
    p_ids: list[int] = []
    e_ids: list[int] = []
    max_freqs: list[float] = []
    if cpu_count:
        for cpu_id in range(cpu_count):
            f = _cpu_max_freq_path(cpu_id)
            if f is None:
                continue
            try:
                khz = int(f.read_text().strip())
                max_freqs.append(khz / 1000.0)
            except (OSError, ValueError):
                continue
        if max_freqs:
            # Group CPUs by max freq; the highest-freq cluster is P.
            threshold = max(max_freqs) * 0.85
            for i, f in enumerate(max_freqs):
                if f >= threshold:
                    p_ids.append(i)
                else:
                    e_ids.append(i)

    # Fallback when cpufreq isn't readable (containers, WSL1):
    # assume all physical cores are P.
    if not p_ids and not e_ids:
        p_ids = list(range(physical_cores))

    simd = _detect_simd_linux()

    # core_max_freqs_mhz mirrors max_freqs by CPU ID. Cores whose
    # cpufreq file is unreadable default to 0 so the freq-tier filter
    # ignores them — partial data must never inflate the thread count.
    core_max_freqs_mhz: list[float] = list(max_freqs)
    if len(core_max_freqs_mhz) < cpu_count:
        core_max_freqs_mhz.extend([0.0] * (cpu_count - len(core_max_freqs_mhz)))

    return CPUProfile(
        vendor=vendor, name=name, family=family, model=model,
        physical_cores=physical_cores, logical_cores=cpu_count or physical_cores,
        p_core_ids=p_ids, e_core_ids=e_ids,
        max_boost_mhz=max(max_freqs) if max_freqs else 0.0,
        core_max_freqs_mhz=core_max_freqs_mhz,
        simd=simd,
    )


def _cpu_topology_macos() -> CPUProfile:
    info = {}
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        info["name"] = out
    except Exception:
        info["name"] = "Apple Silicon"
    # hw.physicalcpu = physical cores, hw.logicalcpu = with HT (Intel)
    # or perf + efficiency (Apple). On Apple Silicon the OS doesn't
    # expose P/E separately, so we treat all perf cores as P.
    physical_cores = int(os.environ.get("TOKI_TEST_PHYSICAL_CORES", "0"))
    if not physical_cores:
        try:
            physical_cores = int(subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True, text=True, timeout=2,
            ).stdout.strip())
        except Exception:
            physical_cores = os.cpu_count() or _FALLBACK_LOGICAL
    logical_cores = os.cpu_count() or physical_cores

    # Apple Silicon: detect via sysctl. perflevel0 = perf, perflevel1 = eff.
    p_ids = list(range(physical_cores))
    e_ids: list[int] = []
    freqs_mhz: list[float] = []
    if "Apple" in info["name"]:
        perf_count: int | None = None
        eff_count: int | None = None
        try:
            r = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                capture_output=True, text=True, timeout=2,
            )
            if r.returncode == 0 and r.stdout.strip().isdigit():
                perf_count = int(r.stdout.strip())
        except Exception:
            pass
        try:
            r = subprocess.run(
                ["sysctl", "-n", "hw.perflevel1.logicalcpu"],
                capture_output=True, text=True, timeout=2,
            )
            if r.returncode == 0 and r.stdout.strip().isdigit():
                eff_count = int(r.stdout.strip())
        except Exception:
            pass
        if perf_count is not None:
            eff_count = eff_count or 0
            # Efficiency cores start after the performance cores on
            # Apple SoCs (per Apple's published topology).
            e_ids = list(range(perf_count, perf_count + eff_count))
            # No per-core frequency reads on macOS; encode the P/E
            # split as synthetic max-freq values (arbitrary units).
            # The freq-tier filter in effective_cpu_threads compares
            # ratios: 0.65 < _FREQ_TIER_RATIO (0.70) drops the
            # efficiency cores, exactly like Intel LPE cores.
            freqs_mhz = [1000.0] * perf_count + [650.0] * eff_count

    simd = _detect_simd_macos()

    return CPUProfile(
        vendor="Apple", name=info["name"], family=0, model=0,
        physical_cores=physical_cores, logical_cores=logical_cores,
        p_core_ids=p_ids, e_core_ids=e_ids,
        max_boost_mhz=0.0, core_max_freqs_mhz=freqs_mhz,
        simd=simd,
    )


def _cpu_topology_windows() -> CPUProfile:
    """Windows: read environment hints + CPU brand from registry. Real
    P/E detection needs WMI; we approximate with logical_cores as the
    physical count and trust the user to override via TOKI_TEST_*."""
    info = _read_proc_cpuinfo()
    name = info.get("model name", info.get("ProcessorNameString", "Windows CPU"))
    try:
        physical_cores = int(os.environ.get(
            "NUMBER_OF_PROCESSORS", str(os.cpu_count() or _FALLBACK_LOGICAL),
        ))
    except ValueError:
        physical_cores = os.cpu_count() or _FALLBACK_LOGICAL
    logical_cores = os.cpu_count() or physical_cores
    simd = _detect_simd_windows()
    return CPUProfile(
        vendor=info.get("vendor_id", "unknown"), name=name,
        family=0, model=0,
        physical_cores=physical_cores, logical_cores=logical_cores,
        p_core_ids=list(range(physical_cores)), e_core_ids=[],
        max_boost_mhz=0.0, core_max_freqs_mhz=[],
        simd=simd,
    )


# ── Thread-count policy ─────────────────────────────────────────────


# Fraction of the fastest core's max frequency that a core must meet
# to count as "fast tier" for llama.cpp's thread pool. On hybrid Intel
# (P+E+LPE) this drops the 2 LPE cores — they're 40–50 % of the P-core
# boost and gate every ggml matmul on the slowest thread. On a plain
# (non-hybrid) box every core is within ~10 % of the fastest, so the
# filter is a no-op and the count equals physical_cores.
_FREQ_TIER_RATIO = 0.70


def effective_cpu_threads(cpu: CPUProfile) -> int:
    """Thread count to hand to llama-server's ``-t`` on CPU backends.

    Counts cores whose ``cpuinfo_max_freq`` is at least 70 % of the
    fastest core on the box, then subtracts HT siblings so the result
    reflects physical fast cores — not per-logical-CPU fast cores.
    On a Core Ultra 155U (2P+8E+2LPE, 14 logical / 12 physical) the
    filter drops the 2 LPE cores and the 2 P-core HT siblings, so
    llama.cpp's pool sees 10 threads on distinct fast cores instead
    of 14 (which would gate every matmul on the 1.5 GHz LPE cores).

    On a non-hybrid box every core passes the threshold and HT
    siblings cancel out, so the result equals ``physical_cores``.
    When per-core frequencies aren't readable (containers without
    cpufreq) the function falls back to ``physical_cores``. On
    Apple Silicon the macOS profiler encodes the P/E split as
    synthetic max-freq values (1000/650), so the same filter drops
    the efficiency cores there too.
    """
    freqs = [f for f in cpu.core_max_freqs_mhz if f > 0]
    if not freqs:
        return max(1, cpu.physical_cores)
    peak = max(freqs)
    threshold = peak * _FREQ_TIER_RATIO
    fast_logical = sum(1 for f in freqs if f >= threshold)
    # HT siblings live on the fastest tier by Intel convention;
    # subtract the (logical - physical) surplus so the count matches
    # physical fast cores rather than logical fast cores. On a box
    # without HT (logical == physical) the surplus is 0 and the
    # count is unchanged.
    ht_surplus = max(0, cpu.logical_cores - cpu.physical_cores)
    physical_fast = max(1, fast_logical - ht_surplus)
    return physical_fast


# ── GPU ────────────────────────────────────────────────────────────


@dataclass
class GPUProfile:
    vendor: str          # "NVIDIA", "AMD", "Intel", "Apple"
    name: str
    vram_gb: float
    driver: str | None
    vulkan_score: float  # 0–1 heuristic of whether this GPU is worth using
    shared_ram: bool
    use_for_llm: bool
    reason: str


def enumerate_gpus() -> list[GPUProfile]:
    if sys.platform.startswith("linux"):
        return _enumerate_gpus_linux()
    if sys.platform == "darwin":
        return _enumerate_gpus_macos()
    if sys.platform.startswith("win"):
        return _enumerate_gpus_windows()
    return []


def _enumerate_gpus_linux() -> list[GPUProfile]:
    out: list[GPUProfile] = []

    # NVIDIA: nvidia-smi is the source of truth when present. When
    # missing (headless server, minimal driver install), the PCI scan
    # below picks up the card via /sys/class/drm and reads VRAM from
    # mem_info_vram_total.
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2 and parts[1].rstrip("MiB").strip().isdigit():
                        vram = int(parts[1].rstrip("MiB").strip()) / 1024
                        out.append(GPUProfile(
                            vendor="NVIDIA", name=parts[0],
                            vram_gb=vram, driver=parts[2] if len(parts) > 2 else None,
                            vulkan_score=1.0, shared_ram=False,
                            use_for_llm=vram >= 4,
                            reason=f"{vram:.1f} GB VRAM" if vram < 4 else "discrete GPU",
                        ))
        except Exception:
            pass

    # AMD discrete: rocm-smi. Two rocm-smi calls — one for product
    # name, one for VRAM. Without VRAM, the vram_gb >= 4 gate in
    # _recommend_defaults wrongly skips the GPU.
    if shutil.which("rocm-smi"):
        try:
            r = subprocess.run(
                ["rocm-smi", "--showproductname", "--csv"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0 and "GPU" in r.stdout:
                for line in r.stdout.splitlines()[1:]:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2:
                        vram_gb = _amd_rocm_vram_gb()
                        use = vram_gb >= 4 if vram_gb > 0 else True
                        reason = (
                            f"{vram_gb:.1f} GB VRAM"
                            if 0 < vram_gb < 4
                            else "AMD discrete ROCm device"
                        )
                        out.append(GPUProfile(
                            vendor="AMD", name=parts[1],
                            vram_gb=vram_gb, driver=None,
                            vulkan_score=0.9, shared_ram=False,
                            use_for_llm=use,
                            reason=reason,
                        ))
                        break
        except Exception:
            pass

    # Intel/AMD iGPU + anything else: read PCI vendor from /sys/class/drm.
    out.extend(_scan_pci_gpus(Path("/sys/class/drm"), out))
    return out


def _scan_pci_gpus(drm_root: Path, already: list[GPUProfile]) -> list[GPUProfile]:
    """Enumerate GPUs from the DRM PCI bus (/sys/class/drm/card*/device).

    This is the fallback and completeness pass: it catches every GPU
    the vendor tools (nvidia-smi / rocm-smi) miss — Intel iGPUs, AMD
    APUs, discrete Arc — and doubles as the source of truth when the
    vendor tool isn't installed. ``already`` holds GPUs found by the
    vendor tools; NVIDIA and discrete AMD cards found there are not
    duplicated here.
    """
    found: list[GPUProfile] = []
    if not drm_root.is_dir():
        return found
    for card in sorted(drm_root.glob("card*"), key=lambda p: p.name):
        dev = card / "device"
        if not dev.is_dir():
            continue
        try:
            vendor_id = (dev / "vendor").read_text().strip()
        except OSError:
            continue
        try:
            device_id = (dev / "device").read_text().strip()
        except OSError:
            device_id = ""
        name = _pci_device_name(vendor_id, device_id) or f"{vendor_id}/{device_id}"
        vendor = {
            "0x10de": "NVIDIA", "0x1002": "AMD",
            "0x8086": "Intel", "0x106b": "Apple",
        }.get(vendor_id, "unknown")
        if vendor == "NVIDIA" and any(g.vendor == "NVIDIA" for g in already):
            continue   # already covered by nvidia-smi
        if vendor == "AMD" and any(g.vendor == "AMD" and not g.shared_ram for g in already):
            continue   # discrete already covered by rocm-smi
        shared = vendor == "Intel" and device_id.startswith(_INTEL_IGPU_PREFIXES)
        # Discrete GPU vendors (NVIDIA, AMD, Intel Arc) report
        # VRAM from sysfs. UMA/integrated parts report 0.
        if vendor in ("NVIDIA", "AMD"):
            vram = _pci_vram_gb(dev)
        elif vendor == "Intel":
            # Intel iGPU: shared RAM, conservative placeholder.
            # Intel Arc discrete: dedicated VRAM from sysfs.
            if shared:
                vram = _estimate_intel_vram_gb(device_id)
            else:
                vram = _pci_vram_gb(dev)
        else:
            vram = 0.0
        # AMD GPUs on UMA systems (Strix Halo, AMD APUs) report no
        # dedicated VRAM from sysfs/KFD — the GPU shares system RAM
        # and can use the full pool (CPU + GPU draw from the same
        # unified memory). Estimate usable memory from system RAM so
        # the vram_gb >= 4 gate in _recommend_defaults recognises
        # them as usable. The value is a signal, not a runtime cap;
        # llama.cpp manages actual allocation at launch.
        if vendor == "AMD" and vram == 0.0:
            total_ram = _total_ram_gb()
            if total_ram >= 4:
                vram = total_ram
        score = 0.3 if shared else 0.8 if vram >= 4 else 0.5
        use = (
            not shared
            and (vram >= 4 or vendor != "Intel")
        )
        reason_bits = []
        if shared:
            reason_bits.append("shared system RAM")
        if vram and vram < 4:
            reason_bits.append(f"only {vram:.1f} GB VRAM")
        if vendor == "Intel" and shared:
            reason_bits.append("Intel iGPU — CPU is faster for LLM matmul")
        elif vendor == "Intel" and vram and vram < 4:
            reason_bits.append(f"Intel discrete with only {vram:.1f} GB VRAM")
        reason = "; ".join(reason_bits) or "discrete GPU"
        found.append(GPUProfile(
            vendor=vendor, name=name, vram_gb=vram,
            driver=None, vulkan_score=score, shared_ram=shared,
            use_for_llm=use, reason=reason,
        ))
    return found


def _enumerate_gpus_macos() -> list[GPUProfile]:
    out = []
    try:
        r = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=8,
        )
        if r.returncode == 0:
            text = r.stdout
            # Look for "Chipset Model: Apple M2" or similar.
            for m in re.finditer(r"Chipset Model:\s*(.+)", text):
                name = m.group(1).strip()
                # Apple Silicon GPUs are unified memory: the GPU can
                # use the full system-RAM pool, exactly like AMD UMA.
                # sysctl hw.memsize is the whole-pool figure; report
                # it so the vram_gb >= 4 gate and the KV budget
                # recognise the GPU as usable on any Mac.
                vram = _total_ram_gb()
                out.append(GPUProfile(
                    vendor="Apple", name=name, vram_gb=vram,
                    driver=None, vulkan_score=0.95, shared_ram=True,
                    use_for_llm=vram >= 4,
                    reason="Apple Silicon unified memory; Metal backend",
                ))
    except Exception:
        pass
    return out


def _enumerate_gpus_windows() -> list[GPUProfile]:
    """Windows GPU enumeration — partial.

    NVIDIA: nvidia-smi (present on any CUDA install) reports name +
    VRAM. AMD/Intel on Windows would need WMI or DXGI enumeration
    (no /sys/class/drm, no rocm-smi); TODO: implement via
    wmic path win32_VideoController or dxgi. Until then AMD/Intel
    boxes on Windows fall back to CPU — safe, never wrong.
    """
    out = []
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2 and parts[1].isdigit():
                        vram = int(parts[1]) / 1024
                        out.append(GPUProfile(
                            vendor="NVIDIA", name=parts[0],
                            vram_gb=vram, driver=None,
                            vulkan_score=1.0, shared_ram=False,
                            use_for_llm=vram >= 4,
                            reason=f"{vram:.1f} GB VRAM" if vram < 4 else "discrete GPU",
                        ))
        except Exception:
            pass
    # AMD/Intel on Windows via WMI (no DRM). wmic is deprecated but
    # still present on Win10/11; fall back to PowerShell CIM.
    # AdapterRAM is bytes on WMI (0xffffffff = 4 GB sentinel for >4 GB,
    # so treat any >=4 GB as discrete).
    wmi_gpus: list[tuple[str, float]] = []
    try:
        r = subprocess.run(
            ["wmic", "path", "win32_VideoController", "get", "Name,AdapterRAM", "/format:csv"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                if "win32_VideoController" not in line.lower() and "," in line:
                    # CSV: Node,AdapterRAM,Name
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3 and parts[2]:
                        name = parts[2]
                        try:
                            ram = int(parts[1]) / (1024**3) if parts[1].isdigit() else 0.0
                        except ValueError:
                            ram = 0.0
                        # 0xffffffff sentinel: driver reports 4 GB max uint32
                        # for >4 GB cards — treat as discrete.
                        if ram >= 3.9:
                            ram = 4.0
                        wmi_gpus.append((name, ram))
    except Exception:
        pass
    if not wmi_gpus:
        # PowerShell CIM fallback (works when wmic missing on Win11)
        try:
            ps = (
                "Get-CimInstance Win32_VideoController | ForEach-Object { "
                "\"{0},{1}\" -f $_.Name,$_.AdapterRAM }"
            )
            r = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", ps],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    if "," in line:
                        name, ram_s = line.rsplit(",", 1)
                        name, ram_s = name.strip(), ram_s.strip()
                        try:
                            ram = int(ram_s) / (1024**3) if ram_s.lstrip("-").isdigit() else 0.0
                        except ValueError:
                            ram = 0.0
                        if ram >= 3.9:
                            ram = 4.0
                        if name:
                            wmi_gpus.append((name, ram))
        except Exception:
            pass
    for name, vram in wmi_gpus:
        low = name.lower()
        if "nvidia" in low:
            continue  # already handled via nvidia-smi with accurate VRAM
        if "amd" in low or "radeon" in low:
            vendor = "AMD"
        elif "intel" in low:
            vendor = "Intel"
        else:
            continue
        # Shared-RAM heuristic: Intel iGPU names contain UHD/HD/Iris
        is_intel_igpu = vendor == "Intel" and any(k in low for k in ("uhd", "hd graphics", "iris"))
        shared = is_intel_igpu
        use = not shared and vram >= 4
        # AMD UMA with unknown VRAM: fall back to total RAM check elsewhere
        reason = (
            "shared RAM (iGPU)" if shared
            else (f"{vram:.1f} GB VRAM" if vram < 4 else "discrete GPU")
        )
        out.append(GPUProfile(
            vendor=vendor, name=name,
            vram_gb=vram, driver=None,
            vulkan_score=1.0, shared_ram=shared,
            use_for_llm=use, reason=reason,
        ))
    return out


def _pci_device_name(vendor: str, device: str) -> str | None:
    """Look up a PCI device name from /usr/share/hwdata or the kernel's
    modalias. Returns None when neither works."""
    p = Path(f"/sys/bus/pci/devices/{vendor}:{device}/modalias")
    if p.exists():
        # modalias looks like "pci:v000010DEd00002684sv..."
        return None
    hw = Path("/usr/share/hwdata/pci.ids")
    if not hw.exists():
        return None
    vendor_l = vendor.lstrip("0").lower()
    device_l = device.lstrip("0").lower()
    text = hw.read_text(errors="replace")
    lines = text.splitlines()
    in_vendor = False
    for line in lines:
        if line.startswith(vendor_l):
            in_vendor = True
            continue
        if in_vendor:
            if not line.strip():
                break
            if line.startswith("\t"):
                stripped = line.lstrip("\t").strip()
                if stripped.startswith(device_l + " "):
                    return stripped.split(" ", 1)[1]
    return None


def _estimate_intel_vram_gb(device_id: str) -> float:
    """Conservative shared-RAM budget for Intel iGPUs (no dedicated VRAM).
    Reports a small number — the OS will steal system RAM; this is
    just a placeholder for the LLM matmul workload estimate."""
    return 0.5


def _amd_rocm_vram_gb() -> float:
    """Query VRAM in GB from ``rocm-smi --showmeminfo vram --csv``.

    Returns 0.0 on any failure (no rocm-smi, parse error, GPU with
    no VRAM reported). The caller is expected to keep ``use_for_llm=
    True`` for true ROCm devices — a 0 return only means "we couldn't
    measure it", not "the GPU is UMA".
    """
    try:
        r = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return 0.0
    if r.returncode != 0:
        return 0.0
    for line in r.stdout.splitlines():
        if "VRAM" not in line or "Total" not in line:
            continue
        for tok in line.replace(":", ",").split(","):
            tok = tok.strip()
            if tok.isdigit():
                try:
                    return int(tok) / 1024**3
                except ValueError:
                    return 0.0
    return 0.0


def _pci_vram_gb(card_dev_path: Path) -> float:
    """Read a DRM card's dedicated VRAM from sysfs in bytes, return GB.

    Discrete GPUs report a non-zero ``mem_info_vram_total``; UMA GPUs
    (AMD APUs, Intel iGPUs, Strix Halo) report 0. Returns 0.0 when
    the file is missing or unreadable.
    """
    f = card_dev_path / "mem_info_vram_total"
    if not f.exists():
        return 0.0
    try:
        return int(f.read_text().strip()) / 1024**3
    except (OSError, ValueError):
        return 0.0


# ── Profile lifecycle ─────────────────────────────────────────────


@dataclass
class BackendScore:
    """One entry in the recommended-backends table. Populated by
    ``toki doctor --profile --bench`` (micro-benchmark), empty when
    the user hasn't run a bench yet. The installer falls back to
    heuristics when this dict is empty."""
    name: str           # "cpu_p_cores_avx_vnni", "vulkan_intel_icd"
    prompt_ms: float
    decode_tps: float
    threads: int
    simd: str | None


@dataclass
class HardwareProfile:
    schema_version: int
    profiled_at: str         # ISO 8601
    os: str
    cpu: CPUProfile
    gpus: list[GPUProfile]
    ram_gb: float
    benchmarks: dict[str, BackendScore]
    recommended: dict = field(default_factory=dict)

    def age_seconds(self) -> float:
        try:
            t = time.mktime(time.strptime(self.profiled_at, "%Y-%m-%dT%H:%M:%SZ"))
        except ValueError:
            return float("inf")
        return time.time() - t

    def is_fresh(self) -> bool:
        return self.age_seconds() < PROFILE_TTL_SECONDS

    def backend_for(self, model_size_class: str) -> str:
        """Pick the recommended backend for this profile."""
        if self.recommended.get("backend"):
            return self.recommended["backend"]
        return _pick_backend(self.cpu, self.gpus, self.os)["backend"]


def profile_box(force: bool = False) -> HardwareProfile:
    """Build a HardwareProfile from the actual machine.

    If a cached profile exists in ``~/.config/tokirecur/hardware.json``
    and is younger than ``PROFILE_TTL_SECONDS``, return it. Otherwise
    re-profile and persist.
    """
    cached = load_profile() if not force else None
    live_gpus: list | None = None
    if cached is not None and cached.is_fresh() \
            and cached.schema_version == PROFILE_VERSION:
        live_gpus = enumerate_gpus()
        if live_gpus == cached.gpus:
            return cached
        logger.info("hardware profile gpus changed %r -> %r — re-profiling",
                    [g.name for g in cached.gpus], [g.name for g in live_gpus])

    cpu = cpu_topology()
    gpus = live_gpus if live_gpus is not None else enumerate_gpus()
    ram_gb = _available_gb()
    profile = HardwareProfile(
        schema_version=PROFILE_VERSION,
        profiled_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        os=_os_name(),
        cpu=cpu,
        gpus=gpus,
        ram_gb=ram_gb,
        benchmarks={},
        recommended=_recommend_defaults(cpu, gpus),
    )
    save_profile(profile)
    return profile


def _os_name() -> str:
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform.startswith("win"):
        return "windows"
    return "unknown"


# Vendor → backend per OS. The picker walks eligible GPUs in
# preference order (discrete first) and consults this table.
_BACKEND_TABLE: dict[str, dict[str, str]] = {
    "NVIDIA": {"linux": "cuda", "windows": "cuda", "darwin": "vulkan"},
    "AMD": {"linux": "rocm", "windows": "vulkan", "darwin": "vulkan"},
    "Intel": {"linux": "vulkan", "windows": "vulkan", "darwin": "vulkan"},
    "Apple": {"darwin": "metal", "linux": "vulkan", "windows": "vulkan"},
}


def _gpu_eligible(gpu: GPUProfile) -> bool:
    """Is this GPU worth using for LLM offload?

    - NVIDIA/Apple: any use_for_llm GPU with >= 4 GB VRAM. Apple's
      unified memory is shared by design but Metal handles it fine.
    - AMD: same — UMA parts (Strix Halo, APUs) are genuinely fast
      for LLM matmul; shared_ram is NOT a disqualifier.
    - Intel: only discrete Arc (shared iGPU is slower than the CPU).
    """
    if not gpu.use_for_llm or gpu.vram_gb < 4:
        return False
    if gpu.vendor == "Intel":
        return not gpu.shared_ram
    return True


def _pick_backend(cpu: CPUProfile, gpus: list[GPUProfile],
                  os_name: str) -> dict:
    """Choose backend + ngl + threads from the profile.

    Table-driven: _BACKEND_TABLE maps vendor → per-OS backend.
    Preference order for multi-GPU boxes: discrete before UMA, then
    most VRAM first. Falls back to CPU when no GPU is eligible.
    """
    eligible = [g for g in gpus if _gpu_eligible(g)]
    eligible.sort(key=lambda g: (g.shared_ram, -g.vram_gb))
    for gpu in eligible:
        backend = _BACKEND_TABLE.get(gpu.vendor, {}).get(os_name, "vulkan")
        return {"backend": backend, "ngl": 99,
                "threads": cpu.physical_cores}

    # No usable GPU. CPU it is. Drop the slowest tier on
    # hybrid chips so the LPE cores don't gate ggml's matmul sync —
    # every other core passes the freq filter, so a non-hybrid box
    # keeps all its cores. We used to cap threads at the P-core
    # cluster size and pin matmul to it with taskset; on a Core
    # Ultra 155U (2P+8E) that left the 8 E-cores idle and dropped a
    # 9B to ~1 TPS. The freq-tier filter keeps E-cores, drops LPE,
    # and leaves non-hybrid boxes untouched.
    threads = effective_cpu_threads(cpu)
    return {
        "backend": "cpu",
        "ngl": 0,
        "threads": threads,
        "build_hint": "ubuntu-x64.tar.gz (CPU)",
    }


def _recommend_defaults(cpu: CPUProfile, gpus: list[GPUProfile]) -> dict:
    """Heuristic recommendation when no bench data exists yet.

    The rule set is intentionally conservative — preferring CPU over a
    marginal GPU — because the bench data, when present, overrides it.
    Wrong-in-the-wrong-direction means the user pays a few percent of
    CPU throughput, not their whole session.
    """
    return _pick_backend(cpu, gpus, _os_name())


# ── Persistence ───────────────────────────────────────────────────


def profile_path() -> Path:
    """Where the cached profile lives. ``~/.config/tokirecur/`` is the
    XDG default and matches the rest of the project."""
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA", str(Path.home())))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    return base / "tokirecur" / "hardware.json"


def load_profile() -> HardwareProfile | None:
    p = profile_path()
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return _from_dict(data)


def save_profile(profile: HardwareProfile) -> None:
    p = profile_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_to_dict(profile), indent=2))


def _to_dict(profile: HardwareProfile) -> dict:
    d = asdict(profile)
    d["schema_version"] = profile.schema_version
    return d


def _from_dict(data: dict) -> HardwareProfile:
    """Tolerant deserialisation.

    Missing keys take safe defaults; unknown keys are ignored. The
    schema version is read but NOT enforced here — callers decide
    whether a cached profile is fresh enough (``profile_box``
    re-profiles when ``schema_version != PROFILE_VERSION``, which
    is the v2 → v3 migration path: the next load re-profiles and
    re-saves, so the freq-tier data ``core_max_freqs_mhz`` (new in
    v3) is never silently lost from a cached v2 file).
    """
    cpu = data.get("cpu", {})
    cpu_p = CPUProfile(
        vendor=cpu.get("vendor", "unknown"),
        name=cpu.get("name", "unknown"),
        family=cpu.get("family", 0),
        model=cpu.get("model", 0),
        physical_cores=cpu.get("physical_cores", 1),
        logical_cores=cpu.get("logical_cores", 1),
        p_core_ids=cpu.get("p_core_ids", []),
        e_core_ids=cpu.get("e_core_ids", []),
        max_boost_mhz=cpu.get("max_boost_mhz", 0.0),
        core_max_freqs_mhz=cpu.get("core_max_freqs_mhz", []),
        simd=cpu.get("simd", []),
    )
    gpus = [
        GPUProfile(**g) for g in data.get("gpus", [])
    ]
    benches = {
        k: BackendScore(**v) for k, v in data.get("benchmarks", {}).items()
    }
    return HardwareProfile(
        schema_version=data.get("schema_version", PROFILE_VERSION),
        profiled_at=data.get("profiled_at", ""),
        os=data.get("os", _os_name()),
        cpu=cpu_p,
        gpus=gpus,
        ram_gb=data.get("ram_gb", 0.0),
        benchmarks=benches,
        recommended=data.get("recommended", {}),
    )


def _available_gb() -> float:
    """Best-effort free RAM. Used to size thread count and KV cache
    decisions. Falls back to 16 GB (a common midrange desktop) when
    the platform doesn't expose it cleanly."""
    if sys.platform == "darwin":
        try:
            r = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=2,
            ).stdout.strip()
            if r.isdigit():
                return int(r) / 1024**3
        except Exception:
            pass
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / 1024**3
        except OSError:
            pass
    if sys.platform.startswith("win"):
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return stat.ullAvailPhys / 1024**3
        except Exception:
            pass
    return 16.0


# ── Inspect llama-server binary for SIMD backends ──────────────────


def llama_simd_backends(binary_dir: Path) -> list[str]:
    """List the CPU-specific libggml-cpu-*.so libraries shipped alongside
    ``llama-server``. Their filenames encode the CPU family they
    optimise for (alderlake, cascadelake, icelake, …). Returns an
    empty list when no per-family library is found (generic CPU build).

    Used by install-llama to warn when the chosen binary lacks an
    optimised path for the detected CPU family.
    """
    out: list[str] = []
    if not binary_dir.exists():
        return out
    pat = re.compile(r"libggml-cpu-([a-z0-9_]+)\.so(\.\d+)?$", re.I)
    for f in binary_dir.iterdir():
        m = pat.match(f.name)
        if m:
            out.append(m.group(1))
    # Windows uses the same naming with .dll.
    pat_dll = re.compile(r"libggml-cpu-([a-z0-9_]+)\.dll$", re.I)
    for f in binary_dir.iterdir():
        m = pat_dll.match(f.name)
        if m and m.group(1) not in out:
            out.append(m.group(1))
    return sorted(set(out))


def has_simd_for(profile: HardwareProfile) -> bool:
    """Return True when the profile's CPU has a SIMD extension that
    any recent llama.cpp release will use. AVX2 is the floor — every
    x86_64 build since 2017 supports it. AVX-VNNI is the meaningful
    speedup on Meteor Lake/Raptor Lake."""
    return any(s in profile.cpu.simd for s in ("avx2", "avx_vnni", "amx_tile",
                                             "avx512f", "neon_hpfp", "neon"))
