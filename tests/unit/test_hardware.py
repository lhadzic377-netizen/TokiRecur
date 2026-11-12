"""Unit tests for hardware.py profiling.

The hardware module does real I/O (reads /proc/cpuinfo, /sys/class/drm,
spawns nvidia-smi). Tests stub those paths and verify the structured
output shape is correct.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from tokirecur import hardware
from tokirecur.hardware import (
    CPUProfile,
    GPUProfile,
    HardwareProfile,
    profile_box,
)


class _DrmPathShim:
    """Patch hardware.Path to redirect /sys/class/drm lookups to a
    controlled directory. Hardware uses ``Path("/sys/class/drm")`` and
    then ``.glob("card*")``; we redirect the root to ``fake_root`` so
    tests can build a controlled device tree without touching the
    real /sys."""

    def __init__(self, fake_root):
        self.fake_root = Path(str(fake_root))

    def __call__(self, p=""):
        s = str(p)
        if s == "/sys/class/drm":
            return self.fake_root
        return Path(s)


# ── CPU topology ──────────────────────────────────────────────────


def test_cpu_profile_picks_p_cores_by_max_frequency(monkeypatch, tmp_path):
    """The P-core detector picks the higher-freq cluster as P.

    Simulates a 12-core Meteor Lake: 4 high-freq CPUs (P-cores with HT)
    at 4.8 GHz and 8 lower-freq E-cores at 2.7 GHz.
    """
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    monkeypatch.setattr(hardware, "_read_proc_cpuinfo", lambda: {
        "vendor_id": "GenuineIntel",
        "model name": "Intel(R) Core(TM) Ultra 7 155U",
        "cpu family": "6", "model": "189",
        "cpu cores": "6",   # 6 physical cores (2P + 4E)
    })
    monkeypatch.setattr(hardware, "_count_logical_cpus", lambda: 12)
    # CPUs 0..11 with P-cores at 0,1,2,3 (4.8 GHz) and E-cores 4..11 (2.7 GHz).
    def fake_max_freq(cpu_id):
        freq_path = tmp_path / f"cpu{cpu_id}.freq"
        freq_path.parent.mkdir(parents=True, exist_ok=True)
        # 4.8 GHz = 4_800_000 kHz, 2.7 GHz = 2_700_000 kHz.
        khz = 4_800_000 if cpu_id < 4 else 2_700_000
        freq_path.write_text(str(khz))
        return freq_path

    monkeypatch.setattr(hardware, "_cpu_max_freq_path", fake_max_freq)
    monkeypatch.setattr(hardware, "_detect_simd_linux",
                        lambda: ["sse4_2", "avx", "avx2", "avx_vnni"])

    cpu = hardware.cpu_topology()
    assert cpu.vendor == "GenuineIntel"
    assert "Ultra" in cpu.name
    assert cpu.physical_cores == 6   # 12 logical / 2 hyperthreading
    assert cpu.logical_cores == 12
    # CPUs 0..3 are P (4.8 GHz); 4..11 are E.
    assert cpu.p_core_ids == [0, 1, 2, 3]
    assert cpu.e_core_ids == [4, 5, 6, 7, 8, 9, 10, 11]
    assert "avx_vnni" in cpu.simd


def test_cpu_profile_no_cpufreq_falls_back_to_all_physical(monkeypatch):
    """When /sys cpufreq isn't readable (containers, WSL1), every
    physical core is treated as P-core. Better to over-thread than
    under-utilise."""
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    monkeypatch.setattr(hardware, "_read_proc_cpuinfo", lambda: {
        "vendor_id": "GenuineIntel",
        "model name": "Xeon E5-2690",
        "cpu family": "6", "model": "45",
        "cpu cores": "4",
    })
    monkeypatch.setattr(hardware, "_count_logical_cpus", lambda: 8)
    monkeypatch.setattr(hardware, "_cpu_max_freq_path", lambda _: None)
    monkeypatch.setattr(hardware, "_detect_simd_linux",
                        lambda: ["sse4_2", "avx"])
    cpu = hardware.cpu_topology()
    assert cpu.p_core_ids == [0, 1, 2, 3]
    assert cpu.e_core_ids == []


# ── GPU enumeration ───────────────────────────────────────────────


def test_enumerate_gpus_drops_intel_igpu(monkeypatch, tmp_path):
    """The enumerate_gpus step records every GPU it can find, but
    flags Intel iGPUs as use_for_llm=False — they share system RAM and
    are slower than the host CPU for LLM matmul."""
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)
    # Symlink /sys/class/drm to a controlled tree in tmp_path.
    fake_drm = tmp_path / "drm"
    fake_drm.mkdir()
    card = fake_drm / "card0" / "device"
    card.mkdir(parents=True)
    (card / "vendor").write_text("0x8086")     # Intel
    (card / "device").write_text("0x9a49")     # Meteor Lake Arc iGPU
    monkeypatch.setattr(hardware, "_pci_device_name", lambda *_: "Intel Arc iGPU")
    monkeypatch.setattr(hardware, "_estimate_intel_vram_gb", lambda *_: 0.5)
    monkeypatch.setattr(hardware, "Path", _DrmPathShim(fake_drm))

    gpus = hardware.enumerate_gpus()
    assert len(gpus) == 1
    arc = gpus[0]
    assert arc.vendor == "Intel"
    assert arc.shared_ram is True
    assert arc.use_for_llm is False
    assert "shared" in arc.reason or "iGPU" in arc.reason


def test_enumerate_gpus_keeps_discrete_nvidia(monkeypatch, tmp_path):
    """An NVIDIA card with its own VRAM gets use_for_llm=True."""
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    # nvidia-smi output
    def fake_which(name):
        return "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None

    def fake_run(cmd, **kwargs):
        class R:
            returncode = 0
            stdout = "NVIDIA GeForce RTX 4070, 12282, 555.42\n"
        return R()

    monkeypatch.setattr(hardware.shutil, "which", fake_which)
    monkeypatch.setattr(hardware.subprocess, "run", fake_run)
    # No Intel/AMD iGPUs in the fake /sys tree.
    empty_drm = tmp_path / "drm-empty"
    empty_drm.mkdir()
    monkeypatch.setattr(hardware, "Path", _DrmPathShim(empty_drm))

    gpus = hardware.enumerate_gpus()
    nvidia = [g for g in gpus if g.vendor == "NVIDIA"]
    assert len(nvidia) == 1
    assert nvidia[0].vram_gb >= 4
    assert nvidia[0].use_for_llm is True


def test_enumerate_gpus_uma_amd_uses_full_system_ram(monkeypatch, tmp_path):
    """An AMD GPU on a UMA system (Strix Halo, AMD APU) reports no
    dedicated VRAM from sysfs. The profiler must estimate the full
    unified-memory pool — not cap it at half — so the GPU is
    recognised as usable and llama.cpp can claim the full pool at
    launch. Generic: any device with unified memory benefits."""
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)
    # Fake /sys/class/drm with one AMD card (no rocm-smi).
    fake_drm = tmp_path / "drm"
    fake_drm.mkdir()
    card = fake_drm / "card0" / "device"
    card.mkdir(parents=True)
    (card / "vendor").write_text("0x1002")     # AMD
    (card / "device").write_text("0x1586")     # Strix Halo / gfx1151
    monkeypatch.setattr(hardware, "_pci_device_name", lambda *_: "Radeon 8060S")
    monkeypatch.setattr(hardware, "Path", _DrmPathShim(fake_drm))
    # 128 GB system — the GPU can use all of it (UMA).
    monkeypatch.setattr(hardware, "_total_ram_gb", lambda: 128.0)

    gpus = hardware.enumerate_gpus()
    amd = [g for g in gpus if g.vendor == "AMD"]
    assert len(amd) == 1
    # Full RAM, not half — capping at half would silently truncate
    # the KV cache budget on memory-rich boxes like Strix Halo.
    assert amd[0].vram_gb == 128.0
    assert amd[0].use_for_llm is True


def test_enumerate_gpus_uma_amd_works_on_small_boxes_too(monkeypatch, tmp_path):
    """An AMD UMA box with only 16 GB system RAM still reports the
    full pool as vram_gb — the GPU can use it all, and the value
    passes the vram_gb >= 4 gate so the GPU is recognised as usable."""
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)
    fake_drm = tmp_path / "drm"
    fake_drm.mkdir()
    card = fake_drm / "card0" / "device"
    card.mkdir(parents=True)
    (card / "vendor").write_text("0x1002")
    (card / "device").write_text("0x1586")
    monkeypatch.setattr(hardware, "_pci_device_name", lambda *_: "Radeon 780M")
    monkeypatch.setattr(hardware, "Path", _DrmPathShim(fake_drm))
    monkeypatch.setattr(hardware, "_total_ram_gb", lambda: 16.0)

    gpus = hardware.enumerate_gpus()
    amd = [g for g in gpus if g.vendor == "AMD"]
    assert len(amd) == 1
    assert amd[0].vram_gb == 16.0
    assert amd[0].use_for_llm is True


def test_amd_rocm_vram_gb_parses_csv(monkeypatch):
    """_amd_rocm_vram_gb parses the 'VRAM Total Memory (B)' cell from
    rocm-smi --showmeminfo vram --csv, regardless of which column
    carries the number."""
    monkeypatch.setattr(hardware.sys, "platform", "linux")

    class R:
        returncode = 0
        stdout = "GPU[0], VRAM Total Memory (B), 17179869184\n"
    monkeypatch.setattr(hardware.subprocess, "run",
                        lambda *a, **kw: R())
    vram = hardware._amd_rocm_vram_gb()
    assert abs(vram - 16.0) < 0.01


def test_amd_rocm_vram_gb_parses_real_header_and_data_rows(monkeypatch):
    """Real rocm-smi puts the label on a header row and the number on a
    separate data row. Regression: the old same-line scan reported 0 GB
    here, which demoted a usable AMD GPU to a CPU build."""
    monkeypatch.setattr(hardware.sys, "platform", "linux")

    class R:
        returncode = 0
        stdout = (
            "device,VRAM Total Memory (B),VRAM Total Used Memory (B)\n"
            "card0,17179869184,370290688\n"
        )
    monkeypatch.setattr(hardware.subprocess, "run",
                        lambda *a, **kw: R())
    vram = hardware._amd_rocm_vram_gb()
    assert abs(vram - 16.0) < 0.01


def test_amd_rocm_vram_gb_ignores_used_column(monkeypatch):
    """The total-memory column must be read, not the 'Used' column that
    follows it."""
    monkeypatch.setattr(hardware.sys, "platform", "linux")

    class R:
        returncode = 0
        stdout = (
            "device,VRAM Total Used Memory (B),VRAM Total Memory (B)\n"
            "card0,1073741824,8589934592\n"
        )
    monkeypatch.setattr(hardware.subprocess, "run",
                        lambda *a, **kw: R())
    vram = hardware._amd_rocm_vram_gb()
    assert abs(vram - 8.0) < 0.01


def test_amd_rocm_vram_gb_handles_colon_separator(monkeypatch):
    """_amd_rocm_vram_gb also handles the 'key: value' output format
    some ROCm versions emit."""
    monkeypatch.setattr(hardware.sys, "platform", "linux")

    class R:
        returncode = 0
        stdout = "GPU[0], VRAM Total Memory (B): 8589934592\n"
    monkeypatch.setattr(hardware.subprocess, "run",
                        lambda *a, **kw: R())
    vram = hardware._amd_rocm_vram_gb()
    assert abs(vram - 8.0) < 0.01


def test_amd_rocm_vram_gb_returns_zero_on_missing_gpu(monkeypatch):
    """rocm-smi exit code != 0 → returns 0.0 (caller keeps
    use_for_llm=True for the discrete GPU; the value is a signal, not
    a gate)."""
    monkeypatch.setattr(hardware.sys, "platform", "linux")

    class R:
        returncode = 1
        stdout = ""
    monkeypatch.setattr(hardware.subprocess, "run",
                        lambda *a, **kw: R())
    assert hardware._amd_rocm_vram_gb() == 0.0


def test_amd_rocm_vram_gb_returns_zero_on_unparseable(monkeypatch):
    """rocm-smi returns garbage → returns 0.0 instead of crashing."""
    monkeypatch.setattr(hardware.sys, "platform", "linux")

    class R:
        returncode = 0
        stdout = "GPU[0], VRAM Total Memory (B), not-a-number\n"
    monkeypatch.setattr(hardware.subprocess, "run",
                        lambda *a, **kw: R())
    assert hardware._amd_rocm_vram_gb() == 0.0


def test_amd_discrete_gpu_with_vram_is_usable(monkeypatch, tmp_path):
    """AMD discrete GPU with rocm-smi reporting 16 GB VRAM gets
    vram_gb=16 and use_for_llm=True. The old hardcoded vram_gb=0
    bug would have failed the vram_gb >= 4 gate and skipped the GPU."""
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    monkeypatch.setattr(hardware.shutil, "which",
                        lambda name: "/usr/bin/rocm-smi" if name == "rocm-smi" else None)

    def fake_run(cmd, **kwargs):
        class R:
            returncode = 0
            stdout = ""
        if "--showproductname" in cmd:
            r = R()
            # Real rocm-smi --showproductname --csv has a header line.
            r.stdout = "GPU, Product Name\nGPU[0], AMD Radeon RX 7900 XTX\n"
            return r
        if "--showmeminfo" in cmd:
            r = R()
            r.stdout = "GPU[0], VRAM Total Memory (B), 17179869184\n"
            return r
        return R()

    monkeypatch.setattr(hardware.subprocess, "run", fake_run)
    # No PCI scan needed — rocm-smi covers it.
    monkeypatch.setattr(hardware, "Path", _DrmPathShim(tmp_path / "empty"))

    gpus = hardware.enumerate_gpus()
    amd = [g for g in gpus if g.vendor == "AMD"]
    assert len(amd) == 1
    assert "RX 7900" in amd[0].name
    assert amd[0].vram_gb >= 16
    assert amd[0].use_for_llm is True


def test_nvidia_pci_fallback_when_nvidia_smi_missing(monkeypatch, tmp_path):
    """When nvidia-smi is not on PATH (headless server, minimal driver
    install), the PCI scan still detects NVIDIA with VRAM from sysfs."""
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)
    fake_drm = tmp_path / "drm"
    fake_drm.mkdir()
    card = fake_drm / "card0" / "device"
    card.mkdir(parents=True)
    (card / "vendor").write_text("0x10de")
    (card / "device").write_text("0x2684")
    # 24 GB VRAM in bytes: 24 * 1024^3
    (card / "mem_info_vram_total").write_text(str(24 * 1024**3))
    monkeypatch.setattr(hardware, "_pci_device_name", lambda *_: "NVIDIA RTX 4090")
    monkeypatch.setattr(hardware, "Path", _DrmPathShim(fake_drm))

    gpus = hardware.enumerate_gpus()
    nvidia = [g for g in gpus if g.vendor == "NVIDIA"]
    assert len(nvidia) == 1
    assert nvidia[0].vram_gb >= 24
    assert nvidia[0].use_for_llm is True


def test_nvidia_pci_fallback_zero_vram_is_skipped_at_backend_pick(monkeypatch, tmp_path):
    """NVIDIA 0x10de with mem_info_vram_total=0 (unusual; e.g. a
    Tesla K40 in some configurations) is reported with vram_gb=0.
    The vram_gb >= 4 gate in _recommend_defaults skips it at
    runtime — the GPU is hardware-capable (use_for_llm=True) but
    not recommended for LLM."""
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)
    fake_drm = tmp_path / "drm"
    fake_drm.mkdir()
    card = fake_drm / "card0" / "device"
    card.mkdir(parents=True)
    (card / "vendor").write_text("0x10de")
    (card / "device").write_text("0x1b06")
    (card / "mem_info_vram_total").write_text("0")
    monkeypatch.setattr(hardware, "_pci_device_name", lambda *_: "Tesla K40")
    monkeypatch.setattr(hardware, "Path", _DrmPathShim(fake_drm))

    gpus = hardware.enumerate_gpus()
    nvidia = [g for g in gpus if g.vendor == "NVIDIA"]
    assert len(nvidia) == 1
    assert nvidia[0].vram_gb == 0.0
    assert nvidia[0].use_for_llm is True   # hardware-capable
    # The _recommend_defaults gate is what actually picks the backend.
    cpu = hardware.CPUProfile(
        vendor="GenuineIntel", name="stub",
        family=6, model=183, physical_cores=4, logical_cores=4,
        p_core_ids=[0, 1, 2, 3], e_core_ids=[],
        max_boost_mhz=4000.0, simd=["avx2"],
    )
    rec = hardware._recommend_defaults(cpu, gpus)
    assert rec["backend"] == "cpu"   # gate skipped the zero-VRAM GPU


def test_nvidia_pci_fallback_skipped_when_nvidia_smi_already_present(monkeypatch, tmp_path):
    """When nvidia-smi already added the GPU, the PCI scan skips
    NVIDIA — no duplicate entry."""
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    monkeypatch.setattr(hardware.shutil, "which",
                        lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)

    class R:
        returncode = 0
        stdout = "NVIDIA RTX 4090, 24576, 555.42\n"
    monkeypatch.setattr(hardware.subprocess, "run", lambda *a, **kw: R())

    fake_drm = tmp_path / "drm"
    fake_drm.mkdir()
    card = fake_drm / "card0" / "device"
    card.mkdir(parents=True)
    (card / "vendor").write_text("0x10de")
    (card / "device").write_text("0x2684")
    (card / "mem_info_vram_total").write_text(str(24 * 1024**3))
    monkeypatch.setattr(hardware, "_pci_device_name", lambda *_: "NVIDIA RTX 4090")
    monkeypatch.setattr(hardware, "Path", _DrmPathShim(fake_drm))

    gpus = hardware.enumerate_gpus()
    nvidia = [g for g in gpus if g.vendor == "NVIDIA"]
    assert len(nvidia) == 1   # no duplicate


def test_intel_arc_discrete_is_usable(monkeypatch, tmp_path):
    """Intel Arc discrete (Alchemist 0x56, Battlemage 0xe2, etc.) is
    NOT in the iGPU prefix list — it has dedicated VRAM and is
    usable for LLM. Previously all Intel GPUs were hardcoded as
    use_for_llm=False, which wrongly skipped discrete Arc."""
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)
    fake_drm = tmp_path / "drm"
    fake_drm.mkdir()
    card = fake_drm / "card0" / "device"
    card.mkdir(parents=True)
    (card / "vendor").write_text("0x8086")
    (card / "device").write_text("0x5690")     # Arc A770
    # 16 GB VRAM
    (card / "mem_info_vram_total").write_text(str(16 * 1024**3))
    monkeypatch.setattr(hardware, "_pci_device_name", lambda *_: "Intel Arc A770")
    monkeypatch.setattr(hardware, "Path", _DrmPathShim(fake_drm))

    gpus = hardware.enumerate_gpus()
    arc = [g for g in gpus if g.vendor == "Intel"]
    assert len(arc) == 1
    assert arc[0].vram_gb >= 16
    assert arc[0].shared_ram is False
    assert arc[0].use_for_llm is True


def test_intel_igpu_still_dropped_with_shared_ram(monkeypatch, tmp_path):
    """Regression guard: Intel iGPU (e.g. 155U 0x9a49) still gets
    use_for_llm=False and shared_ram=True. The new discrete-Arc
    path must not regress the iGPU rejection."""
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)
    fake_drm = tmp_path / "drm"
    fake_drm.mkdir()
    card = fake_drm / "card0" / "device"
    card.mkdir(parents=True)
    (card / "vendor").write_text("0x8086")
    (card / "device").write_text("0x9a49")     # Meteor Lake Arc iGPU
    # iGPU has no dedicated VRAM (mem_info_vram_total=0)
    (card / "mem_info_vram_total").write_text("0")
    monkeypatch.setattr(hardware, "_pci_device_name", lambda *_: "Intel Arc iGPU")
    monkeypatch.setattr(hardware, "Path", _DrmPathShim(fake_drm))

    gpus = hardware.enumerate_gpus()
    igpu = [g for g in gpus if g.vendor == "Intel"]
    assert len(igpu) == 1
    assert igpu[0].shared_ram is True
    assert igpu[0].use_for_llm is False
    assert "iGPU" in igpu[0].reason or "shared" in igpu[0].reason


# ── Recommendation ────────────────────────────────────────────────


def test_recommend_defaults_picks_cpu_for_intel_igpu_only_box():
    """The recommendation loop prefers CPU when the only GPU is an
    Intel iGPU. This is the path the Core Ultra 155U user is on."""
    cpu = CPUProfile(
        vendor="GenuineIntel", name="Ultra 7 155U",
        family=6, model=189,
        physical_cores=6, logical_cores=12,
        p_core_ids=[0, 1, 2, 3], e_core_ids=list(range(4, 12)),
        max_boost_mhz=4800.0, simd=["avx2", "avx_vnni"],
    )
    gpus = [GPUProfile(
        vendor="Intel", name="Arc iGPU",
        vram_gb=0.5, driver=None, vulkan_score=0.3,
        shared_ram=True, use_for_llm=False,
        reason="shared system RAM",
    )]
    rec = hardware._recommend_defaults(cpu, gpus)
    assert rec["backend"] == "cpu"
    assert rec["ngl"] == 0
    # All physical cores — 6 of them, P+E together.
    assert rec["threads"] == 6
    assert "P-core" in rec.get("build_hint", "") or "CPU" in rec.get("build_hint", "")


def test_recommend_defaults_picks_cuda_for_discrete_nvidia(monkeypatch):
    # Apple silicon is Metal; the recommendation under test is the CUDA
    # one, so the test has to declare the platform it means.
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    cpu = CPUProfile(
        vendor="GenuineIntel", name="i7-13700K",
        family=6, model=183,
        physical_cores=16, logical_cores=24,
        p_core_ids=list(range(16)), e_core_ids=[],
        max_boost_mhz=5400.0, simd=["avx2"],
    )
    gpus = [GPUProfile(
        vendor="NVIDIA", name="RTX 4070",
        vram_gb=12.0, driver=None, vulkan_score=1.0,
        shared_ram=False, use_for_llm=True, reason="discrete GPU",
    )]
    rec = hardware._recommend_defaults(cpu, gpus)
    assert rec["backend"] == "cuda"
    assert rec["ngl"] == 99


def test_recommend_defaults_picks_metal_for_apple_silicon(monkeypatch):
    monkeypatch.setattr(hardware.sys, "platform", "darwin")
    cpu = CPUProfile(
        vendor="Apple", name="Apple M2",
        family=0, model=0, physical_cores=4, logical_cores=8,
        p_core_ids=[0, 1, 2, 3], e_core_ids=[4, 5, 6, 7],
        max_boost_mhz=0.0, simd=["neon_hpfp"],
    )
    gpus = [GPUProfile(
        vendor="Apple", name="Apple M2",
        vram_gb=4.0, driver=None, vulkan_score=0.95,
        shared_ram=True, use_for_llm=True,
        reason="Apple Silicon unified memory; Metal backend",
    )]
    rec = hardware._recommend_defaults(cpu, gpus)
    assert rec["backend"] == "metal"
    assert rec["ngl"] == 99


def test_recommend_defaults_amd_uma_is_eligible(monkeypatch):
    """Strix Halo / APU: shared_ram=True is NOT a disqualifier for AMD
    — UMA GPUs are genuinely fast for LLM matmul. ROCm is picked."""
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    cpu = CPUProfile(
        vendor="AuthenticAMD", name="Ryzen AI MAX+ 395",
        family=26, model=32, physical_cores=16, logical_cores=16,
        p_core_ids=list(range(16)), e_core_ids=[],
        max_boost_mhz=5100.0, simd=["avx512"],
    )
    gpus = [GPUProfile(
        vendor="AMD", name="Radeon 8060S",
        vram_gb=125.0, driver=None, vulkan_score=0.9,
        shared_ram=True, use_for_llm=True,
        reason="AMD UMA — full unified memory pool",
    )]
    rec = hardware._recommend_defaults(cpu, gpus)
    assert rec["backend"] == "rocm"
    assert rec["ngl"] == 99


def test_recommend_defaults_multi_gpu_prefers_discrete(monkeypatch):
    """NVIDIA discrete + Intel iGPU: discrete is picked (cuda), even
    though the iGPU comes first in enumeration order."""
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    cpu = CPUProfile(
        vendor="GenuineIntel", name="i9-14900K",
        family=6, model=183, physical_cores=24, logical_cores=32,
        p_core_ids=list(range(16)), e_core_ids=list(range(16, 24)),
        max_boost_mhz=6000.0, simd=["avx2"],
    )
    igpu = GPUProfile(
        vendor="Intel", name="UHD 770",
        vram_gb=0.5, driver=None, vulkan_score=0.3,
        shared_ram=True, use_for_llm=False,
        reason="Intel iGPU — CPU is faster for LLM matmul",
    )
    discrete = GPUProfile(
        vendor="NVIDIA", name="RTX 4090",
        vram_gb=24.0, driver=None, vulkan_score=1.0,
        shared_ram=False, use_for_llm=True, reason="discrete GPU",
    )
    rec = hardware._recommend_defaults(cpu, [igpu, discrete])
    assert rec["backend"] == "cuda"


def test_pick_backend_table_driven_vendor_os(monkeypatch):
    """_pick_backend consults _BACKEND_TABLE per vendor + OS."""
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    cpu = CPUProfile(
        vendor="GenuineIntel", name="stub",
        family=6, model=183, physical_cores=8, logical_cores=8,
        p_core_ids=list(range(8)), e_core_ids=[],
        max_boost_mhz=4000.0, simd=["avx2"],
    )
    gpus = [GPUProfile(
        vendor="AMD", name="Radeon RX 7900 XTX",
        vram_gb=24.0, driver=None, vulkan_score=0.9,
        shared_ram=False, use_for_llm=True, reason="discrete GPU",
    )]
    assert hardware._pick_backend(cpu, gpus, "linux")["backend"] == "rocm"
    assert hardware._pick_backend(cpu, gpus, "windows")["backend"] == "vulkan"
    assert hardware._pick_backend(cpu, gpus, "darwin")["backend"] == "metal"


def test_freq_tier_drop_apple_efficiency_cores():
    """Apple Silicon P/E split (synthetic 1000/650 MHz) drops the
    efficiency cores through the same freq-tier filter as Intel LPE."""
    cpu = CPUProfile(
        vendor="Apple", name="Apple M3 Pro",
        family=0, model=0, physical_cores=12, logical_cores=12,
        p_core_ids=list(range(6)), e_core_ids=list(range(6, 12)),
        max_boost_mhz=0.0,
        core_max_freqs_mhz=[1000.0] * 6 + [650.0] * 6,
        simd=["neon_hpfp"],
    )
    assert hardware.effective_cpu_threads(cpu) == 6


def test_freq_tier_keeps_apple_all_perf_when_no_eff_reported():
    """A Mac where perflevel1 doesn't exist (or old macOS) has no
    freqs → falls back to physical_cores."""
    cpu = CPUProfile(
        vendor="Apple", name="Apple M2",
        family=0, model=0, physical_cores=8, logical_cores=8,
        p_core_ids=list(range(8)), e_core_ids=[],
        max_boost_mhz=0.0, core_max_freqs_mhz=[], simd=["neon_hpfp"],
    )
    assert hardware.effective_cpu_threads(cpu) == 8


def test_cpu_topology_macos_sets_freq_tiers_from_perflevel(monkeypatch):
    """_cpu_topology_macos encodes hw.perflevel0/1 into
    core_max_freqs_mhz so the freq-tier filter can drop E-cores."""
    monkeypatch.setattr(hardware.sys, "platform", "darwin")
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 10)

    def fake_run(cmd, **kwargs):
        class R:
            returncode = 0
            stdout = ""
        if cmd == ["sysctl", "-n", "machdep.cpu.brand_string"]:
            R.stdout = "Apple M2 Pro"
        elif cmd == ["sysctl", "-n", "hw.physicalcpu"]:
            R.stdout = "10"
        elif cmd == ["sysctl", "-n", "hw.perflevel0.logicalcpu"]:
            R.stdout = "8"
        elif cmd == ["sysctl", "-n", "hw.perflevel1.logicalcpu"]:
            R.stdout = "2"
        return R()
    monkeypatch.setattr(hardware.subprocess, "run", fake_run)
    monkeypatch.setattr(hardware, "_detect_simd_macos", lambda: [])

    cpu = hardware._cpu_topology_macos()
    assert cpu.vendor == "Apple"
    assert cpu.physical_cores == 10
    assert cpu.e_core_ids == [8, 9]
    assert cpu.core_max_freqs_mhz == [1000.0] * 8 + [650.0] * 2
    assert hardware.effective_cpu_threads(cpu) == 8


def test_pick_backend_no_gpu_returns_cpu():
    cpu = CPUProfile(
        vendor="GenuineIntel", name="stub",
        family=6, model=183, physical_cores=8, logical_cores=8,
        p_core_ids=list(range(8)), e_core_ids=[],
        max_boost_mhz=4000.0, simd=["avx2"],
    )
    rec = hardware._pick_backend(cpu, [], "linux")
    assert rec["backend"] == "cpu"
    assert rec["ngl"] == 0


def test_recommend_defaults_threads_use_all_physical_cores():
    """Threads aren't capped by RAM or P-core count — every physical
    core is used, on hybrid and plain chips alike."""
    cpu = CPUProfile(
        vendor="GenuineIntel", name="many cores",
        family=6, model=183, physical_cores=32, logical_cores=64,
        p_core_ids=list(range(32)), e_core_ids=[],
        max_boost_mhz=4000.0, simd=["avx2"],
    )
    # Even with only 4 GB free RAM we keep all physical cores.
    rec = hardware._recommend_defaults(cpu, [])
    assert rec["threads"] == 32


# ── Thread-count policy (effective_cpu_threads) ────────────────────


def _cpu(vendor, name, physical, logical, p_ids, e_ids, freqs):
    """Compact CPUProfile builder for the freq-tier tests."""
    return CPUProfile(
        vendor=vendor, name=name, family=6, model=183,
        physical_cores=physical, logical_cores=logical,
        p_core_ids=p_ids, e_core_ids=e_ids,
        max_boost_mhz=max(freqs) if freqs else 0.0,
        core_max_freqs_mhz=list(freqs),
        simd=["avx2"],
    )


def test_effective_cpu_threads_drops_lpe_on_155u():
    """Core Ultra 155U: 2P + 8E + 2LPE. The 2 LPE cores run at
    ~2.1 GHz vs P at 4.8 — they must NOT enter the thread pool.
    The 2 P-core HT siblings are also subtracted so the count
    reflects distinct physical fast cores (2P + 8E = 10), not
    logical fast cores (12)."""
    freqs = [4800, 4800, 4800, 4800,    # 2P with HT
             3800, 3800, 3800, 3800,    # 4 E-cores
             3800, 3800, 3800, 3800,    # 4 more E-cores
             2100, 2100]                # 2 LPE cores
    cpu = _cpu("GenuineIntel", "Core Ultra 7 155U", 12, 14,
               p_ids=list(range(4)), e_ids=list(range(4, 14)), freqs=freqs)
    assert hardware.effective_cpu_threads(cpu) == 10


def test_effective_cpu_threads_keeps_all_on_non_hybrid():
    """A box where every core boosts within ~10 % of the fastest
    must keep every physical core — no filter, no shrink."""
    freqs = [5700.0] * 16  # Halo-style AMD, all P-tier
    cpu = _cpu("AuthenticAMD", "Ryzen AI MAX", 16, 16,
               p_ids=list(range(16)), e_ids=[], freqs=freqs)
    assert hardware.effective_cpu_threads(cpu) == 16


def test_effective_cpu_threads_handles_ht_siblings():
    """A non-hybrid HT box (8P, 16 threads): all cores are P-tier,
    so the fast count is 16 logical; subtract the 8 HT siblings and
    the result is 8 physical — the right pool size."""
    freqs = [5200.0] * 16  # 8P with HT
    cpu = _cpu("GenuineIntel", "i7-13700K (non-hybrid box)", 8, 16,
               p_ids=list(range(8)), e_ids=list(range(8, 16)), freqs=freqs)
    assert hardware.effective_cpu_threads(cpu) == 8


def test_effective_cpu_threads_falls_back_when_no_freq_data():
    """Containers / macOS / Windows leave core_max_freqs_mhz empty.
    The filter must NOT shrink the pool below physical_cores."""
    cpu = _cpu("Apple", "M2", 4, 8,
               p_ids=list(range(4)), e_ids=list(range(4, 8)), freqs=[])
    cpu = CPUProfile(
        vendor=cpu.vendor, name=cpu.name, family=cpu.family, model=cpu.model,
        physical_cores=cpu.physical_cores, logical_cores=cpu.logical_cores,
        p_core_ids=cpu.p_core_ids, e_core_ids=cpu.e_core_ids,
        max_boost_mhz=cpu.max_boost_mhz, simd=cpu.simd,
    )  # core_max_freqs_mhz defaults to []
    assert hardware.effective_cpu_threads(cpu) == cpu.physical_cores


def test_recommend_defaults_cpu_uses_filtered_threads_on_hybrid():
    """The CPU branch of _recommend_defaults must apply the freq-tier
    filter — a 155U with per-core freqs reported gets LPE cores and
    HT siblings excluded from the recommended thread count."""
    freqs = [4800, 4800, 4800, 4800] + [3800] * 8 + [2100, 2100]
    cpu = _cpu("GenuineIntel", "Core Ultra 7 155U", 12, 14,
               p_ids=list(range(4)), e_ids=list(range(4, 14)), freqs=freqs)
    rec = hardware._recommend_defaults(cpu, [])
    assert rec["backend"] == "cpu"
    assert rec["ngl"] == 0
    assert rec["threads"] == 10  # 2P + 8E physical fast


def test_recommend_defaults_cpu_threads_unchanged_on_non_hybrid():
    """A box with all cores in the same tier must keep physical_cores
    threads — no shrink, no surprise."""
    freqs = [5700.0] * 16
    cpu = _cpu("AuthenticAMD", "Ryzen AI MAX", 16, 16,
               p_ids=list(range(16)), e_ids=[], freqs=freqs)
    rec = hardware._recommend_defaults(cpu, [])
    assert rec["threads"] == 16


# ── Profile persistence ───────────────────────────────────────────


def test_profile_save_and_load_roundtrip(tmp_path, monkeypatch):
    """Saving and loading yields an equivalent profile."""
    monkeypatch.setattr(hardware, "profile_path", lambda: tmp_path / "hw.json")
    prof = HardwareProfile(
        schema_version=1,
        profiled_at="2026-01-01T00:00:00Z",
        os="linux",
        cpu=CPUProfile(
            vendor="GenuineIntel", name="Ultra 7 155U",
            family=6, model=189, physical_cores=6, logical_cores=12,
            p_core_ids=[0, 1, 2, 3], e_core_ids=list(range(4, 12)),
            max_boost_mhz=4800.0, simd=["avx2", "avx_vnni"],
        ),
        gpus=[GPUProfile(
            vendor="Intel", name="Arc iGPU", vram_gb=0.5,
            driver=None, vulkan_score=0.3, shared_ram=True,
            use_for_llm=False, reason="shared system RAM",
        )],
        ram_gb=16.0,
        benchmarks={},
        recommended={"backend": "cpu", "ngl": 0, "threads": 4},
    )
    hardware.save_profile(prof)
    loaded = hardware.load_profile()
    assert loaded is not None
    assert loaded.cpu.name == "Ultra 7 155U"
    assert loaded.cpu.p_core_ids == [0, 1, 2, 3]
    assert loaded.gpus[0].vendor == "Intel"
    assert loaded.recommended["threads"] == 4


def test_profile_box_returns_loaded_when_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(hardware, "profile_path", lambda: tmp_path / "hw.json")
    # Stub every live I/O call so the test never touches the real box.
    monkeypatch.setattr(hardware, "cpu_topology", lambda: CPUProfile(
        vendor="GenuineIntel", name="stub",
        family=6, model=189, physical_cores=4, logical_cores=4,
        p_core_ids=[0, 1], e_core_ids=[],
        max_boost_mhz=4000.0, simd=["avx2"],
    ))
    monkeypatch.setattr(hardware, "enumerate_gpus", lambda: [])
    monkeypatch.setattr(hardware, "_available_gb", lambda: 16.0)
    monkeypatch.setattr(hardware, "_recommend_defaults",
                        lambda *a, **k: {"backend": "cpu", "ngl": 0, "threads": 4})

    prof = profile_box(force=True)
    assert prof.is_fresh()
    # Subsequent call should return the cached one without re-probing.
    monkeypatch.setattr(hardware, "cpu_topology",
                        lambda: (_ for _ in ()).throw(AssertionError("should not re-probe")))
    cached = profile_box()
    assert cached.cpu == prof.cpu
    assert cached.recommended == prof.recommended


def test_profile_box_skips_cache_when_stale(tmp_path, monkeypatch):
    """A 31-day-old profile triggers a re-profile."""
    monkeypatch.setattr(hardware, "profile_path", lambda: tmp_path / "hw.json")
    stale = HardwareProfile(
        schema_version=1,
        profiled_at="2025-01-01T00:00:00Z",  # ancient
        os="linux", cpu=CPUProfile(
            vendor="unknown", name="unknown", family=0, model=0,
            physical_cores=1, logical_cores=1,
            p_core_ids=[0], e_core_ids=[],
            max_boost_mhz=0.0, simd=[],
        ),
        gpus=[], ram_gb=1.0, benchmarks={}, recommended={},
    )
    hardware.save_profile(stale)
    assert not stale.is_fresh()
    # Stub every live I/O call so the test never touches the real box.
    monkeypatch.setattr(hardware, "cpu_topology", lambda: CPUProfile(
        vendor="GenuineIntel", name="fresh",
        family=6, model=189, physical_cores=4, logical_cores=4,
        p_core_ids=[0, 1], e_core_ids=[],
        max_boost_mhz=4000.0, simd=["avx2"],
    ))
    monkeypatch.setattr(hardware, "enumerate_gpus", lambda: [])
    monkeypatch.setattr(hardware, "_available_gb", lambda: 8.0)
    monkeypatch.setattr(hardware, "_recommend_defaults",
                        lambda *a, **k: {"backend": "cpu", "ngl": 0, "threads": 2})
    fresh = profile_box()
    assert fresh.cpu.name == "fresh"
    assert fresh.recommended["threads"] == 2


def test_profile_box_reprofiles_on_gpu_swap(tmp_path, monkeypatch):
    """A GPU swap must re-profile immediately, not after the 30-day
    TTL — the cached profile would keep recommending the old backend
    (e.g. cuda) on a box whose discrete card was removed."""
    monkeypatch.setattr(hardware, "profile_path", lambda: tmp_path / "hw.json")
    cached_gpu = GPUProfile(
        vendor="NVIDIA", name="RTX 3090", vram_gb=24.0, driver=None,
        vulkan_score=1.0, shared_ram=False, use_for_llm=True,
        reason="discrete GPU",
    )
    cached = HardwareProfile(
        schema_version=hardware.PROFILE_VERSION,
        profiled_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        os="linux", cpu=CPUProfile(
            vendor="unknown", name="old", family=0, model=0,
            physical_cores=1, logical_cores=1,
            p_core_ids=[0], e_core_ids=[],
            max_boost_mhz=0.0, simd=[],
        ),
        gpus=[cached_gpu], ram_gb=32.0, benchmarks={},
        recommended={"backend": "cuda", "ngl": 99, "threads": 2},
    )
    hardware.save_profile(cached)
    assert cached.is_fresh()
    # GPU disappeared — enumerate now returns empty while the cache
    # still lists the RTX 3090.
    enumerate_calls = []
    real_cpu = cached.cpu

    def fake_enumerate():
        enumerate_calls.append(1)
        return []

    monkeypatch.setattr(hardware, "enumerate_gpus", fake_enumerate)
    monkeypatch.setattr(hardware, "cpu_topology", lambda: real_cpu)
    monkeypatch.setattr(hardware, "_available_gb", lambda: 16.0)
    monkeypatch.setattr(hardware, "_recommend_defaults",
                        lambda *a, **k: {"backend": "cpu", "ngl": 0, "threads": 2})

    prof = profile_box()  # NOT force=True — the swap must be detected
    assert prof.gpus == []
    assert prof.recommended["backend"] == "cpu"


# ── SIMD backend inspection ───────────────────────────────────────


def test_llama_simd_backends_returns_per_family_libs(tmp_path):
    """The release asset ships per-CPU-family libggml-cpu-<family>.so;
    inspect them by filename."""
    (tmp_path / "libggml-cpu.so").write_bytes(b"")
    (tmp_path / "libggml-cpu-alderlake.so").write_bytes(b"")
    (tmp_path / "libggml-cpu-cascadelake.so").write_bytes(b"")
    (tmp_path / "libggml-cpu-sapphirerapids.so").write_bytes(b"")
    # Windows uses .dll.
    (tmp_path / "libggml-cpu-icelake.dll").write_bytes(b"")
    out = hardware.llama_simd_backends(tmp_path)
    assert out == ["alderlake", "cascadelake", "icelake", "sapphirerapids"]


def test_llama_simd_backends_empty_on_unknown_build(tmp_path):
    """A generic CPU build has no per-family libraries."""
    (tmp_path / "libggml-cpu.so").write_bytes(b"")
    (tmp_path / "libggml-base.so").write_bytes(b"")
    assert hardware.llama_simd_backends(tmp_path) == []


# ── Apple Silicon ─────────────────────────────────────────────────


def test_apple_silicon_vram_is_full_unified_memory(monkeypatch):
    """Apple Silicon GPU reports the full system-RAM pool as vram_gb
    (sysctl hw.memsize) — not the old hardcoded 4 GB. Any Mac, any
    RAM size."""
    monkeypatch.setattr(hardware.sys, "platform", "darwin")
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)

    class R:
        returncode = 0
        stdout = "Chipset Model: Apple M3 Max\n"
    # 48 GB in bytes
    monkeypatch.setattr(hardware.subprocess, "run",
                        lambda *a, **kw: R())
    monkeypatch.setattr(hardware, "_total_ram_gb", lambda: 48.0)

    gpus = hardware.enumerate_gpus()
    apple = [g for g in gpus if g.vendor == "Apple"]
    assert len(apple) == 1
    assert apple[0].vram_gb == 48.0
    assert apple[0].use_for_llm is True


def test_apple_silicon_vram_reports_small_macs_too(monkeypatch):
    """An 8 GB Mac Mini still reports 8 GB and stays usable."""
    monkeypatch.setattr(hardware.sys, "platform", "darwin")
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)

    class R:
        returncode = 0
        stdout = "Chipset Model: Apple M1\n"
    monkeypatch.setattr(hardware.subprocess, "run",
                        lambda *a, **kw: R())
    monkeypatch.setattr(hardware, "_total_ram_gb", lambda: 8.0)

    gpus = hardware.enumerate_gpus()
    apple = [g for g in gpus if g.vendor == "Apple"]
    assert len(apple) == 1
    assert apple[0].vram_gb == 8.0
    assert apple[0].use_for_llm is True


def test_total_ram_gb_macos_parses_sysctl(monkeypatch):
    """sysctl -n hw.memsize returns bytes; _total_ram_gb converts."""
    monkeypatch.setattr(hardware.sys, "platform", "darwin")

    class R:
        returncode = 0
        stdout = "17179869184\n"   # 16 GiB
    monkeypatch.setattr(hardware.subprocess, "run",
                        lambda *a, **kw: R())
    assert abs(hardware._total_ram_gb() - 16.0) < 0.01


def test_total_ram_gb_macos_handles_sysctl_failure(monkeypatch):
    """sysctl failure or garbage → 0.0, no crash."""
    monkeypatch.setattr(hardware.sys, "platform", "darwin")

    class R:
        returncode = 1
        stdout = ""
    monkeypatch.setattr(hardware.subprocess, "run",
                        lambda *a, **kw: R())
    assert hardware._total_ram_gb() == 0.0


# ── Centralised PCI scan ──────────────────────────────────────────


def test_scan_pci_gpus_multi_gpu_prefers_discrete(monkeypatch, tmp_path):
    """A box with an NVIDIA discrete + Intel iGPU finds both; the
    discrete is usable, the iGPU is not. Discrete-first ordering."""
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)

    fake_drm = tmp_path / "drm"
    fake_drm.mkdir()
    nv = fake_drm / "card0" / "device"
    nv.mkdir(parents=True)
    (nv / "vendor").write_text("0x10de")
    (nv / "device").write_text("0x2684")
    (nv / "mem_info_vram_total").write_text(str(24 * 1024**3))
    igpu = fake_drm / "card1" / "device"
    igpu.mkdir(parents=True)
    (igpu / "vendor").write_text("0x8086")
    (igpu / "device").write_text("0x9a49")

    def fake_name(vendor, device):
        return {"0x10de": "NVIDIA RTX 4090", "0x8086": "Intel Arc iGPU"}.get(vendor)
    monkeypatch.setattr(hardware, "_pci_device_name", fake_name)
    monkeypatch.setattr(hardware, "Path", _DrmPathShim(fake_drm))

    gpus = hardware._scan_pci_gpus(hardware.Path("/sys/class/drm"), [])
    assert [g.vendor for g in gpus] == ["NVIDIA", "Intel"]
    assert gpus[0].use_for_llm is True
    assert gpus[1].use_for_llm is False


def test_scan_pci_gpus_dedup_against_vendor_tools(monkeypatch, tmp_path):
    """GPUs already found by nvidia-smi/rocm-smi (passed as `already`)
    are not duplicated by the PCI scan."""
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)

    fake_drm = tmp_path / "drm"
    fake_drm.mkdir()
    nv = fake_drm / "card0" / "device"
    nv.mkdir(parents=True)
    (nv / "vendor").write_text("0x10de")
    (nv / "device").write_text("0x2684")
    (nv / "mem_info_vram_total").write_text(str(24 * 1024**3))
    monkeypatch.setattr(hardware, "_pci_device_name", lambda *_: "NVIDIA RTX 4090")
    monkeypatch.setattr(hardware, "Path", _DrmPathShim(fake_drm))

    already = [GPUProfile(
        vendor="NVIDIA", name="NVIDIA RTX 4090", vram_gb=24.0,
        driver="555.42", vulkan_score=1.0, shared_ram=False,
        use_for_llm=True, reason="discrete GPU",
    )]
    gpus = hardware._scan_pci_gpus(hardware.Path("/sys/class/drm"), already)
    assert gpus == []   # already covered → no duplicate


# ── Windows ───────────────────────────────────────────────────────


def test_windows_returns_empty_with_warning(monkeypatch):
    """Windows without nvidia-smi returns [] — AMD/Intel fall back to
    CPU (safe, never wrong) until the WMI/DXGI stub lands."""
    monkeypatch.setattr(hardware.sys, "platform", "win32")
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)
    assert hardware.enumerate_gpus() == []


def test_windows_nvidia_via_nvidia_smi(monkeypatch):
    """Windows with nvidia-smi reports the NVIDIA GPU with VRAM."""
    monkeypatch.setattr(hardware.sys, "platform", "win32")
    monkeypatch.setattr(hardware.shutil, "which",
                        lambda name: "C:\\nvidia\\nvidia-smi.exe" if name == "nvidia-smi" else None)

    class R:
        returncode = 0
        stdout = "NVIDIA GeForce RTX 3080, 10240\n"
    monkeypatch.setattr(hardware.subprocess, "run", lambda *a, **kw: R())

    gpus = hardware.enumerate_gpus()
    nvidia = [g for g in gpus if g.vendor == "NVIDIA"]
    assert len(nvidia) == 1
    assert nvidia[0].vram_gb >= 10
    assert nvidia[0].use_for_llm is True


# ── Schema v2 → v3 migration ──────────────────────────────────────


def test_from_dict_v3_reads_core_max_freqs_mhz():
    """v3 profiles round-trip core_max_freqs_mhz — the freq-tier data
    the CPU thread-count policy needs."""
    data = {
        "schema_version": 3,
        "profiled_at": "2026-08-18T00:00:00Z",
        "os": "linux",
        "cpu": {
            "vendor": "GenuineIntel", "name": "Ultra 7 155U",
            "family": 6, "model": 189,
            "physical_cores": 12, "logical_cores": 14,
            "p_core_ids": [0, 1], "e_core_ids": list(range(2, 12)),
            "max_boost_mhz": 4800.0,
            "core_max_freqs_mhz": [4800.0, 4800.0] + [3800.0] * 10 + [1500.0, 1500.0],
            "simd": ["avx2", "avx_vnni"],
        },
        "gpus": [], "ram_gb": 16.0, "benchmarks": {}, "recommended": {},
    }
    prof = hardware._from_dict(data)
    assert len(prof.cpu.core_max_freqs_mhz) == 14
    assert hardware.effective_cpu_threads(prof.cpu) == 10


def test_from_dict_v2_profile_loads_tolerantly():
    """A cached v2 file (no core_max_freqs_mhz key) loads fine — the
    field defaults to [] and profile_box re-profiles on the version
    mismatch."""
    data = {
        "schema_version": 2,
        "profiled_at": "2026-01-01T00:00:00Z",
        "os": "linux",
        "cpu": {
            "vendor": "GenuineIntel", "name": "old",
            "family": 6, "model": 189,
            "physical_cores": 8, "logical_cores": 8,
            "p_core_ids": list(range(8)), "e_core_ids": [],
            "max_boost_mhz": 4000.0, "simd": ["avx2"],
        },
        "gpus": [], "ram_gb": 16.0, "benchmarks": {}, "recommended": {},
    }
    prof = hardware._from_dict(data)
    assert prof.schema_version == 2
    assert prof.cpu.core_max_freqs_mhz == []
    assert prof.cpu.name == "old"


def test_profile_box_migrates_v2_to_current_on_next_load(tmp_path, monkeypatch):
    """A cached v2 profile is re-profiled (and re-saved at the current
    schema version) on the next profile_box() call — the migration path.
    The target is PROFILE_VERSION, never a literal: schema bumps must
    not turn this test red."""
    monkeypatch.setattr(hardware, "profile_path", lambda: tmp_path / "hw.json")
    v2 = {
        "schema_version": 2,
        "profiled_at": "2026-08-18T00:00:00Z",
        "os": "linux",
        "cpu": {
            "vendor": "GenuineIntel", "name": "old",
            "family": 6, "model": 189,
            "physical_cores": 8, "logical_cores": 8,
            "p_core_ids": list(range(8)), "e_core_ids": [],
            "max_boost_mhz": 4000.0, "simd": ["avx2"],
        },
        "gpus": [], "ram_gb": 16.0, "benchmarks": {}, "recommended": {},
    }
    (tmp_path / "hw.json").write_text(json.dumps(v2))
    monkeypatch.setattr(hardware, "cpu_topology", lambda: CPUProfile(
        vendor="GenuineIntel", name="fresh",
        family=6, model=189, physical_cores=8, logical_cores=8,
        p_core_ids=list(range(8)), e_core_ids=[],
        max_boost_mhz=4000.0,
        core_max_freqs_mhz=[4000.0] * 8, simd=["avx2"],
    ))
    monkeypatch.setattr(hardware, "enumerate_gpus", lambda: [])
    monkeypatch.setattr(hardware, "_available_gb", lambda: 16.0)
    monkeypatch.setattr(hardware, "_recommend_defaults",
                        lambda *a, **k: {"backend": "cpu", "ngl": 0, "threads": 8})

    prof = hardware.profile_box()
    assert prof.cpu.name == "fresh"          # re-profiled, not the cached v2
    assert prof.schema_version == hardware.PROFILE_VERSION
    re_saved = json.loads((tmp_path / "hw.json").read_text())
    assert re_saved["schema_version"] == hardware.PROFILE_VERSION
    assert re_saved["cpu"]["core_max_freqs_mhz"] == [4000.0] * 8


def test_windows_gpu_enumeration_via_wmi(monkeypatch):
    """Windows AMD/Intel GPUs are found via wmic (AdapterRAM is bytes,
    0xffffffff sentinel = >4 GB). NVIDIA entries are skipped because
    nvidia-smi reports them more accurately; Intel UHD/HD/Iris names
    are marked shared-RAM and never use_for_llm."""
    wmic_csv = (
        "Node,AdapterRAM,Name\n"
        "DESKTOP,4293918720,NVIDIA GeForce RTX 4070\n"
        "DESKTOP,1073741824,Intel(R) UHD Graphics\n"
        "DESKTOP,8589934592,AMD Radeon RX 7600\n"
        "DESKTOP,,VirtualDisplay\n"
    )
    calls: list[list[str]] = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if cmd[0] == "wmic":
            return type("R", (), {"returncode": 0, "stdout": wmic_csv})()
        return type("R", (), {"returncode": 0, "stdout": ""})()

    monkeypatch.setattr(hardware.shutil, "which", lambda name: None if name == "nvidia-smi" else "/fake/wnode")
    monkeypatch.setattr(hardware.subprocess, "run", fake_run)
    monkeypatch.setattr(hardware.os.name, "nt", True) if False else None
    gpus = hardware._enumerate_gpus_windows()

    by_vendor = {g.vendor: g for g in gpus}
    # NVIDIA skipped (would come from nvidia-smi which is absent here,
    # but wmic's uint32 sentinel makes its VRAM a lie).
    assert "NVIDIA" not in by_vendor
    # Intel iGPU: shared RAM, never usable for LLM.
    intel = next(g for g in gpus if g.vendor == "Intel")
    assert "UHD" in intel.name
    assert intel.shared_ram is True
    assert intel.use_for_llm is False
    # AMD discrete with real VRAM: usable.
    amd = next(g for g in gpus if g.vendor == "AMD")
    assert "Radeon" in amd.name
    assert amd.shared_ram is False
    assert amd.use_for_llm is True


def test_windows_gpu_enumeration_powershell_fallback(monkeypatch):
    """When wmic is missing (Win11 22H2+ removed it), the PowerShell
    CIM query provides Name,AdapterRAM lines instead."""
    def fake_which(name):
        return None  # no nvidia-smi, no wmic on PATH

    ps_out = "AMD Radeon 8060S,8589934592\r\nIntel(R) Arc(tm) Graphics,2147483648\r\n"

    def fake_run(cmd, **kw):
        if cmd[0] == "powershell.exe":
            return type("R", (), {"returncode": 0, "stdout": ps_out})()
        return type("R", (), {"returncode": 1, "stdout": ""})()

    monkeypatch.setattr(hardware.shutil, "which", fake_which)
    monkeypatch.setattr(hardware.subprocess, "run", fake_run)
    gpus = hardware._enumerate_gpus_windows()
    vendors = sorted(g.vendor for g in gpus)
    assert vendors == ["AMD", "Intel"]


def test_windows_registry_vram_overrides_adapterram(monkeypatch):
    """AdapterRAM caps at 4 GB; the registry QWORD carries the real size.

    An RX 7900 XTX or Arc A770 must not be profiled as a 4 GB card, or
    every memory-fit decision (and the doctor's VRAM line) is wrong.
    """
    wmic_csv = (
        "Node,AdapterRAM,Name\n"
        "DESKTOP,4293918720,AMD Radeon RX 7900 XTX\n"
        "DESKTOP,4293918720,Intel(R) Arc(TM) A770 Graphics\n"
    )

    def fake_run(cmd, **kw):
        return type("R", (), {"returncode": 0, "stdout": wmic_csv})()

    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)
    monkeypatch.setattr(hardware.subprocess, "run", fake_run)
    monkeypatch.setattr(hardware, "_windows_registry_vram", lambda: {
        "amd radeon rx 7900 xtx": 24.0,
        "intel(r) arc(tm) a770 graphics": 16.0,
    })
    gpus = {g.name: g for g in hardware._enumerate_gpus_windows()}
    amd = gpus["AMD Radeon RX 7900 XTX"]
    arc = gpus["Intel(R) Arc(TM) A770 Graphics"]
    assert amd.vram_gb == 24.0
    assert arc.vram_gb == 16.0
    assert arc.shared_ram is False
    assert arc.use_for_llm is True


def test_windows_arc_igpu_is_shared_ram(monkeypatch):
    """Meteor Lake's iGPU is "Intel(R) Arc(TM) Graphics", not HD/UHD/Iris.

    A discrete Arc always carries a model number (A770, B580), so the
    bare phrase must be treated as an iGPU and kept on the CPU build.
    """
    wmic_csv = (
        "Node,AdapterRAM,Name\n"
        "DESKTOP,4293918720,Intel(R) Arc(TM) Graphics\n"
    )

    def fake_run(cmd, **kw):
        return type("R", (), {"returncode": 0, "stdout": wmic_csv})()

    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)
    monkeypatch.setattr(hardware.subprocess, "run", fake_run)
    monkeypatch.setattr(hardware, "_windows_registry_vram", lambda: {})
    arc = hardware._enumerate_gpus_windows()[0]
    assert arc.shared_ram is True
    assert arc.use_for_llm is False


def test_windows_registry_vram_reads_display_class(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("winreg")
    fake.HKEY_LOCAL_MACHINE = object()

    class Key:
        def __init__(self, sub):
            self.sub = sub

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    fake.OpenKey = lambda root, sub: Key(sub)
    fake.QueryInfoKey = lambda key: (1, 0, 0)
    fake.EnumKey = lambda key, i: "0000"

    def query_value(key, name):
        if name == "DriverDesc":
            return "AMD Radeon RX 7600", 1
        if name == "HardwareInformation.qwMemorySize":
            return 8589934592, 1
        raise FileNotFoundError(name)

    fake.QueryValueEx = query_value
    monkeypatch.setitem(sys.modules, "winreg", fake)
    monkeypatch.setattr(hardware.sys, "platform", "win32")
    assert hardware._windows_registry_vram() == {"amd radeon rx 7600": 8.0}


def test_registry_vram_matches_adapter_name():
    table = {"intel(r) arc(tm) a770 graphics": 16.0}
    assert hardware._registry_vram_for(
        "Intel(R) Arc(TM) A770 Graphics", table) == 16.0
    assert hardware._registry_vram_for(
        "Intel(R) Arc(TM) A770 Graphics (driver 32.0)", table) == 16.0
    assert hardware._registry_vram_for("NVIDIA GeForce RTX 4070", table) is None


def test_intel_lunar_lake_igpu_is_not_called_a_discrete_gpu(monkeypatch, tmp_path):
    """A Core Ultra 155U (Lunar Lake, 0x7d45) has no dedicated VRAM.

    It used to be described as "discrete GPU" with 0 GB, because the
    VRAM clause was skipped when the number was zero — and that is how
    a box with a working Vulkan build was told it had a CPU-only GPU.
    """
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)
    fake_drm = tmp_path / "drm"
    fake_drm.mkdir()
    card = fake_drm / "card0" / "device"
    card.mkdir(parents=True)
    (card / "vendor").write_text("0x8086")
    (card / "device").write_text("0x7d45")     # Lunar Lake Arc 140V iGPU
    (card / "mem_info_vram_total").write_text("0")
    monkeypatch.setattr(hardware, "_pci_device_name", lambda *_: "Intel Arc 140V")
    monkeypatch.setattr(hardware, "Path", _DrmPathShim(fake_drm))

    gpus = hardware.enumerate_gpus()
    igpu = [g for g in gpus if g.vendor == "Intel"]
    assert len(igpu) == 1
    assert igpu[0].shared_ram is True
    assert igpu[0].use_for_llm is False
    assert "discrete GPU" not in igpu[0].reason
    assert "integrated" in igpu[0].reason


def test_gpu_reason_names_an_intel_gpu_without_vram_as_integrated():
    assert "integrated GPU" in hardware._gpu_reason("Intel", 0.0, False)
    assert "discrete GPU" not in hardware._gpu_reason("Intel", 0.0, False)
    # A discrete Arc card with real VRAM is still a discrete GPU.
    assert hardware._gpu_reason("Intel", 16.0, False) == "discrete GPU"


def test_free_ram_reads_meminfo_in_gigabytes():
    """/proc/meminfo reports kibibytes. Dividing by 1024**3 made every
    number a thousand times too small, so an 8 GB box reported 0.008 GB
    and the profile printed "ram=0GB"."""
    from tokirecur.hardware import _ram_from_meminfo

    text = "MemTotal:    33554432 kB\nMemAvailable:  8388608 kB\n"
    assert _ram_from_meminfo(text) == 8.0


def test_free_ram_is_never_zero():
    """MemAvailable reported as zero (or missing) must not size every
    KV and thread decision to nothing."""
    from tokirecur.hardware import _ram_from_meminfo

    zero_available = "MemTotal:    33554432 kB\nMemAvailable:        0 kB\n"
    assert _ram_from_meminfo(zero_available) == 16.0
    assert _ram_from_meminfo("MemTotal:    33554432 kB\n") == 16.0
    assert _ram_from_meminfo("") == 0.0


def test_uma_amd_with_a_small_carve_is_judged_on_the_pool(monkeypatch, tmp_path):
    """The firmware reserves a slice (0.5 GB on a Strix Halo) and the
    sysfs figure is that slice, not the budget. Judging eligibility by
    it is how a Ryzen AI box that runs 35B models on the iGPU was told
    it had no usable GPU and handed the CPU build."""
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)
    fake_drm = tmp_path / "drm"
    fake_drm.mkdir()
    card = fake_drm / "card0" / "device"
    card.mkdir(parents=True)
    (card / "vendor").write_text("0x1002")
    (card / "device").write_text("0x1586")
    (card / "mem_info_vram_total").write_text(str(int(0.5 * 1024**3)))
    monkeypatch.setattr(hardware, "_pci_device_name", lambda *_: "Radeon 8060S")
    monkeypatch.setattr(hardware, "Path", _DrmPathShim(fake_drm))
    monkeypatch.setattr(hardware, "_total_ram_gb", lambda: 125.0)

    gpus = hardware.enumerate_gpus()
    amd = [g for g in gpus if g.vendor == "AMD"]
    assert len(amd) == 1
    assert amd[0].vram_gb == 125.0
    assert amd[0].use_for_llm is True
    assert "carve (0.5 GB reserved)" in amd[0].reason
    assert "unified memory" in amd[0].reason
    assert hardware._gpu_eligible(amd[0]) is True


def test_uma_amd_on_linux_recommends_rocm(monkeypatch, tmp_path):
    """And the recommendation follows: the GPU the box actually uses
    is the one the profile names."""
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)
    fake_drm = tmp_path / "drm"
    fake_drm.mkdir()
    card = fake_drm / "card0" / "device"
    card.mkdir(parents=True)
    (card / "vendor").write_text("0x1002")
    (card / "device").write_text("0x1586")
    (card / "mem_info_vram_total").write_text(str(int(0.5 * 1024**3)))
    monkeypatch.setattr(hardware, "_pci_device_name", lambda *_: "Radeon 8060S")
    monkeypatch.setattr(hardware, "Path", _DrmPathShim(fake_drm))
    monkeypatch.setattr(hardware, "_total_ram_gb", lambda: 125.0)

    cpu = CPUProfile(
        vendor="AuthenticAMD", name="RYZEN AI MAX+ 395",
        family=26, model=112, physical_cores=16, logical_cores=32,
        p_core_ids=list(range(16)), e_core_ids=[],
        max_boost_mhz=5187.0, simd=["avx512"],
    )
    rec = hardware._recommend_defaults(cpu, hardware.enumerate_gpus())
    assert rec["backend"] == "rocm"
    assert rec["ngl"] == 99
