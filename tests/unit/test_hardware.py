"""Unit tests for hardware.py profiling.

The hardware module does real I/O (reads /proc/cpuinfo, /sys/class/drm,
spawns nvidia-smi). Tests stub those paths and verify the structured
output shape is correct.
"""

from __future__ import annotations

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


def test_recommend_defaults_picks_cuda_for_discrete_nvidia():
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


def test_recommend_defaults_picks_metal_for_apple_silicon():
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
