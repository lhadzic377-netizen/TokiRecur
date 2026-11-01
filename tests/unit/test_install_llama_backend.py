"""Unit tests for `toki model install-llama` backend/asset selection.

Covers: backend auto-detection, the OS/arch -> asset pattern map, the
newest-release walk, ROCm env wiring, ETXTBSY-safe binary replace,
and retry-on-disconnect for the streaming download. No network calls
and no real processes are spawned.
"""

import os
from pathlib import Path

import pytest
import yaml

from tokirecur.cli import _asset_patterns, _pick_asset, _rocm_env

# Asset names taken from real llama.cpp releases (b10289 and b10375) so
# the map is tested against reality, not invented names. b10289 used
# win-hip-radeon-x64; b10375 renamed it win-rocm-7.14-x64.
B10289_ASSETS = [
    "llama-b10289-bin-android-arm64.tar.gz",
    "llama-b10289-bin-macos-arm64.tar.gz",
    "llama-b10289-bin-macos-x64.tar.gz",
    "llama-b10289-bin-ubuntu-arm64.tar.gz",
    "llama-b10289-bin-ubuntu-openvino-2026.2.1-x64.tar.gz",
    "llama-b10289-bin-ubuntu-rocm-7.2-x64.tar.gz",
    "llama-b10289-bin-ubuntu-s390x.tar.gz",
    "llama-b10289-bin-ubuntu-sycl-fp16-x64.tar.gz",
    "llama-b10289-bin-ubuntu-sycl-fp32-x64.tar.gz",
    "llama-b10289-bin-ubuntu-vulkan-arm64.tar.gz",
    "llama-b10289-bin-ubuntu-vulkan-x64.tar.gz",
    "llama-b10289-bin-ubuntu-x64.tar.gz",
    "llama-b10289-bin-win-cpu-arm64.zip",
    "llama-b10289-bin-win-cpu-x64.zip",
    "llama-b10289-bin-win-cuda-12.4-x64.zip",
    "llama-b10289-bin-win-cuda-13.3-x64.zip",
    "llama-b10289-bin-win-hip-radeon-x64.zip",
    "llama-b10289-bin-win-opencl-adreno-arm64.zip",
    "llama-b10289-bin-win-sycl-x64.zip",
    "llama-b10289-bin-win-vulkan-x64.zip",
]

# Asset names from b10375, which introduced the current win-rocm-* naming
# and still shipped Linux ROCm. Later releases dropped Linux ROCm until
# ROCm 10.0 resumed it in b12024.
B10375_ASSETS = [
    "llama-b10375-bin-ubuntu-rocm-7.14-x64.tar.gz",
    "llama-b10375-bin-ubuntu-vulkan-x64.tar.gz",
    "llama-b10375-bin-win-rocm-7.14-x64.zip",
    "llama-b10375-bin-win-vulkan-x64.zip",
]

# b12024 is the matrix release: Linux CUDA on x64 and arm64 with a
# separate cudart archive, Linux SYCL (fp16/fp32), OpenVINO on both
# Windows and Linux, Windows ARM64 CUDA, and ROCm 10.0 on both OSes.
B12024_ASSETS = [
    "cudart-llama-b12024-bin-ubuntu-cuda-12.8-x64.tar.gz",
    "cudart-llama-b12024-bin-ubuntu-cuda-13.4-arm64.tar.gz",
    "cudart-llama-b12024-bin-ubuntu-cuda-13.4-x64.tar.gz",
    "cudart-llama-bin-win-cuda-12.4-x64.zip",
    "cudart-llama-bin-win-cuda-13.4-arm64.zip",
    "cudart-llama-bin-win-cuda-13.4-x64.zip",
    "llama-b12024-bin-macos-arm64.tar.gz",
    "llama-b12024-bin-macos-x64.tar.gz",
    "llama-b12024-bin-ubuntu-arm64.tar.gz",
    "llama-b12024-bin-ubuntu-cuda-12.8-x64.tar.gz",
    "llama-b12024-bin-ubuntu-cuda-13.4-arm64.tar.gz",
    "llama-b12024-bin-ubuntu-cuda-13.4-x64.tar.gz",
    "llama-b12024-bin-ubuntu-openvino-2026.4.1-x64.tar.gz",
    "llama-b12024-bin-ubuntu-rocm-10.0-x64.tar.gz",
    "llama-b12024-bin-ubuntu-sycl-fp16-x64.tar.gz",
    "llama-b12024-bin-ubuntu-sycl-fp32-x64.tar.gz",
    "llama-b12024-bin-ubuntu-vulkan-arm64.tar.gz",
    "llama-b12024-bin-ubuntu-vulkan-x64.tar.gz",
    "llama-b12024-bin-ubuntu-x64.tar.gz",
    "llama-b12024-bin-win-cpu-arm64.zip",
    "llama-b12024-bin-win-cpu-x64.zip",
    "llama-b12024-bin-win-cuda-12.4-x64.zip",
    "llama-b12024-bin-win-cuda-13.4-arm64.zip",
    "llama-b12024-bin-win-cuda-13.4-x64.zip",
    "llama-b12024-bin-win-openvino-2026.4.1-x64.zip",
    "llama-b12024-bin-win-rocm-10.0-x64.zip",
    "llama-b12024-bin-win-sycl-x64.zip",
    "llama-b12024-bin-win-vulkan-arm64.zip",
    "llama-b12024-bin-win-vulkan-x64.zip",
]


def _assets() -> list[dict]:
    return [{"name": n, "size": 1000} for n in B10289_ASSETS]


@pytest.mark.parametrize(
    "os_name,arch,backend,expected",
    [
        ("ubuntu", "x64", "rocm", "llama-b10289-bin-ubuntu-rocm-7.2-x64.tar.gz"),
        ("ubuntu", "x64", "vulkan", "llama-b10289-bin-ubuntu-vulkan-x64.tar.gz"),
        ("ubuntu", "x64", "cpu", "llama-b10289-bin-ubuntu-x64.tar.gz"),
        ("ubuntu", "arm64", "vulkan", "llama-b10289-bin-ubuntu-vulkan-arm64.tar.gz"),
        ("ubuntu", "arm64", "cpu", "llama-b10289-bin-ubuntu-arm64.tar.gz"),
        ("windows", "x64", "cuda", "llama-b10289-bin-win-cuda-13.3-x64.zip"),
        ("windows", "x64", "rocm", "llama-b10289-bin-win-hip-radeon-x64.zip"),
        ("windows", "x64", "sycl", "llama-b10289-bin-win-sycl-x64.zip"),
        ("windows", "x64", "vulkan", "llama-b10289-bin-win-vulkan-x64.zip"),
        ("windows", "x64", "cpu", "llama-b10289-bin-win-cpu-x64.zip"),
        ("windows", "arm64", "cpu", "llama-b10289-bin-win-cpu-arm64.zip"),
        ("macos", "arm64", "metal", "llama-b10289-bin-macos-arm64.tar.gz"),
        ("macos", "x64", "metal", "llama-b10289-bin-macos-x64.tar.gz"),
    ],
)
def test_pick_asset_matrix(os_name, arch, backend, expected):
    patterns = _asset_patterns(os_name, arch)
    requested = [p for p in patterns if p[0] == backend]
    asset, matched = _pick_asset(_assets(), requested)
    assert asset is not None
    assert asset["name"] == expected
    assert matched == backend


@pytest.mark.parametrize(
    "os_name,arch,backend,expected",
    [
        ("ubuntu", "x64", "cuda", "llama-b12024-bin-ubuntu-cuda-13.4-x64.tar.gz"),
        ("ubuntu", "x64", "sycl", "llama-b12024-bin-ubuntu-sycl-fp16-x64.tar.gz"),
        ("ubuntu", "x64", "openvino",
         "llama-b12024-bin-ubuntu-openvino-2026.4.1-x64.tar.gz"),
        ("ubuntu", "x64", "rocm", "llama-b12024-bin-ubuntu-rocm-10.0-x64.tar.gz"),
        ("ubuntu", "x64", "vulkan", "llama-b12024-bin-ubuntu-vulkan-x64.tar.gz"),
        ("ubuntu", "x64", "cpu", "llama-b12024-bin-ubuntu-x64.tar.gz"),
        ("ubuntu", "arm64", "cuda",
         "llama-b12024-bin-ubuntu-cuda-13.4-arm64.tar.gz"),
        ("ubuntu", "arm64", "vulkan",
         "llama-b12024-bin-ubuntu-vulkan-arm64.tar.gz"),
        ("ubuntu", "arm64", "cpu", "llama-b12024-bin-ubuntu-arm64.tar.gz"),
        ("windows", "x64", "cuda", "llama-b12024-bin-win-cuda-13.4-x64.zip"),
        ("windows", "x64", "cuda", "llama-b12024-bin-win-cuda-13.4-x64.zip"),
        ("windows", "x64", "rocm", "llama-b12024-bin-win-rocm-10.0-x64.zip"),
        ("windows", "x64", "sycl", "llama-b12024-bin-win-sycl-x64.zip"),
        ("windows", "x64", "openvino",
         "llama-b12024-bin-win-openvino-2026.4.1-x64.zip"),
        ("windows", "x64", "vulkan", "llama-b12024-bin-win-vulkan-x64.zip"),
        ("windows", "x64", "cpu", "llama-b12024-bin-win-cpu-x64.zip"),
        ("windows", "arm64", "cuda",
         "llama-b12024-bin-win-cuda-13.4-arm64.zip"),
        ("windows", "arm64", "vulkan",
         "llama-b12024-bin-win-vulkan-arm64.zip"),
        ("windows", "arm64", "cpu", "llama-b12024-bin-win-cpu-arm64.zip"),
        ("macos", "arm64", "metal", "llama-b12024-bin-macos-arm64.tar.gz"),
        ("macos", "x64", "metal", "llama-b12024-bin-macos-x64.tar.gz"),
    ],
)
def test_pick_asset_b12024_matrix(os_name, arch, backend, expected):
    assets = [{"name": n, "size": 1000} for n in B12024_ASSETS]
    patterns = [p for p in _asset_patterns(os_name, arch) if p[0] == backend]
    asset, matched = _pick_asset(assets, patterns)
    assert asset is not None, f"no asset for {os_name}/{arch}/{backend}"
    assert asset["name"] == expected
    assert matched == backend


def test_pick_asset_b12024_prefers_cuda12_for_old_driver():
    from tokirecur.cli import _prefer_cuda12

    assets = [{"name": n, "size": 1000} for n in B12024_ASSETS]
    patterns = [p for p in _asset_patterns("windows", "x64") if p[0] == "cuda"]
    asset, matched = _pick_asset(assets, _prefer_cuda12(patterns))
    assert matched == "cuda"
    assert asset["name"] == "llama-b12024-bin-win-cuda-12.4-x64.zip"


def test_pick_asset_b12024_cudart_pairing():
    from tokirecur.cli import _find_cuda_runtime_asset

    assets = [{"name": n, "size": 1000} for n in B12024_ASSETS]
    pairs = {
        "llama-b12024-bin-ubuntu-cuda-13.4-x64.tar.gz":
            "cudart-llama-b12024-bin-ubuntu-cuda-13.4-x64.tar.gz",
        "llama-b12024-bin-ubuntu-cuda-13.4-arm64.tar.gz":
            "cudart-llama-b12024-bin-ubuntu-cuda-13.4-arm64.tar.gz",
        "llama-b12024-bin-ubuntu-cuda-12.8-x64.tar.gz":
            "cudart-llama-b12024-bin-ubuntu-cuda-12.8-x64.tar.gz",
        "llama-b12024-bin-win-cuda-13.4-x64.zip":
            "cudart-llama-bin-win-cuda-13.4-x64.zip",
        "llama-b12024-bin-win-cuda-12.4-x64.zip":
            "cudart-llama-bin-win-cuda-12.4-x64.zip",
        "llama-b12024-bin-win-cuda-13.4-arm64.zip":
            "cudart-llama-bin-win-cuda-13.4-arm64.zip",
    }
    for main, expected in pairs.items():
        got = _find_cuda_runtime_asset(assets, main)
        assert got is not None, main
        assert got["name"] == expected


def test_fallback_patterns_are_vendor_compatible():
    """A missing GPU asset must not fall through to another vendor."""
    from tokirecur.cli import _fallback_patterns

    arm = _asset_patterns("ubuntu", "arm64")
    rocm = _fallback_patterns(arm, "rocm")
    assert {b for b, _ in rocm} == {"vulkan", "cpu"}
    assert not any(b == "cuda" for b, _ in rocm)

    openvino = _fallback_patterns(arm, "openvino")
    assert {b for b, _ in openvino} == {"vulkan", "cpu"}

    # macOS has no CPU asset; an explicit CPU request keeps the Metal
    # patterns so the universal build lands with the CPU-only note.
    mac = _asset_patterns("macos", "arm64")
    assert _fallback_patterns(mac, "cpu") == mac


def test_pick_asset_windows_rocm_current_naming():
    """Post-b10375 releases name the Windows ROCm asset win-rocm-*; the
    pattern map must resolve it to the ROCm build (not fall through to
    Vulkan)."""
    assets = [{"name": n, "size": 1000} for n in B10375_ASSETS]
    asset, backend = _pick_asset(assets, _asset_patterns("windows", "x64"))
    assert backend == "rocm"
    assert asset["name"] == "llama-b10375-bin-win-rocm-7.14-x64.zip"


def test_pick_asset_prefers_rocm_over_cpu_when_both_present():
    assets = _assets() + [{"name": "llama-b99999-bin-ubuntu-x64.tar.gz", "size": 1}]
    asset, backend = _pick_asset(assets, _asset_patterns("ubuntu", "x64"))
    assert asset["name"].startswith("llama-b10289-bin-ubuntu-rocm-")
    assert backend == "rocm"


def test_pick_asset_fallback_when_requested_backend_missing():
    # No ROCm asset in this (mini) release -> Vulkan wins for x64.
    assets = [
        {"name": "llama-b9000-bin-ubuntu-vulkan-x64.tar.gz", "size": 1},
        {"name": "llama-b9000-bin-ubuntu-x64.tar.gz", "size": 1},
    ]
    asset, backend = _pick_asset(assets, _asset_patterns("ubuntu", "x64"))
    assert backend == "vulkan"
    assert asset["name"] == "llama-b9000-bin-ubuntu-vulkan-x64.tar.gz"


def test_pick_asset_none_when_no_match():
    assets = [{"name": "llama-b9000-bin-macos-arm64.tar.gz", "size": 1}]
    asset, backend = _pick_asset(assets, _asset_patterns("ubuntu", "x64"))
    assert asset is None
    assert backend is None


def test_pick_asset_skips_cudart_runtime_archives():
    # CUDA releases also ship cudart-llama-bin-* / cuda-cublas-llama-bin-*
    # runtime archives that match the same "win-cuda" fragment but contain
    # no llama-server.exe. The llama-<tag>-bin-* archive must win even when
    # the runtime archives sort first (GitHub lists them alphabetically).
    assets = [
        {"name": "cudart-llama-bin-win-cuda-13.3-x64.zip", "size": 390_000_000},
        {"name": "cuda-cublas-llama-bin-win-cuda-13.3-x64.zip", "size": 1_000_000},
        {"name": "llama-b10311-bin-win-cuda-13.3-x64.zip", "size": 140_000_000},
    ]
    asset, backend = _pick_asset(assets, _asset_patterns("windows", "x64"))
    assert backend == "cuda"
    assert asset["name"].startswith("llama-b10311-bin-")

    # And the runtime archives must NOT shadow any other backend's archive.
    assets = [
        {"name": "cudart-llama-bin-win-cuda-13.3-x64.zip", "size": 390_000_000},
        {"name": "llama-b10311-bin-win-vulkan-x64.zip", "size": 50_000_000},
    ]
    asset, backend = _pick_asset(assets, _asset_patterns("windows", "x64"))
    assert asset["name"] == "llama-b10311-bin-win-vulkan-x64.zip"


def test_pick_asset_never_picks_runtime_archive_alone():
    """A release with only the cudart runtime archive has no server."""
    assets = [
        {"name": "cudart-llama-bin-win-cuda-13.4-x64.zip", "size": 1},
    ]
    asset, backend = _pick_asset(assets, _asset_patterns("windows", "x64"))
    assert asset is None
    assert backend is None


def test_rocm_env_defaults():
    env = _rocm_env(None)
    assert env == {"GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1"}


def test_rocm_env_with_gfx():
    env = _rocm_env("11.5.1")
    assert env["HSA_OVERRIDE_GFX_VERSION"] == "11.5.1"


@pytest.mark.parametrize(
    "os_name,arch",
    [("ubuntu", "x64"), ("ubuntu", "arm64"), ("windows", "x64"),
     ("windows", "arm64"), ("macos", "arm64"), ("macos", "x64")],
)
def test_asset_patterns_cover_every_os_arch(os_name, arch):
    patterns = _asset_patterns(os_name, arch)
    assert patterns, f"no patterns for {os_name}/{arch}"
    for backend, frag in patterns:
        assert frag and backend in (
            "rocm", "cuda", "vulkan", "sycl", "openvino", "metal", "cpu",
        )


def test_set_binary_merges_env(tmp_path, monkeypatch):
    from tokirecur import ctl

    reg = tmp_path / "models.yaml"
    state = tmp_path / "state"
    marker = tmp_path / "active-model"
    reg.write_text(yaml.safe_dump({"models": ctl.DEFAULT_MODELS}, sort_keys=False))
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", reg)
    monkeypatch.setattr(ctl, "STATE_ROOT", state)
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", marker)

    spec = ctl.set_binary("9b", "/opt/bin/llama-server",
                          env={"HSA_OVERRIDE_GFX_VERSION": "11.5.1"})
    assert spec.binary == "/opt/bin/llama-server"
    assert spec.env["HSA_OVERRIDE_GFX_VERSION"] == "11.5.1"
    # Second call keeps existing keys.
    spec = ctl.set_binary("9b", "/opt/bin/llama-server",
                          env={"GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1"})
    assert spec.env["HSA_OVERRIDE_GFX_VERSION"] == "11.5.1"
    assert spec.env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] == "1"
    # No env passed -> untouched.
    spec = ctl.set_binary("9b", "/opt/bin/llama-server2")
    assert spec.env["HSA_OVERRIDE_GFX_VERSION"] == "11.5.1"


def _stub_profile(monkeypatch, *, backend="cpu", ngl=0, threads=4):
    """Stub the hardware profile so _detect_gpu_backend picks
    deterministically. Real profiling happens on a real machine."""
    from tokirecur.hardware import CPUProfile, HardwareProfile
    prof = HardwareProfile(
        schema_version=1,
        profiled_at="2026-01-01T00:00:00Z",
        os="linux",
        cpu=CPUProfile(
            vendor="GenuineIntel", name="stub", family=0, model=0,
            physical_cores=4, logical_cores=4,
            p_core_ids=[0, 1], e_core_ids=[],
            max_boost_mhz=0.0, simd=["avx2"],
        ),
        gpus=[],
        ram_gb=16.0,
        benchmarks={},
        recommended={"backend": backend, "ngl": ngl, "threads": threads},
    )
    # Force the helper to return our stub regardless of cache.
    monkeypatch.setattr(
        "tokirecur.hardware.profile_box",
        lambda force=False: prof,
    )
    # Ensure cached profile doesn't leak between tests.
    monkeypatch.setattr(
        "tokirecur.hardware.load_profile",
        lambda: None,
    )


def test_detect_backend_macos_always_metal(monkeypatch):
    from tokirecur.cli import _detect_gpu_backend
    _stub_profile(monkeypatch, backend="metal", ngl=99)
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert _detect_gpu_backend("macos") == "metal"


def test_detect_backend_linux_rocm_first(monkeypatch):
    from tokirecur.cli import _detect_gpu_backend
    _stub_profile(monkeypatch, backend="rocm", ngl=99)
    monkeypatch.setattr("shutil.which", lambda name: "hipconfig" if name == "hipconfig" else None)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    assert _detect_gpu_backend("ubuntu") == "rocm"


def test_detect_backend_linux_vulkan_then_cpu(monkeypatch):
    from tokirecur.cli import _detect_gpu_backend
    _stub_profile(monkeypatch, backend="vulkan", ngl=99)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    assert _detect_gpu_backend("ubuntu") == "vulkan"
    # Profile now says CPU — picks CPU.
    _stub_profile(monkeypatch, backend="cpu", ngl=0)
    assert _detect_gpu_backend("ubuntu") == "cpu"


def test_detect_backend_windows_cuda_then_cpu(monkeypatch):
    from tokirecur.cli import _detect_gpu_backend
    _stub_profile(monkeypatch, backend="cuda", ngl=99)
    assert _detect_gpu_backend("windows") == "cuda"
    _stub_profile(monkeypatch, backend="cpu", ngl=0)
    assert _detect_gpu_backend("windows") == "cpu"


def test_detect_backend_windows_intel_igpu_gets_cpu_build(monkeypatch):
    """The profile decides Arc iGPU boxes get CPU — even when vulkaninfo
    is present. The old vendor-based Intel check is gone; the profile
    is the only authority now."""
    from tokirecur.cli import _detect_gpu_backend
    _stub_profile(monkeypatch, backend="cpu", ngl=0)
    assert _detect_gpu_backend("windows") == "cpu"


def test_detect_backend_windows_non_intel_vulkan(monkeypatch):
    """A discrete-GPU Windows box keeps Vulkan."""
    from tokirecur.cli import _detect_gpu_backend
    _stub_profile(monkeypatch, backend="vulkan", ngl=99)
    assert _detect_gpu_backend("windows") == "vulkan"


@pytest.mark.parametrize(
    "main,cudart",
    [
        ("llama-b12020-bin-win-cuda-13.4-x64.zip",
         "cudart-llama-bin-win-cuda-13.4-x64.zip"),
        ("llama-b12020-bin-win-cuda-13.4-arm64.zip",
         "cudart-llama-bin-win-cuda-13.4-arm64.zip"),
        ("llama-b10326-bin-win-cuda-12.4-x64.zip",
         "cudart-llama-bin-win-cuda-12.4-x64.zip"),
        ("llama-b12024-bin-ubuntu-cuda-13.4-x64.tar.gz",
         "cudart-llama-b12024-bin-ubuntu-cuda-13.4-x64.tar.gz"),
        ("llama-b12024-bin-ubuntu-cuda-12.8-x64.tar.gz",
         "cudart-llama-b12024-bin-ubuntu-cuda-12.8-x64.tar.gz"),
        ("llama-b12024-bin-ubuntu-cuda-13.4-arm64.tar.gz",
         "cudart-llama-b12024-bin-ubuntu-cuda-13.4-arm64.tar.gz"),
        ("llama-b10326-bin-ubuntu-rocm-7.2-x64.tar.gz", None),
        ("llama-b10326-bin-win-cpu-x64.zip", None),
    ],
)
def test_cuda_runtime_archive_name(main, cudart):
    from tokirecur.cli import _cuda_runtime_archive_name
    assert _cuda_runtime_archive_name(main) == cudart


def test_find_cuda_runtime_asset_exact_and_fallback():
    from tokirecur.cli import _find_cuda_runtime_asset

    main = "llama-b12020-bin-win-cuda-13.4-x64.zip"
    assets = [
        {"name": main},
        {"name": "cudart-llama-bin-win-cuda-13.4-x64.zip"},
    ]
    got = _find_cuda_runtime_asset(assets, main)
    assert got["name"] == "cudart-llama-bin-win-cuda-13.4-x64.zip"

    # Renamed (build-numbered) runtime archive: same version + arch wins.
    assets[1]["name"] = "cudart-llama-b12020-bin-win-cuda-13.4-x64.zip"
    got = _find_cuda_runtime_asset(assets, main)
    assert got["name"].endswith("win-cuda-13.4-x64.zip")

    # Releases that predate the split have no runtime archive.
    assert _find_cuda_runtime_asset(assets[:1], main) is None

    # A runtime archive for a different CUDA major is not a substitute:
    # its DLLs are named cudart64_12.dll and the 13.x build can't load
    # them, so the server would silently fall back to CPU.
    wrong_major = [
        {"name": main},
        {"name": "cudart-llama-bin-win-cuda-12.4-x64.zip"},
    ]
    assert _find_cuda_runtime_asset(wrong_major, main) is None


def test_pick_asset_current_cuda_release_names():
    """b12020 ships cuda-13.4; the old hardcoded 13.3 fragment made
    install-llama miss CUDA entirely and fall back to Vulkan/CPU."""
    assets = [
        {"name": "cudart-llama-bin-win-cuda-13.4-x64.zip", "size": 1},
        {"name": "llama-b12020-bin-win-cuda-12.4-x64.zip", "size": 1},
        {"name": "llama-b12020-bin-win-cuda-13.4-x64.zip", "size": 1},
        {"name": "llama-b12020-bin-win-vulkan-x64.zip", "size": 1},
        {"name": "llama-b12020-bin-win-cpu-x64.zip", "size": 1},
    ]
    asset, backend = _pick_asset(assets, _asset_patterns("windows", "x64"))
    assert backend == "cuda"
    assert asset["name"] == "llama-b12020-bin-win-cuda-13.4-x64.zip"


def test_pick_asset_windows_arch_is_not_crossed():
    """The same CUDA name exists for x64 and arm64; never cross them."""
    assets = [
        {"name": "llama-b12020-bin-win-cuda-13.4-arm64.zip", "size": 1},
        {"name": "llama-b12020-bin-win-cuda-13.4-x64.zip", "size": 1},
    ]
    asset, _ = _pick_asset(assets, _asset_patterns("windows", "x64"))
    assert asset["name"].endswith("-x64.zip")
    asset, _ = _pick_asset(assets, _asset_patterns("windows", "arm64"))
    assert asset["name"].endswith("-arm64.zip")


def test_prefer_cuda12_for_old_driver():
    from tokirecur.cli import _prefer_cuda12
    assets = [
        {"name": "llama-b12020-bin-win-cuda-13.4-x64.zip", "size": 1},
        {"name": "llama-b12020-bin-win-cuda-12.4-x64.zip", "size": 1},
    ]
    patterns = _prefer_cuda12(_asset_patterns("windows", "x64"))
    asset, backend = _pick_asset(assets, patterns)
    assert backend == "cuda"
    assert asset["name"].endswith("cuda-12.4-x64.zip")


def test_pick_asset_future_cuda_minor_matches():
    assets = [
        {"name": "llama-b99999-bin-win-cuda-14.1-x64.zip", "size": 1},
    ]
    asset, backend = _pick_asset(assets, _asset_patterns("windows", "x64"))
    assert backend == "cuda"
    assert asset["name"].endswith("cuda-14.1-x64.zip")


def test_prune_cuda_runtime_major(tmp_path):
    from tokirecur.cli import _prune_cuda_runtime_major
    for name in (
        "cudart64_12.dll", "cublas64_12.dll", "cudart64_13.dll",
        "cublasLt64_13.dll", "ggml-cuda.dll",
    ):
        (tmp_path / name).write_bytes(b"x")
    _prune_cuda_runtime_major(tmp_path, "13")
    left = {p.name for p in tmp_path.iterdir()}
    assert left == {"cudart64_13.dll", "cublasLt64_13.dll", "ggml-cuda.dll"}


def test_prune_foreign_gpu_libs_cuda(tmp_path):
    from tokirecur.cli import _prune_foreign_gpu_libs
    for name in (
        "ggml-cuda.dll", "cudart64_13.dll", "cublas64_13.dll",
        "ggml-vulkan.dll", "ggml-hip.dll", "amdhip64_7.dll",
        "ggml-cpu-x64.dll", "llama-server.exe",
    ):
        (tmp_path / name).write_bytes(b"x")
    _prune_foreign_gpu_libs(tmp_path, "cuda")
    left = {p.name for p in tmp_path.iterdir()}
    assert left == {
        "ggml-cuda.dll", "cudart64_13.dll", "cublas64_13.dll",
        "ggml-cpu-x64.dll", "llama-server.exe",
    }


def test_prune_foreign_gpu_libs_vulkan_removes_cudart(tmp_path):
    from tokirecur.cli import _prune_foreign_gpu_libs
    for name in (
        "ggml-vulkan.dll", "ggml-cuda.dll", "cudart64_13.dll",
        "cublas64_13.dll", "ggml-cpu-x64.dll",
    ):
        (tmp_path / name).write_bytes(b"x")
    _prune_foreign_gpu_libs(tmp_path, "vulkan")
    left = {p.name for p in tmp_path.iterdir()}
    assert left == {"ggml-vulkan.dll", "ggml-cpu-x64.dll"}


def test_prune_foreign_gpu_libs_cpu_removes_all(tmp_path):
    from tokirecur.cli import _prune_foreign_gpu_libs
    for name in (
        "ggml-cuda.dll", "cudart64_13.dll", "ggml-vulkan.dll",
        "ggml-hip.dll", "amdhip64_7.dll", "ggml-sycl.dll",
        "sycl9.dll", "ggml-openvino.dll", "openvino.dll",
        "ggml-cpu-x64.dll",
    ):
        (tmp_path / name).write_bytes(b"x")
    _prune_foreign_gpu_libs(tmp_path, "cpu")
    left = {p.name for p in tmp_path.iterdir()}
    assert left == {"ggml-cpu-x64.dll"}


def test_prune_foreign_gpu_libs_sycl(tmp_path):
    from tokirecur.cli import _prune_foreign_gpu_libs
    for name in (
        "ggml-sycl.dll", "sycl9.dll", "mkl_sycl_blas.6.dll",
        "ur_loader.dll", "dnnl.dll", "ggml-vulkan.dll", "ggml-hip.dll",
        "amdhip64_7.dll", "ggml-openvino.dll", "openvino.dll",
        "tbb12.dll", "ggml-cpu-x64.dll",
    ):
        (tmp_path / name).write_bytes(b"x")
    _prune_foreign_gpu_libs(tmp_path, "sycl")
    left = {p.name for p in tmp_path.iterdir()}
    assert {"ggml-sycl.dll", "sycl9.dll", "tbb12.dll"} <= left
    assert "ggml-vulkan.dll" not in left
    assert "ggml-hip.dll" not in left
    assert "ggml-openvino.dll" not in left


def test_prune_foreign_gpu_libs_openvino(tmp_path):
    from tokirecur.cli import _prune_foreign_gpu_libs
    for name in (
        "ggml-openvino.dll", "openvino.dll",
        "openvino_intel_gpu_plugin.dll", "tbb12.dll", "tbbmalloc.dll",
        "ggml-sycl.dll", "sycl9.dll", "ggml-vulkan.dll",
        "ggml-cpu-x64.dll",
    ):
        (tmp_path / name).write_bytes(b"x")
    _prune_foreign_gpu_libs(tmp_path, "openvino")
    left = {p.name for p in tmp_path.iterdir()}
    assert {"ggml-openvino.dll", "openvino.dll", "tbb12.dll"} <= left
    assert "ggml-sycl.dll" not in left
    assert "sycl9.dll" not in left
    assert "ggml-vulkan.dll" not in left


def test_install_pulls_cudart_runtime_for_cuda_13_4(tmp_path, monkeypatch):
    """Full CUDA install flow: main archive + cudart runtime, stale Vulkan
    DLL pruned, runtime verified, and the downloaded zip not installed."""
    import io
    import types
    import zipfile

    from tokirecur import cli as cli_mod
    from tokirecur.cli import install_llama

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ggml-vulkan.dll").write_bytes(b"old")
    (bin_dir / "cudart64_12.dll").write_bytes(b"old12")
    (bin_dir / "llama-server.exe").write_bytes(b"old")

    def make_zip(files):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for name, data in files.items():
                zf.writestr(name, data)
        return buf.getvalue()

    main_zip = make_zip({
        "llama-server.exe": b"MZ",
        "ggml-cuda.dll": b"cuda",
        "ggml-base.dll": b"base",
    })
    cudart_zip = make_zip({
        "cudart64_13.dll": b"rt",
        "cublas64_13.dll": b"blas",
    })

    RELEASE = {
        "tag_name": "b12020",
        "assets": [
            {"name": "llama-b12020-bin-win-cuda-13.4-x64.zip",
             "size": len(main_zip),
             "browser_download_url": "https://example.test/main.zip"},
            {"name": "cudart-llama-bin-win-cuda-13.4-x64.zip",
             "size": len(cudart_zip),
             "browser_download_url": "https://example.test/cudart.zip"},
        ],
    }

    class FakeClient:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False

        def get(self, url, headers=None):
            class _Resp:
                def json(self_inner): return RELEASE
                def raise_for_status(self_inner): pass
            return _Resp()

    downloads: list[str] = []

    def fake_download(api, url, dest, label=""):
        downloads.append(url)
        Path(dest).write_bytes(
            main_zip if url.endswith("main.zip") else cudart_zip
        )

    def fake_run(argv, **kw):
        if "--list-devices" in argv:
            return types.SimpleNamespace(
                returncode=0,
                stdout="Available devices:\n  CUDA0: RTX 4090\n",
                stderr="",
            )
        return types.SimpleNamespace(returncode=0, stdout="v1\n", stderr="")

    import httpx as _httpx
    monkeypatch.setattr(_httpx, "Client", FakeClient)
    monkeypatch.setattr(cli_mod, "_stream_download_with_retry", fake_download)
    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr("platform.machine", lambda: "AMD64")
    monkeypatch.setattr(
        "shutil.which", lambda n: "nvidia-smi" if n == "nvidia-smi" else None,
    )
    from tokirecur import diagnose
    monkeypatch.setattr(
        diagnose, "cuda_driver_verdict", lambda d, os_name="windows": (False, False),
    )

    install_llama.callback(tag="b12020", dest=str(bin_dir), name=None,
                           backend="cuda", gfx=None, profile=False,
                           force_cpu=False)

    assert any("cudart.zip" in u for u in downloads)
    names = {p.name for p in bin_dir.iterdir()}
    assert {"llama-server.exe", "ggml-cuda.dll", "cudart64_13.dll",
            "cublas64_13.dll"} <= names
    assert "ggml-vulkan.dll" not in names, "stale Vulkan DLL survived"
    assert "cudart64_12.dll" not in names, "stale CUDA 12 runtime survived"
    assert not any(n.endswith(".zip") for n in names), (
        "the downloaded archive was installed into bin_dir"
    )
    assert diagnose.binary_backend(
        str(bin_dir / "llama-server.exe")
    ) == "CUDA"


def _tar_bytes(files: dict[str, bytes]) -> bytes:
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_install_linux_cuda_pulls_paired_cudart(tmp_path, monkeypatch):
    """Linux CUDA ships cudart/cublas as a separate archive too. The
    install must fetch it, drop the stale major, and verify a real
    CUDA device line — not leave a build that silently runs on CPU."""
    import types

    from tokirecur import cli as cli_mod
    from tokirecur import diagnose
    from tokirecur.cli import install_llama

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "llama-server").write_bytes(b"old")
    (bin_dir / "libggml-vulkan.so").write_bytes(b"old-vk")
    (bin_dir / "libcudart.so.12").write_bytes(b"old-12")

    main_tar = _tar_bytes({
        "build/bin/llama-server": b"ELF",
        "build/bin/libggml-cuda.so": b"cuda",
        "build/bin/libggml-base.so": b"base",
    })
    cudart_tar = _tar_bytes({
        "lib/libcudart.so.13": b"rt",
        "lib/libcublas.so.13": b"blas",
        "lib/libcublasLt.so.13": b"blaslt",
    })
    RELEASE = {
        "tag_name": "b12024",
        "assets": [
            {"name": "llama-b12024-bin-ubuntu-cuda-13.4-x64.tar.gz",
             "size": len(main_tar),
             "browser_download_url": "https://example.test/main.tar.gz"},
            {"name": "cudart-llama-b12024-bin-ubuntu-cuda-13.4-x64.tar.gz",
             "size": len(cudart_tar),
             "browser_download_url": "https://example.test/cudart.tar.gz"},
        ],
    }

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def get(self, url, headers=None):
            class _Resp:
                def json(self_inner):
                    return RELEASE

                def raise_for_status(self_inner):
                    pass

            return _Resp()

    downloads: list[str] = []

    def fake_download(api, url, dest, label=""):
        downloads.append(url)
        Path(dest).write_bytes(
            main_tar if url.endswith("main.tar.gz") else cudart_tar
        )

    def fake_run(argv, **kw):
        if "--list-devices" in argv:
            return types.SimpleNamespace(
                returncode=0,
                stdout="Available devices:\n  CUDA0: NVIDIA RTX 4090\n",
                stderr="",
            )
        return types.SimpleNamespace(returncode=0, stdout="v1\n", stderr="")

    import httpx as _httpx
    monkeypatch.setattr(_httpx, "Client", FakeClient)
    monkeypatch.setattr(cli_mod, "_stream_download_with_retry", fake_download)
    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    monkeypatch.setattr("shutil.which", lambda n: None)
    monkeypatch.setattr(cli_mod, "_find_local_llama_server", lambda: None)
    monkeypatch.setattr(
        diagnose, "cuda_driver_verdict", lambda d, os_name="windows": (False, False),
    )
    monkeypatch.setattr(diagnose, "driver_version", lambda: (580, 65))

    install_llama.callback(
        tag="b12024", dest=str(bin_dir), name=None, backend="cuda",
        gfx=None, profile=False, force_cpu=False,
    )

    assert any("cudart.tar.gz" in u for u in downloads)
    names = {p.name for p in bin_dir.iterdir()}
    assert {"llama-server", "libggml-cuda.so", "libcudart.so.13",
            "libcublas.so.13", "libcublasLt.so.13"} <= names
    assert "libggml-vulkan.so" not in names, "stale Vulkan module survived"
    assert "libcudart.so.12" not in names, "stale CUDA 12 runtime survived"
    assert diagnose.binary_backend(str(bin_dir / "llama-server")) == "CUDA"


def test_install_linux_sycl_requires_oneapi_runtime(tmp_path, monkeypatch):
    """Linux SYCL is not self-contained: without oneAPI the install must
    fail loudly instead of leaving a CPU fallback."""
    from tokirecur import cli as cli_mod
    from tokirecur import diagnose
    from tokirecur.cli import install_llama

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    payload = _tar_bytes({
        "build/bin/llama-server": b"ELF",
        "build/bin/libggml-sycl.so": b"sycl",
    })
    RELEASE = {
        "tag_name": "b12024",
        "assets": [{
            "name": "llama-b12024-bin-ubuntu-sycl-fp16-x64.tar.gz",
            "size": len(payload),
            "browser_download_url": "https://example.test/sycl.tar.gz",
        }],
    }

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def get(self, url, headers=None):
            class _Resp:
                def json(self_inner):
                    return RELEASE

            return _Resp()

    def fake_download(api, url, dest, label=""):
        Path(dest).write_bytes(payload)

    monkeypatch.setattr(cli_mod.httpx, "Client", FakeClient)
    monkeypatch.setattr(cli_mod, "_stream_download_with_retry", fake_download)
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    monkeypatch.setattr("shutil.which", lambda n: None)
    monkeypatch.setattr(cli_mod, "_find_local_llama_server", lambda: None)
    monkeypatch.setattr(diagnose, "sycl_runtime_ready", lambda b: False)
    monkeypatch.setattr(
        diagnose, "sycl_runtime_missing",
        lambda b: ["libsycl.so (oneAPI DPC++ runtime)"],
    )

    try:
        install_llama.callback(
            tag="b12024", dest=str(bin_dir), name=None, backend="sycl",
            gfx=None, profile=False, force_cpu=False,
        )
    except Exception as exc:  # click.ClickException
        message = str(exc)
    else:
        raise AssertionError("SYCL install without oneAPI should fail")
    assert "oneAPI" in message
    assert "libsycl.so" in message


def test_install_linux_openvino_requires_gpu_plugin(tmp_path, monkeypatch):
    """OpenVINO core alone enumerates the CPU plugin. The install must
    refuse a build without the Intel GPU plugin."""
    from tokirecur import cli as cli_mod
    from tokirecur import diagnose
    from tokirecur.cli import install_llama

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    payload = _tar_bytes({
        "build/bin/llama-server": b"ELF",
        "build/bin/libggml-openvino.so": b"ov",
        "build/bin/libopenvino.so.2600": b"core",
    })
    RELEASE = {
        "tag_name": "b12024",
        "assets": [{
            "name": "llama-b12024-bin-ubuntu-openvino-2026.4.1-x64.tar.gz",
            "size": len(payload),
            "browser_download_url": "https://example.test/ov.tar.gz",
        }],
    }

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def get(self, url, headers=None):
            class _Resp:
                def json(self_inner):
                    return RELEASE

            return _Resp()

    def fake_download(api, url, dest, label=""):
        Path(dest).write_bytes(payload)

    monkeypatch.setattr(cli_mod.httpx, "Client", FakeClient)
    monkeypatch.setattr(cli_mod, "_stream_download_with_retry", fake_download)
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    monkeypatch.setattr("shutil.which", lambda n: None)
    monkeypatch.setattr(cli_mod, "_find_local_llama_server", lambda: None)
    monkeypatch.setattr(diagnose, "openvino_runtime_ready", lambda b: True)
    monkeypatch.setattr(diagnose, "openvino_gpu_plugin_ready", lambda b: False)

    try:
        install_llama.callback(
            tag="b12024", dest=str(bin_dir), name=None, backend="openvino",
            gfx=None, profile=False, force_cpu=False,
        )
    except Exception as exc:
        message = str(exc)
    else:
        raise AssertionError("OpenVINO install without GPU plugin should fail")
    assert "Intel GPU" in message or "GPU plugin" in message


def test_binary_backend_reports_mlx_for_module_name():
    from tokirecur import diagnose
    # Bare string identifier (no real file) is what the registry stores
    # when install-llama --backend mlx rewrites the binary field.
    assert diagnose.binary_backend("mlx_lm.server") == "MLX"


def test_binary_backend_reports_mlx_for_path_to_shim(tmp_path):
    from tokirecur import diagnose
    shim = tmp_path / "mlx_lm.server"
    shim.write_text("#!/bin/sh\nexec python -m mlx_lm.server \"$@\"\n")
    assert diagnose.binary_backend(str(shim)) == "MLX"


def test_mlx_backend_choice_in_install_llama():
    """--backend mlx is now a valid choice for install-llama."""
    from click.testing import CliRunner

    from tokirecur.cli import cli
    runner = CliRunner()
    # Just verify the choice is accepted (--help would also work but
    # this avoids printing all the help text).
    res = runner.invoke(cli, ["model", "install-llama", "--help"])
    assert "mlx" in res.output


def test_mlx_backend_rejected_on_linux(monkeypatch):
    """--backend mlx must refuse to run off macOS."""
    from click.testing import CliRunner

    from tokirecur.cli import cli as _cli

    monkeypatch.setattr("sys.platform", "linux")

    runner = CliRunner()
    res = runner.invoke(_cli, ["model", "install-llama", "--backend", "mlx"])
    assert res.exit_code != 0
    assert "mlx is only available on macOS" in res.output


@pytest.mark.parametrize(
    "platform,arch,backend,needle",
    [
        ("darwin", "arm64", "cuda", "only available on Windows and Linux"),
        ("darwin", "arm64", "vulkan", "not available on macOS"),
        ("darwin", "arm64", "sycl", "only available on x64 Windows and Linux"),
        ("darwin", "arm64", "openvino",
         "only available on x64 Windows and Linux"),
        ("linux", "aarch64", "sycl", "only available on x64 Windows and Linux"),
        ("linux", "aarch64", "openvino",
         "only available on x64 Windows and Linux"),
    ],
)
def test_install_backend_guards_reject_impossible_assets(
    monkeypatch, platform, arch, backend, needle,
):
    from click.testing import CliRunner

    from tokirecur.cli import cli as _cli

    monkeypatch.setattr("sys.platform", platform)
    monkeypatch.setattr("platform.machine", lambda: arch)
    res = CliRunner().invoke(
        _cli, ["model", "install-llama", "--backend", backend],
    )
    assert res.exit_code != 0
    assert needle in res.output


def test_llama_args_use_mlx_for_mlx_binary():
    """When spec.binary is mlx_lm.server, _llama_args should build
    the mlx_lm.server invocation, not the llama-server one."""
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    from tokirecur.ctl import _llama_args

    model_dir = Path("/tmp/fake-mlx-model")
    spec = SimpleNamespace(
        name="mlx-test", gguf=str(model_dir),
        llama_port=8091, ctx_size=4096, parallel=1,
        kv="f16", reasoning="default", extra_args=[],
        binary="mlx_lm.server",
    )
    # Pretend the model directory exists so the function doesn't bail.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        spec.gguf = td
        args = _llama_args(spec)
    assert args[:3] == [sys.executable, "-m", "mlx_lm.server"]
    assert "--model" in args
    assert td in args
    assert "--port" in args
    assert "8091" in args
    # And NOT the llama-server flag set:
    assert "-c" not in args
    assert "-ngl" not in args


# ── install-llama: ETXTBSY-safe replace + retry-on-disconnect ─────────────


def test_safe_copy_replace_renames_existing_dst_before_writing(tmp_path, monkeypatch):
    """When the destination exists, ``_safe_copy_replace`` must rename
    it aside first so the kernel doesn't refuse the write with
    ETXTBSY (errno 26) — that's the fix for the "Text file busy" traceback
    users hit when an old llama-server is still running."""
    from tokirecur.cli import _safe_copy_replace

    src = tmp_path / "src.bin"
    src.write_bytes(b"new-content")
    dst = tmp_path / "dst.bin"
    dst.write_bytes(b"old-content-being-executed")

    # Capture the rename order to prove the rename happens BEFORE the copy.
    import shutil as _shutil
    events: list[str] = []
    real_rename = Path.rename

    def tracking_rename(self, target):
        events.append(f"rename:{self.name}")
        return real_rename(self, target)

    real_copy2 = _shutil.copy2

    def tracking_copy2(s, d):
        events.append(f"copy:{Path(s).name}->{Path(d).name}")
        return real_copy2(s, d)

    monkeypatch.setattr(Path, "rename", tracking_rename)
    monkeypatch.setattr(_shutil, "copy2", tracking_copy2)

    _safe_copy_replace(src, dst)

    # Rename happened first, then copy.
    assert events[0].startswith("rename:"), events
    assert any(e.startswith("copy:") for e in events), events
    # The renamed-aside file still has the old content (proves rename
    # actually moved the old bytes, not just the new ones).
    backup = dst.with_name(dst.name + f".stale.{os.getpid()}")
    assert backup.exists()
    assert backup.read_bytes() == b"old-content-being-executed"
    # The destination now has the new bytes.
    assert dst.read_bytes() == b"new-content"


def test_safe_copy_replace_no_existing_dst(tmp_path):
    """When the destination doesn't exist yet (fresh install), copy
    directly — no rename, no backup."""
    from tokirecur.cli import _safe_copy_replace

    src = tmp_path / "src.bin"
    src.write_bytes(b"fresh")
    dst = tmp_path / "dst.bin"

    _safe_copy_replace(src, dst)
    assert dst.read_bytes() == b"fresh"
    assert not list(tmp_path.glob("dst.bin.stale.*"))


def test_sweep_stale_backups_removes_dead_pid(tmp_path):
    """``.stale.<pid>`` backups whose owning pid is no longer alive
    get cleaned up; live-pid orphans stay so the running daemon's mmap
    isn't broken."""
    from tokirecur.cli import _sweep_stale_backups

    # Pick a pid that almost certainly isn't alive on the test box.
    dead_pid = 2_000_000_000  # way past max valid pid on Linux
    backup = tmp_path / "llama-server.stale.dead"
    backup.write_bytes(b"orphan")

    # Rename to use the dead-pid suffix.
    target = tmp_path / f"llama-server.stale.{dead_pid}"
    backup.rename(target)

    _sweep_stale_backups(tmp_path, "llama-server")
    assert not target.exists()


def test_sweep_stale_backups_keeps_live_pid(tmp_path):
    """A backup whose owning pid is alive (the current test process)
    stays on disk — deleting it would break the process's mmap."""
    from tokirecur.cli import _sweep_stale_backups

    live_pid = os.getpid()
    target = tmp_path / f"llama-server.stale.{live_pid}"
    target.write_bytes(b"live")

    _sweep_stale_backups(tmp_path, "llama-server")
    assert target.exists()
    target.unlink()  # tidy up


def test_stream_download_with_retry_recovers_after_remote_protocol_error(tmp_path, monkeypatch):
    """A transient RemoteProtocolError on the first attempt should
    trigger a retry; the second attempt succeeds."""
    import httpx

    from tokirecur import cli as cli_mod
    from tokirecur.cli import _stream_download_with_retry

    dest = tmp_path / "blob"
    attempts: list[int] = []

    class FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def iter_bytes(self, chunk_size):
            # Always yield a single small chunk.
            yield b"ok"

    class FakeStreamCtx:
        def __enter__(self): return FakeResp()
        def __exit__(self, *a): return False

    class FakeApi:
        def stream(self, method, url):
            attempts.append(len(attempts) + 1)
            if len(attempts) == 1:
                # First attempt: stream tears down mid-body.
                class BadResp:
                    def raise_for_status(self): pass
                    def iter_bytes(self, chunk_size):
                        raise httpx.RemoteProtocolError(
                            "Server disconnected without sending a response."
                        )
                return BadResp()  # Not a context manager — caller will get
                                  # the iter_bytes exception directly.
            return FakeStreamCtx()

    # Make the first FakeApi.stream return an object whose iter_bytes
    # raises (no __enter__ needed since api.stream is called with `as`).
    class BadApi:
        def stream(self, method, url):
            attempts.append(len(attempts) + 1)
            if len(attempts) == 1:
                class R:
                    def __enter__(self): return self
                    def __exit__(self, *a): return False
                    def raise_for_status(self): pass
                    def iter_bytes(self, chunk_size):
                        raise httpx.RemoteProtocolError(
                            "Server disconnected without sending a response."
                        )
                return R()
            return FakeStreamCtx()

    # Speed up retry sleeps in tests.
    monkeypatch.setattr(cli_mod, "_DOWNLOAD_ATTEMPTS", 3)
    monkeypatch.setattr(cli_mod.time, "sleep", lambda s: None)

    _stream_download_with_retry(BadApi(), "https://example.test/x.tar.gz", dest)
    assert attempts == [1, 2]
    assert dest.read_bytes() == b"ok"


def test_stream_download_with_retry_exhausts_attempts_then_raises(tmp_path, monkeypatch):
    """If every attempt fails with a retryable error, surface a single
    ClickException with the actionable hint — not the raw 60-line
    httpx/httpcore traceback the old code raised."""
    import httpx
    from click.exceptions import ClickException

    from tokirecur import cli as cli_mod
    from tokirecur.cli import _stream_download_with_retry

    dest = tmp_path / "blob"

    # Force the curl path to fail so we exercise the httpx fallback.
    monkeypatch.setattr(cli_mod.shutil, "which", lambda name: None)

    class BadApi:
        def stream(self, method, url):
            class R:
                def __enter__(self): return self
                def __exit__(self, *a): return False
                def raise_for_status(self): pass
                def iter_bytes(self, chunk_size):
                    raise httpx.RemoteProtocolError(
                        "Server disconnected without sending a response."
                    )
            return R()

    monkeypatch.setattr(cli_mod, "_DOWNLOAD_ATTEMPTS", 3)
    monkeypatch.setattr(cli_mod.time, "sleep", lambda s: None)

    with pytest.raises(ClickException) as exc_info:
        _stream_download_with_retry(
            BadApi(), "https://example.test/x.tar.gz", dest,
            label="download foo.tar.gz",
        )
    msg = str(exc_info.value)
    # New error message format: actionable hint, not raw exception text.
    assert "failed to download" in msg
    assert "RemoteProtocolError" in msg
    assert "download foo.tar.gz" in msg
    assert "TOKI_GITHUB_TOKEN" in msg
    # Partial file is removed on final failure so a later retry
    # doesn't mistake a 0-byte file for a cached asset.
    assert not dest.exists()


def test_curl_download_writes_file_and_returns(tmp_path, monkeypatch):
    """Happy path: curl runs, exits 0, dest is non-empty."""
    from tokirecur.cli import _curl_download

    dest = tmp_path / "blob"
    calls: list[list[str]] = []

    class _Result:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        dest.write_bytes(b"hello-curl")
        return _Result()

    monkeypatch.setattr("tokirecur.cli.subprocess.run", fake_run)
    _curl_download("/usr/bin/curl", "https://example.test/x", dest,
                   label="download x", token=None)
    assert dest.read_bytes() == b"hello-curl"
    assert calls and "--retry-all-errors" in calls[0]
    assert "--max-time" in calls[0]
    # No token → no Authorization header.
    assert not any("Authorization" in a for a in calls[0])


def test_curl_download_passes_bearer_token(tmp_path, monkeypatch):
    """When GITHUB_TOKEN is set, curl gets an Authorization header."""
    from tokirecur.cli import _curl_download

    dest = tmp_path / "blob"
    calls: list[list[str]] = []

    class _Result:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        dest.write_bytes(b"x")
        return _Result()

    monkeypatch.setenv("TOKI_GITHUB_TOKEN", "ghp_test123")
    monkeypatch.setattr("tokirecur.cli.subprocess.run", fake_run)
    _curl_download("/usr/bin/curl", "https://example.test/x", dest,
                   label="download x", token="ghp_test123")
    assert any(a == "Authorization: Bearer ghp_test123" for a in calls[0])


def test_curl_download_raises_on_zero_byte_output(tmp_path, monkeypatch):
    """curl --fail aborts on 4xx/5xx, leaving dest empty. That's our
    signal to fall back to httpx, not a hard failure."""
    from tokirecur.cli import _curl_download, _RetryableDownloadError

    dest = tmp_path / "blob"

    class _Result:
        returncode = 0  # curl didn't error, but produced no body
        stderr = ""

    def fake_run(cmd, **kwargs):
        # Don't create dest — simulates curl aborting on 4xx.
        return _Result()

    monkeypatch.setattr("tokirecur.cli.subprocess.run", fake_run)
    with pytest.raises(_RetryableDownloadError) as exc_info:
        _curl_download("/usr/bin/curl", "https://example.test/x", dest,
                       label="download x", token=None)
    assert "0 bytes" in str(exc_info.value)


def test_curl_download_raises_on_nonzero_exit(tmp_path, monkeypatch):
    """Non-zero curl exit (timeout, connection refused) is a retryable
    signal, not a hard failure."""
    from tokirecur.cli import _curl_download, _RetryableDownloadError

    dest = tmp_path / "blob"

    class _Result:
        returncode = 56  # CURLE_RECV_ERROR
        stderr = "Failure when receiving data from the peer"

    def fake_run(cmd, **kwargs):
        return _Result()

    monkeypatch.setattr("tokirecur.cli.subprocess.run", fake_run)
    with pytest.raises(_RetryableDownloadError) as exc_info:
        _curl_download("/usr/bin/curl", "https://example.test/x", dest,
                       label="download x", token=None)
    msg = str(exc_info.value)
    assert "56" in msg
    assert "Failure when receiving data" in msg


def test_stream_download_uses_curl_when_available(tmp_path, monkeypatch):
    """When curl is on PATH, the primary path runs curl and never
    touches the httpx client at all."""
    from tokirecur import cli as cli_mod
    from tokirecur.cli import _stream_download_with_retry

    dest = tmp_path / "blob"
    curl_called: list[str] = []
    httpx_called: list[str] = []

    class _Result:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **kwargs):
        curl_called.append(cmd[0])
        dest.write_bytes(b"from-curl")
        return _Result()

    def fake_which(name):
        return "/usr/bin/curl" if name == "curl" else None

    class BadApi:
        def stream(self, *a, **kw):
            httpx_called.append("should-not-be-called")
            raise AssertionError("api.stream should not be called when curl succeeds")

    monkeypatch.setattr(cli_mod.shutil, "which", fake_which)
    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    _stream_download_with_retry(
        BadApi(), "https://example.test/x", dest, label="download x",
    )
    assert curl_called == ["/usr/bin/curl"]
    assert httpx_called == []
    assert dest.read_bytes() == b"from-curl"


def test_stream_download_falls_back_to_httpx_when_curl_fails(tmp_path, monkeypatch):
    """When curl fails (e.g. CDN keeps dropping the connection even
    with --retry-all-errors), fall back to the in-process httpx path
    before giving up."""
    from tokirecur import cli as cli_mod
    from tokirecur.cli import _stream_download_with_retry

    dest = tmp_path / "blob"

    class _Result:
        returncode = 22  # HTTP 4xx — curl --fail aborts
        stderr = "HTTP error 404"

    def fake_run(cmd, **kwargs):
        return _Result()

    class GoodApi:
        def stream(self, method, url):
            class R:
                def __enter__(self): return self
                def __exit__(self, *a): return False
                def raise_for_status(self): pass
                def iter_bytes(self, chunk_size):
                    yield b"from-"
                    yield b"httpx"
            return R()

    def fake_which(name):
        return "/usr/bin/curl" if name == "curl" else None

    monkeypatch.setattr(cli_mod.shutil, "which", fake_which)
    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    _stream_download_with_retry(
        GoodApi(), "https://example.test/x", dest, label="download x",
    )
    assert dest.read_bytes() == b"from-httpx"


# ── Linux Vulkan detection (loader-only vs working ICD) ──────────────


def test_linux_vulkan_loader_only_returns_false(tmp_path, monkeypatch):
    """libvulkan.so.1 is installed (Steam, OBS, browsers all pull it
    in) but no ICD JSON exists — the Vulkan build would hang at
    runtime. Detection should return False so the install picks CPU."""
    from tokirecur import ctl

    # No loader, no ICD. Helper short-circuits to False.
    monkeypatch.setattr(
        "tokirecur.ctl.Path",
        _make_path_stub(tmp_path, loader_exists=False, icd_dir_exists=False),
    )
    assert ctl._linux_vulkan_works() is False


def test_linux_vulkan_loader_only_no_icd_returns_false(tmp_path, monkeypatch):
    """Loader is there (common on Ubuntu) but no ICD JSONs at all —
    must return False."""
    from tokirecur import ctl

    monkeypatch.setattr(
        "tokirecur.ctl.Path",
        _make_path_stub(tmp_path, loader_exists=True, icd_dir_exists=False),
    )
    assert ctl._linux_vulkan_works() is False


def test_linux_vulkan_icd_present_but_driver_missing_returns_false(tmp_path, monkeypatch):
    """ICD JSON present but the library_path it references doesn't exist —
    the loader will fail to load this ICD. Must return False."""
    from tokirecur import ctl

    icd_dir = tmp_path / "icd.d"
    icd_dir.mkdir()
    (icd_dir / "fake_icd.json").write_text(
        '{"ICD": {"library_path": "/nonexistent/libvulkan_fake.so"}}'
    )
    monkeypatch.setattr(
        "tokirecur.ctl.Path",
        _make_path_stub(
            tmp_path, loader_exists=True, icd_dir=icd_dir, icd_has_loadable=False,
        ),
    )
    assert ctl._linux_vulkan_works() is False


def test_linux_vulkan_icd_with_loadable_driver_returns_true(tmp_path, monkeypatch):
    """A real ICD JSON pointing at a library that actually exists on disk
    means llama.cpp's Vulkan build can initialise a device."""
    from tokirecur import ctl

    driver = tmp_path / "libvulkan_fake.so"
    driver.write_bytes(b"")
    icd_dir = tmp_path / "icd.d"
    icd_dir.mkdir()
    (icd_dir / "fake_icd.json").write_text(
        f'{{"ICD": {{"library_path": "{driver}"}}}}'
    )
    monkeypatch.setattr(
        "tokirecur.ctl.Path",
        _make_path_stub(
            tmp_path, loader_exists=True, icd_dir=icd_dir, icd_has_loadable=True,
        ),
    )
    assert ctl._linux_vulkan_works() is True


def test_linux_vulkan_lavapipe_software_icd_returns_false(tmp_path, monkeypatch):
    """A Mesa lavapipe (llvmpipe) software-Vulkan ICD — the only Vulkan on
    many headless CPU-only Ubuntu boxes — must NOT count as a working GPU.
    Offloading to software Vulkan is slower than native CPU kernels."""
    from tokirecur import ctl

    driver = tmp_path / "libvulkan_lvp.so"
    driver.write_bytes(b"")
    icd_dir = tmp_path / "icd.d"
    icd_dir.mkdir()
    (icd_dir / "lvp_icd.json").write_text(
        f'{{"ICD": {{"library_path": "{driver}"}}}}'
    )
    monkeypatch.setattr(
        "tokirecur.ctl.Path",
        _make_path_stub(
            tmp_path, loader_exists=True, icd_dir=icd_dir, icd_has_loadable=True,
        ),
    )
    assert ctl._linux_vulkan_works() is False


def test_linux_vulkan_swiftshader_software_icd_returns_false(tmp_path, monkeypatch):
    """Google SwiftShader (``vk_swiftshader.so``) is software Vulkan too —
    excluded even though its ICD JSON names a loadable library."""
    from tokirecur import ctl

    driver = tmp_path / "vk_swiftshader.so"
    driver.write_bytes(b"")
    icd_dir = tmp_path / "icd.d"
    icd_dir.mkdir()
    (icd_dir / "vk_swiftshader_icd.json").write_text(
        f'{{"ICD": {{"library_path": "{driver}"}}}}'
    )
    monkeypatch.setattr(
        "tokirecur.ctl.Path",
        _make_path_stub(
            tmp_path, loader_exists=True, icd_dir=icd_dir, icd_has_loadable=True,
        ),
    )
    assert ctl._linux_vulkan_works() is False


def test_linux_vulkan_lavapipe_and_real_icd_returns_true(tmp_path, monkeypatch):
    """Lavapipe alongside a real discrete-GPU ICD (radeon/amdvlk/nouveau)
    must still count: the real ICD is what matters."""
    from tokirecur import ctl

    lvp = tmp_path / "libvulkan_lvp.so"
    lvp.write_bytes(b"")
    radeon = tmp_path / "libvulkan_radeon.so"
    radeon.write_bytes(b"")
    icd_dir = tmp_path / "icd.d"
    icd_dir.mkdir()
    (icd_dir / "lvp_icd.json").write_text(
        f'{{"ICD": {{"library_path": "{lvp}"}}}}'
    )
    (icd_dir / "radeon_icd.json").write_text(
        f'{{"ICD": {{"library_path": "{radeon}"}}}}'
    )
    monkeypatch.setattr(
        "tokirecur.ctl.Path",
        _make_path_stub(
            tmp_path, loader_exists=True, icd_dir=icd_dir, icd_has_loadable=True,
        ),
    )
    assert ctl._linux_vulkan_works() is True


def test_intel_igpu_only_linux_true_for_intel_only(tmp_path, monkeypatch):
    from tokirecur import ctl

    _fake_drm(tmp_path, vendors=["0x8086"])
    monkeypatch.setattr("tokirecur.ctl.Path", _make_drm_stub(tmp_path, ["0x8086"]))
    assert ctl._intel_igpu_only_linux() is True


def test_intel_igpu_only_linux_false_with_discrete_gpu(tmp_path, monkeypatch):
    from tokirecur import ctl

    monkeypatch.setattr(
        "tokirecur.ctl.Path",
        _make_drm_stub(tmp_path, ["0x8086", "0x10de"]),
    )
    assert ctl._intel_igpu_only_linux() is False


def test_intel_igpu_only_linux_false_with_no_intel(tmp_path, monkeypatch):
    from tokirecur import ctl

    monkeypatch.setattr("tokirecur.ctl.Path", _make_drm_stub(tmp_path, ["0x1002"]))
    assert ctl._intel_igpu_only_linux() is False


# ── Helpers for stubbing pathlib in the detection tests ──────────────


def _make_path_stub(tmp_path, *, loader_exists, icd_dir_exists=True,
                    icd_dir=None, icd_has_loadable=False):
    """Build a Path subclass whose existence semantics match the scenario.

    The stub treats the canonical /usr paths as if they existed under
    ``tmp_path``: ``libvulkan.so.1`` (the loader) and the ICD JSON
    directory. Library-path candidates resolve relative to the stub
    tmp dir if ``icd_has_loadable`` is True.
    """
    base = Path(str(tmp_path))
    icd_dir_path = icd_dir if icd_dir is not None else (base / "vulkan" / "icd.d")
    # Only create the icd_dir if the scenario says it exists; the
    # test then drives the helper via the stub.
    if icd_dir_exists:
        icd_dir_path.mkdir(parents=True, exist_ok=True)
        # If the scenario says the ICD has a loadable driver, create
        # a stub library so is_file() returns True for the library_path.
        if icd_has_loadable:
            for f in icd_dir_path.glob("*.json"):
                import json as _json
                try:
                    data = _json.loads(f.read_text())
                except Exception:
                    continue
                lib = (data.get("ICD") or {}).get("library_path") or ""
                if lib:
                    Path(lib).parent.mkdir(parents=True, exist_ok=True)
                    Path(lib).write_bytes(b"")

    real_Path = Path

    class _Stub(real_Path):
        def exists(self):
            s = str(self)
            if "libvulkan.so.1" in s:
                return bool(loader_exists)
            # If the ICD helper probes the ICD JSON or the library_path
            # it names, the real file at that path exists (we created
            # it above when icd_has_loadable=True).
            return real_Path(str(self)).exists()

        def is_dir(self):
            s = str(self)
            # /usr/share/vulkan/icd.d is a directory only when the
            # scenario says the icd_dir exists.
            if s == "/usr/share/vulkan/icd.d":
                return bool(icd_dir_exists)
            if s == str(icd_dir_path):
                return bool(icd_dir_exists)
            return real_Path(str(self)).is_dir()

        def is_file(self):
            s = str(self)
            if s.endswith("libvulkan.so.1"):
                return bool(loader_exists)
            return real_Path(str(self)).is_file()

        def glob(self, pattern):
            s = str(self)
            if s == "/usr/share/vulkan/icd.d" or s == str(icd_dir_path):
                if icd_dir_exists:
                    return iter(list(icd_dir_path.glob(pattern)))
                return iter(())
            return real_Path(str(self)).glob(pattern)

    return _Stub


def _make_drm_stub(tmp_path, vendors):
    """Build a Path subclass that reports a list of PCI vendor IDs
    under /sys/class/drm/card* for the Intel-iGPU check."""
    base = Path(str(tmp_path))
    drm_root = base / "class" / "drm"
    drm_root.mkdir(parents=True, exist_ok=True)
    cards = []
    for i, v in enumerate(vendors):
        card = drm_root / f"card{i}"
        device = card / "device"
        device.mkdir(parents=True, exist_ok=True)
        (device / "vendor").write_text(v)
        cards.append(card)

    real_Path = Path

    class _DRM(real_Path):
        def is_dir(self):
            return real_Path(str(self)).is_dir()

        def glob(self, pattern):
            s = str(self)
            if s == "/sys/class/drm" or s == str(drm_root):
                return iter(cards)
            return real_Path(str(self)).glob(pattern)

    return _DRM


def _fake_drm(tmp_path, vendors):
    """Drop a real /sys/class/drm/card*/device/vendor tree under tmp."""
    drm_root = tmp_path / "class" / "drm"
    drm_root.mkdir(parents=True, exist_ok=True)
    for i, v in enumerate(vendors):
        card = drm_root / f"card{i}"
        device = card / "device"
        device.mkdir(parents=True, exist_ok=True)
        (device / "vendor").write_text(v)


# ── Force CPU build when no usable GPU is present ────────────────


def _stub_profile_with_gpus(monkeypatch, gpus):
    """Build a 155U-shaped profile with the given GPU list and stub
    hardware.profile_box to return it."""
    from tokirecur.hardware import CPUProfile, HardwareProfile
    prof = HardwareProfile(
        schema_version=1,
        profiled_at="2026-01-01T00:00:00Z",
        os="linux",
        cpu=CPUProfile(
            vendor="GenuineIntel", name="Core Ultra 7 155U",
            family=6, model=189, physical_cores=12, logical_cores=14,
            p_core_ids=list(range(4)), e_core_ids=list(range(4, 14)),
            max_boost_mhz=4800.0, simd=["avx2", "avx_vnni"],
        ),
        gpus=gpus,
        ram_gb=16.0,
        benchmarks={},
        recommended={"backend": "vulkan", "ngl": 99, "threads": 12},
    )
    monkeypatch.setattr(
        "tokirecur.hardware.profile_box",
        lambda force=False: prof,
    )


def test_force_cpu_when_intel_igpu_only(monkeypatch, capsys):
    """A Linux x64 box with only an Intel iGPU (shared RAM, not usable
    for LLM) gets the Vulkan build auto-switched to CPU. The notice
    explains why."""
    from tokirecur.hardware import GPUProfile
    _stub_profile_with_gpus(monkeypatch, [
        GPUProfile(
            vendor="Intel", name="Arc iGPU",
            vram_gb=0.5, driver=None, vulkan_score=0.3,
            shared_ram=True, use_for_llm=False,
            reason="shared system RAM",
        ),
    ])
    from tokirecur.cli import _force_cpu_when_no_usable_gpu
    result = _force_cpu_when_no_usable_gpu("ubuntu", "x64", "vulkan")
    assert result == "cpu"
    stderr = capsys.readouterr().err
    assert "no usable GPU" in stderr
    assert "vulkan" in stderr
    assert "cpu" in stderr


def test_force_cpu_when_no_gpu_at_all(monkeypatch, capsys):
    """A box with no GPUs at all — Vulkan -> CPU."""
    _stub_profile_with_gpus(monkeypatch, [])
    from tokirecur.cli import _force_cpu_when_no_usable_gpu
    result = _force_cpu_when_no_usable_gpu("ubuntu", "x64", "vulkan")
    assert result == "cpu"
    stderr = capsys.readouterr().err
    assert "no usable GPU" in stderr


def test_force_cpu_silent_when_real_gpu_present(monkeypatch, capsys):
    """A Linux x64 box with a real discrete NVIDIA GPU — Vulkan is the
    right choice, no auto-switch, no notice."""
    from tokirecur.hardware import GPUProfile
    _stub_profile_with_gpus(monkeypatch, [
        GPUProfile(
            vendor="NVIDIA", name="RTX 4070",
            vram_gb=12.0, driver=None, vulkan_score=1.0,
            shared_ram=False, use_for_llm=True,
            reason="discrete GPU",
        ),
    ])
    from tokirecur.cli import _force_cpu_when_no_usable_gpu
    result = _force_cpu_when_no_usable_gpu("ubuntu", "x64", "vulkan")
    assert result == "vulkan"
    assert capsys.readouterr().err == ""


def test_force_cpu_silent_for_cpu_backend(monkeypatch, capsys):
    """CPU installs on the same hardware produce no notice."""
    from tokirecur.hardware import GPUProfile
    _stub_profile_with_gpus(monkeypatch, [
        GPUProfile(
            vendor="Intel", name="Arc iGPU",
            vram_gb=0.5, driver=None, vulkan_score=0.3,
            shared_ram=True, use_for_llm=False,
            reason="shared system RAM",
        ),
    ])
    from tokirecur.cli import _force_cpu_when_no_usable_gpu
    result = _force_cpu_when_no_usable_gpu("ubuntu", "x64", "cpu")
    assert result == "cpu"
    assert capsys.readouterr().err == ""


def test_force_cpu_works_for_cuda_and_rocm_too(monkeypatch, capsys):
    """The rule is generic: any GPU backend (vulkan, cuda, rocm) on a
    box with no usable GPU gets auto-switched. No per-vendor
    special-casing."""
    from tokirecur.hardware import GPUProfile
    _stub_profile_with_gpus(monkeypatch, [
        GPUProfile(
            vendor="Intel", name="Arc iGPU",
            vram_gb=0.5, driver=None, vulkan_score=0.3,
            shared_ram=True, use_for_llm=False,
            reason="shared system RAM",
        ),
    ])
    from tokirecur.cli import _force_cpu_when_no_usable_gpu
    assert _force_cpu_when_no_usable_gpu("ubuntu", "x64", "vulkan") == "cpu"
    assert _force_cpu_when_no_usable_gpu("windows", "x64", "cuda") == "cpu"
    assert _force_cpu_when_no_usable_gpu("ubuntu", "x64", "rocm") == "cpu"


def test_force_cpu_works_on_amd_apu_only(monkeypatch, capsys):
    """An AMD APU (shared RAM, not usable for LLM) also triggers the
    auto-switch — the rule is box-level, not vendor-specific."""
    from tokirecur.hardware import GPUProfile
    _stub_profile_with_gpus(monkeypatch, [
        GPUProfile(
            vendor="AMD", name="Radeon 780M",
            vram_gb=0.5, driver=None, vulkan_score=0.3,
            shared_ram=True, use_for_llm=False,
            reason="shared system RAM",
        ),
    ])
    from tokirecur.cli import _force_cpu_when_no_usable_gpu
    result = _force_cpu_when_no_usable_gpu("ubuntu", "x64", "vulkan")
    assert result == "cpu"
    stderr = capsys.readouterr().err
    assert "no usable GPU" in stderr
    # No mention of "Intel" — the rule is generic.
    assert "Intel" not in stderr


def test_env_backend_override_honours_known_backend(monkeypatch):
    """TOKI_BACKEND=rocm is honoured."""
    monkeypatch.setenv("TOKI_BACKEND", "rocm")
    from tokirecur.cli import _env_backend_override
    assert _env_backend_override() == "rocm"


def test_env_backend_override_honours_cpu(monkeypatch):
    """TOKI_BACKEND=cpu is honoured (forces a CPU build)."""
    monkeypatch.setenv("TOKI_BACKEND", "cpu")
    from tokirecur.cli import _env_backend_override
    assert _env_backend_override() == "cpu"


def test_env_backend_override_case_insensitive(monkeypatch):
    monkeypatch.setenv("TOKI_BACKEND", "CUDA")
    from tokirecur.cli import _env_backend_override
    assert _env_backend_override() == "cuda"


def test_env_backend_override_unknown_warns_and_returns_none(monkeypatch, capsys):
    """An unknown TOKI_BACKEND warns on stderr and falls through to
    auto-detection (None)."""
    monkeypatch.setenv("TOKI_BACKEND", "quantum")
    from tokirecur.cli import _env_backend_override
    assert _env_backend_override() is None
    assert "not a known backend" in capsys.readouterr().err


def test_env_backend_override_unset_returns_none(monkeypatch):
    monkeypatch.delenv("TOKI_BACKEND", raising=False)
    from tokirecur.cli import _env_backend_override
    assert _env_backend_override() is None


def test_profile_ttl_env_override():
    """TOKI_PROFILE_TTL (seconds) replaces the 30-day default.

    Run in a subprocess: importlib.reload() swaps the module's dataclass
    objects, so tests that imported CPUProfile from the original module
    then compare it with instances built after the reload fail with
    "identical reprs but not equal".
    """
    import subprocess
    import sys

    def ttl(value):
        env = os.environ.copy()
        if value is None:
            env.pop("TOKI_PROFILE_TTL", None)
        else:
            env["TOKI_PROFILE_TTL"] = value
        out = subprocess.run(
            [sys.executable, "-c",
             "from tokirecur import hardware; "
             "print(hardware.PROFILE_TTL_SECONDS)"],
            capture_output=True, text=True, env=env, timeout=60,
        )
        assert out.returncode == 0, out.stderr
        return int(out.stdout.strip())

    assert ttl("60") == 60
    assert ttl(None) == 30 * 24 * 3600


def test_prune_foreign_gpu_libs_removes_gpu_libs_on_cpu_install(tmp_path):
    """A CPU install must leave zero GPU runtime libs in the bin dir —
    a stale libggml-vulkan.so would make every backend probe (doctor,
    launcher, the ngl warning) report "Vulkan" on a CPU-only box."""
    from tokirecur.cli import _prune_foreign_gpu_libs
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("llama-server", "libggml-cpu.so", "libggml-vulkan.so",
                 "libggml-vulkan.so.stale.53508", "libggml-cuda.so",
                 "libggml-cuda.so.stale.99999", "ggml-vulkan.dll"):
        (bin_dir / name).write_bytes(b"x")
    _prune_foreign_gpu_libs(bin_dir, "cpu")
    remaining = {p.name for p in bin_dir.iterdir()}
    assert remaining == {"llama-server", "libggml-cpu.so"}


def test_prune_foreign_gpu_libs_keeps_gpu_libs_on_gpu_backend(tmp_path):
    """Pruning only applies to CPU installs; a Vulkan/CUDA install's
    own libs must be left alone."""
    from tokirecur.cli import _prune_foreign_gpu_libs
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("llama-server", "libggml-vulkan.so"):
        (bin_dir / name).write_bytes(b"x")
    _prune_foreign_gpu_libs(bin_dir, "vulkan")
    assert {p.name for p in bin_dir.iterdir()} == {
        "llama-server", "libggml-vulkan.so"}


def test_prune_foreign_gpu_libs_missing_bin_dir_is_noop(tmp_path):
    from tokirecur.cli import _prune_foreign_gpu_libs
    _prune_foreign_gpu_libs(tmp_path / "does-not-exist", "cpu")


def test_find_local_llama_server_prefers_LLAMA_CPP_BLD_env(tmp_path, monkeypatch):
    """``LLAMA_CPP_BLD`` env var wins over the default search paths —
    this is how ``scripts/build_llama_cpp.sh`` points at its build dir."""
    from tokirecur.cli import _find_local_llama_server
    bld = tmp_path / "llama-bld" / "bin" / "llama-server"
    bld.parent.mkdir(parents=True)
    bld.write_bytes(b"#!/bin/sh\nexit 0\n")
    bld.chmod(0o755)
    monkeypatch.setenv("LLAMA_CPP_BLD", str(tmp_path / "llama-bld"))
    # A second candidate in $HOME that should NOT be chosen.
    fallback = tmp_path / "fallback" / "bin" / "llama-server"
    fallback.parent.mkdir(parents=True)
    fallback.write_bytes(b"#!/bin/sh\nexit 0\n")
    fallback.chmod(0o755)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "fallback")
    assert _find_local_llama_server() == str(bld.resolve())


def test_find_local_llama_server_returns_none_when_absent(tmp_path, monkeypatch):
    """No local build anywhere — the helper returns None and the
    installer falls back to (or fails on) the GitHub release."""
    from tokirecur.cli import _find_local_llama_server
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("LLAMA_CPP_BLD", raising=False)
    assert _find_local_llama_server() is None


def test_find_local_llama_server_skips_non_executable(tmp_path, monkeypatch):
    """A binary file without the executable bit is skipped — without
    that check the installer would copy it and llama-server would
    fail to spawn on POSIX with EACCES."""
    from tokirecur.cli import _find_local_llama_server
    bld = tmp_path / "build-tr" / "bin" / "llama-server"
    bld.parent.mkdir(parents=True)
    bld.write_bytes(b"data")
    bld.chmod(0o644)  # not executable
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("LLAMA_CPP_BLD", raising=False)
    assert _find_local_llama_server() is None


def test_install_refuses_to_overwrite_gpu_binary_with_cpu(tmp_path, monkeypatch):
    """The bin dir already has a working ROCm/CUDA/Vulkan/Metal
    llama-server. ``install-llama`` (auto or ``--backend cpu``) must
    refuse to replace it with a CPU build unless the user forces the
    downgrade — otherwise every spec pinning ``-ngl 99`` would refuse
    to start after the install.
    """
    from click.testing import CliRunner

    from tokirecur.cli import cli

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    existing = bin_dir / "llama-server"
    existing.write_bytes(b"existing-gpu-binary")
    existing.chmod(0o755)
    # Stub binary_backend to report the existing binary as GPU.
    monkeypatch.setattr(
        "tokirecur.diagnose.binary_backend",
        lambda p: "ROCm" if str(p) == str(existing) else None,
    )

    runner = CliRunner()
    monkeypatch.setenv("TOKI_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("TOKI_INSTALL_LLAMA_FORCE_DOWNGRADE", raising=False)
    result = runner.invoke(
        cli, ["model", "install-llama", "--backend", "cpu", "--to", str(bin_dir)],
    )
    assert result.exit_code != 0, result.output
    # The pre-existing GPU binary must still be on disk.
    assert existing.exists()
    assert "refusing to overwrite" in result.output.lower() or \
           "refusing to overwrite" in (result.output + result.stderr).lower()


def test_install_force_downgrade_env_overrides_gpu_to_cpu_guard(tmp_path, monkeypatch):
    """``TOKI_INSTALL_LLAMA_FORCE_DOWNGRADE=1`` lets the user explicitly
    accept the downgrade — needed for CI / scripted rebuilds that
    genuinely want a CPU binary."""
    from click.testing import CliRunner

    from tokirecur.cli import cli

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    existing = bin_dir / "llama-server"
    existing.write_bytes(b"existing-gpu-binary")
    existing.chmod(0o755)
    monkeypatch.setattr(
        "tokirecur.diagnose.binary_backend",
        lambda p: "ROCm" if str(p) == str(existing) else None,
    )
    # Stub the network/asset path entirely so we test only the guard.
    monkeypatch.setattr("tokirecur.cli._find_local_llama_server", lambda: None)

    def fake_download(*args, **kwargs):
        # Drop a fake CPU llama-server into bin_dir and return.
        (bin_dir / "llama-server").write_bytes(b"new-cpu-binary")
        (bin_dir / "llama-server").chmod(0o755)
        return None

    monkeypatch.setattr("tokirecur.cli._download_and_unpack_zip", fake_download)
    monkeypatch.setattr(
        "tokirecur.cli._pick_asset",
        lambda assets, patterns: ({"name": "llama-cpu.zip"}, "cpu"),
    )
    monkeypatch.setattr(
        "tokirecur.cli.httpx.Client",
        lambda *a, **kw: type("C", (), {
            "get": staticmethod(lambda *a, **kw: type("R", (), {
                "json": staticmethod(lambda self: {
                    "tag_name": "b99999",
                    "assets": [{"name": "llama-cpu.zip"}],
                })
            })())
        })(),
    )

    runner = CliRunner()
    monkeypatch.setenv("TOKI_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("TOKI_INSTALL_LLAMA_FORCE_DOWNGRADE", "1")
    result = runner.invoke(
        cli, ["model", "install-llama", "--backend", "cpu", "--to", str(bin_dir)],
    )
    # The downgrade should now be allowed (the guard exits cleanly
    # and the rest of the install runs against the fake download).
    assert "refusing to overwrite" not in result.output.lower()



def test_probe_gpu_device_prints_real_device(monkeypatch, capsys):
    from tokirecur.cli import _probe_gpu_device

    def fake_run(argv, **kw):
        return type("R", (), {
            "returncode": 0,
            "stdout": "  - SYCL0 : Intel(R) Arc(TM) A770 Graphics "
                      "(16384 MiB, 15000 MiB free)\n",
            "stderr": "",
        })()

    monkeypatch.setattr("tokirecur.cli.subprocess.run", fake_run)
    _probe_gpu_device(
        Path("/tmp/llama-server.exe"), r"\bSYCL\d+\b", "SYCL", "hint",
    )
    out = capsys.readouterr().out
    assert "verify: - SYCL0 : Intel(R) Arc(TM) A770 Graphics" in out


def test_probe_gpu_device_warns_when_no_device(monkeypatch, capsys):
    from tokirecur.cli import _probe_gpu_device

    def fake_run(argv, **kw):
        return type("R", (), {
            "returncode": 0, "stdout": "ggml_vulkan: no devices\n",
            "stderr": "",
        })()

    monkeypatch.setattr("tokirecur.cli.subprocess.run", fake_run)
    _probe_gpu_device(
        Path("/tmp/llama-server.exe"), r"\bVulkan\d+\b", "Vulkan", "update",
    )
    err = capsys.readouterr().err
    assert "did not enumerate a Vulkan device" in err
    assert "update" in err


@pytest.mark.parametrize(
    "backend,asset_name,files,device_line",
    [
        (
            "sycl",
            "llama-b12020-bin-win-sycl-x64.zip",
            {
                "llama-server.exe": b"MZ",
                "ggml-sycl.dll": b"sycl",
                "sycl9.dll": b"rt",
                "mkl_sycl_blas.6.dll": b"mkl",
                "dnnl.dll": b"dnnl",
            },
            "  - SYCL0 : Intel(R) Arc(TM) A770 Graphics "
            "(16384 MiB, 15000 MiB free)",
        ),
        (
            "rocm",
            "llama-b12020-bin-win-rocm-10.0-x64.zip",
            {
                "llama-server.exe": b"MZ",
                "ggml-hip.dll": b"hip",
                "amdhip64_7.dll": b"hiprt",
                "amd_comgr.dll": b"comgr",
                "rocm_kpack.dll": b"kpack",
            },
            "  - ROCm0 : AMD Radeon RX 7900 XTX (24576 MiB, 24000 MiB free)",
        ),
        (
            "vulkan",
            "llama-b12020-bin-win-vulkan-x64.zip",
            {
                "llama-server.exe": b"MZ",
                "ggml-vulkan.dll": b"vk",
            },
            "  - Vulkan0 : AMD Radeon RX 7600 (8192 MiB, 8000 MiB free)",
        ),
        (
            "openvino",
            "llama-b12024-bin-win-openvino-2026.4.1-x64.zip",
            {
                "llama-server.exe": b"MZ",
                "ggml-openvino.dll": b"ov",
                "openvino.dll": b"core",
                "openvino_intel_gpu_plugin.dll": b"gpu",
            },
            "  - OpenVINO0 : Intel(R) Arc(TM) A770 Graphics "
            "(16384 MiB, 15000 MiB free)",
        ),
    ],
)
def test_install_windows_gpu_backend_verifies_runtime(
    tmp_path, monkeypatch, backend, asset_name, files, device_line,
):
    """Each Windows GPU path must land its build, keep its runtime, prune
    the others, and prove a real device line before reporting success."""
    import io
    import types
    import zipfile

    from tokirecur import cli as cli_mod
    from tokirecur import diagnose
    from tokirecur.cli import install_llama

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for stale in ("ggml-cuda.dll", "cudart64_13.dll", "ggml-vulkan.dll",
                  "ggml-hip.dll", "amdhip64_7.dll", "ggml-sycl.dll",
                  "sycl9.dll", "ggml-openvino.dll", "openvino.dll"):
        (bin_dir / stale).write_bytes(b"old")
    (bin_dir / "llama-server.exe").write_bytes(b"old")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    payload = buf.getvalue()

    RELEASE = {
        "tag_name": "b12020",
        "assets": [{
            "name": asset_name,
            "size": len(payload),
            "browser_download_url": "https://example.test/gpu.zip",
        }],
    }

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, headers=None):
            class _Resp:
                def json(self_inner):
                    return RELEASE

                def raise_for_status(self_inner):
                    pass

            return _Resp()

    def fake_download(api, url, dest, label=""):
        Path(dest).write_bytes(payload)

    def fake_run(argv, **kw):
        if "--list-devices" in argv:
            return types.SimpleNamespace(
                returncode=0,
                stdout="Available devices:\n" + device_line + "\n",
                stderr="",
            )
        return types.SimpleNamespace(returncode=0, stdout="v1\n", stderr="")

    import httpx as _httpx
    monkeypatch.setattr(_httpx, "Client", FakeClient)
    monkeypatch.setattr(cli_mod, "_stream_download_with_retry", fake_download)
    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr("platform.machine", lambda: "AMD64")
    monkeypatch.setattr("shutil.which", lambda n: None)
    monkeypatch.setattr(diagnose.winproc, "IS_WINDOWS", True)
    if backend == "vulkan":
        monkeypatch.setattr(diagnose, "vulkan_runtime_ready", lambda b: True)

    install_llama.callback(
        tag="b12020", dest=str(bin_dir), name=None, backend=backend,
        gfx=None, profile=False, force_cpu=False,
    )

    names = {p.name for p in bin_dir.iterdir()}
    assert set(files) <= names
    if backend != "openvino":
        assert "ggml-openvino.dll" not in names
    assert not any(n.endswith(".zip") for n in names)
    landed = diagnose.binary_backend(str(bin_dir / "llama-server.exe"))
    assert landed.lower() == backend


def test_prune_foreign_gpu_libs_cpu_removes_metal_dylibs(tmp_path):
    from tokirecur.cli import _prune_foreign_gpu_libs

    for name in (
        "libggml-metal.dylib", "libggml-metal.0.28.0.dylib",
        "libggml-cpu.dylib", "libggml-base.dylib",
        "ggml-opencl.dll", "OpenCL.dll",
    ):
        (tmp_path / name).write_bytes(b"x")
    _prune_foreign_gpu_libs(tmp_path, "cpu")
    left = {p.name for p in tmp_path.iterdir()}
    assert "libggml-cpu.dylib" in left
    assert "libggml-base.dylib" in left
    assert "libggml-metal.dylib" not in left
    assert "libggml-metal.0.28.0.dylib" not in left
    assert "ggml-opencl.dll" not in left
    assert "OpenCL.dll" not in left


@pytest.mark.parametrize(
    "requested_backend,device_output,expected",
    [
        (
            "metal",
            "Available devices:\n"
            "  MTL0: Apple M2 (24576 MiB, 23000 MiB free)",
            "verify: MTL0",
        ),
        (
            "metal",
            "Available devices:\n  (none)",
            "did not enumerate an MTL device",
        ),
        (
            "cpu",
            "Available devices:\n"
            "  MTL0: Apple M2 (24576 MiB, 23000 MiB free)",
            "macOS has no separate CPU build",
        ),
    ],
)
def test_install_macos_metal_and_cpu_fallback(
    tmp_path, monkeypatch, capsys, requested_backend, device_output, expected,
):
    """macOS ships one universal Metal build. With a Metal device it
    verifies MTL0; on a Mac/VM without Metal the same build runs CPU and
    the install must succeed with an explicit note, not a false error."""
    import io
    import tarfile
    import types

    from tokirecur import cli as cli_mod
    from tokirecur.cli import install_llama

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "llama-server").write_bytes(b"old")
    (bin_dir / "libggml-metal.dylib").write_bytes(b"old-metal")

    files = {
        "llama-server": b"#!/bin/sh\nexit 0\n",
        "libggml-metal.dylib": b"metal",
        "libggml-metal.0.28.0.dylib": b"metal-versioned",
        "libggml-cpu.dylib": b"cpu",
        "libggml-base.dylib": b"base",
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name=f"llama-b12020/{name}")
            info.size = len(data)
            info.mode = 0o755
            tf.addfile(info, io.BytesIO(data))
    payload = buf.getvalue()

    RELEASE = {
        "tag_name": "b12020",
        "assets": [{
            "name": "llama-b12020-bin-macos-arm64.tar.gz",
            "size": len(payload),
            "browser_download_url": "https://example.test/macos.tar.gz",
        }],
    }

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, headers=None):
            class _Resp:
                def json(self_inner):
                    return RELEASE

                def raise_for_status(self_inner):
                    pass

            return _Resp()

    def fake_download(api, url, dest, label=""):
        Path(dest).write_bytes(payload)

    def fake_run(argv, **kw):
        if "--list-devices" in argv:
            return types.SimpleNamespace(
                returncode=0, stdout=device_output + "\n", stderr="",
            )
        return types.SimpleNamespace(returncode=0, stdout="v1\n", stderr="")

    import httpx as _httpx
    monkeypatch.setattr(_httpx, "Client", FakeClient)
    monkeypatch.setattr(cli_mod, "_stream_download_with_retry", fake_download)
    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setattr("shutil.which", lambda n: None)
    # A local llama.cpp build on the test box must not shadow the
    # release asset this test is exercising.
    monkeypatch.setattr("tokirecur.cli._find_local_llama_server", lambda: None)

    install_llama.callback(
        tag="b12020", dest=str(bin_dir), name=None, backend=requested_backend,
        gfx=None, profile=False, force_cpu=False,
    )

    names = {p.name for p in bin_dir.iterdir()}
    assert "libggml-metal.dylib" in names
    assert "libggml-metal.0.28.0.dylib" in names
    combined = capsys.readouterr()
    text = combined.out + combined.err
    assert expected in text
    if "MTL0" in expected:
        assert "landed backend metal" in text
    elif "CPU build" in expected:
        assert "installing metal instead" in text
    else:
        assert "will run on CPU" in text


def test_install_macos_cpu_uses_local_cpu_build(tmp_path, monkeypatch, capsys):
    """macOS has no CPU release asset; a local CPU-only build is the
    only true CPU binary, so --backend cpu must install it instead of
    falling back to the universal Metal asset."""
    from tokirecur import cli as cli_mod
    from tokirecur import diagnose
    from tokirecur.cli import install_llama

    local_dir = tmp_path / "local"
    local_dir.mkdir()
    (local_dir / "llama-server").write_bytes(b"cpu-server")
    (local_dir / "libggml-cpu.dylib").write_bytes(b"cpu")

    bin_dir = tmp_path / "bin"
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setattr(
        cli_mod, "_find_local_llama_server",
        lambda: str(local_dir / "llama-server"),
    )
    monkeypatch.setattr(diagnose, "_run", lambda argv, timeout=5.0: None)

    install_llama.callback(
        tag=None, dest=str(bin_dir), name=None, backend="cpu",
        gfx=None, profile=False, force_cpu=False,
    )

    assert (bin_dir / "libggml-cpu.dylib").exists()
    assert not (bin_dir / "libggml-metal.dylib").exists()
    text = "".join(capsys.readouterr())
    assert "local CPU build" in text
    assert "landed backend cpu" in text
