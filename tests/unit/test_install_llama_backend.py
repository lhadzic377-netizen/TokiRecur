"""Unit tests for `toki model install-llama` backend/asset selection.

Covers: backend auto-detection, the OS/arch -> asset pattern map, the
newest-release walk, and ROCm env wiring. No network calls and no
processes are spawned.
"""

from pathlib import Path

import pytest

from tokirecur.cli import _asset_patterns, _pick_asset, _rocm_env

# Asset names taken from a real llama.cpp release (b10289) so the map
# is tested against reality, not invented names.
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
        assert frag and backend in ("rocm", "cuda", "vulkan", "metal", "cpu")


def test_set_binary_merges_env(tmp_path, monkeypatch):
    from tokirecur import ctl

    reg = tmp_path / "models.yaml"
    state = tmp_path / "state"
    marker = tmp_path / "active-model"
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


def test_detect_backend_macos_always_metal(monkeypatch):
    from tokirecur.cli import _detect_gpu_backend

    monkeypatch.setattr("shutil.which", lambda name: None)
    assert _detect_gpu_backend("macos") == "metal"


def test_detect_backend_linux_rocm_first(monkeypatch):
    from tokirecur.cli import _detect_gpu_backend

    monkeypatch.setattr("shutil.which", lambda name: "hipconfig" if name == "hipconfig" else None)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    assert _detect_gpu_backend("ubuntu") == "rocm"


def test_detect_backend_linux_vulkan_then_cpu(monkeypatch):
    from tokirecur.cli import _detect_gpu_backend

    monkeypatch.setattr(Path, "exists", lambda self: False)
    monkeypatch.setattr("shutil.which", lambda name: "vulkaninfo" if name == "vulkaninfo" else None)
    assert _detect_gpu_backend("ubuntu") == "vulkan"
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert _detect_gpu_backend("ubuntu") == "cpu"


def test_detect_backend_windows_cuda_then_cpu(monkeypatch):
    from tokirecur.cli import _detect_gpu_backend

    monkeypatch.setattr(Path, "exists", lambda self: False)
    monkeypatch.setattr("shutil.which", lambda name: "nvidia-smi" if name == "nvidia-smi" else None)
    assert _detect_gpu_backend("windows") == "cuda"
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert _detect_gpu_backend("windows") == "cpu"
