"""Unit tests for `toki model install-llama` backend/asset selection.

Covers: backend auto-detection, the OS/arch -> asset pattern map, the
newest-release walk, and ROCm env wiring. No network calls and no
processes are spawned.
"""

from pathlib import Path

import pytest
import yaml

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


@pytest.mark.parametrize(
    "main,cudart",
    [
        ("llama-b10326-bin-win-cuda-13.3-x64.zip",
         "cudart-llama-bin-win-cuda-13.3-x64.zip"),
        ("llama-b10326-bin-win-cuda-12.4-x64.zip",
         "cudart-llama-bin-win-cuda-12.4-x64.zip"),
        ("llama-b10326-bin-ubuntu-rocm-7.2-x64.tar.gz", None),
        ("llama-b10326-bin-win-cpu-x64.zip", None),
    ],
)
def test_cuda_runtime_archive_name(main, cudart):
    from tokirecur.cli import _cuda_runtime_archive_name
    assert _cuda_runtime_archive_name(main) == cudart


def test_install_pulls_cudart_runtime_for_cuda_13_3(tmp_path, monkeypatch):
    """CUDA 13.x main archives don't bundle cudart64_13.dll; install-llama
    must download the paired cudart-llama-bin runtime archive too, otherwise
    llama-server can't load CUDA and falls back to CPU."""
    from tokirecur import cli as cli_mod
    from tokirecur.cli import install_llama

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    downloads: list[str] = []

    class FakeClient:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False

        def get(self, url, headers=None):
            class _Resp:
                status_code = 200
                def json(self_inner): return RELEASE
                def raise_for_status(self_inner): pass
            return _Resp()

        def stream(self, method, url, headers=None):
            downloads.append(url)
            # Build a real zip in memory containing llama-server.exe so
            # the main archive extraction succeeds and the cudart hook runs.
            import io as _io
            import zipfile as _zipfile
            buf = _io.BytesIO()
            with _zipfile.ZipFile(buf, "w") as zf:
                zf.writestr("llama-server.exe", b"MZ")
            data = buf.getvalue()

            class _Resp:
                def raise_for_status(self_inner): pass
                def __enter__(self_inner): return self_inner
                def __exit__(self_inner, *a): return False
                def iter_bytes(self_inner, chunk):
                    yield data
            return _Resp()

    RELEASE = {
        "tag_name": "b10326",
        "assets": [
            {"name": "llama-b10326-bin-win-cuda-13.3-x64.zip",
             "size": 146_540_671,
             "browser_download_url": "https://example.test/main.zip"},
            {"name": "cudart-llama-bin-win-cuda-13.3-x64.zip",
             "size": 390_970_417,
             "browser_download_url": "https://example.test/cudart.zip"},
        ],
    }

    monkeypatch.setattr("httpx.Client", FakeClient, raising=False)
    import httpx as _httpx
    monkeypatch.setattr(_httpx, "Client", FakeClient)
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr("platform.machine", lambda: "AMD64")
    # Auto-detect backend: pretend nvidia-smi is present → cuda.
    monkeypatch.setattr("shutil.which", lambda n: "nvidia-smi" if n == "nvidia-smi" else None)
    # Force a known-good driver so the 13.3 asset is chosen over 12.4.
    from tokirecur import diagnose
    monkeypatch.setattr(diagnose, "cuda_driver_verdict", lambda d: (False, False))

    # Skip the real extraction/zipfile plumbing for the cudart archive;
    # record any URL the dispatch hands to it.
    extra_downloads: list[str] = []

    def fake_unpack(api, asset, dest_dir, progress_prefix=""):
        extra_downloads.append(asset["browser_download_url"])

    monkeypatch.setattr(cli_mod, "_download_and_unpack_zip", fake_unpack)

    install_llama.callback(tag="b10326", dest=str(bin_dir), name=None,
                          backend="cuda", gfx=None)

    urls = extra_downloads
    assert any("cudart.zip" in u for u in urls), (
        f"cudart runtime archive not downloaded: {urls}"
    )
