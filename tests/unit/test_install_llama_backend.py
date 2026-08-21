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

# Asset names from b10375 — the last release that shipped both Linux and
# Windows ROCm, using the current win-rocm-* naming.
B10375_ASSETS = [
    "llama-b10375-bin-ubuntu-rocm-7.14-x64.tar.gz",
    "llama-b10375-bin-ubuntu-vulkan-x64.tar.gz",
    "llama-b10375-bin-win-rocm-7.14-x64.zip",
    "llama-b10375-bin-win-vulkan-x64.zip",
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
                          backend="cuda", gfx=None, profile=False,
                          force_cpu=False)

    urls = extra_downloads
    assert any("cudart.zip" in u for u in urls), (
        f"cudart runtime archive not downloaded: {urls}"
    )


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


def test_profile_ttl_env_override(monkeypatch):
    """TOKI_PROFILE_TTL (seconds) replaces the 30-day default."""
    monkeypatch.setenv("TOKI_PROFILE_TTL", "60")
    import importlib

    from tokirecur import hardware
    importlib.reload(hardware)
    assert hardware.PROFILE_TTL_SECONDS == 60
    monkeypatch.delenv("TOKI_PROFILE_TTL")
    importlib.reload(hardware)
    assert hardware.PROFILE_TTL_SECONDS == 30 * 24 * 3600


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

