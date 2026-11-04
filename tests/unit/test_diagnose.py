"""Unit tests for GPU-state log parsing and diagnostics."""

import os
import shutil
import subprocess
import sys

import pytest

from tokirecur import ctl, diagnose


@pytest.fixture
def iso_env(tmp_path, monkeypatch):
    """Route STATE_ROOT (and friends) into tmp so llama.log lands where
    gpu_state() reads it."""
    monkeypatch.setattr(ctl, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", tmp_path / "models.yaml")
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", tmp_path / "active")
    return tmp_path


def _spec(tmp_path, name="qwen"):
    gguf = tmp_path / f"{name}.gguf"
    gguf.write_bytes(b"dummy")
    return ctl.ModelSpec(
        name=name, label=name, gguf=str(gguf),
        llama_port=1, daemon_port=2, ctx_size=100, parallel=2,
        kv="f16", max_tokens=90, model_id=f"{name}-GGUF",
    )


def _log_dir(spec, tmp_path):
    d = spec.state_dir(root=tmp_path)
    d.mkdir(parents=True)
    return d


def test_gpu_state_missing_log(iso_env):
    spec = _spec(iso_env)
    assert ctl.gpu_state(spec) == {
        "backend": None, "device": None, "layers": None,
    }


def test_gpu_state_modern_cuda_lines(iso_env):
    spec = _spec(iso_env)
    d = _log_dir(spec, iso_env)
    (d / "llama.log").write_text(
        "0.00.001.000 I   - CUDA0   : NVIDIA GeForce GTX 1650 Mobile (sm_75, "
        "4096 MiB, 3057 MiB free)\n"
        "0.00.001.001 I system_info: n_threads = 8 | CUDA0 : NO_VMM = 1\n"
    )
    st = ctl.gpu_state(spec)
    assert st["backend"] == "CUDA"
    assert "GTX 1650" in (st["device"] or "")
    assert st["layers"] is None


def test_gpu_state_classic_offload_line(iso_env):
    spec = _spec(iso_env)
    d = _log_dir(spec, iso_env)
    (d / "llama.log").write_text(
        "llama_model_load: offloaded 34/37 layers to GPU\n"
    )
    st = ctl.gpu_state(spec)
    assert st["backend"] is None
    assert st["layers"] == "34/37"


def test_gpu_state_modern_rocm(iso_env):
    spec = _spec(iso_env)
    d = _log_dir(spec, iso_env)
    (d / "llama.log").write_text(
        "0.00.001.000 I   - ROCm0   : Radeon 8060S Graphics (112640 MiB, "
        "121602 MiB free)\n"
    )
    st = ctl.gpu_state(spec)
    assert st["backend"] == "ROCm"
    assert "Radeon" in (st["device"] or "")


def test_gpu_state_cpu_only(iso_env):
    spec = _spec(iso_env)
    d = _log_dir(spec, iso_env)
    (d / "llama.log").write_text(
        "0.00.001.000 I system_info: n_threads = 16 | CPU : AVX2 = 1\n"
    )
    st = ctl.gpu_state(spec)
    assert st == {"backend": None, "device": None, "layers": None}


def test_gpu_hardware_nvidia(monkeypatch):
    # The darwin branch of gpu_hardware() returns Metal before it ever
    # looks for a GPU, so this test has to say it is on Linux.
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kw: type("R", (), {
            "returncode": 0,
            "stdout": "NVIDIA GeForce GTX 1650 Mobile, 4096 MiB, 527.56\n",
        })(),
    )
    hw = diagnose.gpu_hardware()
    assert hw["backend"] == "CUDA"
    assert "1650" in (hw["name"] or "")
    assert hw["vram"] == "4096 MiB"


def test_gpu_hardware_none(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    monkeypatch.setattr(os.path, "exists", lambda _: False)
    monkeypatch.setattr(sys, "platform", "linux")
    hw = diagnose.gpu_hardware()
    assert hw["backend"] is None


def test_driver_version_parses(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(diagnose, "_run", lambda argv, timeout=5.0: "461.40")
    assert diagnose.driver_version() == (461, 40)


def test_driver_version_none_when_missing(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    assert diagnose.driver_version() is None


def test_driver_version_none_on_query_failure(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(diagnose, "_run", lambda argv, timeout=5.0: None)
    assert diagnose.driver_version() is None


@pytest.mark.parametrize(
    "driver,expected",
    [
        ((461, 40), (True, True)),
        ((551, 60), (True, True)),
        ((551, 61), (False, True)),
        ((569, 99), (False, True)),
        ((580, 64), (False, True)),
        ((580, 65), (False, False)),
        ((610, 0), (False, False)),
        (None, (False, False)),
    ],
)
def test_cuda_driver_verdict(driver, expected):
    assert diagnose.cuda_driver_verdict(driver) == expected


def test_doctor_notes_old_driver(monkeypatch):
    monkeypatch.setattr(diagnose, "gpu_hardware", lambda: {
        "backend": "CUDA", "name": "GTX 1650", "vram": "4096 MiB",
        "driver": "461.40",
    })
    monkeypatch.setattr(diagnose, "driver_version", lambda: (461, 40))
    monkeypatch.setattr(
        diagnose, "installed_build",
        lambda: {"state_bin": "C:/bin/llama-server.exe", "registry": []},
    )
    monkeypatch.setattr(diagnose, "_active_spec", lambda: None)
    monkeypatch.setattr(diagnose.sys, "platform", "win32")
    report = diagnose.run()
    assert "gpu note:" in report
    assert "551.61" in report
    assert "install-llama --backend cuda" in report


def test_doctor_notes_old_driver_linux(monkeypatch):
    monkeypatch.setattr(diagnose, "gpu_hardware", lambda: {
        "backend": "CUDA", "name": "RTX 3090", "vram": "24576 MiB",
        "driver": "550.54",
    })
    monkeypatch.setattr(diagnose, "driver_version", lambda: (550, 54))
    monkeypatch.setattr(
        diagnose, "installed_build",
        lambda: {"state_bin": "/opt/bin/llama-server", "registry": []},
    )
    monkeypatch.setattr(diagnose, "_active_spec", lambda: None)
    monkeypatch.setattr(diagnose.sys, "platform", "linux")
    report = diagnose.run()
    assert "570.26" in report
    assert "CUDA 12.8+" in report
    assert "551.61" not in report


def test_hint_no_gpu_returns_empty(monkeypatch):
    monkeypatch.setattr(diagnose, "gpu_hardware", lambda: {
        "backend": None, "name": None, "vram": None, "driver": None,
    })
    assert diagnose.hint() == []


def test_hint_gpu_but_cpu_log(monkeypatch, iso_env):
    spec = _spec(iso_env)
    monkeypatch.setattr(diagnose, "_active_spec", lambda: spec)
    monkeypatch.setattr(diagnose, "gpu_hardware", lambda: {
        "backend": "CUDA", "name": "GTX 1650", "vram": None, "driver": None,
    })
    lines = diagnose.hint()
    assert any("CPU" in line for line in lines)
    assert any("install-llama --backend cuda" in line for line in lines)
    assert any(f"--name {spec.name}" in line for line in lines)


def test_hint_gpu_used_returns_empty(monkeypatch, iso_env):
    spec = _spec(iso_env)
    d = _log_dir(spec, iso_env)
    (d / "llama.log").write_text("0.00.000 I   - CUDA0 : GTX 1650 (sm_75)\n")
    monkeypatch.setattr(diagnose, "_active_spec", lambda: spec)
    monkeypatch.setattr(diagnose, "gpu_hardware", lambda: {
        "backend": "CUDA", "name": "GTX 1650", "vram": None, "driver": None,
    })
    assert diagnose.hint() == []


def test_binary_backend_windows_cuda(monkeypatch, tmp_path):
    monkeypatch.setattr(diagnose.winproc, "IS_WINDOWS", True)
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"x")
    (tmp_path / "ggml-cuda.dll").write_bytes(b"x")
    (tmp_path / "cudart64_13.dll").write_bytes(b"x")
    (tmp_path / "cublas64_13.dll").write_bytes(b"x")
    assert diagnose.binary_backend(str(exe)) == "CUDA"
    assert diagnose.cuda_runtime_ready(str(exe))


def test_binary_backend_windows_cuda_without_runtime(monkeypatch, tmp_path):
    """The backend module is still CUDA, but the runtime is not ready —
    the launch check must refuse -ngl 99 instead of silently going CPU."""
    monkeypatch.setattr(diagnose.winproc, "IS_WINDOWS", True)
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"x")
    (tmp_path / "ggml-cuda.dll").write_bytes(b"x")
    assert diagnose.binary_backend(str(exe)) == "CUDA"
    assert not diagnose.cuda_runtime_ready(str(exe))


def test_cuda_runtime_ready_accepts_toolkit_on_path(monkeypatch, tmp_path):
    """A locally-built CUDA server finds cudart through the toolkit PATH."""
    monkeypatch.setattr(diagnose.winproc, "IS_WINDOWS", True)
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"x")
    toolkit = tmp_path / "cuda" / "bin"
    toolkit.mkdir(parents=True)
    (toolkit / "cudart64_13.dll").write_bytes(b"x")
    monkeypatch.setenv("PATH", str(toolkit))
    assert diagnose.cuda_runtime_ready(str(exe))


def test_binary_backend_windows_stale_cudart_does_not_win(monkeypatch, tmp_path):
    """A leftover cudart DLL next to a Vulkan build must not read as CUDA."""
    monkeypatch.setattr(diagnose.winproc, "IS_WINDOWS", True)
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"x")
    (tmp_path / "ggml-vulkan.dll").write_bytes(b"x")
    (tmp_path / "cudart64_12.dll").write_bytes(b"x")
    assert diagnose.binary_backend(str(exe)) == "Vulkan"


def test_binary_backend_windows_hip_is_rocm(monkeypatch, tmp_path):
    """The Windows ROCm release ships ggml-hip.dll, not ggml-rocm.dll."""
    monkeypatch.setattr(diagnose.winproc, "IS_WINDOWS", True)
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"x")
    (tmp_path / "ggml-hip.dll").write_bytes(b"x")
    (tmp_path / "amdhip64_7.dll").write_bytes(b"x")
    assert diagnose.binary_backend(str(exe)) == "ROCm"


def test_binary_backend_windows_cpu(monkeypatch, tmp_path):
    monkeypatch.setattr(diagnose.winproc, "IS_WINDOWS", True)
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"x")
    assert diagnose.binary_backend(str(exe)) == "CPU"


def test_binary_backend_linux_rocm(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "name", "posix")
    exe = tmp_path / "llama-server"
    exe.write_bytes(b"x")
    monkeypatch.setattr(
        diagnose,
        "_run",
        lambda argv, timeout=5.0: "libggml-base.so\nlibggml-cpu.so\n"
        "libggml-rocm.so (HSA)\nlibhipblas.so.2",
    )
    assert diagnose.binary_backend(str(exe)) == "ROCm"


def test_binary_backend_linux_cpu(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "name", "posix")
    exe = tmp_path / "llama-server"
    exe.write_bytes(b"x")
    monkeypatch.setattr(
        diagnose, "_run", lambda argv, timeout=5.0: "libggml-base.so\nlibggml-cpu.so",
    )
    assert diagnose.binary_backend(str(exe)) == "CPU"


def test_binary_backend_missing(tmp_path):
    assert diagnose.binary_backend(str(tmp_path / "nope")) is None


def test_binary_backend_linux_cuda_sibling_wins_over_none(monkeypatch, tmp_path):
    # The sibling libggml-*.so check is Linux-only; say so here so
    # the branch is exercised wherever the test runs.
    monkeypatch.setattr(diagnose.sys, "platform", "linux")
    """A GPU build with no runtime enumerates (none); the sibling module
    still identifies the build so the doctor can name the missing lib."""
    exe = tmp_path / "llama-server"
    exe.write_bytes(b"x")
    (tmp_path / "libggml-cuda.so").write_bytes(b"x")
    monkeypatch.setattr(
        diagnose, "_run",
        lambda argv, timeout=5.0: "Available devices:\n  (none)",
    )
    assert diagnose.binary_backend(str(exe)) == "CUDA"


def test_binary_backend_linux_sycl_sibling(monkeypatch, tmp_path):
    # The sibling libggml-*.so check is Linux-only; say so here so
    # the branch is exercised wherever the test runs.
    monkeypatch.setattr(diagnose.sys, "platform", "linux")
    exe = tmp_path / "llama-server"
    exe.write_bytes(b"x")
    (tmp_path / "libggml-sycl.so").write_bytes(b"x")
    monkeypatch.setattr(diagnose, "_run", lambda argv, timeout=5.0: None)
    assert diagnose.binary_backend(str(exe)) == "SYCL"


def test_binary_backend_linux_openvino_sibling(monkeypatch, tmp_path):
    # The sibling libggml-*.so check is Linux-only; say so here so
    # the branch is exercised wherever the test runs.
    monkeypatch.setattr(diagnose.sys, "platform", "linux")
    exe = tmp_path / "llama-server"
    exe.write_bytes(b"x")
    (tmp_path / "libggml-openvino.so").write_bytes(b"x")
    monkeypatch.setattr(diagnose, "_run", lambda argv, timeout=5.0: None)
    assert diagnose.binary_backend(str(exe)) == "OpenVINO"


def test_cuda_runtime_ready_linux_sibling(monkeypatch, tmp_path):
    monkeypatch.setattr(diagnose, "_cuda_lib_dirs", lambda: [])
    monkeypatch.setattr(diagnose, "_ldconfig_has", lambda *names: False)
    exe = tmp_path / "llama-server"
    exe.write_bytes(b"x")
    assert not diagnose.cuda_runtime_ready(str(exe))
    (tmp_path / "libcudart.so.13").write_bytes(b"x")
    assert diagnose.cuda_runtime_ready(str(exe))
    assert diagnose.cuda_runtime_libs(tmp_path) == ["libcudart.so.13"]


def test_rocm_runtime_ready_linux_sibling(monkeypatch, tmp_path):
    monkeypatch.setattr(diagnose, "_ldconfig_has", lambda *names: False)
    monkeypatch.setattr(
        diagnose, "_posix_lib_dirs", lambda *extra: [],
    )
    exe = tmp_path / "llama-server"
    exe.write_bytes(b"x")
    assert not diagnose.rocm_runtime_ready(str(exe))
    (tmp_path / "libamdhip64.so.7").write_bytes(b"x")
    assert diagnose.rocm_runtime_ready(str(exe))


def test_vulkan_runtime_ready_linux_sibling(monkeypatch, tmp_path):
    monkeypatch.setattr(diagnose, "_ldconfig_has", lambda *names: False)
    monkeypatch.setattr(
        diagnose, "_posix_lib_dirs", lambda *extra: [],
    )
    monkeypatch.setattr(diagnose.sys, "platform", "linux")
    exe = tmp_path / "llama-server"
    exe.write_bytes(b"x")
    assert not diagnose.vulkan_runtime_ready(str(exe))
    (tmp_path / "libvulkan.so.1").write_bytes(b"x")
    assert diagnose.vulkan_runtime_ready(str(exe))


def test_sycl_runtime_ready_linux_sibling(monkeypatch, tmp_path):
    monkeypatch.setattr(diagnose, "_sycl_lib_dirs", lambda: [])
    monkeypatch.setattr(diagnose, "_ldconfig_has", lambda *names: False)
    exe = tmp_path / "llama-server"
    exe.write_bytes(b"x")
    assert not diagnose.sycl_runtime_ready(str(exe))
    missing = diagnose.sycl_runtime_missing(str(exe))
    assert len(missing) == 3
    for name in ("libsycl.so.9", "libmkl_sycl.so.2", "libze_loader.so.1"):
        (tmp_path / name).write_bytes(b"x")
    assert diagnose.sycl_runtime_ready(str(exe))
    assert diagnose.sycl_runtime_missing(str(exe)) == []


def test_openvino_runtime_and_gpu_plugin_linux(monkeypatch, tmp_path):
    monkeypatch.setattr(diagnose, "_openvino_lib_dirs", lambda: [])
    monkeypatch.setattr(diagnose, "_ldconfig_has", lambda *names: False)
    exe = tmp_path / "llama-server"
    exe.write_bytes(b"x")
    assert not diagnose.openvino_runtime_ready(str(exe))
    (tmp_path / "libopenvino.so.2600").write_bytes(b"x")
    assert diagnose.openvino_runtime_ready(str(exe))
    assert not diagnose.openvino_gpu_plugin_ready(str(exe))
    (tmp_path / "libopenvino_intel_gpu_plugin.so").write_bytes(b"x")
    assert diagnose.openvino_gpu_plugin_ready(str(exe))


def test_hint_cpu_build_advises_reinstall(monkeypatch, iso_env):
    spec = _spec(iso_env)
    monkeypatch.setattr(diagnose, "_active_spec", lambda: spec)
    monkeypatch.setattr(diagnose, "gpu_hardware", lambda: {
        "backend": "CUDA", "name": "GTX 1650", "vram": None, "driver": None,
    })
    monkeypatch.setattr(ctl, "_resolve_binary", lambda spec: "/tmp/llama-server")
    monkeypatch.setattr(diagnose, "binary_backend", lambda b: "CPU")
    lines = diagnose.hint()
    assert any("install-llama --backend cuda" in line for line in lines)


def test_hint_gpu_build_advises_ngl_check(monkeypatch, iso_env):
    spec = _spec(iso_env)
    monkeypatch.setattr(diagnose, "_active_spec", lambda: spec)
    monkeypatch.setattr(diagnose, "gpu_hardware", lambda: {
        "backend": "CUDA", "name": "GTX 1650", "vram": None, "driver": None,
    })
    monkeypatch.setattr(ctl, "_resolve_binary", lambda spec: "/tmp/llama-server")
    monkeypatch.setattr(diagnose, "binary_backend", lambda b: "CUDA")
    monkeypatch.setattr(diagnose, "cuda_runtime_ready", lambda b: True)
    lines = diagnose.hint()
    assert any("TOKI_NGL" in line for line in lines)
    assert not any("install-llama" in line for line in lines)


def test_hint_cuda_missing_runtime(monkeypatch, iso_env):
    spec = _spec(iso_env)
    monkeypatch.setattr(diagnose, "_active_spec", lambda: spec)
    monkeypatch.setattr(diagnose, "gpu_hardware", lambda: {
        "backend": "CUDA", "name": "GTX 1650", "vram": None, "driver": None,
    })
    monkeypatch.setattr(ctl, "_resolve_binary", lambda spec: "/tmp/llama-server")
    monkeypatch.setattr(diagnose, "binary_backend", lambda b: "CUDA")
    monkeypatch.setattr(diagnose.winproc, "IS_WINDOWS", True)
    monkeypatch.setattr(diagnose, "cuda_runtime_ready", lambda b: False)
    lines = diagnose.hint()
    assert any("cudart64" in line for line in lines)
    assert any("install-llama --backend cuda" in line for line in lines)


def test_run_includes_sections(monkeypatch, iso_env):
    spec = _spec(iso_env)
    d = _log_dir(spec, iso_env)
    (d / "llama.log").write_text("0.00.000 I   - CUDA0 : GTX 1650 (sm_75)\n")
    monkeypatch.setattr(diagnose, "_active_spec", lambda: spec)
    monkeypatch.setattr(diagnose, "gpu_hardware", lambda: {
        "backend": "CUDA", "name": "GTX 1650", "vram": "4096 MiB",
        "driver": "527.56",
    })
    out = diagnose.run()
    assert "gpu: CUDA · GTX 1650" in out
    assert f"model: {spec.name}" in out
    assert "llama-server (state bin)" in out
    assert "model gpu: CUDA" in out


def test_binary_backend_asks_the_binary_before_ldd(monkeypatch, tmp_path):
    """A GPU build dlopens its backend; ldd alone reads it as CPU.

    llama.cpp ships libggml-vulkan/cuda/hip as separate libraries that
    are loaded at runtime, so a working Vulkan build links only the CPU
    core. install-llama used the link table as proof and rejected a
    correct download while the binary itself enumerated Vulkan0.
    """
    monkeypatch.setattr(os, "name", "posix")
    exe = tmp_path / "llama-server"
    exe.write_bytes(b"x")

    def fake_run(argv, timeout=5.0):
        if argv[-1] == "--list-devices":
            return (
                "Available devices:\n"
                "  Vulkan0: Intel(R) Graphics (MTL) (11615 MiB, 7091 MiB free)"
            )
        return "libggml-base.so\nlibggml-cpu.so"

    monkeypatch.setattr(diagnose, "_run", fake_run)
    assert diagnose.binary_backend(str(exe)) == "Vulkan"


def test_gpu_state_modern_sycl(iso_env):
    spec = _spec(iso_env)
    d = _log_dir(spec, iso_env)
    (d / "llama.log").write_text(
        "0.00.001.000 I   - SYCL0   : Intel(R) Arc(TM) A770 Graphics "
        "(16384 MiB, 15000 MiB free)\n"
    )
    st = ctl.gpu_state(spec)
    assert st["backend"] == "SYCL"
    assert "A770" in (st["device"] or "")


def test_gpu_state_modern_openvino(iso_env):
    spec = _spec(iso_env)
    d = _log_dir(spec, iso_env)
    (d / "llama.log").write_text(
        "0.00.001.000 I   - OPENVINO0 : Intel(R) Arc(TM) A770 Graphics\n"
    )
    st = ctl.gpu_state(spec)
    assert st["backend"] == "OPENVINO"
    assert "A770" in (st["device"] or "")


def test_binary_backend_windows_sycl(monkeypatch, tmp_path):
    monkeypatch.setattr(diagnose.winproc, "IS_WINDOWS", True)
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"x")
    (tmp_path / "ggml-sycl.dll").write_bytes(b"x")
    (tmp_path / "sycl9.dll").write_bytes(b"x")
    assert diagnose.binary_backend(str(exe)) == "SYCL"
    assert diagnose.sycl_runtime_ready(str(exe))


def test_binary_backend_windows_openvino(monkeypatch, tmp_path):
    monkeypatch.setattr(diagnose.winproc, "IS_WINDOWS", True)
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"x")
    (tmp_path / "ggml-openvino.dll").write_bytes(b"x")
    (tmp_path / "openvino.dll").write_bytes(b"x")
    assert diagnose.binary_backend(str(exe)) == "OpenVINO"
    assert diagnose.openvino_runtime_ready(str(exe))


def test_rocm_runtime_ready(monkeypatch, tmp_path):
    monkeypatch.setattr(diagnose.winproc, "IS_WINDOWS", True)
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"x")
    (tmp_path / "ggml-hip.dll").write_bytes(b"x")
    assert not diagnose.rocm_runtime_ready(str(exe))
    (tmp_path / "amdhip64_7.dll").write_bytes(b"x")
    (tmp_path / "amd_comgr.dll").write_bytes(b"x")
    assert diagnose.rocm_runtime_ready(str(exe))
    assert diagnose.rocm_runtime_dlls(tmp_path) == [
        "amd_comgr.dll", "amdhip64_7.dll",
    ]


def test_vulkan_runtime_ready_finds_system_loader(monkeypatch, tmp_path):
    monkeypatch.setattr(diagnose.winproc, "IS_WINDOWS", True)
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"x")
    root = tmp_path / "Windows"
    (root / "System32").mkdir(parents=True)
    (root / "System32" / "vulkan-1.dll").write_bytes(b"x")
    monkeypatch.setenv("SystemRoot", str(root))
    assert diagnose.vulkan_runtime_ready(str(exe))
    (root / "System32" / "vulkan-1.dll").unlink()
    assert not diagnose.vulkan_runtime_ready(str(exe))


def test_sycl_runtime_ready(monkeypatch, tmp_path):
    monkeypatch.setattr(diagnose.winproc, "IS_WINDOWS", True)
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"x")
    assert not diagnose.sycl_runtime_ready(str(exe))
    (tmp_path / "sycl9.dll").write_bytes(b"x")
    assert diagnose.sycl_runtime_ready(str(exe))
    assert diagnose.sycl_runtime_dlls(tmp_path) == ["sycl9.dll"]


def _win_gpu(monkeypatch, gpus):
    monkeypatch.setattr(diagnose.winproc, "IS_WINDOWS", True)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(diagnose.shutil, "which", lambda _: None)
    monkeypatch.setattr(diagnose.os.path, "exists", lambda _: False)
    monkeypatch.setattr(
        "tokirecur.hardware._enumerate_gpus_windows", lambda: gpus,
    )
    return diagnose.gpu_hardware()


def test_gpu_hardware_windows_amd_uses_vulkan(monkeypatch):
    from tokirecur.hardware import GPUProfile

    hw = _win_gpu(monkeypatch, [GPUProfile(
        vendor="AMD", name="AMD Radeon RX 7600", vram_gb=8.0,
        driver=None, vulkan_score=1.0, shared_ram=False,
        use_for_llm=True, reason="discrete GPU",
    )])
    assert hw["backend"] == "Vulkan"
    assert hw["name"] == "AMD Radeon RX 7600"
    assert hw["vram"] == "8.0 GB"


def test_gpu_hardware_windows_intel_arc_uses_sycl(monkeypatch):
    from tokirecur.hardware import GPUProfile

    hw = _win_gpu(monkeypatch, [GPUProfile(
        vendor="Intel", name="Intel(R) Arc(TM) A770 Graphics",
        vram_gb=16.0, driver=None, vulkan_score=1.0, shared_ram=False,
        use_for_llm=True, reason="discrete GPU",
    )])
    assert hw["backend"] == "SYCL"
    assert hw["name"] == "Intel(R) Arc(TM) A770 Graphics"


def test_gpu_hardware_windows_intel_igpu_reports_name_not_backend(monkeypatch):
    from tokirecur.hardware import GPUProfile

    hw = _win_gpu(monkeypatch, [GPUProfile(
        vendor="Intel", name="Intel(R) UHD Graphics", vram_gb=1.0,
        driver=None, vulkan_score=1.0, shared_ram=True,
        use_for_llm=False, reason="shared RAM (iGPU)",
    )])
    assert hw["backend"] is None
    assert hw["name"] == "Intel(R) UHD Graphics"


def test_run_reports_present_but_unused_gpu(monkeypatch, iso_env):
    monkeypatch.setattr(diagnose, "gpu_hardware", lambda: {
        "backend": None, "name": "Intel(R) UHD Graphics",
        "vram": "1.0 GB", "driver": None,
    })
    out = diagnose.run()
    assert "present, not used for LLM offload" in out


def test_hint_rocm_missing_runtime(monkeypatch, iso_env):
    spec = _spec(iso_env)
    monkeypatch.setattr(diagnose, "_active_spec", lambda: spec)
    monkeypatch.setattr(diagnose, "gpu_hardware", lambda: {
        "backend": "Vulkan", "name": "RX 7600", "vram": None, "driver": None,
    })
    monkeypatch.setattr(ctl, "_resolve_binary", lambda spec: "/tmp/llama-server")
    monkeypatch.setattr(diagnose, "binary_backend", lambda b: "ROCm")
    monkeypatch.setattr(diagnose.winproc, "IS_WINDOWS", True)
    monkeypatch.setattr(diagnose, "rocm_runtime_ready", lambda b: False)
    lines = diagnose.hint()
    assert any("amdhip64" in line for line in lines)
    assert any("--backend rocm" in line for line in lines)


def test_hint_vulkan_missing_loader(monkeypatch, iso_env):
    spec = _spec(iso_env)
    monkeypatch.setattr(diagnose, "_active_spec", lambda: spec)
    monkeypatch.setattr(diagnose, "gpu_hardware", lambda: {
        "backend": "Vulkan", "name": "RX 7600", "vram": None, "driver": None,
    })
    monkeypatch.setattr(ctl, "_resolve_binary", lambda spec: "/tmp/llama-server")
    monkeypatch.setattr(diagnose, "binary_backend", lambda b: "Vulkan")
    monkeypatch.setattr(diagnose.winproc, "IS_WINDOWS", True)
    monkeypatch.setattr(diagnose, "vulkan_runtime_ready", lambda b: False)
    lines = diagnose.hint()
    assert any("vulkan-1.dll" in line for line in lines)
    assert any("GPU driver" in line for line in lines)


def test_hint_sycl_missing_runtime(monkeypatch, iso_env):
    spec = _spec(iso_env)
    monkeypatch.setattr(diagnose, "_active_spec", lambda: spec)
    monkeypatch.setattr(diagnose, "gpu_hardware", lambda: {
        "backend": "SYCL", "name": "Arc A770", "vram": None, "driver": None,
    })
    monkeypatch.setattr(ctl, "_resolve_binary", lambda spec: "/tmp/llama-server")
    monkeypatch.setattr(diagnose, "binary_backend", lambda b: "SYCL")
    monkeypatch.setattr(diagnose.winproc, "IS_WINDOWS", True)
    monkeypatch.setattr(diagnose, "sycl_runtime_ready", lambda b: False)
    lines = diagnose.hint()
    assert any("sycl*.dll" in line for line in lines)
    assert any("--backend sycl" in line for line in lines)


def test_gpu_state_modern_metal(iso_env):
    """llama.cpp names the Metal device MTL0, not Metal0."""
    spec = _spec(iso_env)
    d = _log_dir(spec, iso_env)
    (d / "llama.log").write_text(
        "0.00.001.000 I   - MTL0    : Apple M2 (24576 MiB, 23000 MiB free)\n"
    )
    st = ctl.gpu_state(spec)
    assert st["backend"] == "Metal"
    assert st["device"] == "Apple M2"


def test_binary_backend_macos_metal(monkeypatch, tmp_path):
    exe = tmp_path / "llama-server"
    exe.write_bytes(b"x")

    def fake_run(argv, timeout=5.0):
        if argv[-1] == "--list-devices":
            return (
                "Available devices:\n"
                "  MTL0: Apple M2 (24576 MiB, 23000 MiB free)"
            )
        return None

    monkeypatch.setattr(diagnose, "_run", fake_run)
    assert diagnose.binary_backend(str(exe)) == "Metal"


def test_binary_backend_macos_cpu_from_none(monkeypatch, tmp_path):
    """--list-devices prints (none) when only the CPU backend is present."""
    exe = tmp_path / "llama-server"
    exe.write_bytes(b"x")

    def fake_run(argv, timeout=5.0):
        if argv[-1] == "--list-devices":
            return "Available devices:\n  (none)"
        return None

    monkeypatch.setattr(diagnose, "_run", fake_run)
    assert diagnose.binary_backend(str(exe)) == "CPU"


def test_binary_backend_macos_metal_from_sibling_dylib(monkeypatch, tmp_path):
    """No ldd on macOS; the release ships libggml-metal*.dylib."""
    exe = tmp_path / "llama-server"
    exe.write_bytes(b"x")
    (tmp_path / "libggml-metal.0.28.0.dylib").write_bytes(b"x")
    (tmp_path / "libggml-cpu.dylib").write_bytes(b"x")
    monkeypatch.setattr(diagnose.sys, "platform", "darwin")
    monkeypatch.setattr(diagnose, "_run", lambda argv, timeout=5.0: None)
    assert diagnose.binary_backend(str(exe)) == "Metal"


def test_binary_backend_macos_cpu_from_sibling_dylib(monkeypatch, tmp_path):
    exe = tmp_path / "llama-server"
    exe.write_bytes(b"x")
    (tmp_path / "libggml-cpu.dylib").write_bytes(b"x")
    monkeypatch.setattr(diagnose.sys, "platform", "darwin")
    monkeypatch.setattr(diagnose, "_run", lambda argv, timeout=5.0: None)
    assert diagnose.binary_backend(str(exe)) == "CPU"


def test_binary_backend_macos_otool_metal(monkeypatch, tmp_path):
    exe = tmp_path / "llama-server"
    exe.write_bytes(b"x")

    def fake_run(argv, timeout=5.0):
        if argv[0] == "otool":
            return (
                f"{exe}:\n"
                "\t@rpath/libggml-metal.0.dylib (compatibility version 0.0.0)\n"
                "\t/usr/lib/libSystem.B.dylib"
            )
        return None

    monkeypatch.setattr(diagnose.sys, "platform", "darwin")
    monkeypatch.setattr(diagnose, "_run", fake_run)
    assert diagnose.binary_backend(str(exe)) == "Metal"


def test_hint_macos_metal_without_mtl_device(monkeypatch, iso_env):
    """A Metal build on a Mac/VM that cannot enumerate MTL must say so."""
    spec = _spec(iso_env)
    monkeypatch.setattr(diagnose, "_active_spec", lambda: spec)
    monkeypatch.setattr(diagnose, "gpu_hardware", lambda: {
        "backend": "Metal", "name": "Apple M2", "vram": None, "driver": None,
    })
    monkeypatch.setattr(ctl, "_resolve_binary", lambda spec: "/tmp/llama-server")
    monkeypatch.setattr(diagnose, "binary_backend", lambda b: "Metal")
    monkeypatch.setattr(diagnose, "metal_device_ready", lambda b: False)
    monkeypatch.setattr(diagnose.sys, "platform", "darwin")
    lines = diagnose.hint()
    assert any("cannot enumerate an MTL device" in line for line in lines)
    assert any("TOKI_NGL=0" in line for line in lines)


def test_hint_explicit_cpu_env_returns_empty(monkeypatch, iso_env):
    """TOKI_NGL=0 is an intentional CPU launch, not a missing GPU."""
    spec = _spec(iso_env)
    monkeypatch.setattr(diagnose, "_active_spec", lambda: spec)
    monkeypatch.setattr(diagnose, "gpu_hardware", lambda: {
        "backend": "CUDA", "name": "GTX 1650", "vram": None, "driver": None,
    })
    monkeypatch.setenv("TOKI_NGL", "0")
    assert diagnose.hint() == []


def test_hint_explicit_cpu_registry_returns_empty(monkeypatch, iso_env):
    """A registry -ngl 0 pin is an intentional CPU launch."""
    spec = _spec(iso_env)
    spec.extra_args = ["-ngl", "0"]
    monkeypatch.setattr(diagnose, "_active_spec", lambda: spec)
    monkeypatch.setattr(diagnose, "gpu_hardware", lambda: {
        "backend": "CUDA", "name": "GTX 1650", "vram": None, "driver": None,
    })
    monkeypatch.delenv("TOKI_NGL", raising=False)
    assert diagnose.hint() == []
