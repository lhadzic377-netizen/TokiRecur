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
        ((570, 0), (False, False)),
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
    report = diagnose.run()
    assert "gpu note:" in report
    assert "551.61" in report
    assert "install-llama --backend cuda" in report


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
    monkeypatch.setattr(os, "name", "nt")
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"x")
    (tmp_path / "cudart64_13.dll").write_bytes(b"x")
    (tmp_path / "cublas64_13.dll").write_bytes(b"x")
    assert diagnose.binary_backend(str(exe)) == "CUDA"


def test_binary_backend_windows_cpu(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "name", "nt")
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
    lines = diagnose.hint()
    assert any("TOKI_NGL" in line for line in lines)
    assert not any("install-llama" in line for line in lines)


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
