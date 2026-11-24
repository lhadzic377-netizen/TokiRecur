"""MemT native memory: the weights, the build that loads them, the API.

Upstream llama.cpp has never known the MemT tensor type, so a MemT model
needs the ``memt-native-memory`` branch of Brishen/llama.cpp. These
tests cover the three things toki has to get right: telling a MemT
weights file from an ordinary one, refusing to start it on a build that
cannot load it, and passing the memory fields through to the server.
"""

import struct

import pytest

from tokirecur import memt

_I32, _STRING = 5, 8


def _gguf(tensor_names: list[str], kv: dict[str, object] | None = None) -> bytes:
    """A minimal GGUF v3 header with a tensor directory."""
    kv = kv or {}
    out = bytearray(b"GGUF")
    out += struct.pack("<I", 3)
    out += struct.pack("<Q", len(tensor_names))
    out += struct.pack("<Q", len(kv))
    for key, value in kv.items():
        b = key.encode()
        out += struct.pack("<Q", len(b)) + b
        if isinstance(value, int):
            out += struct.pack("<I", _I32) + struct.pack("<i", value)
        elif isinstance(value, str):
            sb = value.encode()
            out += struct.pack("<I", _STRING) + struct.pack("<Q", len(sb)) + sb
        else:
            raise AssertionError(f"unsupported test value {value!r}")
    for name in tensor_names:
        b = name.encode()
        out += struct.pack("<Q", len(b)) + b
        out += struct.pack("<I", 1)      # n_dims
        out += struct.pack("<Q", 2)      # dims
        out += struct.pack("<I", 0)      # ggml type
        out += struct.pack("<Q", 0)      # offset
    return bytes(out)


def test_a_plain_model_has_no_memt_weights(tmp_path):
    path = tmp_path / "m.gguf"
    path.write_bytes(_gguf(["blk.0.attn_q.weight", "token_embd.weight"]))
    assert not memt.has_memt_weights(path)


def test_memt_weights_are_recognised(tmp_path):
    path = tmp_path / "memt.gguf"
    path.write_bytes(_gguf([
        "blk.0.attn_q.weight",
        "blk.0.memt_0.weight",
        "blk.1.memt_1.weight",
    ]))
    assert memt.has_memt_weights(path)
    assert memt.memt_tensor_names(path) == [
        "blk.0.memt_0.weight",
        "blk.1.memt_1.weight",
    ]


def test_a_damaged_header_is_not_a_memt_model(tmp_path):
    path = tmp_path / "broken.gguf"
    path.write_bytes(b"NOTGGUF")
    assert not memt.has_memt_weights(path)


def test_the_build_steps_say_what_the_branch_says():
    steps = memt.build_steps(
        __import__("pathlib").Path("/tmp/src"), backend="cuda", jobs=8,
    )
    joined = [" ".join(step) for step in steps]
    assert "memt-native-memory" in joined[0]
    assert "Brishen/llama.cpp" in joined[0]
    assert "-DGGML_CUDA=ON" in joined[1]
    assert "--target" in joined[2] and "llama-server" in joined[2]
    assert "llama-quantize" in joined[2]
    assert "-j 8" in joined[2]


def test_the_cpu_build_asks_for_no_gpu():
    flags = memt.cmake_flags("cpu")
    assert flags == []


def test_the_conversion_pins_the_memory_weights_to_q8_0(tmp_path):
    import pathlib

    steps = memt.convert_steps(
        pathlib.Path("/src/convert_hf_to_gguf.py"),
        pathlib.Path("/bin/llama-quantize"),
        pathlib.Path("/ckpt"),
        tmp_path / "out.gguf",
        quant="Q4_K_M",
    )
    convert, quantize = steps
    assert "--outtype" in convert and "bf16" in convert
    assert "--tensor-type" in quantize
    assert "memt_=q8_0" in quantize
    assert quantize[-1] == "Q4_K_M"


def test_the_build_hint_names_the_fork_and_the_branch():
    hint = memt.build_hint()
    assert memt.MEMT_FORK in hint
    assert memt.MEMT_BRANCH in hint


def test_a_build_that_never_heard_of_memt_is_told_so(tmp_path):
    from tokirecur.arch import binary_supports_memt

    binary = tmp_path / "llama-server"
    binary.write_bytes(b"\x7fELF" + b"nothing to see here" * 100)
    (tmp_path / "libllama.so").write_bytes(b"upstream llama.cpp" * 100)
    assert binary_supports_memt(str(binary)) is False


def test_a_memt_build_is_recognised(tmp_path):
    from tokirecur.arch import binary_supports_memt

    binary = tmp_path / "llama-server"
    binary.write_bytes(b"\x7fELF" + b"wrapper" * 100)
    (tmp_path / "libllama.so").write_bytes(b"memt_q8_0" + b"x" * 100)
    assert binary_supports_memt(str(binary)) is True


def test_starting_a_memt_model_on_an_ordinary_build_is_refused(tmp_path, monkeypatch):
    """A MemT model on an upstream build dies inside the load.

    The refusal happens before the server starts, with the way out.
    """
    from tokirecur import ctl

    gguf = tmp_path / "memt.gguf"
    gguf.write_bytes(_gguf(["blk.0.memt_0.weight"]))
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"\x7fELF" + b"upstream llama.cpp" * 100)
    (tmp_path / "libllama.so").write_bytes(b"upstream llama.cpp" * 100)

    spec = ctl.ModelSpec(
        name="memt",
        label="MemT",
        gguf=str(gguf),
        llama_port=9100,
        daemon_port=9101,
        ctx_size=1000,
        parallel=1,
        kv="f16",
        max_tokens=1000,
        model_id="memt",
        binary=str(binary),
    )
    with pytest.raises(RuntimeError) as err:
        ctl._check_memt(spec)
    assert "memt-native-memory" in str(err.value)


def test_an_ordinary_model_is_not_refused(tmp_path):
    from tokirecur import ctl

    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(_gguf(["blk.0.attn_q.weight"]))
    spec = ctl.ModelSpec(
        name="m",
        label="M",
        gguf=str(gguf),
        llama_port=9100,
        daemon_port=9101,
        ctx_size=1000,
        parallel=1,
        kv="f16",
        max_tokens=1000,
        model_id="m",
        binary=str(tmp_path / "llama-server"),
    )
    ctl._check_memt(spec)


async def test_the_launcher_says_which_models_have_a_memory(tmp_path, monkeypatch):
    """A model with a memory is not an ordinary model."""
    import yaml
    from textual.widgets import Static

    from tokirecur import ctl
    from tokirecur.tui.launcher import LauncherApp

    registry = tmp_path / "models.yaml"
    registry.write_text(yaml.safe_dump({"models": [{
        "name": "memt", "label": "MemT", "gguf": str(tmp_path / "m.gguf"),
        "llama_port": 9100, "daemon_port": 9101, "ctx_size": 1000,
        "max_tokens": 1000, "model_id": "memt", "parallel": 1, "kv": "f16",
        "memt": True,
    }]}))
    (tmp_path / "m.gguf").write_bytes(b"x")
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", registry)
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", tmp_path / "active")
    monkeypatch.setenv("TOKI_MODEL_REGISTRY", str(registry))
    monkeypatch.setenv("TOKI_ACTIVE_MODEL", str(tmp_path / "active"))

    app = LauncherApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.focus_pane = "models"
        app._refresh += 1
        await pilot.pause()
        pane = str(app.query_one("#pane", Static).render())
    assert "[memt]" in pane
