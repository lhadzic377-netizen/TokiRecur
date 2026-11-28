"""A vision projector the installed build cannot load.

llama.cpp keeps the multimodal projector in its mtmd library, and each
projector kind ("glm5next", "bailing_qwen3_vit", "qwen2vl", ...) is a
string in it. A build that predates the kind answers "unknown projector
type" and the server exits - so the model never starts at all, over a
projector the model would happily run without.

The fix has three parts, and this file pins all of them:

* the projector kind is read from the mmproj's own metadata;
* the build is asked whether it knows that kind;
* the model starts text-only instead of not at all, and says so.
"""

import struct

import pytest

from tokirecur import arch, ctl, diagnose

# ── a minimal GGUF with a projector kind ─────────────────────────────

_I32 = 4
_STRING = 8


def _gguf_bytes(kv: dict[str, object]) -> bytes:
    out = bytearray(b"GGUF")
    out += struct.pack("<I", 3)
    out += struct.pack("<Q", 0)
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
            raise AssertionError(f"unsupported value {value!r}")
    return bytes(out)


def _write_mmproj(tmp_path, kind: str = "glm5next"):
    p = tmp_path / "mmproj-F16.gguf"
    p.write_bytes(_gguf_bytes({
        "general.architecture": "clip",
        "general.type": "mmproj",
        "clip.projector_type": kind,
    }))
    return p


def _fake_build(tmp_path, kinds: tuple[str, ...] = ()):
    """A llama-server beside a libmtmd that knows ``kinds``."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    binary = bindir / "llama-server"
    binary.write_bytes(b"\x7fELF not really")
    mtmd = bindir / "libmtmd.so.1.0.0"
    mtmd.write_bytes(b"\x00" + b" ".join(k.encode() for k in kinds))
    return binary


# ── the metadata ──────────────────────────────────────────────────────


def test_the_projector_kind_is_read_from_the_mmproj(tmp_path):
    from tokirecur.ggufmeta import read_gguf_meta

    p = _write_mmproj(tmp_path, "bailing_qwen3_vit")
    assert read_gguf_meta(p).projector_type == "bailing_qwen3_vit"


def test_a_model_without_a_projector_kind_reads_as_none(tmp_path):
    from tokirecur.ggufmeta import read_gguf_meta

    p = tmp_path / "model.gguf"
    p.write_bytes(_gguf_bytes({"general.architecture": "glm5-next"}))
    assert read_gguf_meta(p).projector_type is None


# ── asking the build ─────────────────────────────────────────────────


def test_the_build_is_asked_about_the_projector_kind(tmp_path):
    binary = _fake_build(tmp_path, ("glm5next", "qwen2vl"))
    assert arch.binary_supports_projector(str(binary), "glm5next") is True
    assert arch.binary_supports_projector(str(binary), "bailing_qwen3_vit") is False


def test_no_opinion_without_an_mtmd_library(tmp_path):
    """No mtmd to read: no opinion, and no refusal on a guess."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    binary = bindir / "llama-server"
    binary.write_bytes(b"x")
    assert arch.binary_supports_projector(str(binary), "glm5next") is None


def test_no_opinion_without_a_binary():
    assert arch.binary_supports_projector(None, "glm5next") is None
    assert arch.binary_supports_projector("/nope/llama-server", "glm5next") is None


# ── the launcher asks for the model, not the projector ───────────────


def _spec(tmp_path, mmproj, binary):
    """A model with a projector, pointing at ``binary``."""
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"dummy")
    raw = {
        "name": "m", "label": "M", "gguf": str(gguf),
        "llama_port": 9100, "daemon_port": 9101, "ctx_size": 1024,
        "max_tokens": 512, "model_id": "m", "parallel": 1, "kv": "f16",
        "binary": str(binary), "mmproj": str(mmproj), "env": {},
    }
    return ctl.ModelSpec(raw=raw, **raw)


def test_the_projector_is_dropped_when_the_build_cannot_load_it(tmp_path):
    """The model runs text-only instead of not at all."""
    mmproj = _write_mmproj(tmp_path)
    binary = _fake_build(tmp_path, ("qwen2vl",))  # no glm5next
    spec = _spec(tmp_path, mmproj, binary)

    args = ctl._llama_args(spec, ctx_override=1024)
    assert "--mmproj" not in args


def test_the_projector_is_kept_when_the_build_knows_it(tmp_path):
    mmproj = _write_mmproj(tmp_path, "glm5next")
    binary = _fake_build(tmp_path, ("glm5next",))
    spec = _spec(tmp_path, mmproj, binary)

    args = ctl._llama_args(spec, ctx_override=1024)
    assert "--mmproj" in args
    assert args[args.index("--mmproj") + 1] == str(mmproj)


# ── and the doctor explains it ───────────────────────────────────────


def _spec_with_log(tmp_path, text: str):
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "llama.log").write_text(text, encoding="utf-8")

    class S:
        name = "m"

        def state_dir(self):
            return state

    return S()


def test_the_doctor_names_the_projector_the_build_lacks(tmp_path):
    spec = _spec_with_log(tmp_path, (
        "E mtmd_init_from_file: error: load_hparams: unknown projector "
        "type: glm5next\n"
        "E srv load_model: failed to load multimodal model\n"
    ))
    reason = diagnose.model_load_failure(spec)
    assert reason is not None
    assert "glm5next" in reason
    assert "text-only" in reason


def test_the_doctor_explains_a_tensor_layout_mismatch(tmp_path):
    """A file made for another llama.cpp says nothing useful on its own."""
    spec = _spec_with_log(tmp_path, (
        "E llama_model_load: error loading model: check_tensor_dims: "
        "tensor 'blk.0.ssm_a' has wrong shape; expected 1, 32, got 32, 1, 1, 1\n"
    ))
    reason = diagnose.model_load_failure(spec)
    assert reason is not None
    assert "blk.0.ssm_a" in reason
    assert "another llama.cpp" in reason


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
