"""Unit tests for speculative-decoding drafts (DSpine / DFlash2).

A draft is a second model tied to one target. Three things have to hold
or the pairing is worse than no pairing at all:

1. the draft's flags reach llama-server as separate argv tokens;
2. the fit planner reserves the drafter's VRAM, because auto-fit that
   ignores it produces a server that cannot allocate (upstream's own
   model card warns about this);
3. a build that cannot load the draft is detected before launch, not by
   a failure deep in the model load.
"""

from pathlib import Path

import pytest
import yaml

from tokirecur import draft as draft_mod
from tokirecur.ctl import (
    _build_spec,
    _check_draft,
    _draft_n_max_for,
    _fit_plan,
    _pull_draft,
    draft_for,
    load_registry,
    pull_model,
    save_registry,
)
from tokirecur.draft import (
    DRAFT_N_MAX_DEFAULT,
    DraftSpec,
    draft_needs_dspine,
    draft_supported_by,
    draft_vram_gb,
    dspine_adapter_hint,
    looks_like_dspine_adapter,
)
from tokirecur.draftmatch import DraftCandidate, find_draft, target_family

# ── draft model ────────────────────────────────────────────────────────


def test_draft_spec_argv_is_one_token_per_flag(tmp_path):
    draft = DraftSpec(gguf=str(tmp_path / "draft.gguf"))
    args = draft.llama_args()
    assert args[:2] == ["--spec-draft-model", str(tmp_path / "draft.gguf")]
    assert "--spec-type" in args
    assert args[args.index("--spec-type") + 1] == "draft-dflash"
    assert args[args.index("--spec-draft-n-max") + 1] == str(DRAFT_N_MAX_DEFAULT)
    # A joined "--flag value" is a startup failure; every flag is alone.
    assert all(" " not in a for a in args)


def test_draft_spec_round_trips_through_the_registry_form():
    raw = {
        "gguf": "/models/draft.gguf",
        "spec_type": "draft-dflash",
        "n_max": 7,
        "repo": "aj9o9/Qwen3.8-27B-DSpine-b16-GGUF",
        "file": "draft-Q4_K_M.gguf",
        "extra_args": ["--spec-draft-type-k", "q8_0"],
    }
    draft = DraftSpec.from_dict(raw)
    assert draft is not None
    assert draft.n_max == 7
    assert draft.llama_args()[-2:] == ["--spec-draft-type-k", "q8_0"]
    assert draft.to_dict() == raw


def test_draft_spec_reads_a_bare_path_and_tolerates_junk():
    assert DraftSpec.from_dict("/models/draft.gguf").gguf == "/models/draft.gguf"
    assert DraftSpec.from_dict({}) is None
    assert DraftSpec.from_dict(None) is None
    assert DraftSpec.from_dict(42) is None
    assert DraftSpec.from_dict({"gguf": ""}) is None
    # A broken n_max is not a reason to refuse the draft.
    bad = DraftSpec.from_dict({"gguf": "/d.gguf", "n_max": "many"})
    assert bad is not None and bad.n_max == DRAFT_N_MAX_DEFAULT


def test_model_spec_reads_the_flat_draft_fields():
    spec = _build_spec({
        "name": "qwen38", "label": "Qwen3.8 27B", "gguf": "/t.gguf",
        "llama_port": 8099, "daemon_port": 8799, "ctx_size": 32768,
        "max_tokens": 1,
        "draft": "/d.gguf", "draft_n_max": 7,
        "draft_repo": "aj9o9/repo", "draft_file": "d.gguf",
    })
    draft = draft_for(spec)
    assert draft is not None
    assert draft.n_max == 7
    assert draft.repo == "aj9o9/repo"
    # A model without a draft has none, not an empty one.
    assert draft_for(_build_spec({
        "name": "plain", "llama_port": 1, "daemon_port": 2,
        "ctx_size": 4096, "max_tokens": 1, "gguf": "/t.gguf",
    })) is None


def test_registry_round_trip_keeps_the_draft(tmp_path):
    registry = tmp_path / "models.yaml"
    registry.write_text(yaml.safe_dump({"models": [{
        "name": "qwen38", "label": "Qwen3.8 27B", "gguf": "/t.gguf",
        "llama_port": 8099, "daemon_port": 8799, "ctx_size": 32768,
        "parallel": 1, "kv": "q8_0", "max_tokens": 1, "model_id": "qwen38",
        "draft": "/models/draft.gguf", "draft_n_max": 15,
        "draft_spec_type": "draft-dflash", "draft_repo": "aj9o9/repo",
        "draft_file": "draft.gguf",
    }]}))
    spec = load_registry(registry)[0]
    save_registry([spec], registry)
    back = yaml.safe_load(registry.read_text())["models"][0]
    assert back["draft"] == "/models/draft.gguf"
    assert back["draft_n_max"] == 15
    assert back["draft_repo"] == "aj9o9/repo"


def test_draft_vram_counts_weights_kv_and_recurrent_copies(
        tmp_path, monkeypatch):
    weights = tmp_path / "draft.gguf"
    weights.write_bytes(b"x" * 1_000_000_000)  # 1 GB on disk
    draft = DraftSpec(gguf=str(weights), n_max=15)

    class Meta:
        n_embd = 5120
        n_layer = 5
        n_head = 32
        n_head_kv = 8
        complete = True

    monkeypatch.setattr(draft_mod, "read_gguf_meta", lambda _p: Meta())
    got = draft_vram_gb(draft, ctx=32768, parallel=1)
    # 1.2 GB weights + (12800 elements x 2 bytes) x 8192 tokens
    # + 15 x 0.15 GB recurrent copies. The window is capped at 8k.
    expected = 1.2 + (12800 * 2 * 8192) / 1e9 + 15 * 0.15
    assert got == pytest.approx(expected, rel=1e-6)
    # Two slots keep two KV caches and two sets of recurrent state.
    two = draft_vram_gb(draft, ctx=32768, parallel=2)
    per_slot = (12800 * 2 * 8192) / 1e9 + 15 * 0.15
    assert two - got == pytest.approx(per_slot, rel=1e-6)


def test_draft_kv_precision_follows_the_flags(tmp_path, monkeypatch):
    weights = tmp_path / "draft.gguf"
    weights.write_bytes(b"x" * 10)
    q8 = DraftSpec(gguf=str(weights), n_max=1,
                   extra_args=["--spec-draft-type-k", "q8_0",
                               "--spec-draft-type-v", "q8_0"])
    f16 = DraftSpec(gguf=str(weights), n_max=1)

    class Meta:
        n_embd = 5120
        n_layer = 5
        n_head = 32
        n_head_kv = 8
        complete = True

    monkeypatch.setattr(draft_mod, "read_gguf_meta", lambda _p: Meta())
    q8_kv = draft_mod._kv_bytes_per_token(q8)
    f16_kv = draft_mod._kv_bytes_per_token(f16)
    assert q8_kv < f16_kv
    assert q8_kv / f16_kv == pytest.approx(34 / 32 / 2, rel=1e-6)


def test_missing_draft_file_costs_nothing(tmp_path):
    assert draft_vram_gb(DraftSpec(gguf=str(tmp_path / "nope.gguf")), 4096) == 0.0


# ── binary capability ─────────────────────────────────────────────────


def _binary_with(folder: Path, needle: bytes | None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    binary = folder / "llama-server"
    binary.write_bytes(b"#!/bin/sh\n")
    lib = folder / "libllama.so"
    lib.write_bytes(b"\x00" + (needle or b"") + b"\x00")
    return binary


def test_stock_build_is_reported_as_unable_to_load_dspine(tmp_path):
    assert draft_supported_by(str(_binary_with(tmp_path, None))) is False
    assert draft_supported_by(str(tmp_path / "missing")) is False
    assert draft_supported_by("") is False


def test_dspine_build_is_recognised_in_the_library(tmp_path):
    binary = _binary_with(tmp_path, b"dspine_codebook")
    assert draft_supported_by(str(binary)) is True


def test_needle_across_a_chunk_boundary_is_found(tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"x")
    lib = tmp_path / "libllama.so"
    # The first chunk is 1 MiB; straddle its end.
    lib.write_bytes(b"x" * ((1 << 20) - 4) + b"dspine_codebook" + b"x")
    assert draft_supported_by(str(binary)) is True


def test_check_draft_refuses_a_missing_draft(tmp_path):
    spec = _build_spec({
        "name": "qwen38", "llama_port": 1, "daemon_port": 2,
        "ctx_size": 4096, "max_tokens": 1,
        "gguf": str(tmp_path / "target.gguf"),
        "binary": str(tmp_path / "llama-server"),
        "draft": str(tmp_path / "missing.gguf"),
    })
    with pytest.raises(RuntimeError, match="missing"):
        _check_draft(spec)


def test_check_draft_refuses_a_build_without_dspine(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tokirecur.ggufmeta.read_gguf_tensor_sizes",
        lambda _p: {"blk.0.dspine_read": 1},
    )
    binary = _binary_with(tmp_path / "bin", None)
    draft = tmp_path / "draft.gguf"
    draft.write_bytes(b"x")
    spec = _build_spec({
        "name": "qwen38", "llama_port": 1, "daemon_port": 2,
        "ctx_size": 4096, "max_tokens": 1,
        "gguf": str(tmp_path / "target.gguf"),
        "binary": str(binary), "draft": str(draft),
    })
    with pytest.raises(RuntimeError, match="predates DSpine"):
        _check_draft(spec)


def test_check_draft_accepts_a_capable_build(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tokirecur.ggufmeta.read_gguf_tensor_sizes",
        lambda _p: {"blk.0.dspine_read": 1},
    )
    binary = _binary_with(tmp_path / "bin", b"dspine_codebook")
    draft = tmp_path / "draft.gguf"
    draft.write_bytes(b"x")
    spec = _build_spec({
        "name": "qwen38", "llama_port": 1, "daemon_port": 2,
        "ctx_size": 4096, "max_tokens": 1,
        "gguf": str(tmp_path / "target.gguf"),
        "binary": str(binary), "draft": str(draft),
    })
    _check_draft(spec)  # does not raise


def test_ordinary_draft_is_not_refused_by_a_stock_build(tmp_path):
    draft = tmp_path / "draft.gguf"
    draft.write_bytes(b"not a dspine gguf")
    assert draft_needs_dspine(DraftSpec(gguf=str(draft))) is False
    spec = _build_spec({
        "name": "qwen38", "llama_port": 1, "daemon_port": 2,
        "ctx_size": 4096, "max_tokens": 1,
        "gguf": str(tmp_path / "target.gguf"),
        "binary": str(_binary_with(tmp_path / "bin", None)),
        "draft": str(draft),
    })
    _check_draft(spec)  # does not raise


def test_vulkan_starts_with_a_short_draft(monkeypatch):
    monkeypatch.setattr(
        "tokirecur.diagnose.binary_backend", lambda _b: "Vulkan",
    )
    assert _draft_n_max_for("/b/llama-server") == 3
    monkeypatch.setattr(
        "tokirecur.diagnose.binary_backend", lambda _b: "ROCm",
    )
    assert _draft_n_max_for("/b/llama-server") == DRAFT_N_MAX_DEFAULT


# ── pairing by name ────────────────────────────────────────────────────


def test_target_family_strips_quant_and_split_suffixes():
    assert target_family("Qwen3.8-27B-Q4_K_M.gguf") == "Qwen3.8-27B"
    assert target_family("Qwen3.8-27B-IQ2_XXS-00001-of-00006.gguf") == (
        "Qwen3.8-27B"
    )
    assert target_family("model.gguf") == "model"
    assert target_family("Q4_K_M.gguf") == "Q4_K_M"


class _FakeApi:
    def __init__(self, repos: dict[str, list[str]]):
        self.repos = repos
        self.listed: list[str] = []

    def list_repo_files(self, repo: str) -> list[str]:
        self.listed.append(repo)
        if repo not in self.repos:
            raise RuntimeError("not found")
        return self.repos[repo]


def test_find_draft_matches_the_family_and_quant():
    api = _FakeApi({
        "generalinference/Qwen3.8-27B-DSpine-b16-GGUF": [
            "Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf",
        ],
    })
    found = find_draft("Qwen3.8-27B-Q8_0.gguf", api=api)
    assert found == DraftCandidate(
        repo="generalinference/Qwen3.8-27B-DSpine-b16-GGUF",
        file="Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf",
        family="Qwen3.8-27B", quant="Q4_K_M",
    )


def test_find_draft_returns_none_when_no_repo_has_it():
    api = _FakeApi({})
    assert find_draft("Llama-9B-Q4_K_M.gguf", api=api) is None
    # Every candidate owner was tried before giving up.
    assert len(api.listed) >= 2


def test_dspine_files_are_recognised_as_adapters():
    assert looks_like_dspine_adapter(
        "Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf",
    )
    assert not looks_like_dspine_adapter("Qwen3.8-27B-Q4_K_M.gguf")
    assert "Qwen3.8-27B" in dspine_adapter_hint(
        "Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf",
    )


def test_pulling_an_adapter_with_no_target_says_which_base_to_pull(
        tmp_path, monkeypatch):
    registry = tmp_path / "models.yaml"
    registry.write_text(yaml.safe_dump({"models": []}))

    class FakeApi:
        def __init__(self, endpoint=None):
            pass

        def list_repo_files(self, _repo):
            return ["Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf"]

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    with pytest.raises(RuntimeError, match="Qwen3.8-27B target"):
        pull_model(
            "aj9o9/Qwen3.8-27B-DSpine-b16-GGUF",
            registry=registry,
        )


# ── pull wiring ────────────────────────────────────────────────────────


def test_pull_draft_skips_attachment_when_build_cannot_load_it(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tokirecur.draftmatch.find_draft",
        lambda *a, **k: DraftCandidate(
            repo="aj9o9/Qwen3.8-27B-DSpine-b16-GGUF",
            file="draft.gguf", family="Qwen3.8-27B", quant="Q4_K_M",
        ),
    )
    stock = _binary_with(tmp_path / "bin", None)
    fields = _pull_draft(
        target_file="Qwen3.8-27B-Q4_K_M.gguf", local_dir=tmp_path,
        endpoint=None, want=None, quant="Q4_K_M", n_max=None,
        binary=str(stock),
    )
    assert fields == {}


def test_pull_draft_attaches_the_file_when_the_build_can_load_it(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tokirecur.draftmatch.find_draft",
        lambda *a, **k: DraftCandidate(
            repo="aj9o9/Qwen3.8-27B-DSpine-b16-GGUF",
            file="draft.gguf", family="Qwen3.8-27B", quant="Q4_K_M",
        ),
    )
    capable = _binary_with(tmp_path / "bin", b"dspine_codebook")
    fetched = tmp_path / "draft.gguf"
    fetched.write_bytes(b"x")

    def fake_download(repo, filename, **kwargs):
        assert repo == "aj9o9/Qwen3.8-27B-DSpine-b16-GGUF"
        assert filename == "draft.gguf"
        return str(fetched)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    fields = _pull_draft(
        target_file="Qwen3.8-27B-Q4_K_M.gguf", local_dir=tmp_path,
        endpoint=None, want=None, quant="Q4_K_M", n_max=None,
        binary=str(capable),
    )
    assert fields["draft"] == str(fetched)
    assert fields["draft_n_max"] == DRAFT_N_MAX_DEFAULT
    assert fields["draft_repo"] == "aj9o9/Qwen3.8-27B-DSpine-b16-GGUF"


def test_pull_draft_explicit_request_registers_but_does_not_break_a_stock_build(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tokirecur.draftmatch.find_draft",
        lambda *a, **k: DraftCandidate(
            repo="aj9o9/Qwen3.8-27B-DSpine-b16-GGUF",
            file="draft.gguf", family="Qwen3.8-27B", quant="Q4_K_M",
        ),
    )
    stock = _binary_with(tmp_path / "bin", None)
    fetched = tmp_path / "draft.gguf"
    fetched.write_bytes(b"x")
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda *a, **k: str(fetched),
    )
    fields = _pull_draft(
        target_file="Qwen3.8-27B-Q4_K_M.gguf", local_dir=tmp_path,
        endpoint=None, want=True, quant="Q4_K_M", n_max=7,
        binary=str(stock),
    )
    # The adapter is installed as a mode, but the stack is not switched
    # to a file this build cannot load: a working model must keep
    # working until `binary:` points at a DSpine-capable build.
    assert fields["adapter_mode"] == ""
    assert "draft" not in fields
    assert fields["adapters"][0]["gguf"] == str(fetched)
    assert fields["adapters"][0]["n_max"] == 7
    assert fields["adapters"][0]["kind"] == "draft"


# ── fit planner ───────────────────────────────────────────────────────


def test_fit_plan_reserves_room_for_the_drafter(tmp_path, monkeypatch):
    target = tmp_path / "target.gguf"
    target.write_bytes(b"x")
    draft_file = tmp_path / "draft.gguf"
    draft_file.write_bytes(b"x" * 1_000_000_000)
    base = {
        "name": "qwen38", "label": "Qwen3.8 27B", "gguf": str(target),
        "llama_port": 1, "daemon_port": 2, "ctx_size": 131072,
        "parallel": 1, "kv": "f16", "model_id": "qwen38",
        "max_tokens": 1,
    }
    plain = _build_spec(dict(base))
    paired = _build_spec(dict(base, draft=str(draft_file), draft_n_max=15))

    monkeypatch.setattr(
        "tokirecur.ctl._kv_bytes_per_token",
        lambda _s, kv_type=None, parallel=None: 200_000.0,
    )
    monkeypatch.setattr("tokirecur.ctl._model_max_ctx", lambda _s: 131072)
    plain_ctx = _fit_plan(plain, avail_gb=40, weight_gb=20)[0]
    paired_ctx = _fit_plan(paired, avail_gb=40, weight_gb=20)[0]
    assert paired_ctx < plain_ctx
