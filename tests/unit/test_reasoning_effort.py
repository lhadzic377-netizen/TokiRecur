"""Reasoning effort: the levels a model understands, and the variants
they become.

An effort is a request-level thing: llama-server hands "high" to the
model's chat template, which decides what that means for its family.
The registry says which levels a model understands, and the loader turns
those levels into variants a harness can pick by model id.
"""

import pytest

from tokirecur.api.app import _resolve_effort
from tokirecur.ctl import (
    EFFORT_LEVELS,
    ModelSpec,
    expand_variants,
    resolve_model_id,
)


def _spec(**over) -> ModelSpec:
    base = dict(
        name="qwen3.8-27b",
        label="Qwen3.8 27B",
        gguf="/models/qwen3.8-27b.gguf",
        llama_port=8090,
        daemon_port=8790,
        ctx_size=200000,
        parallel=1,
        kv="f16",
        max_tokens=100000,
        model_id="qwen3.8-27b",
    )
    base.update(over)
    return ModelSpec(**base)


def test_a_model_without_levels_has_one_variant():
    """No levels declared means the on/off switch only — the model as
    it is. Nothing to expand, nothing to confuse anyone with."""
    variants = expand_variants([_spec()])
    assert [v.model_id for v in variants] == ["qwen3.8-27b"]
    assert variants[0].effort is None


def test_levels_become_variants():
    spec = _spec(reasoning_efforts=["none", "low", "high"])
    variants = expand_variants([spec])
    assert [v.model_id for v in variants] == [
        "qwen3.8-27b",
        "qwen3.8-27b:none",
        "qwen3.8-27b:low",
        "qwen3.8-27b:high",
    ]
    assert [v.name for v in variants][1] == "qwen3.8-27b · none"


def test_variants_share_the_weights():
    """One server, one set of weights, one slot: a variant is the same
    model with the reasoning steered, not another model."""
    spec = _spec(reasoning_efforts=["high"])
    base, high = expand_variants([spec])
    assert high.spec is base.spec
    assert high.spec.gguf == base.spec.gguf
    assert high.spec.llama_port == base.spec.llama_port


def test_resolve_a_variant_id():
    spec = _spec(reasoning_efforts=["low", "high"])
    resolved, effort = resolve_model_id([spec], "qwen3.8-27b:high")
    assert resolved is spec
    assert effort == "high"


def test_resolve_the_base_id():
    spec = _spec(reasoning_efforts=["high"])
    resolved, effort = resolve_model_id([spec], "qwen3.8-27b")
    assert resolved is spec
    assert effort is None


def test_unknown_ids_resolve_to_nothing():
    spec = _spec(reasoning_efforts=["high"])
    assert resolve_model_id([spec], "something-else") == (None, None)
    assert resolve_model_id([spec], "") == (None, None)


def test_a_level_the_model_did_not_declare_is_not_a_variant():
    """A model that only understands low/high must not answer to
    "xhigh" just because the vocabulary knows the word."""
    spec = _spec(reasoning_efforts=["low", "high"])
    resolved, effort = resolve_model_id([spec], "qwen3.8-27b:xhigh")
    assert resolved is None
    assert effort is None


def test_effort_levels_are_the_vocabulary_llama_cpp_understands():
    assert EFFORT_LEVELS[0] == "none"
    for level in ("low", "medium", "high", "xhigh", "max"):
        assert level in EFFORT_LEVELS


def test_an_explicit_effort_is_honoured():
    """A client that asks for an effort gets it, whatever the model id
    says — the id may be a name the daemon does not serve."""
    assert _resolve_effort("qwen3.8-27b", "high", "qwen3.8-27b") == "high"
    assert _resolve_effort("qwen3.8-27b", "none", "qwen3.8-27b") == "none"


def test_an_unknown_effort_is_ignored_not_guessed():
    """A level nobody defined is dropped: a template that does not
    understand it would answer with its own default, and saying nothing
    is better than pretending."""
    assert _resolve_effort("qwen3.8-27b", "enormous", "qwen3.8-27b") is None


def test_a_variant_id_carries_its_effort(monkeypatch, tmp_path):
    spec = _spec(reasoning_efforts=["low", "high"])
    registry = tmp_path / "models.yaml"
    registry.write_text(
        "models:\n"
        "- name: qwen3.8-27b\n"
        "  gguf: /models/qwen3.8-27b.gguf\n"
        "  llama_port: 8090\n"
        "  daemon_port: 8790\n"
        "  ctx_size: 200000\n"
        "  parallel: 1\n"
        "  kv: f16\n"
        "  max_tokens: 100000\n"
        "  model_id: qwen3.8-27b\n"
        "  reasoning_efforts: [low, high]\n"
    )
    monkeypatch.setattr(
        "tokirecur.ctl.DEFAULT_REGISTRY", registry, raising=False
    )
    assert _resolve_effort("qwen3.8-27b:high", None, "qwen3.8-27b") == "high"
    assert _resolve_effort("qwen3.8-27b", None, "qwen3.8-27b") is None


def test_a_variant_of_another_model_is_ignored(monkeypatch, tmp_path):
    """The daemon is single-tenant: it serves one model. An effort meant
    for a model it does not serve would be answered by the wrong one."""
    registry = tmp_path / "models.yaml"
    registry.write_text(
        "models:\n"
        "- name: other\n"
        "  gguf: /models/other.gguf\n"
        "  llama_port: 8091\n"
        "  daemon_port: 8791\n"
        "  ctx_size: 200000\n"
        "  parallel: 1\n"
        "  kv: f16\n"
        "  max_tokens: 100000\n"
        "  model_id: other\n"
        "  reasoning_efforts: [high]\n"
    )
    monkeypatch.setattr(
        "tokirecur.ctl.DEFAULT_REGISTRY", registry, raising=False
    )
    assert _resolve_effort("other:high", None, "qwen3.8-27b") is None


@pytest.mark.parametrize("level", ["none", "minimal", "low", "medium", "high"])
def test_every_level_is_a_valid_choice(level):
    assert level in EFFORT_LEVELS
