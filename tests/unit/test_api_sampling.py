"""Sampling defaults: the daemon's anti-loop profile.

Greedy decode (temperature 0.0) with no repetition penalty is the loop
engine behind agent harnesses — identical context reproduces identical
tokens and the model re-sends the same tool call forever. These tests
pin the defaults that break that lock and the merge rules: client
values win, per-model registry blocks override globals, and only
non-None fields reach llama-server.
"""

from tokirecur import ctl
from tokirecur.api.app import (
    DEFAULT_SAMPLING,
    _resolve_sampling,
    _sampling_defaults,
)
from tokirecur.backend import _merge_sampling


def test_defaults_profile_is_anti_loop():
    """The shipped defaults must be non-greedy with a repetition
    penalty — the exact inverse of the loop amplifier."""
    assert DEFAULT_SAMPLING["temperature"] > 0.0
    assert DEFAULT_SAMPLING["repeat_penalty"] > 1.0
    assert DEFAULT_SAMPLING["top_k"] > 0
    assert 0.0 < DEFAULT_SAMPLING["min_p"] <= 1.0


def test_resolve_sampling_applies_defaults_when_client_sends_nothing():
    temp, sampling = _resolve_sampling()
    assert temp == 0.4
    assert sampling == {
        "repeat_penalty": 1.15,
        "top_k": 40,
        "min_p": 0.05,
    }


def test_resolve_sampling_client_values_win():
    temp, sampling = _resolve_sampling(
        temperature=0.0, repeat_penalty=1.0, top_k=0, min_p=0.0,
    )
    assert temp == 0.0
    assert sampling == {"repeat_penalty": 1.0, "top_k": 0, "min_p": 0.0}


def test_resolve_sampling_partial_client_keeps_rest_of_defaults():
    temp, sampling = _resolve_sampling(temperature=0.7)
    assert temp == 0.7
    assert sampling["repeat_penalty"] == 1.15
    assert sampling["top_k"] == 40


def test_resolve_sampling_custom_defaults_and_profile(monkeypatch):
    monkeypatch.setenv("TOKI_SAMPLING", '{"repeat_penalty": 1.3}')
    profile = _sampling_defaults()
    assert profile["repeat_penalty"] == 1.3
    assert profile["temperature"] == 0.4  # partial env merges over built-ins
    temp, sampling = _resolve_sampling()
    assert sampling["repeat_penalty"] == 1.3


def test_resolve_sampling_ignores_malformed_env(monkeypatch):
    monkeypatch.setenv("TOKI_SAMPLING", "not-json")
    assert _sampling_defaults() == DEFAULT_SAMPLING


def test_responses_keeps_openai_temperature_default():
    # The Responses shim keeps OpenAI's temperature=1.0 when the client
    # sends nothing, but still gets the repetition controls.
    from tokirecur.api.app import _responses_sampling_defaults

    temp, sampling = _resolve_sampling(
        defaults=_responses_sampling_defaults(), default_temperature=1.0,
    )
    assert temp == 1.0
    assert sampling["repeat_penalty"] == 1.15


def test_merged_sampling_overrides_globals_per_model():
    spec = ctl.ModelSpec(
        name="35b", label="35b", gguf="/tmp/x", llama_port=8088,
        daemon_port=8787, ctx_size=200000, parallel=2, kv="f16",
        max_tokens=100000, model_id="qwen3.6-35b-a3b",
        sampling={"repeat_penalty": 1.2, "temperature": 0.5},
    )
    merged = ctl.merged_sampling(spec)
    assert merged["repeat_penalty"] == 1.2
    assert merged["temperature"] == 0.5
    # Unset keys still come from the globals.
    assert merged["top_k"] == 40
    assert merged["min_p"] == 0.05


def test_merged_sampling_empty_spec_uses_globals():
    spec = ctl.ModelSpec(
        name="9b", label="9b", gguf="/tmp/x", llama_port=8088,
        daemon_port=8787, ctx_size=200000, parallel=2, kv="f16",
        max_tokens=100000, model_id="qwen3.5-9b",
    )
    assert ctl.merged_sampling(spec) == DEFAULT_SAMPLING


def test_merge_sampling_only_adds_non_none_fields():
    payload = {"prompt": "hi", "n_predict": 10, "temperature": 0.4}
    out = _merge_sampling(payload, {"repeat_penalty": 1.15, "top_p": None})
    assert out["repeat_penalty"] == 1.15
    assert "top_p" not in out
    assert "top_k" not in out
    assert out["prompt"] == "hi"  # untouched


def test_merge_sampling_none_is_noop():
    payload = {"prompt": "hi", "temperature": 0.4}
    assert _merge_sampling(payload, None) == payload
    assert _merge_sampling(payload, {}) == payload
