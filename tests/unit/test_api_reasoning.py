"""Daemon thinking-mode defaults driven by the model's `reasoning` field."""

import pytest

from tokirecur.api.app import (
    MAX_TOKENS_FLOOR,
    THINKING_TOKENS_FLOOR,
    _default_chat_template_kwargs,
    _resolve_max_tokens,
)


@pytest.mark.parametrize("reasoning,expected", [
    ("auto", True),
    ("on", True),
    ("off", False),
    ("", True),
])
def test_default_chat_template_kwargs(reasoning, expected):
    kwargs = _default_chat_template_kwargs(reasoning)
    assert kwargs == {"enable_thinking": expected}


@pytest.mark.parametrize("requested", [1, 32, 64, 511])
def test_thinking_floor_raises_small_budgets(requested):
    assert _resolve_max_tokens(requested, reasoning="auto") == THINKING_TOKENS_FLOOR
    assert _resolve_max_tokens(requested, reasoning="on") == THINKING_TOKENS_FLOOR


@pytest.mark.parametrize("requested", [1, 32, 64, 511])
def test_off_reasoning_keeps_small_floor(requested):
    assert _resolve_max_tokens(requested, reasoning="off") == MAX_TOKENS_FLOOR


def test_large_budgets_never_capped():
    assert _resolve_max_tokens(THINKING_TOKENS_FLOOR + 100, reasoning="auto") == (
        THINKING_TOKENS_FLOOR + 100
    )
    assert _resolve_max_tokens(MAX_TOKENS_FLOOR + 100, reasoning="off") == (
        MAX_TOKENS_FLOOR + 100
    )


def test_verbose_thinking_model_budget():
    """The 35B-A3B reasons ~570 tokens for a trivial prompt; the floor
    must leave room for the answer so content is never empty."""
    assert THINKING_TOKENS_FLOOR >= 2048
    assert THINKING_TOKENS_FLOOR > 570


def test_output_cap_clamps_runaway_budgets(monkeypatch):
    """TOKI_MAX_OUTPUT_TOKENS (per-model max_output_tokens) is a hard
    per-turn cap: the 32k-token runaway generation that stalled a
    session must be impossible once the registry sets the budget."""
    monkeypatch.setenv("TOKI_MAX_OUTPUT_TOKENS", "16384")
    assert _resolve_max_tokens(32000, reasoning="off") == 16384
    assert _resolve_max_tokens(100000, reasoning="auto") == 16384


def test_output_cap_never_breaks_floor(monkeypatch):
    monkeypatch.setenv("TOKI_MAX_OUTPUT_TOKENS", "64")
    assert _resolve_max_tokens(100, reasoning="off") == MAX_TOKENS_FLOOR
    assert _resolve_max_tokens(100, reasoning="auto") == THINKING_TOKENS_FLOOR


def test_output_cap_unset_means_no_cap(monkeypatch):
    monkeypatch.delenv("TOKI_MAX_OUTPUT_TOKENS", raising=False)
    assert _resolve_max_tokens(32000, reasoning="off") == 32000


def test_output_cap_ignores_malformed_env(monkeypatch):
    monkeypatch.setenv("TOKI_MAX_OUTPUT_TOKENS", "not-a-number")
    assert _resolve_max_tokens(32000, reasoning="off") == 32000
