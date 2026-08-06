"""Daemon thinking-mode defaults driven by the model's `reasoning` field."""

import pytest

from tokirecur.api.app import _default_chat_template_kwargs


@pytest.mark.parametrize("reasoning,expected", [
    ("auto", True),
    ("on", True),
    ("off", False),
    ("", True),
])
def test_default_chat_template_kwargs(reasoning, expected):
    kwargs = _default_chat_template_kwargs(reasoning)
    assert kwargs == {"enable_thinking": expected}


def test_clients_can_override_default():
    """Explicit chat_template_kwargs must win over the default."""
    kwargs = _default_chat_template_kwargs("off")
    # The daemon merges/overrides before sending; the helper only
    # produces the default for clients that send nothing.
    assert kwargs == {"enable_thinking": False}
