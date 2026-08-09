"""Toki Recur — KV cache reuse for LLM inference."""

import os

__version__ = "0.2.24"


def cache_prompt_enabled() -> bool:
    """Whether raw-mode requests should ask llama-server to reuse KV.

    Defaults to True, except on Windows where the CPU llama-server build
    can hang on the multi-turn KV-reuse path (first message works, the
    second spins the CPU without producing tokens). Set TOKI_CACHE_PROMPT
    to "1" or "0" to force the choice on any platform.
    """
    forced = os.environ.get("TOKI_CACHE_PROMPT")
    if forced == "1":
        return True
    if forced == "0":
        return False
    return os.name != "nt"
