"""Non-interactive completion helpers for `toki diff`.

One-shot completion against either the native llama-server endpoint
(``api="native"``) or the toki daemon's OpenAI-compatible endpoint
(``api="openai"``). The interactive chat lives in `toki tui` and the
web UI.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import httpx

#: The reasoning a model writes when the prompt did not give it a chat
#: template with a reasoning channel. llama-server splits this out for the
#: chat endpoints; on the native completion endpoint it arrives as part of
#: the text.
#:
#: It matters for `toki diff`: the raw side is a native completion and the
#: cached side is a chat turn through the daemon, so the raw side carried
#: the model's thinking and the cached side did not. The diff then reported
#: that the two outputs diverged — at character zero — when the answers
#: were the same.
_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def split_thinking(text: str) -> tuple[str, str]:
    """The answer and the reasoning, separated.

    Returns ``(answer, reasoning)``. The answer is what the model said
    after it stopped thinking; the reasoning is what it thought while
    doing so.
    """
    if "<think>" not in text.lower():
        return text, ""
    reasoning = "\n".join(
        m.group(1).strip() for m in _THINK_BLOCK.finditer(text) if m.group(1).strip()
    )
    answer = _THINK_BLOCK.sub("", text).strip()
    return answer, reasoning


def _auth_headers() -> dict[str, str]:
    """Bearer header for the toki daemon when a token is configured.

    Same read order as the daemon: TOKI_AUTH_TOKEN env var, then
    ~/.config/tokirecur/auth, then /etc/toki-recur/auth.
    """
    token = os.environ.get("TOKI_AUTH_TOKEN")
    if not token:
        for path in (
            Path.home() / ".config" / "tokirecur" / "auth",
            Path("/etc/toki-recur/auth"),
        ):
            try:
                if path.is_file():
                    content = path.read_text().strip()
                    if content:
                        token = content
                        break
            except OSError:
                pass
    return {"Authorization": f"Bearer {token}"} if token else {}


def _openai_messages(
    prompt: str,
    *,
    system: str | None = None,
    history: list[dict] | None = None,
) -> list[dict]:
    """Build OpenAI chat messages, system first, history, then user."""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    return messages


def completion_once(
    base_url: str,
    prompt: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
    system: str | None = None,
    history: list[dict] | None = None,
    timeout: float = 120.0,
    api: str = "native",
) -> tuple[str, dict[str, Any]]:
    """Send a non-streaming completion. Used by `toki diff`.

    ``api="openai"`` targets the toki daemon's ``/v1/chat/completions``.
    """
    if api == "openai":
        payload: dict[str, Any] = {
            "model": "default",
            "messages": _openai_messages(prompt, system=system, history=history),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        started = time.perf_counter()
        try:
            r = httpx.post(f"{base_url}/v1/chat/completions", json=payload,
                           timeout=timeout, headers=_auth_headers())
            r.raise_for_status()
            body = r.json()
            cache = body.get("cache") or {}
            usage = body.get("usage") or {}
            message = (body.get("choices") or [{}])[0].get("message", {}) or {}
            answer, reasoning = split_thinking(message.get("content", "") or "")
            return answer, {
                "reasoning": reasoning or (message.get("reasoning_content") or ""),
                "ttft_ms": float(cache.get("ttft_ms", 0.0)),
                "total_ms": (time.perf_counter() - started) * 1000.0,
                "tokens_generated": int(usage.get("completion_tokens", 0)),
                "prefill_tokens": int(usage.get("prompt_tokens", 0)),
                "hit": bool(cache.get("hit")),
                "reuse_depth": int(cache.get("reuse_depth", 0)),
                "cache_id": cache.get("cache_id"),
            }
        except httpx.HTTPError as e:
            return f"[error: {e}]", {}
    # The same chat endpoint the daemon serves, with the cache bypassed.
    # It used to be the native /completion endpoint, which is a different
    # request in a different format: no chat template, no reasoning
    # channel, a different prompt shape. The two sides of the diff were
    # not the same question, so the "speedup" was measuring two different
    # requests and the outputs were reported as different whenever the
    # model thought.
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload: dict[str, Any] = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    started = time.perf_counter()
    try:
        r = httpx.post(f"{base_url}/v1/chat/completions", json=payload,
                       timeout=timeout, headers=_auth_headers())
        r.raise_for_status()
        body = r.json()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        message = (body.get("choices") or [{}])[0].get("message", {}) or {}
        usage = body.get("usage") or {}
        answer, reasoning = split_thinking(message.get("content", "") or "")
        return answer, {
            "reasoning": reasoning or (message.get("reasoning_content") or ""),
            "ttft_ms": float((body.get("timings") or {}).get("prompt_ms", 0.0)),
            "total_ms": elapsed_ms,
            # This side posts /v1/chat/completions too, so the counts live
            # in `usage` — the native /completion fields (tokens_evaluated /
            # tokens_predicted) are absent and parsed as a flat 0, which
            # made "prefill tokens saved" always 0.
            "prefill_tokens": int(usage.get("prompt_tokens", 0)),
            "tokens_generated": int(usage.get("completion_tokens", 0)),
        }
    except httpx.HTTPError as e:
        return "", {"error": str(e), "total_ms": (time.perf_counter() - started) * 1000.0}
