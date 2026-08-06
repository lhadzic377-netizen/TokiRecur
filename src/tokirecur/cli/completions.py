"""Non-interactive completion helpers for `toki diff`.

One-shot completion against either the native llama-server endpoint
(``api="native"``) or the toki daemon's OpenAI-compatible endpoint
(``api="openai"``). The interactive chat lives in `toki tui` and the
web UI.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import httpx


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
            return body.get("choices", [{}])[0].get("message", {}).get("content", ""), {
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
    payload: dict[str, Any] = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if system:
        payload["system_prompt"] = system
    started = time.perf_counter()
    try:
        r = httpx.post(f"{base_url}/completion", json=payload, timeout=timeout,
                       headers=_auth_headers())
        r.raise_for_status()
        body = r.json()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        timings = body.get("timings", {}) or {}
        return body.get("content", ""), {
            "ttft_ms": float(timings.get("prompt_ms", 0.0)),
            "total_ms": elapsed_ms,
            "prefill_tokens": int(body.get("tokens_evaluated", 0)),
            "tokens_generated": int(body.get("tokens_predicted", 0)),
        }
    except httpx.HTTPError as e:
        return "", {"error": str(e), "total_ms": (time.perf_counter() - started) * 1000.0}
