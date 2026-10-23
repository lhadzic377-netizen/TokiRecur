"""Async HTTP client for the toki daemon.

Wraps the daemon's HTTP API with methods the TUI needs:
health probes, cache state, live metrics, precision check,
and streaming chat (SSE for both llama-server and OpenAI formats).

All methods gracefully handle network errors and return sentinel
values rather than raising — the TUI needs to keep rendering even
when a backend is down.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from tokirecur import cache_prompt_enabled


def _make_client(base_url: str, auth_token: str | None):
    """Build an httpx.AsyncClient with our defaults."""
    import httpx
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    return httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        timeout=httpx.Timeout(180.0, connect=5.0),
        headers=headers,
    )


def _resolve_auth_token(explicit: str | None) -> str | None:
    """Auth-token lookup order:
      1. explicit arg (e.g. --auth-token CLI flag)
      2. $TOKI_AUTH_TOKEN env var
      3. ~/.config/tokirecur/auth (first line, trimmed)
      4. /etc/toki-recur/auth (system-wide fallback)
    Matches the daemon's lookup so a single auth file works for both.
    """
    if explicit:
        return explicit
    env = os.environ.get("TOKI_AUTH_TOKEN")
    if env:
        return env
    for path in (
        Path.home() / ".config" / "tokirecur" / "auth",
        Path("/etc/toki-recur/auth"),
    ):
        try:
            if path.is_file():
                content = path.read_text().strip()
                if content:
                    return content
        except OSError:
            pass
    return None


class DaemonClient:
    """Thin async wrapper over the toki daemon's HTTP API."""

    def __init__(
        self,
        daemon_url: str = "http://127.0.0.1:8787",
        raw_url: str = "http://127.0.0.1:8088",
        auth_token: str | None = None,
    ):
        self.daemon_url = daemon_url
        self.raw_url = raw_url
        # Fall back to env var, then auth file (matches daemon lookup).
        self._auth_token = _resolve_auth_token(auth_token)

    def _auth_headers(self) -> dict[str, str]:
        if not self._auth_token:
            return {}
        return {"Authorization": f"Bearer {self._auth_token}"}

    # ── Health probes ─────────────────────────────────────────────────

    async def health_llama(self) -> str:
        """Returns 'ok' or 'unreachable'."""
        try:
            async with _make_client(self.raw_url, None) as c:
                r = await c.get("/health")
                return "ok" if r.status_code == 200 else "unreachable"
        except Exception:
            return "unreachable"

    async def health_daemon(self) -> str:
        """Returns 'ok', 'auth_required', or 'unreachable'."""
        try:
            async with _make_client(self.daemon_url, self._auth_token) as c:
                r = await c.get("/healthz")
                if r.status_code == 200:
                    return "ok"
                if r.status_code == 401:
                    return "auth_required"
                return "unreachable"
        except Exception:
            return "unreachable"

    # ── Cache state + metrics ─────────────────────────────────────────

    async def cache_state(self) -> dict[str, Any]:
        """Return slot state. Empty dict on error.

        Adds 'daemon_status' key so the UI can tell apart 'no slots'
        from 'daemon unreachable'.
        """
        try:
            async with _make_client(self.daemon_url, self._auth_token) as c:
                r = await c.get("/v1/cache/state")
                if r.status_code == 200:
                    body = r.json()
                    body.setdefault("daemon_status", "ok")
                    return body
                if r.status_code == 401:
                    return {"slots": [], "sessions": {}, "spill_count": 0,
                            "max_context_tokens": 0, "daemon_status": "auth_required"}
        except Exception:
            pass
        return {"slots": [], "sessions": {}, "spill_count": 0,
                "max_context_tokens": 0, "daemon_status": "unreachable"}

    async def metrics_live(self) -> dict[str, Any]:
        """Parse the daemon's Prometheus /metrics into a small dict.

        ``daemon`` says where the numbers came from: "ok", or why there are
        none. Without it, a daemon that is down and a daemon that has never
        served a request look identical — all zeros — and a command that
        reports zeros reads as "no traffic" when it means "no answer".
        """
        empty = {
            "requests_total": 0, "cache_hits": 0, "cache_misses": 0,
            "tokens_skipped": 0, "active_slots": 0, "auth_failures": 0,
            "daemon": "unreachable",
        }
        try:
            async with _make_client(self.daemon_url, self._auth_token) as c:
                r = await c.get("/metrics")
                if r.status_code != 200:
                    empty["daemon"] = "auth_required" if r.status_code == 401 else "error"
                    return empty
                text = r.text
        except Exception:
            return empty

        for line in text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            token, value = parts[0], parts[1]
            # Prometheus labels its counters — toki_requests_total{route=…,
            # status=…} — and the label block is part of the token. The
            # name has to be cut off before it is compared, or the request
            # counter never matches anything and always reads zero.
            name = token.split("{", 1)[0]
            labels = token[len(name):]
            try:
                v = float(value)
            except ValueError:
                continue
            if name == "toki_requests_total":
                empty["requests_total"] += v
            elif name == "toki_cache_hits_total" and 'kind="hit"' in labels:
                empty["cache_hits"] += v
            elif name == "toki_cache_hits_total" and 'kind="miss"' in labels:
                empty["cache_misses"] += v
            elif name == "toki_tokens_skipped_total":
                empty["tokens_skipped"] = int(v)
            elif name == "toki_active_slots":
                empty["active_slots"] = int(v)
            elif name == "toki_auth_failures_total":
                empty["auth_failures"] = int(v)
        empty["daemon"] = "ok"
        return empty

    async def precision_check(self, *, max_tokens: int = 32) -> dict[str, Any]:
        """Run the precision retention check via the daemon.

        The daemon proxies this to llama-server internally and adds
        auth, so we hit the daemon's /v1/precision endpoint rather
        than calling llama-server directly.
        """
        try:
            async with _make_client(self.daemon_url, self._auth_token) as c:
                r = await c.post(
                    "/v1/precision",
                    json={"max_tokens": max_tokens},
                )
                if r.status_code == 200:
                    return r.json()
        except Exception:
            pass
        return {
            "baseline_deterministic": 0.0,
            "recycled_deterministic": 0.0,
            "byte_identical_to_baseline": 0.0,
            "prompts_tested": 0,
            "per_prompt": [],
            "error": "could not run precision check",
        }

    async def reset_cache(self) -> bool:
        try:
            async with _make_client(self.daemon_url, self._auth_token) as c:
                r = await c.post("/v1/cache/reset")
                return r.status_code == 200
        except Exception:
            return False

    # ── Streaming chat ────────────────────────────────────────────────

    async def chat_stream(
        self,
        prompt: str,
        *,
        mode: str,
        session_id: str,
        system: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        history: list[dict] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a completion. Yields dict events.

        Event types:
          {"event": "start", "mode": ..., "target": ..., "session_id": ...}
          {"event": "token", "content": "..."}
          {"event": "done", "text": "...", "metrics": {...}, "cache": {...}}
          {"event": "error", "message": "..."}

        For both 'raw' and 'cached' modes, hits llama-server's
        OpenAI-compatible /v1/chat/completions endpoint so llama-server
        applies the correct chat template for the loaded model (Qwen3.5,
        Llama, Mistral, etc.). The 'cached' mode additionally routes
        through the daemon so session_id and max_slots reach the engine
        for slot pinning.
        """
        history = history or []
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        for m in history:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role in ("user", "assistant", "system"):
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": prompt})

        if mode == "raw":
            # Hit llama-server's OpenAI-compatible endpoint. llama-server
            # applies the correct chat template for the loaded model, so
            # chat-tuned models like Qwen3.5 get correct templating
            # (no more "User: ... Assistant: ..." garbage continuations).
            endpoint = f"{self.raw_url}/v1/chat/completions"
            payload = {
                "model": "default",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
                # Reuse llama-server's in-memory KV prefix across turns:
                # consecutive messages in a session share a prefix, so
                # multi-turn requests skip re-prefilling the history.
                # Off by default on Windows (the CPU llama-server build
                # can hang on the KV-reuse path on second messages).
                "cache_prompt": cache_prompt_enabled(),
            }
            target_label = "raw"
        elif mode == "cached":
            endpoint = f"{self.daemon_url}/v1/chat/completions"
            payload = {
                "model": "default",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
                "session_id": session_id,
                "max_slots": 4,
            }
            target_label = "cached"
        else:
            yield {"event": "error", "message": f"unknown mode: {mode}"}
            return

        yield {
            "event": "start",
            "mode": mode,
            "target": target_label,
            "session_id": session_id,
        }

        started_at = _now()
        first_token_at: float | None = None
        text_chunks: list[str] = []
        reasoning_chunks: list[str] = []
        error_msg: str | None = None
        prompt_tokens = 0
        cache_verdict: dict | None = None
        # Raw mode bypasses the daemon (no auth needed for loopback).
        # Cached mode forwards the token (TUI shares auth file with daemon).
        auth_for_request = self._auth_token if mode == "cached" else None
        try:
            async with _make_client(
                self.raw_url if mode == "raw" else self.daemon_url,
                auth_for_request,
            ) as c:
                async with c.stream("POST", endpoint, json=payload) as r:
                    if r.status_code != 200:
                        body = await r.aread()
                        error_msg = (
                            f"upstream {r.status_code}: "
                            f"{body[:200].decode('utf-8', errors='replace')}"
                        )
                        yield {"event": "error", "message": error_msg}
                        cache_marker = (
                            {"applicable": False, "mode": "raw"}
                            if mode == "raw"
                            else {"hit": False, "skipped_tokens": 0}
                        )
                        yield {
                            "event": "done",
                            "text": "",
                            "metrics": {
                                "ttft_ms": 0,
                                "total_ms": (_now() - started_at) * 1000.0,
                                "tokens_generated": 0,
                            },
                            "cache": cache_marker,
                            "error": error_msg,
                        }
                        return

                    pending = ""
                    async for chunk_text in r.aiter_text():
                        pending += chunk_text
                        lines, pending = _split_sse(pending)
                        for line in lines:
                            if not line.strip():
                                continue
                            if line.startswith("data: "):
                                line = line[6:]
                            if line.strip() in ("[DONE]", ""):
                                continue
                            obj = _parse_sse_line(line)
                            if obj is None:
                                continue
                            usage = obj.get("usage")
                            if isinstance(usage, dict):
                                # OpenAI-style usage chunk arrives after
                                # the finish_reason chunk.
                                prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                            if isinstance(obj.get("cache"), dict):
                                # The daemon's terminal chunk carries the
                                # cache verdict (hit, soft_hit, cache_id,
                                # reuse_depth, skipped_tokens, ...). Preserve
                                # it so the TUI can render hits/misses.
                                cache_verdict = obj["cache"]
                            text = _extract_text(obj)
                            if text:
                                if first_token_at is None:
                                    first_token_at = _now()
                                text_chunks.append(text)
                                yield {"event": "token", "content": text}
                            reasoning = _extract_reasoning(obj)
                            if reasoning:
                                if first_token_at is None:
                                    first_token_at = _now()
                                reasoning_chunks.append(reasoning)
                                yield {"event": "reasoning", "content": reasoning}
                            if obj.get("stop") or obj.get("finish_reason"):
                                break
        except Exception as e:  # noqa: BLE001
            error_msg = f"{type(e).__name__}: {e}"
            yield {"event": "error", "message": error_msg}
            yield {
                "event": "done",
                "text": "",
                "metrics": {
                    "ttft_ms": 0,
                    "total_ms": (_now() - started_at) * 1000.0,
                    "tokens_generated": 0,
                },
                "cache": (
                    {"applicable": False, "mode": "raw"}
                    if mode == "raw"
                    else {"hit": False, "skipped_tokens": 0}
                ),
                "error": error_msg,
            }
            return

        total_ms = (_now() - started_at) * 1000.0
        ttft_ms = (
            (first_token_at - started_at) * 1000.0
            if first_token_at is not None else 0.0
        )
        # Raw mode bypasses the cache entirely — there's no "hit" or
        # "miss" verdict because the request never touched the cache
        # layer. Emit a marker that the TUI can render as "↯ raw"
        # instead of "✓ hit" / "✗ miss".
        if mode == "raw":
            cache_marker: dict = {"applicable": False, "mode": "raw"}
        else:
            # Cached mode: prefer the daemon's verdict chunk (hit,
            # soft_hit, reuse_depth, cache_id, ...); fall back to a
            # conservative miss if no verdict arrived (e.g. upstream
            # closed early).
            cache_marker = cache_verdict or {"hit": False, "skipped_tokens": 0}
        yield {
            "event": "done",
            "text": "".join(text_chunks),
            "reasoning": "".join(reasoning_chunks),
            "metrics": {
                "ttft_ms": ttft_ms,
                "total_ms": total_ms,
                "tokens_generated": len(text_chunks),
                "prompt_tokens": prompt_tokens,
            },
            "cache": cache_marker,
        }


# ── helpers ──────────────────────────────────────────────────────────


def _now() -> float:
    import time
    return time.perf_counter()


def _split_sse(chunk: str) -> tuple[list[str], str]:
    """Split an SSE buffer into complete lines plus a partial remainder.

    A line is complete once it ends with ``\\n``. The trailing partial
    line (if any) is returned separately so the caller can prepend it
    to the next chunk — otherwise a JSON event split across two HTTP
    chunks would be parsed as invalid JSON and silently dropped.
    """
    if "\n" not in chunk:
        return [], chunk
    complete, _, remainder = chunk.rpartition("\n")
    if not complete:
        return [], remainder
    return complete.split("\n"), remainder


def _parse_sse_line(line: str) -> dict | None:
    try:
        return json.loads(line)
    except Exception:
        return None


def _extract_text(obj: dict) -> str:
    """Pull text from a streaming chunk.

    llama-server /completion:        {"content": "..."} or {"response": "..."}
    OpenAI /v1/chat/completions:      {"choices": [{"delta": {"content": "..."}}]}
    """
    if not isinstance(obj, dict):
        return ""
    if obj.get("content"):
        return str(obj["content"])
    if obj.get("response"):
        return str(obj["response"])
    choices = obj.get("choices")
    if choices and isinstance(choices, list):
        delta = choices[0].get("delta") or {}
        if isinstance(delta, dict) and delta.get("content"):
            return str(delta["content"])
    return ""


def _extract_reasoning(obj: dict) -> str:
    """Pull thinking/reasoning content from a streaming chunk.

    Sent in its own ``reasoning_content`` SSE delta by the daemon (or
    directly by llama-server). Forwarded verbatim so thinking models
    can render their chain of thought.
    """
    if not isinstance(obj, dict):
        return ""
    rc = obj.get("reasoning_content")
    if rc:
        return str(rc)
    choices = obj.get("choices")
    if choices and isinstance(choices, list):
        delta = choices[0].get("delta") or {}
        if isinstance(delta, dict):
            rc = delta.get("reasoning_content")
            if rc:
                return str(rc)
    return ""
