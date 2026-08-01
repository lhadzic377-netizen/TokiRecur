"""Backend adapters for llama.cpp.

* :class:`LlamaServerBackend` — talks to llama-server over HTTP using
  the upstream endpoints (including the built-in slot save/restore
  available since llama.cpp b3xxx behind the `--slot-save-path` flag).

The recycle engine depends only on the :class:`LlamaBackend` protocol;
swapping backends is a one-line change.

Endpoint reference (upstream llama.cpp >= b3524):
  POST /slots/{id_slot}?action=save     body: {"filename": "..."}
  POST /slots/{id_slot}?action=restore  body: {"filename": "..."}
  POST /slots/{id_slot}?action=erase    body: {"filename": "..."}
  GET  /slots
  POST /tokenize    body: {"content": "..."}
  POST /embedding   body: {"content": "..."}
  POST /completion  body: {"prompt": "...", ...}
"""

from __future__ import annotations

import logging
import os
import time
import json
from pathlib import Path
from typing import Any

import httpx

from ..core.recycling import LlamaBackend

logger = logging.getLogger(__name__)


class LlamaServerBackend(LlamaBackend):
    """HTTP backend against llama-server (upstream).

    Uses the built-in slot save/restore endpoints to persist KV cache
    state to files. The server must be started with `--slot-save-path <dir>`
    to enable these endpoints; otherwise save/restore calls return 501.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        timeout: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._client = httpx.Client(timeout=timeout)

    def _post(self, path: str, payload: dict) -> dict[str, Any]:
        r = self._client.post(f"{self.base_url}{path}", json=payload)
        r.raise_for_status()
        return r.json()

    def _get(self, path: str) -> dict[str, Any]:
        r = self._client.get(f"{self.base_url}{path}")
        r.raise_for_status()
        return r.json()

    def cache_path(self, cache_id: str) -> str:
        """Return the relative filename to pass to llama-server's
        `?action=save` / `?action=restore` endpoints. The server
        prepends `--slot-save-path`, so the filename must be relative.
        The full path on disk is `<slot_save_path>/<cache_id>.bin`."""
        if self.cache_dir is None:
            raise RuntimeError(
                "cache_dir not configured; pass --cache-dir or set "
                "TOKI_CACHE_DIR so the backend knows where to store KV blobs."
            )
        return f"{cache_id}.bin"

    # ---- LlamaBackend protocol ----

    _MAX_EMBED_TOKENS = 2000  # safe upper bound for /embedding
    _FALLBACK_DIM = 4096  # matches Qwen3.5-9B's embedding; safe default

    def embed(self, text: str) -> list[float]:
        """Embed `text`. If it exceeds the server's batch size, chunk it
        into segments and average the per-segment embeddings (the standard
        long-text strategy). On any error, fall back to a hash-based
        pseudo-embedding so the request degrades to a cache miss rather
        than crashing the run."""
        try:
            return self._embed_chunked(text)
        except Exception:
            # Last-resort fallback: deterministic pseudo-embedding from
            # the SHA-256 of the text. This is not semantically meaningful
            # but gives every text a unique vector (no false-positive
            # cache hits) and lets the request complete.
            import hashlib
            h = hashlib.sha256(text.encode("utf-8")).digest()
            return [
                ((h[i % len(h)] ^ (i * 13)) - 128) / 128.0
                for i in range(self._FALLBACK_DIM)
            ]

    def _embed_chunked(self, text: str) -> list[float]:
        # Tokenize to find the chunk boundaries.
        tokens = self.tokenize(text)
        if len(tokens) <= self._MAX_EMBED_TOKENS:
            return self._embed_one(text)

        # Split into overlapping chunks and average their embeddings.
        stride = self._MAX_EMBED_TOKENS - 256  # 256-token overlap
        chunk_vecs: list[list[float]] = []
        for start in range(0, len(tokens), stride):
            if start + self._MAX_EMBED_TOKENS >= len(tokens):
                chunk_tokens = tokens[start:]
            else:
                chunk_tokens = tokens[start : start + self._MAX_EMBED_TOKENS]
            if not chunk_tokens:
                break
            char_start = int(start / len(tokens) * len(text))
            char_end = int(len(chunk_tokens) / len(tokens) * len(text))
            chunk_text = text[char_start:char_end]
            if not chunk_text.strip():
                continue
            try:
                vec = self._embed_one(chunk_text)
                chunk_vecs.append(vec)
            except Exception:
                # Skip chunk on failure; we'll fall back below if no chunks.
                continue

        if not chunk_vecs:
            # No successful chunks. Truncate to the first chunk-worth
            # of characters and try once more.
            return self._embed_one(text[: self._MAX_EMBED_TOKENS * 4])
        if len(chunk_vecs) == 1:
            return chunk_vecs[0]
        # All chunks must have the same dim; if not, fall back to the
        # first one rather than crashing.
        dim = len(chunk_vecs[0])
        chunk_vecs = [v for v in chunk_vecs if len(v) == dim]
        if not chunk_vecs:
            return self._embed_one(text[: self._MAX_EMBED_TOKENS * 4])
        if len(chunk_vecs) == 1:
            return chunk_vecs[0]
        return [
            sum(v[i] for v in chunk_vecs) / len(chunk_vecs)
            for i in range(dim)
        ]

    def _embed_one(self, text: str) -> list[float]:
        body = self._post("/embedding", {"content": text})
        if isinstance(body, list):
            if body and isinstance(body[0], dict) and "embedding" in body[0]:
                rows = body[0]["embedding"]
                if rows and isinstance(rows[0], list):
                    return [float(x) for x in rows[0]]
                return [float(x) for x in rows]
        elif isinstance(body, dict):
            if "embedding" in body:
                return [float(x) for x in body["embedding"]]
            if "results" in body and body["results"]:
                rows = body["results"][0].get("embedding", [])
                if rows and isinstance(rows[0], list):
                    return [float(x) for x in rows[0]]
                return [float(x) for x in rows]
        raise RuntimeError(f"unexpected /embedding response: {body}")

    def tokenize(self, text: str) -> list[int]:
        body = self._post("/tokenize", {"content": text})
        return [int(t) for t in body["tokens"]]

    def cache_save(self, slot_id: int, cache_id: str) -> dict[str, Any]:
        """Save the slot's KV cache to `<cache_dir>/<cache_id>.bin` using
        llama-server's built-in `?action=save` endpoint."""
        filename = self.cache_path(cache_id)
        body = self._post(
            f"/slots/{slot_id}?action=save",
            {"filename": filename},
        )
        info = self.cache_info(slot_id)
        return {
            "n_past": int(body.get("n_saved", info["n_past"])),
            "bytes": int(body.get("n_written", 0)),
            "model_id": info["model_id"],
        }

    def cache_load(self, slot_id: int, cache_id: str) -> dict[str, Any]:
        """Restore the slot's KV cache from a previously saved file
        using llama-server's built-in `?action=restore` endpoint."""
        filename = self.cache_path(cache_id)
        body = self._post(
            f"/slots/{slot_id}?action=restore",
            {"filename": filename},
        )
        info = self.cache_info(slot_id)
        return {
            "n_past": int(body.get("n_restored", info["n_past"])),
            "model_id": info["model_id"],
        }

    def cache_info(self, slot_id: int) -> dict[str, Any]:
        """Return n_past, model_id, vocab_size for the slot."""
        try:
            slots = self._get("/slots")
            for s in slots:
                if int(s.get("id", s.get("id_slot", -1))) == slot_id:
                    return {
                        "n_past": int(s.get("n_prompt_tokens_processed", 0)),
                        "model_id": s.get("model", "unknown"),
                        "vocab_size": int(s.get("n_probs", 0)) or 152064,
                    }
        except Exception:
            pass
        # Fallback: ask /v1/models for the model id.
        try:
            models = self._get("/v1/models")
            model_id = (
                models["data"][0]["id"] if models.get("data") else "unknown"
            )
        except Exception:
            model_id = "unknown"
        return {"n_past": 0, "model_id": model_id, "vocab_size": 152064}

    def clear_slot(self, slot_id: int) -> dict[str, Any]:
        """Force-clear a slot's KV state.

        llama-server does not expose an endpoint to clear a slot's
        in-memory KV cache (POST /slots/{id}?action=clear returns 501
        on current builds). The only way to reset a slot's KV is to
        send a request with cache_prompt=false, which forces a fresh
        prefill and overwrites the existing KV state.

        This method returns success without doing anything — the actual
        reset happens on the next call to complete() with the same
        slot_id and cache_prompt=False. Callers should track which
        slots are "stale" via engine bookkeeping and send the next
        request with cache_prompt=False.
        """
        # Try the endpoint anyway in case future llama-server versions
        # support it; report the result for diagnostics.
        try:
            body = self._post(
                f"/slots/{slot_id}?action=clear", {"slot_id": slot_id}
            )
            return {"cleared": True, "body": body, "method": "endpoint"}
        except Exception:
            # Endpoint not supported — caller must use cache_prompt=false
            # on the next request to reset the slot.
            return {"cleared": False, "method": "noop", "note": "next request must use cache_prompt=False"}

    def list_slots(self) -> list[dict[str, Any]]:
        """Return the list of slots and their current state."""
        try:
            return self._get("/slots")
        except Exception as e:
            logger.warning("list_slots failed: %s", e)
            return []

    def complete(
        self,
        prompt: str,
        *,
        slot_id: int | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        prefix_token_ids: list[int] | None = None,
        cached_n_past: int = 0,
    ) -> dict[str, Any]:
        """Run a completion. If `cached_n_past > 0` and the caller has
        already loaded a saved KV state into this slot, the server still
        needs to see the full prompt — but the server's own slot cache
        will hit the prefix matches and skip the prefill work for them.

        Returns the same dict shape the engine expects."""
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stream": False,
            "slot_id": slot_id if slot_id is not None else 0,
            "cache_prompt": True,
        }
        started = time.perf_counter()
        r = self._client.post(
            f"{self.base_url}/completion",
            json=payload,
            timeout=600.0,
        )
        r.raise_for_status()
        elapsed = (time.perf_counter() - started) * 1000.0
        body = r.json()
        timings = body.get("timings", {}) or {}
        return {
            "text": body.get("content", ""),
            "tokens_generated": int(body.get("tokens_predicted", 0)),
            "prefill_tokens": int(body.get("tokens_evaluated", 0)),
            "ttft_ms": float(timings.get("prompt_ms", 0.0)),
            "total_ms": elapsed,
            "prefill_tokens_per_sec": (
                body.get("tokens_evaluated", 0)
                / (timings.get("prompt_ms", 0.0) / 1000.0)
                if timings.get("prompt_ms", 0.0) > 0
                else 0.0
            ),
        }

    def complete_native(
        self,
        prompt: str,
        *,
        slot_id: int,
        max_tokens: int = 256,
        temperature: float = 0.0,
        cache_prompt: bool = True,
    ) -> dict[str, Any]:
        """Run a completion that relies on llama-server's IN-MEMORY
        slot cache (no save/restore round-trip through disk).

        The caller is responsible for picking a slot_id whose current
        prompt shares a prefix with `prompt` — that's the only way
        llama-server can reuse KV state without save/restore.

        Pass `cache_prompt=False` to force a fresh prefill, overwriting
        any existing KV state for the slot. Used after a slot spill to
        reset the slot.

        Note: this hits llama-server's legacy ``/completion`` endpoint,
        which does NOT apply chat templates. Use :meth:`complete_openai`
        for chat-tuned models.

        Returns the same dict shape as `complete()`.
        """
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stream": False,
            "slot_id": slot_id,
            "cache_prompt": cache_prompt,
        }
        started = time.perf_counter()
        r = self._client.post(
            f"{self.base_url}/completion",
            json=payload,
            timeout=600.0,
        )
        r.raise_for_status()
        elapsed = (time.perf_counter() - started) * 1000.0
        body = r.json()
        timings = body.get("timings", {}) or {}
        return {
            "text": body.get("content", ""),
            "tokens_generated": int(body.get("tokens_predicted", 0)),
            "prefill_tokens": int(body.get("tokens_evaluated", 0)),
            "ttft_ms": float(timings.get("prompt_ms", 0.0)),
            "total_ms": elapsed,
            "prefill_tokens_per_sec": (
                body.get("tokens_evaluated", 0)
                / (timings.get("prompt_ms", 0.0) / 1000.0)
                if timings.get("prompt_ms", 0.0) > 0
                else 0.0
            ),
        }

    def complete_openai(
        self,
        messages: list[dict],
        *,
        slot_id: int,
        max_tokens: int = 256,
        temperature: float = 0.0,
        cache_prompt: bool = True,
        chat_template_kwargs: dict | None = None,
    ) -> dict[str, Any]:
        """Run a chat completion via llama-server's OpenAI-compatible
        endpoint (``/v1/chat/completions``).

        Unlike :meth:`complete_native`, this sends OpenAI-format
        ``messages`` and lets llama-server apply the chat template
        appropriate for the loaded model (Qwen3.5, Llama, Mistral, etc.).
        Slot pinning via ``slot_id`` is honored via the
        ``cache_prompt`` flag, so prefix-reuse still works.

        Args:
            messages: list of ``{"role": ..., "content": ...}`` dicts.
            slot_id: which llama-server slot to use. Caller is responsible
                for picking a slot whose existing prefix matches the
                new messages' prefix.
            cache_prompt: ``True`` to attempt prefix-reuse (recommended
                for the recycled flow); ``False`` to force a fresh
                prefill (used after slot spill).
            chat_template_kwargs: pass-through kwargs for llama-server's
                chat template (e.g. ``{"enable_thinking": False}`` for
                Qwen3.5). None means don't send the field at all.

        Returns the same dict shape as :meth:`complete_native`.
        """
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "slot_id": slot_id,
            "cache_prompt": cache_prompt,
        }
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        started = time.perf_counter()
        r = self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=600.0,
        )
        r.raise_for_status()
        elapsed = (time.perf_counter() - started) * 1000.0
        body = r.json()
        timings = body.get("timings", {}) or {}
        # OpenAI response shape: {"choices": [{"message": {"content": "..."}}]}
        text = ""
        choices = body.get("choices") or []
        if choices and isinstance(choices, list):
            msg = choices[0].get("message") or {}
            text = msg.get("content", "")
        # OpenAI response shape: {"choices": [{"message": {"content": "..."}}],
        # "usage": {"completion_tokens": N, "prompt_tokens": M, "total_tokens": K},
        # "timings": {...}}. llama-server also returns the llama.cpp
        # fields (tokens_predicted / tokens_evaluated) for compatibility.
        usage = body.get("usage") or {}
        return {
            "text": text,
            "tokens_generated": int(
                usage.get("completion_tokens", 0)
                or body.get("tokens_predicted", 0)
            ),
            "prefill_tokens": int(
                usage.get("prompt_tokens", 0)
                or body.get("tokens_evaluated", 0)
            ),
            "ttft_ms": float(timings.get("prompt_ms", 0.0)),
            "total_ms": elapsed,
            "prefill_tokens_per_sec": (
                body.get("tokens_evaluated", 0)
                / (timings.get("prompt_ms", 0.0) / 1000.0)
                if timings.get("prompt_ms", 0.0) > 0
                else 0.0
            ),
        }

    async def complete_openai_stream(
        self,
        messages: list[dict],
        *,
        slot_id: int,
        max_tokens: int = 256,
        temperature: float = 0.0,
        cache_prompt: bool = True,
        chat_template_kwargs: dict | None = None,
    ):
        """Stream a chat completion via llama-server's OpenAI-compatible
        endpoint (``/v1/chat/completions`` with ``stream: True``).

        Yields dicts:
          ``{"kind": "delta", "delta": str}`` — one per content chunk.
          ``{"kind": "result", "result": {...}}`` — final dict with the
          same shape as :meth:`complete_openai`'s return value.

        Raises ``RuntimeError`` if llama-server answers with an error
        status before streaming begins.
        """
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            "slot_id": slot_id,
            "cache_prompt": cache_prompt,
        }
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        started = time.perf_counter()
        first_token_at: float | None = None
        text_parts: list[str] = []
        timings: dict = {}
        usage: dict = {}
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            ) as r:
                if r.status_code != 200:
                    body = await r.aread()
                    raise RuntimeError(
                        f"llama-server {r.status_code}: "
                        f"{body[:300].decode('utf-8', errors='replace')}"
                    )
                async for line in r.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices") or []
                    if choices and isinstance(choices[0], dict):
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content")
                        if content:
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                            text_parts.append(content)
                            yield {"kind": "delta", "delta": content}
                    if obj.get("usage"):
                        usage = obj["usage"]
                    if obj.get("timings"):
                        timings = obj["timings"]
        elapsed = (time.perf_counter() - started) * 1000.0
        ttft_ms = (
            (first_token_at - started) * 1000.0
            if first_token_at is not None
            else float(timings.get("prompt_ms", 0.0))
        )
        yield {
            "kind": "result",
            "result": {
                "text": "".join(text_parts),
                "tokens_generated": int(
                    usage.get("completion_tokens", len(text_parts))
                    or len(text_parts)
                ),
                "prefill_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "ttft_ms": float(ttft_ms),
                "total_ms": elapsed,
                "prefill_tokens_per_sec": 0.0,
            },
        }
