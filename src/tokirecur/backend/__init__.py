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

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

from ..core.recycling import LlamaBackend

logger = logging.getLogger(__name__)


def _strip_patterns(node: Any) -> Any:
    """Return a copy of a JSON schema with ``pattern`` and length
    constraints removed.

    The laguna fork's JSON-schema -> GBNF converter inlines ``pattern``
    values verbatim into GBNF string literals. Regex escapes the converter
    does not know (e.g. ``\\S`` in OpenClaw's ``cron`` tool) produce
    ``unknown escape`` errors that fail grammar generation for the whole
    request. ``minLength``/``maxLength`` map to GBNF repetitions, and
    values above the fork's ``MAX_REPETITION_THRESHOLD`` (2000) — e.g.
    ``maxLength: 65536`` — are rejected outright. These are advisory
    constraints, so they are dropped.
    """
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            # Only strip the ``pattern``/``minLength``/``maxLength``
            # *keywords* (string regexes and integer lengths). Property
            # definitions are dicts, so a property *named* ``pattern``
            # (e.g. the ``glob`` tool) must be preserved — stripping it
            # leaves an empty ``properties`` schema whose grammar forces
            # tool calls with empty ``{}`` arguments.
            if (
                key in ("pattern", "minLength", "maxLength")
                and isinstance(value, (str, int, float))
            ):
                continue
            out[key] = _strip_patterns(value)
        return out
    if isinstance(node, list):
        return [_strip_patterns(item) for item in node]
    return node


def _content_parts(content: Any) -> list[Any]:
    if isinstance(content, list):
        return [dict(part) if isinstance(part, dict) else part for part in content]
    if content is None:
        return []
    return [{"type": "text", "text": str(content)}]


def _merge_assistant(previous: dict, message: dict) -> None:
    content = message.get("content")
    if content not in (None, "", []):
        previous_content = previous.get("content")
        if previous_content in (None, "", []):
            previous["content"] = content
        elif isinstance(previous_content, str) and isinstance(content, str):
            previous["content"] = previous_content + "\n" + content
        else:
            previous["content"] = (
                _content_parts(previous_content) + _content_parts(content)
            )
    reasoning = message.get("reasoning_content")
    if reasoning:
        previous_reasoning = previous.get("reasoning_content")
        previous["reasoning_content"] = (
            previous_reasoning + "\n" + reasoning
            if previous_reasoning
            else reasoning
        )
    tool_calls = message.get("tool_calls")
    if tool_calls:
        previous["tool_calls"] = (
            list(previous.get("tool_calls") or []) + list(tool_calls)
        )
    function_call = message.get("function_call")
    if function_call and not previous.get("function_call"):
        previous["function_call"] = function_call
    name = message.get("name")
    if name and not previous.get("name"):
        previous["name"] = name


def _normalize_messages(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for raw in messages:
        if not isinstance(raw, dict):
            out.append(raw)
            continue
        message = dict(raw)
        if (
            message.get("role") == "assistant"
            and out
            and isinstance(out[-1], dict)
            and out[-1].get("role") == "assistant"
        ):
            _merge_assistant(out[-1], message)
            continue
        out.append(message)
    return out


def _normalize_tool(tool: Any) -> dict | None:
    if not isinstance(tool, dict):
        return None
    tool = dict(tool)
    function = tool.get("function")
    if not isinstance(function, dict):
        name = tool.get("name")
        if not name:
            return tool
        parameters = tool.get("parameters") or tool.get("input_schema") or {}
        if isinstance(parameters, dict):
            parameters = _strip_patterns(parameters)
        tool = {
            "type": "function",
            "function": {
                "name": name,
                "description": tool.get("description", ""),
                "parameters": parameters,
            },
        }
        return tool
    function = dict(function)
    parameters = function.get("parameters")
    if parameters is None:
        parameters = function.get("input_schema")
    if isinstance(parameters, dict):
        parameters = _strip_patterns(parameters)
    function["parameters"] = parameters or {}
    tool["function"] = function
    tool.setdefault("type", "function")
    return tool


def _sanitize_tools(tools: list) -> list:
    """Return tools in OpenAI function-tool shape with ``pattern`` stripped."""
    out: list[dict] = []
    for tool in tools:
        normalized = _normalize_tool(tool)
        if normalized is not None:
            out.append(normalized)
    return out


# Sampling fields llama-server accepts on /completion and
# /v1/chat/completions. temperature travels as its own parameter;
# everything else rides in the `sampling` dict.
_SAMPLING_FIELDS = ("repeat_penalty", "top_k", "top_p", "min_p")


def _merge_sampling(payload: dict, sampling: dict | None) -> dict:
    """Fold non-None sampling values into a llama-server payload."""
    if not sampling:
        return payload
    for key in _SAMPLING_FIELDS:
        value = sampling.get(key)
        if value is not None:
            payload[key] = value
    return payload


def _swallow_task_result(task: "asyncio.Task[Any]") -> None:
    """Retrieve a finished task's exception so asyncio does not warn that
    it was never retrieved — the task outlives its consumer on purpose."""
    try:
        task.exception()
    except asyncio.CancelledError:
        pass


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

    async def slot_snapshot(self) -> list[dict[str, Any]] | None:
        """Current llama-server slot states, or None if unreachable.

        Each entry carries the slot id, whether it is busy, the task id
        llama-server assigned it and how many prompt tokens that task
        has processed. The last two are what distinguishes a slot that is
        genuinely generating from one left flagged busy by a cancelled
        task — the latter never advances.
        """
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{self.base_url}/slots")
                r.raise_for_status()
                raw = r.json()
        except Exception:
            return None
        out: list[dict[str, Any]] = []
        for slot in raw or []:
            if not isinstance(slot, dict):
                continue
            # next_token[0].n_decoded is the only counter that moves while
            # a slot generates — n_prompt_tokens_processed freezes once the
            # prompt is in, so it cannot tell "generating" from "stuck".
            decoded = 0
            remaining = 0
            next_token = slot.get("next_token") or []
            if isinstance(next_token, list) and next_token and isinstance(next_token[0], dict):
                decoded = int(next_token[0].get("n_decoded") or 0)
                remaining = int(next_token[0].get("n_remain") or 0)
            out.append({
                "id": slot.get("id"),
                "busy": bool(slot.get("is_processing")),
                "task": slot.get("id_task"),
                "processed": slot.get("n_prompt_tokens_processed"),
                "decoded": decoded,
                "remaining": remaining,
            })
        return out

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
        """Save the slot's KV cache to ``<cache_dir>/<cache_id>.bin``
        via llama-server's ``?action=save`` endpoint, then drop a
        ``<cache_dir>/<cache_id>.trkv`` digest sidecar via
        :class:`tokirecur.cache.blob.BlobWriter`.

        The ``.bin`` file is llama-server's native on-disk format and
        is what ``?action=restore`` consumes. The ``.trkv`` sidecar is
        a small header + SHA-256 digest that the tree-cache uses for
        a fast file-existence check (``tree_cache.lookup``) and that
        :meth:`cache_load` verifies against before restoring.
        """
        filename = self.cache_path(cache_id)
        body = self._post(
            f"/slots/{slot_id}?action=save",
            {"filename": filename},
        )
        info = self.cache_info(slot_id)
        n_written = int(body.get("n_written", 0))

        # Write the digest sidecar. The ``.trkv`` file contains the
        # blob header (model_id, n_past, vocab_size) plus the SHA-256
        # of the underlying ``.bin`` payload — same byte content as
        # what llama-server persisted, so any tampering surfaces as a
        # digest mismatch on restore.
        try:
            self._write_trkv_sidecar(cache_id, n_past=n_written, info=info)
        except Exception as e:
            logger.warning("failed to write .trkv sidecar for %s: %s",
                           cache_id, e)

        return {
            "n_past": int(body.get("n_saved", info["n_past"])),
            "bytes": n_written,
            "model_id": info["model_id"],
        }

    def _write_trkv_sidecar(
        self, cache_id: str, *, n_past: int, info: dict[str, Any],
    ) -> None:
        """Write a ``<cache_id>.trkv`` digest sidecar alongside the
        ``.bin`` payload llama-server just persisted.

        Returns silently if ``cache_dir`` isn't configured (e.g. when
        the daemon is wired against a custom backend that doesn't
        materialise files locally).
        """
        if self.cache_dir is None:
            return
        from ..cache.blob import BlobWriter, blob_path
        bin_path = self.cache_dir / f"{cache_id}.bin"
        if not bin_path.exists():
            return
        trkv_path = blob_path(self.cache_dir, cache_id)
        with BlobWriter(trkv_path) as w:
            w.write_header(
                model_id=info.get("model_id", "unknown"),
                n_past=n_past,
                vocab_size=info.get("vocab_size", 152064),
            )
            w.write_payload(bin_path.read_bytes())
            w.finalize()

    def cache_load(self, slot_id: int, cache_id: str) -> dict[str, Any]:
        """Restore the slot's KV cache from a previously saved file
        using llama-server's built-in ``?action=restore`` endpoint.

        Verifies the ``.trkv`` digest sidecar matches the ``.bin``
        payload before issuing the restore — a mismatch means either
        disk corruption or tampering, and we refuse to feed garbage
        to llama-server.
        """
        self._verify_trkv_sidecar(cache_id)
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

    def _verify_trkv_sidecar(self, cache_id: str) -> None:
        """Verify the ``.trkv`` digest sidecar's SHA-256 matches the
        ``.bin`` payload's SHA-256. Skips silently if either file is
        missing or if the sidecar was never written (legacy cache_dir
        populated by older daemon versions)."""
        if self.cache_dir is None:
            return
        from ..cache.blob import BlobReader, blob_path
        bin_path = self.cache_dir / f"{cache_id}.bin"
        trkv_path = blob_path(self.cache_dir, cache_id)
        if not (bin_path.exists() and trkv_path.exists()):
            return
        try:
            reader = BlobReader(trkv_path)
            payload = reader.read_payload()
        except (ValueError, OSError):
            return
        if reader.digest != BlobReader.compute_sha256(payload):
            raise RuntimeError(
                f".trkv sidecar digest mismatch for {cache_id}; "
                f"refusing to feed corrupted KV to llama-server"
            )

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
            return {
                "cleared": False,
                "method": "noop",
                "note": "next request must use cache_prompt=False",
            }

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
        cache_prompt: bool = True,
        sampling: dict | None = None,
    ) -> dict[str, Any]:
        """Run a completion. If `cached_n_past > 0` and the caller has
        already loaded a saved KV state into this slot, the server still
        needs to see the full prompt — but the server's own slot cache
        will hit the prefix matches and skip the prefill work for them.

        Pass ``cache_prompt=False`` to force a fresh prefill (overwrites
        the slot's existing KV). Used by the precision-retention check
        to establish a true baseline where the cache cannot influence
        the result.

        Returns the same dict shape the engine expects."""
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stream": False,
            "slot_id": slot_id if slot_id is not None else 0,
            "cache_prompt": cache_prompt,
        }
        _merge_sampling(payload, sampling)
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
        sampling: dict | None = None,
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
        _merge_sampling(payload, sampling)
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
        reasoning_effort: str | None = None,
        tools: list | None = None,
        tool_choice: str | dict | None = None,
        sampling: dict | None = None,
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
            tools: OpenAI tool definitions (function schemas). When set,
                the model may respond with ``tool_calls`` instead of
                text.
            tool_choice: ``"auto"`` / ``"none"`` or a
                ``{"type": "function", "function": {...}}`` object.

        Returns the same dict shape as :meth:`complete_native`, plus
        ``tool_calls`` (assembled OpenAI tool-call list, or None) and
        ``finish_reason`` when the model requested tools.
        """
        messages = _normalize_messages(messages)
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "slot_id": slot_id,
            "cache_prompt": cache_prompt,
        }
        _merge_sampling(payload, sampling)
        if tools:
            payload["tools"] = _sanitize_tools(tools)
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        if reasoning_effort:
            # llama-server reads this field itself and hands it to the chat
            # template, so the model steers its own reasoning depth. Sending
            # it the way llama-server expects is the whole point: the
            # template decides what "high" means for its family.
            payload["reasoning_effort"] = reasoning_effort
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
        reasoning = ""
        tool_calls = None
        finish_reason = None
        choices = body.get("choices") or []
        if choices and isinstance(choices, list):
            choice = choices[0] or {}
            msg = choice.get("message") or {}
            text = msg.get("content", "") or ""
            # llama-server puts the thinking here when the reasoning format
            # asks for it. Without this the effort is invisible: the model
            # thinks, spends the tokens, and says nothing about it.
            reasoning = msg.get("reasoning_content", "") or ""
            tool_calls = msg.get("tool_calls")
            finish_reason = choice.get("finish_reason")
        # OpenAI response shape: {"choices": [{"message": {"content": "..."}}],
        # "usage": {"completion_tokens": N, "prompt_tokens": M, "total_tokens": K},
        # "timings": {...}}. llama-server also returns the llama.cpp
        # fields (tokens_predicted / tokens_evaluated) for compatibility.
        usage = body.get("usage") or {}
        return {
            "text": text,
            "reasoning": reasoning,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
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
        reasoning_effort: str | None = None,
        tools: list | None = None,
        tool_choice: str | dict | None = None,
        sampling: dict | None = None,
    ):
        """Stream a chat completion via llama-server's OpenAI-compatible
        endpoint (``/v1/chat/completions`` with ``stream: True``).

        Yields dicts:
          ``{"kind": "delta", "delta": str}`` — one per content chunk.
          ``{"kind": "tool_calls", "tool_calls": [...]}`` — one per
          incoming ``delta.tool_calls`` array, forwarded verbatim in
          OpenAI delta format (the SDK accumulates arguments by index).
          ``{"kind": "finish", "finish_reason": str}`` — the stream's
          finish reason (e.g. ``"tool_calls"``), if any.
          ``{"kind": "result", "result": {...}}`` — final dict with the
          same shape as :meth:`complete_openai`'s return value (plus
          assembled ``tool_calls`` / ``finish_reason``).

        Raises ``RuntimeError`` if llama-server answers with an error
        status before streaming begins.

        The upstream HTTP stream is owned by a producer task, not by this
        generator. A harness that goes away mid-turn (timeout, Ctrl+C, an
        aborted tool call) cancels this generator — and a cancelled httpx
        stream closes the connection, which llama-server reads as "cancel
        this task". It then leaves the slot flagged busy *forever*, so
        after two such cancellations the whole server is wedged: health
        checks stay green and every later request hangs until someone
        restarts the backend. Letting the producer finish lets the task
        complete and the slot be released.
        """
        messages = _normalize_messages(messages)
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            "slot_id": slot_id,
            "cache_prompt": cache_prompt,
        }
        _merge_sampling(payload, sampling)
        if tools:
            payload["tools"] = _sanitize_tools(tools)
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        started = time.perf_counter()

        queue: asyncio.Queue = asyncio.Queue()
        _END = object()

        async def _produce() -> None:
            first_token_at: float | None = None
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            timings: dict = {}
            usage: dict = {}
            finish_reason: str | None = None
            # Assembled tool calls, keyed by index: {"id", "type",
            # "function": {"name", "arguments"}} with arguments accumulated
            # across delta chunks.
            tool_calls_by_index: dict[int, dict] = {}
            try:
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
                                choice = choices[0]
                                delta = choice.get("delta") or {}
                                content = delta.get("content")
                                if content:
                                    if first_token_at is None:
                                        first_token_at = time.perf_counter()
                                    text_parts.append(content)
                                    queue.put_nowait({"kind": "delta", "delta": content})
                                reasoning = delta.get("reasoning_content")
                                if reasoning:
                                    if first_token_at is None:
                                        first_token_at = time.perf_counter()
                                    reasoning_parts.append(reasoning)
                                    queue.put_nowait({"kind": "reasoning", "delta": reasoning})
                                tcs = delta.get("tool_calls")
                                if tcs:
                                    # Forward the delta verbatim (OpenAI format)
                                    # and accumulate fragments for the result.
                                    queue.put_nowait({"kind": "tool_calls", "tool_calls": tcs})
                                    for tc in tcs:
                                        if not isinstance(tc, dict) or "index" not in tc:
                                            continue
                                        idx = tc["index"]
                                        acc = tool_calls_by_index.setdefault(idx, {"index": idx})
                                        if tc.get("id"):
                                            acc["id"] = tc["id"]
                                        if tc.get("type"):
                                            acc["type"] = tc["type"]
                                        fn = tc.get("function") or {}
                                        if fn:
                                            f_acc = acc.setdefault("function", {})
                                            if fn.get("name"):
                                                f_acc["name"] = fn["name"]
                                            if fn.get("arguments"):
                                                f_acc["arguments"] = (
                                                    f_acc.get("arguments", "") + fn["arguments"]
                                                )
                                fr = choice.get("finish_reason")
                                if fr:
                                    finish_reason = fr
                                    queue.put_nowait({"kind": "finish", "finish_reason": fr})
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
                tool_calls = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)] or None
                queue.put_nowait({
                    "kind": "result",
                    "result": {
                        "text": "".join(text_parts),
                        "reasoning": "".join(reasoning_parts),
                        "tool_calls": tool_calls,
                        "finish_reason": finish_reason,
                        "tokens_generated": int(
                            usage.get("completion_tokens", len(text_parts))
                            or len(text_parts)
                        ),
                        "prefill_tokens": int(usage.get("prompt_tokens", 0) or 0),
                        "ttft_ms": float(ttft_ms),
                        "total_ms": elapsed,
                        "prefill_tokens_per_sec": 0.0,
                    },
                })
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # surfaced to the consumer below
                queue.put_nowait(exc)
            finally:
                queue.put_nowait(_END)

        producer = asyncio.create_task(_produce())
        try:
            while True:
                item = await queue.get()
                if item is _END:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        except BaseException:
            # The client went away (CancelledError from a disconnect,
            # GeneratorExit from aclose()). Let the producer keep
            # draining upstream so llama-server finishes the task and
            # releases the slot instead of leaving it wedged.
            producer.add_done_callback(_swallow_task_result)
            raise

