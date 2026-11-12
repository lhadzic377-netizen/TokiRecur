"""Core recycling engine.

Implements the prefix-reuse algorithm from the academic paper
"KV Cache Recycling to Expand Usable Context Capacity in Low Parameter LLMs"
(arXiv:2512.11851v1) adapted for a llama-server backend.

The algorithm relies on llama-server's IN-MEMORY slot cache (no disk
round-trip). We:
  1. Embed an incoming prompt.
  2. Pick the slot whose current KV context shares the longest prefix
     with the new prompt (via `_pick_slot()`).
  3. Send the new prompt to that slot. llama-server reuses the matching
     prefix automatically — no save/load, no precision loss.

The reuse depth r satisfies:
    r = max{ r' <= min(m, k) : x_{1:r'}^{(t)} == x_{1:r'}^{(c)} }
where m is the slot's KV context length, k is the new prompt length, and
x are token IDs. The condition is conservative: every cached KV tensor
must correspond to a token that is byte-identical in the new prompt.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Minimum reuse depth to count as a "real" cache hit. Below this, the
# save+restore overhead exceeds the prefill savings, so we report it
# as a soft_hit (not counted in hit rate) instead.
MIN_REUSE_FOR_HIT = 10


class LlamaBackend(Protocol):
    """Protocol every backend must implement. The recycling engine is
    backend-agnostic; it calls only these methods."""

    def embed(self, text: str) -> list[float]:
        """Return a fixed-dimensional embedding vector for the given text."""
        ...

    def tokenize(self, text: str) -> list[int]:
        """Return the token IDs the model would use for `text`."""
        ...

    def cache_info(self, slot_id: int) -> dict[str, Any]:
        """Return current slot state: `n_past`, `model_id`, `vocab_size`."""
        ...

    def complete_native(
        self,
        prompt: str,
        *,
        slot_id: int,
        max_tokens: int = 256,
        temperature: float = 0.0,
        cache_prompt: bool = True,
    ) -> dict[str, Any]:
        """Run a completion on the given slot with the slot's existing KV
        cache (cache_prompt=true semantics). llama-server will reuse any
        matching prefix from the slot's history automatically.

        Pass `cache_prompt=False` to force a fresh prefill, overwriting
        the slot's existing KV state. Used after a slot spill to reset
        the slot.

        Returns a dict with at least `text`, `tokens_generated`,
        `prefill_tokens`, `ttft_ms`, `total_ms`, `tokens_per_sec_prefill`.
        """
        ...


@dataclass
class CacheEntry:
    """Metadata for a single cached KV state."""

    cache_id: str
    prompt: str
    token_ids: list[int]
    embedding: list[float]
    n_past: int
    model_id: str
    bytes_on_disk: int
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    hit_count: int = 0


def _content_text(content: Any) -> str:
    """Normalize OpenAI message ``content`` to a plain string for
    routing/indexing. Agentic clients (e.g. opencode) send multi-part
    content as a list of ``{"type": "text", ...}`` parts; llama-server
    accepts those in chat messages, but the backend's /tokenize and
    /embed endpoints only take strings — a raw list there makes
    llama-server answer 500 ("type must be number, but is object").
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "".join(parts)
    return str(content)


def _last_routing_text(
    messages: list[dict], reasoning_effort: str | None = None,
) -> str:
    """Routing key for a message list: the last user message's text
    (normalized), falling back to the final message's content.

    The reasoning effort is part of the key when there is one. The
    effort changes the prompt the template renders — the level's
    instruction goes into the system message — so the same messages at
    two levels are two different prefixes. Without it in the key, the
    daemon would call the second one a hit while the engine re-prefilled
    every token of it: the reuse would be reported and not happen.
    """
    body = ""
    for m in reversed(messages):
        if m.get("role") == "user" and m.get("content"):
            body = _content_text(m["content"])
            break
    if not body and messages:
        body = _content_text(messages[-1].get("content", ""))
    if reasoning_effort:
        return f"[effort:{reasoning_effort}]\n{body}"
    return body


@dataclass
class RecyclingResult:
    """The outcome of a single recycling decision."""

    hit: bool  # True if reuse_depth >= MIN_REUSE_FOR_HIT (a "real" win)
    soft_hit: bool  # True if 1 <= reuse_depth < MIN_REUSE_FOR_HIT
    cache_id: str | None
    reuse_depth: int  # tokens of the new prompt served from cache
    embedding_similarity: float
    prefilled_tokens: int  # tokens the backend had to actually decode
    skipped_tokens: int  # tokens served from cache
    generated_tokens: int
    ttft_ms: float
    total_ms: float
    prefill_tokens_per_sec: float
    generated_tokens_per_sec: float
    output_text: str = ""
    reasoning: str = ""
    output_tokens: list[int] = field(default_factory=list)
    slot_id: int = 0  # which slot served the request
    spilled: bool = False  # True if this slot was cleared due to context budget
    tool_calls: list | None = None  # OpenAI tool-call list (tool mode only)
    finish_reason: str | None = None  # e.g. "tool_calls" / "stop" (tool mode only)


def make_cache_id(prompt: str, model_id: str) -> str:
    """Stable cache id from prompt + model identifier."""
    h = hashlib.sha256()
    h.update(model_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(prompt.encode("utf-8"))
    return h.hexdigest()[:24]


def compute_reuse_depth(
    cached_tokens: list[int],
    new_tokens: list[int],
) -> int:
    """Return the longest prefix length r such that cached_tokens[:r] ==
    new_tokens[:r]. Conservative: the cached KV is only valid for the
    contiguous matched prefix."""
    r = 0
    limit = min(len(cached_tokens), len(new_tokens))
    while r < limit and cached_tokens[r] == new_tokens[r]:
        r += 1
    return r


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Standard cosine similarity over two equal-length vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


class RecyclingEngine:
    """The 5-step algorithm in object form.

    Construct with a backend and an index. The index is responsible for
    embedding lookup and exact-prefix verification; the engine just
    orchestrates the policy.

    Optional components:
      - tree_cache: dormant disk-KV index. Routing is purely in-memory
        slot reuse — disk save/restore round-trips were removed because
        they corrupted KV precision (cached outputs diverged from
        baseline). Retained so the design can be revived behind a
        precision-safe backend. Defaults to a fresh ``TreeCache``
        rooted at ``~/.local/state/tokirecur/slot-save``.
      - persistent_pool: keeps the top-N most-used prefixes in
        dedicated slots as an in-memory routing hint (no disk
        round-trip). Defaults to a fresh ``PersistentSlotPool``.
    """

    def __init__(
        self,
        backend: LlamaBackend,
        index: Index,  # noqa: F821 - forward-declared
        *,
        similarity_threshold: float = 0.85,
        tree_cache: Any | None = None,
        persistent_pool: Any | None = None,
    ) -> None:
        self.backend = backend
        self.index = index
        self.similarity_threshold = similarity_threshold
        if tree_cache is None:
            from pathlib import Path

            from ..cache.tree import TreeCache
            default_dir = Path(
                os.environ.get(
                    "TOKI_SLOT_SAVE_DIR",
                    str(Path.home() / ".local/state/tokirecur/slot-save"),
                )
            )
            tree_cache = TreeCache(default_dir)
        self.tree_cache = tree_cache
        if persistent_pool is None:
            from ..cache.persistent_pool import PersistentSlotPool
            persistent_pool = PersistentSlotPool()
        self.persistent_pool = persistent_pool

    def serve_native(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        max_slots: int = 1,
        session_id: str | None = None,
        max_context_tokens: int | None = None,
        sampling: dict | None = None,
    ) -> RecyclingResult:
        """Run the recycling algorithm using llama-server's IN-MEMORY
        slot cache instead of explicit save/restore.

        Why this exists
        ---------------
        The original `serve()` flow saved the slot's KV state to disk
        and restored it on the next request. That round-trip through
        disk loses precision, causing 23-37% of cached outputs to differ
        from baseline (76% on multi-turn workloads). `serve_native()`
        avoids this entirely by relying on llama-server's built-in
        `cache_prompt=true` flag, which keeps the slot's KV state in
        VRAM across requests. To get prefix reuse, the new prompt must
        be routed to the SAME slot that already holds the matching
        prefix — we pick that slot via `_pick_slot()` below.

        Note: this method passes a flat prompt string to llama-server's
        legacy ``/completion`` endpoint, which does NOT apply chat
        templates. Use :meth:`serve_chat` for chat-tuned models.

        Slot routing
        ------------
        Each slot in llama-server keeps its own prefix KV. To get
        reuse, similar prompts must land on the same slot. We track
        each slot's last prompt and pick the one whose tokens share the
        longest prefix with the new prompt. If no slot has a useful
        prefix, we round-robin to the next free slot.

        Session routing
        ---------------
        For multi-turn conversations, prompts naturally extend each
        other (turn 2 = turn 1 + assistant reply + new question). If
        we route by prefix match alone, turn 2 may pick a different
        slot than turn 1 if another slot happens to share a longer
        accidental prefix. To prevent that, callers can pass
        `session_id=<stable id>` (e.g. an OpenAI `user` field or
        `X-Session-Id` header). All requests with the same session_id
        land on the same slot.

        Slot spilling
        -------------
        Multi-turn conversations can grow their slot's KV context
        without bound. After each request, if the slot's context length
        exceeds `max_context_tokens` (default 4096), we clear the
        slot's KV state via `backend.clear_slot()`. This bounds
        attention cost and prevents TTFT regressions on long-running
        sessions.
        """
        return self._serve_with_backend(
            prompt=prompt,
            messages=None,
            backend_call=lambda slot_id, cache_prompt: self.backend.complete_native(
                prompt,
                slot_id=slot_id,
                max_tokens=max_tokens,
                temperature=temperature,
                cache_prompt=cache_prompt,
                sampling=sampling,
            ),
            tokenize_text=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            max_slots=max_slots,
            session_id=session_id,
            max_context_tokens=max_context_tokens,
        )

    def serve_chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        max_slots: int = 1,
        session_id: str | None = None,
        max_context_tokens: int | None = None,
        chat_template_kwargs: dict | None = None,
        reasoning_effort: str | None = None,
        tools: list | None = None,
        tool_choice: str | dict | None = None,
        sampling: dict | None = None,
    ) -> RecyclingResult:
        """Run the recycling algorithm with OpenAI-format messages.

        Like :meth:`serve_native`, but routes OpenAI ``messages`` to
        llama-server's ``/v1/chat/completions`` endpoint so the model
        gets the right chat template. This is the right method for
        chat-tuned models (Qwen3.5, Llama, Mistral, etc.).

        Args:
            messages: OpenAI-format list of ``{"role": ..., "content": ...}``
                dicts (may include ``tool_calls`` / ``tool_call_id`` /
                ``name`` for agentic workloads). The concatenated text
                is what gets indexed for prefix-reuse matching.
            max_tokens, temperature, max_slots, session_id,
                max_context_tokens: see :meth:`serve_native`.
            chat_template_kwargs: pass-through for llama-server's chat
                template (e.g. ``{"enable_thinking": False}`` for Qwen3.5).
            tools: OpenAI tool definitions; when set the model may reply
                with ``tool_calls`` (see :meth:`RecyclingResult.tool_calls`).
            tool_choice: OpenAI tool-choice selector, passed through.
        """
        # Index the joined user content for prefix matching. We use the
        # last user message's text as the routing key — it's what future
        # turns are most likely to share with. Content is normalized to
        # a string (clients may send multi-part arrays).
        routing_text = _last_routing_text(messages, reasoning_effort)

        return self._serve_with_backend(
            prompt=routing_text,
            messages=messages,
            backend_call=lambda slot_id, cache_prompt: self.backend.complete_openai(
                messages,
                slot_id=slot_id,
                max_tokens=max_tokens,
                temperature=temperature,
                cache_prompt=cache_prompt,
                chat_template_kwargs=chat_template_kwargs,
                reasoning_effort=reasoning_effort,
                tools=tools,
                tool_choice=tool_choice,
                sampling=sampling,
            ),
            tokenize_text=routing_text,
            max_tokens=max_tokens,
            temperature=temperature,
            max_slots=max_slots,
            session_id=session_id,
            max_context_tokens=max_context_tokens,
        )

    def _route_request(
        self,
        *,
        tokenize_text: str,
        max_slots: int,
        session_id: str | None,
        max_context_tokens: int | None,
    ) -> tuple[list[int], list[float], int, int, bool, bool, str]:
        """Shared slot-routing: decide which llama-server slot to use
        and whether to force a fresh prefill.

        Returns ``(new_tokens, new_embedding, slot_id, reuse_depth,
        spilled, cache_prompt, cache_id)``.

        The embedding is deferred until after the spill check — if the
        current slot is going to be spilled, the slot-id is going to
        change, so any embedding work for routing is wasted. The
        /embedding HTTP call is the dominant per-request cost after
        tokenize, so skipping it on spill saves a roundtrip.
        """
        # Initialise slot state lazily.
        self._init_slot_state()

        new_tokens = self.backend.tokenize(tokenize_text)

        # Spill check on the session's current slot, if any. Reserved
        # slots are exempt from spilling.
        spilled = False
        cache_prompt = True
        if (
            session_id is not None
            and session_id in self._session_to_slot
            and not self._is_slot_reserved(self._session_to_slot[session_id])
        ):
            existing_slot = self._session_to_slot[session_id]
            existing_ctx = self._slot_context_tokens.get(existing_slot, 0)
            if max_context_tokens is None:
                max_context_tokens = self._max_context_tokens
            if max_context_tokens and existing_ctx + len(new_tokens) > max_context_tokens:
                self._spill_slot(existing_slot)
                del self._session_to_slot[session_id]
                spilled = True
                cache_prompt = False  # Force fresh prefill on the new slot

        # Compute the content-addressed cache_id once. Used for index
        # lookup, hot-prefix pooling and the result payload.
        cache_id = make_cache_id(tokenize_text, "llama-server-slot")

        # Pick the slot that has the longest matching prefix with the new
        # prompt. llama-server will then reuse the prefix automatically.
        slot_id, reuse_depth = self._pick_slot(
            new_tokens, max_slots, session_id=session_id,
        )

        new_embedding = self.backend.embed(tokenize_text)
        return new_tokens, new_embedding, slot_id, reuse_depth, spilled, cache_prompt, cache_id

    def _finalize_request(
        self,
        *,
        tokenize_text: str,
        new_tokens: list[int],
        new_embedding: list[float],
        slot_id: int,
        reuse_depth: int,
        spilled: bool,
        session_id: str | None,
        backend_result: dict[str, Any],
        started: float,
        cache_id: str,
    ) -> RecyclingResult:
        """Update slot bookkeeping + index after a completion and build
        the :class:`RecyclingResult`."""
        # Update slot state.
        self._slot_history[slot_id] = new_tokens
        self._slot_context_tokens[slot_id] = (
            self._slot_context_tokens.get(slot_id, 0) + len(new_tokens)
        )
        self._slot_use_count[slot_id] = self._slot_use_count.get(slot_id, 0) + 1
        if session_id is not None:
            self._session_to_slot[session_id] = slot_id

        # Hot-prefix detection only counts MISSES — a hit on an already-
        # reserved prefix shouldn't push us past the threshold a second
        # time. Same-prefix hits still flow through, but they don't bump
        # the count.
        real_hit = reuse_depth >= MIN_REUSE_FOR_HIT
        if not real_hit:
            self._maybe_reserve_slot(slot_id, new_tokens, cache_id)

        # Index for future routing — only on miss. On hit the entry
        # already exists in the index; INSERT OR REPLACE was wasted I/O.
        # cache_id is still derived for every request: the result carries
        # it, and on a hit it names the existing index entry.
        model_id = "llama-server-slot"

        if not real_hit:
            entry = CacheEntry(
                cache_id=cache_id,
                prompt=tokenize_text,
                token_ids=new_tokens,
                embedding=new_embedding,
                n_past=backend_result.get("prefill_tokens", len(new_tokens)),
                model_id=model_id,
                bytes_on_disk=0,
            )
            self.index.insert(entry)

        # Best-effort embedding similarity vs the index. The engine
        # routes by token-prefix match, not by embedding similarity, so
        # this number is purely informational — but callers (and the
        # /v1/metrics/live panel) expect it and the index lookup is
        # in-memory numpy, so it's cheap.
        match = self.index.search(new_embedding)
        embedding_similarity = match.similarity if match is not None else 0.0
        if real_hit and match is not None and match.cache_id != cache_id:
            # On a hit we want to report the similarity to the entry
            # we actually matched against (the slot's cached prefix),
            # not to ourselves — they differ when the prompt is the
            # longest known and we just inserted a near-duplicate.
            embedding_similarity = match.similarity

        soft_hit = 1 <= reuse_depth < MIN_REUSE_FOR_HIT
        elapsed = time.perf_counter() - started

        return RecyclingResult(
            hit=real_hit,
            soft_hit=soft_hit,
            cache_id=cache_id,
            reuse_depth=reuse_depth,
            embedding_similarity=embedding_similarity,
            prefilled_tokens=backend_result["prefill_tokens"],
            skipped_tokens=reuse_depth,
            generated_tokens=backend_result["tokens_generated"],
            ttft_ms=backend_result["ttft_ms"],
            total_ms=backend_result["total_ms"],
            prefill_tokens_per_sec=backend_result["prefill_tokens_per_sec"],
            generated_tokens_per_sec=(
                backend_result["tokens_generated"] / (elapsed or 1e-9)
            ),
            output_text=backend_result.get("text", ""),
            reasoning=backend_result.get("reasoning", ""),
            slot_id=slot_id,
            spilled=spilled,
            tool_calls=backend_result.get("tool_calls"),
            finish_reason=backend_result.get("finish_reason"),
        )

    def _serve_with_backend(
        self,
        *,
        prompt: str,
        messages: list[dict[str, str]] | None,
        backend_call,
        tokenize_text: str,
        max_tokens: int,
        temperature: float,
        max_slots: int,
        session_id: str | None,
        max_context_tokens: int | None,
    ) -> RecyclingResult:
        """Shared slot-routing + bookkeeping for serve_native / serve_chat.

        ``backend_call(slot_id, cache_prompt)`` is invoked to dispatch
        the actual completion. The result dict shape is the same as
        :meth:`LlamaServerBackend.complete_native`.
        """
        started = time.perf_counter()
        new_tokens, new_embedding, slot_id, reuse_depth, spilled, cache_prompt, cache_id = (
            self._route_request(
                tokenize_text=tokenize_text,
                max_slots=max_slots,
                session_id=session_id,
                max_context_tokens=max_context_tokens,
            )
        )

        # Send the request via the supplied backend_call.
        backend_result = backend_call(slot_id, cache_prompt)

        return self._finalize_request(
            tokenize_text=tokenize_text,
            new_tokens=new_tokens,
            new_embedding=new_embedding,
            slot_id=slot_id,
            reuse_depth=reuse_depth,
            spilled=spilled,
            session_id=session_id,
            backend_result=backend_result,
            started=started,
            cache_id=cache_id,
        )

    async def serve_chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        max_slots: int = 4,
        session_id: str | None = None,
        max_context_tokens: int | None = None,
        chat_template_kwargs: dict | None = None,
        reasoning_effort: str | None = None,
        tools: list | None = None,
        tool_choice: str | dict | None = None,
        sampling: dict | None = None,
    ) -> AsyncIterator[dict]:
        """Streaming variant of :meth:`serve_chat`.

        Yields dict items (mirrors the backend contract):
          ``{"kind": "delta", "delta": str}`` — one per text chunk.
          ``{"kind": "tool_calls", "tool_calls": [...]}`` — OpenAI
          delta-format tool-call chunks, forwarded verbatim.
          ``{"kind": "finish", "finish_reason": str}`` — the stream's
          finish reason, if the backend reported one.
          ``{"kind": "result", "result": RecyclingResult}`` — the final
          result, whose :attr:`RecyclingResult.tool_calls` /
          ``finish_reason`` carry the assembled tool call(s).

        Slot routing, spilling, session pinning and the index behave
        exactly like :meth:`serve_chat`; only the network call streams.
        """
        # Index the joined user content for prefix matching (mirrors
        # serve_chat).
        routing_text = _last_routing_text(messages, reasoning_effort)

        started = time.perf_counter()
        # Routing tokenizes + embeds over blocking HTTP and may spill a
        # slot; run it off the event loop so a streaming request does not
        # stall every other request and health check (the sync serve_chat
        # path is offloaded the same way by the API layer).
        new_tokens, new_embedding, slot_id, reuse_depth, spilled, cache_prompt, cache_id = (
            await asyncio.to_thread(
                self._route_request,
                tokenize_text=routing_text,
                max_slots=max_slots,
                session_id=session_id,
                max_context_tokens=max_context_tokens,
            )
        )

        backend_result: dict[str, Any] | None = None
        async for item in self.backend.complete_openai_stream(
            messages,
            slot_id=slot_id,
            max_tokens=max_tokens,
            temperature=temperature,
            cache_prompt=cache_prompt,
            chat_template_kwargs=chat_template_kwargs,
            reasoning_effort=reasoning_effort,
            tools=tools,
            tool_choice=tool_choice,
            sampling=sampling,
        ):
            if item.get("kind") == "delta":
                yield {"kind": "delta", "delta": item.get("delta", "")}
            elif item.get("kind") == "reasoning":
                yield {"kind": "reasoning", "delta": item.get("delta", "")}
            elif item.get("kind") == "tool_calls":
                yield {"kind": "tool_calls", "tool_calls": item.get("tool_calls", [])}
            elif item.get("kind") == "finish":
                yield {"kind": "finish", "finish_reason": item.get("finish_reason")}
            else:
                backend_result = item.get("result", {})

        if backend_result is None:
            backend_result = {
                "text": "",
                "tool_calls": None,
                "finish_reason": None,
                "tokens_generated": 0,
                "prefill_tokens": len(new_tokens),
                "ttft_ms": 0.0,
                "total_ms": (time.perf_counter() - started) * 1000.0,
                "prefill_tokens_per_sec": 0.0,
            }

        # Finalizing writes the index (SQLite) and may query llama-server
        # for a free pool slot — both blocking, so keep them off the loop.
        result = await asyncio.to_thread(
            self._finalize_request,
            tokenize_text=routing_text,
            new_tokens=new_tokens,
            new_embedding=new_embedding,
            slot_id=slot_id,
            reuse_depth=reuse_depth,
            spilled=spilled,
            session_id=session_id,
            backend_result=backend_result,
            started=started,
            cache_id=cache_id,
        )
        yield {"kind": "result", "result": result}

    def _init_slot_state(self) -> None:
        """Initialise per-slot state maps. Idempotent; safe to call
        multiple times even if the engine was constructed before the
        reservation fields were added."""
        # slot_id -> last prompt's token_ids
        if not hasattr(self, "_slot_history"):
            self._slot_history: dict[int, list[int]] = {}
        if not hasattr(self, "_slot_context_tokens"):
            self._slot_context_tokens: dict[int, int] = {}
        if not hasattr(self, "_slot_use_count"):
            self._slot_use_count: dict[int, int] = {}
        if not hasattr(self, "_session_to_slot"):
            self._session_to_slot: dict[str, int] = {}
        if not hasattr(self, "_session_order"):
            self._session_order: list[str] = []
        if not hasattr(self, "_max_context_tokens"):
            # Default context budget per slot. Set to leave room for
            # generation: e.g. llama-server -c 8192 → 6144 leaves 2048
            # for output. Bump higher for very long system prompts.
            self._max_context_tokens: int = 6144
        if not hasattr(self, "_spill_count"):
            self._spill_count: int = 0
        # Persistent slot pool owns the "reserved slot" concept; the
        # engine no longer maintains its own _reserved_slots dict.

    def _spill_slot(self, slot_id: int) -> None:
        """Clear a slot's KV state and reset its tracking.

        Called when a slot's context has grown beyond the configured
        budget. Subsequent requests on this session will pick a fresh
        slot via prefix matching (or round-robin if no match).
        Pool-owned slots are NOT spilled — they persist for hot prefixes.
        """
        if self.persistent_pool.owns_slot(slot_id):
            logger.debug("slot %d is pool-owned; skipping spill", slot_id)
            return
        try:
            self.backend.clear_slot(slot_id)
        except Exception as e:
            logger.warning("clear_slot(%d) failed: %s", slot_id, e)
        self._slot_history.pop(slot_id, None)
        self._slot_context_tokens.pop(slot_id, None)
        self._slot_use_count[slot_id] = 0
        self._spill_count += 1
        logger.info(
            "spilled slot %d (total spills=%d)", slot_id, self._spill_count,
        )

    def _pick_slot(
        self,
        new_tokens: list[int],
        max_slots: int,
        *,
        session_id: str | None = None,
    ) -> tuple[int, int]:
        """Pick the slot whose current prompt shares the longest prefix
        with `new_tokens`. Returns (slot_id, reuse_depth).

        Reserved slots (see :meth:`_reserve_slot_if_hot`) take priority:
        if a reserved slot's prefix matches, route to it. If session_id
        is provided and we have a slot pinned for it, route to that
        slot directly. Otherwise pick the slot with the longest matching
        prefix; if no slot has any prefix match, round-robin to the next
        free slot (preferring non-reserved slots to avoid overwriting hot
        prefixes).

        Reuse depth is the number of leading tokens that match — this is
        what llama-server will prefill-skip when we send the new
        prompt to that slot.
        """
        # Session pinning takes priority.
        if session_id is not None and session_id in self._session_to_slot:
            pinned = self._session_to_slot[session_id]
            depth = compute_reuse_depth(
                self._slot_history.get(pinned, []), new_tokens
            )
            return pinned, depth

        # Check persistent-pool slots first (their prefix is hot, reuse it).
        # The pool stores the prefix tokens alongside the slot, so we
        # route by prefix-match without a second lookup.
        for persistent_slot in self.persistent_pool._slots.values():
            if persistent_slot.prefix_tokens is None:
                continue
            depth = compute_reuse_depth(
                persistent_slot.prefix_tokens, new_tokens
            )
            if depth >= MIN_REUSE_FOR_HIT:
                return persistent_slot.slot_id, depth

        # Find the slot with the longest matching prefix.
        best_slot = 0
        best_depth = 0

        for slot_id, prev_tokens in self._slot_history.items():
            # Skip pool-owned slots unless their prefix actually matches.
            if self.persistent_pool.owns_slot(slot_id):
                continue
            depth = compute_reuse_depth(prev_tokens, new_tokens)
            if depth > best_depth:
                best_depth = depth
                best_slot = slot_id

        # If no slot has any prefix match, round-robin among non-pool slots.
        if best_depth == 0:
            used = set(self._slot_history.keys())
            pool_owned = self.persistent_pool.owned_slot_ids
            for slot_id in range(max_slots):
                if slot_id not in used and slot_id not in pool_owned:
                    return slot_id, 0
            # All non-pool slots used; pick the LRU non-pool slot.
            non_reserved = {
                k: v for k, v in self._slot_use_count.items()
                if k not in pool_owned
            }
            if non_reserved:
                return min(non_reserved.items(), key=lambda kv: kv[1])[0], 0
            # Fallback: pick any slot.
            return 0, 0

        return best_slot, best_depth

    def _maybe_reserve_slot(self, slot_id: int, prefix_tokens: list[int], cache_id: str) -> None:
        """Promote ``slot_id`` to a reserved (hot-prefix) slot via the
        ``PersistentSlotPool``.

        The pool enforces its own threshold (default 50 hits across all
        sessions) and capacity (default 4 slots); when a slot crosses
        the threshold, the pool marks it as "owned by the pool" so it
        survives normal spills, and tracks use counts so LRU eviction
        can run when the pool is full. Pool ownership is purely a
        routing hint — no disk save/restore is involved.

        Hot prefixes that are also in the pool short-circuit
        :meth:`_pick_slot` to the dedicated slot, skipping the prefix
        scan and the embedding-based routing entirely.
        """
        try:
            self.persistent_pool.acquire(
                cache_id,
                filename=cache_id,
                n_past=len(prefix_tokens),
                prefix_tokens=prefix_tokens,
            )
        except Exception as e:
            logger.debug("persistent_pool.acquire failed: %s", e)

    def _is_slot_reserved(self, slot_id: int) -> bool:
        # A slot is "reserved" iff it is owned by the pool.
        return self.persistent_pool.owns_slot(slot_id)

    def slot_state(self) -> dict:
        """Return current per-slot state for observability.

        Exposed via the daemon's `GET /v1/cache/state` endpoint. The web
        UI uses this to render a live slot-status panel.
        """
        self._init_slot_state()
        pool_owned = self.persistent_pool.owned_slot_ids
        return {
            "slots": [
                {
                    "slot_id": sid,
                    "context_tokens": self._slot_context_tokens.get(sid, 0),
                    "last_prompt_tokens": len(self._slot_history.get(sid, [])),
                    "use_count": self._slot_use_count.get(sid, 0),
                    "reserved": sid in pool_owned,
                }
                for sid in sorted(
                    set(self._slot_history)
                    | set(self._slot_context_tokens)
                    | set(self._slot_use_count)
                )
            ],
            "sessions": {
                sid: slot
                for sid, slot in sorted(self._session_to_slot.items())
            },
            "spill_count": self._spill_count,
            "max_context_tokens": self._max_context_tokens,
            "reserved_slots": sorted(pool_owned),
        }

    def reset_state(self) -> None:
        """Clear all slot routing state and the session map.

        Does NOT clear llama-server's actual slot KV — only the
        engine's bookkeeping. Useful for tests and for explicit
        `POST /v1/cache/reset` calls.
        """
        self._init_slot_state()
        self._slot_history.clear()
        self._slot_context_tokens.clear()
        self._slot_use_count.clear()
        self._session_to_slot.clear()
        self._session_order.clear()
        # PersistentSlotPool owns its own use-counts / slot registry;
        # clearing it here drops the engine's view of "reserved slots"
        # without touching the on-disk cache files.
        self.persistent_pool._slots.clear()
        self.persistent_pool._use_counts.clear()
        self._spill_count = 0
