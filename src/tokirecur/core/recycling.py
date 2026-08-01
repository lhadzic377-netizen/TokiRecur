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

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol

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
    output_tokens: list[int] = field(default_factory=list)
    slot_id: int = 0  # which slot served the request
    spilled: bool = False  # True if this slot was cleared due to context budget


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
    for x, y in zip(a, b):
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
      - tree_cache: deduplicates KV blocks across prompts with shared
        prefixes; provides a fast tree-hit check before embedding.
      - persistent_pool: keeps the top-N most-used prefixes in dedicated
        slots to skip the save/load cycle.
    """

    def __init__(
        self,
        backend: LlamaBackend,
        index: "Index",  # noqa: F821 - forward-declared
        *,
        similarity_threshold: float = 0.85,
        tree_cache: Any | None = None,
        persistent_pool: Any | None = None,
    ) -> None:
        self.backend = backend
        self.index = index
        self.similarity_threshold = similarity_threshold
        self.tree_cache = tree_cache
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
    ) -> RecyclingResult:
        """Run the recycling algorithm with OpenAI-format messages.

        Like :meth:`serve_native`, but routes OpenAI ``messages`` to
        llama-server's ``/v1/chat/completions`` endpoint so the model
        gets the right chat template. This is the right method for
        chat-tuned models (Qwen3.5, Llama, Mistral, etc.).

        Args:
            messages: OpenAI-format list of ``{"role": ..., "content": ...}``
                dicts. The concatenated text is what gets indexed for
                prefix-reuse matching.
            max_tokens, temperature, max_slots, session_id,
                max_context_tokens: see :meth:`serve_native`.
            chat_template_kwargs: pass-through for llama-server's chat
                template (e.g. ``{"enable_thinking": False}`` for Qwen3.5).
        """
        # Index the joined user content for prefix matching. We use the
        # last user message's text as the routing key — it's what future
        # turns are most likely to share with.
        routing_text = ""
        for m in reversed(messages):
            if m.get("role") == "user" and m.get("content"):
                routing_text = m["content"]
                break
        if not routing_text and messages:
            routing_text = messages[-1].get("content", "")

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
    ) -> tuple[list[int], list[float], int, int, bool, bool]:
        """Shared slot-routing: decide which llama-server slot to use
        and whether to force a fresh prefill.

        Returns ``(new_tokens, new_embedding, slot_id, reuse_depth,
        spilled, cache_prompt)``.
        """
        # Initialise slot state lazily.
        self._init_slot_state()

        new_tokens = self.backend.tokenize(tokenize_text)
        new_embedding = self.backend.embed(tokenize_text)

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

        # Pick the slot that has the longest matching prefix with the new
        # prompt. llama-server will then reuse the prefix automatically.
        slot_id, reuse_depth = self._pick_slot(
            new_tokens, max_slots, session_id=session_id,
        )
        return new_tokens, new_embedding, slot_id, reuse_depth, spilled, cache_prompt

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

        # Hot-prefix detection: if this prefix has been seen N+ times,
        # reserve the slot so it never spills.
        self._maybe_reserve_slot(slot_id, new_tokens)

        # Index for future routing (only on cache miss; on hit, the
        # entry already exists in the index).
        model_id = "llama-server-slot"
        cache_id = make_cache_id(tokenize_text, model_id)
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

        real_hit = reuse_depth >= MIN_REUSE_FOR_HIT
        soft_hit = 1 <= reuse_depth < MIN_REUSE_FOR_HIT
        elapsed = time.perf_counter() - started

        return RecyclingResult(
            hit=real_hit,
            soft_hit=soft_hit,
            cache_id=cache_id,
            reuse_depth=reuse_depth,
            embedding_similarity=0.0,
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
            slot_id=slot_id,
            spilled=spilled,
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
        new_tokens, new_embedding, slot_id, reuse_depth, spilled, cache_prompt = (
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
    ) -> AsyncIterator[tuple[str | None, RecyclingResult | None]]:
        """Streaming variant of :meth:`serve_chat`.

        Yields ``(delta, None)`` per text chunk while tokens stream in,
        then finally ``(None, result)`` with the completed
        :class:`RecyclingResult`. Slot routing, spilling, session
        pinning and the index behave exactly like :meth:`serve_chat`;
        only the network call streams.
        """
        # Index the joined user content for prefix matching (mirrors
        # serve_chat).
        routing_text = ""
        for m in reversed(messages):
            if m.get("role") == "user" and m.get("content"):
                routing_text = m["content"]
                break
        if not routing_text and messages:
            routing_text = messages[-1].get("content", "")

        started = time.perf_counter()
        new_tokens, new_embedding, slot_id, reuse_depth, spilled, cache_prompt = (
            self._route_request(
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
        ):
            if item.get("kind") == "delta":
                yield item.get("delta", ""), None
            else:
                backend_result = item.get("result", {})

        if backend_result is None:
            backend_result = {
                "text": "",
                "tokens_generated": 0,
                "prefill_tokens": len(new_tokens),
                "ttft_ms": 0.0,
                "total_ms": (time.perf_counter() - started) * 1000.0,
                "prefill_tokens_per_sec": 0.0,
            }

        yield None, self._finalize_request(
            tokenize_text=routing_text,
            new_tokens=new_tokens,
            new_embedding=new_embedding,
            slot_id=slot_id,
            reuse_depth=reuse_depth,
            spilled=spilled,
            session_id=session_id,
            backend_result=backend_result,
            started=started,
        )

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
        # Persistent slot pool: reserved slots never spill.
        if not hasattr(self, "_reserved_slots"):
            self._reserved_slots: dict[int, list[int]] = {}
        if not hasattr(self, "_prefix_use_count"):
            self._prefix_use_count: dict[int, int] = {}
        if not hasattr(self, "_reservation_threshold"):
            self._reservation_threshold: int = 10
        if not hasattr(self, "_max_reserved_slots"):
            self._max_reserved_slots: int = 2

    def _spill_slot(self, slot_id: int) -> None:
        """Clear a slot's KV state and reset its tracking.

        Called when a slot's context has grown beyond the configured
        budget. Subsequent requests on this session will pick a fresh
        slot via prefix matching (or round-robin if no match).
        Reserved slots are NOT spilled — they persist for hot prefixes.
        """
        if slot_id in self._reserved_slots:
            logger.debug("slot %d is reserved; skipping spill", slot_id)
            return
        try:
            self.backend.clear_slot(slot_id)
        except Exception as e:
            logger.warning("clear_slot(%d) failed: %s", slot_id, e)
        self._slot_history.pop(slot_id, None)
        self._slot_context_tokens.pop(slot_id, None)
        self._slot_use_count[slot_id] = 0
        self._prefix_use_count.pop(slot_id, None)
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

        # Check reserved slots first (their prefix is hot, reuse it).
        for slot_id, reserved_prefix in self._reserved_slots.items():
            depth = compute_reuse_depth(reserved_prefix, new_tokens)
            if depth >= MIN_REUSE_FOR_HIT:
                return slot_id, depth

        # Find the slot with the longest matching prefix.
        best_slot = 0
        best_depth = 0

        for slot_id, prev_tokens in self._slot_history.items():
            # Skip reserved slots unless their prefix actually matches.
            if slot_id in self._reserved_slots:
                continue
            depth = compute_reuse_depth(prev_tokens, new_tokens)
            if depth > best_depth:
                best_depth = depth
                best_slot = slot_id

        # If no slot has any prefix match, round-robin among NON-reserved slots.
        if best_depth == 0:
            used = set(self._slot_history.keys())
            for slot_id in range(max_slots):
                if slot_id not in used and slot_id not in self._reserved_slots:
                    return slot_id, 0
            # All non-reserved slots used; pick the LRU non-reserved slot.
            non_reserved = {
                k: v for k, v in self._slot_use_count.items()
                if k not in self._reserved_slots
            }
            if non_reserved:
                return min(non_reserved.items(), key=lambda kv: kv[1])[0], 0
            # Fallback: pick any slot.
            return 0, 0

        return best_slot, best_depth

    def _maybe_reserve_slot(self, slot_id: int, prefix_tokens: list[int]) -> None:
        """If the prefix on `slot_id` has been hit enough times, reserve
        the slot. Reserved slots never spill and are prioritised for
        matching prefixes."""
        self._init_slot_state()
        # Track use count for this slot's prefix.
        current_count = self._prefix_use_count.get(slot_id, 0) + 1
        self._prefix_use_count[slot_id] = current_count

        # Already reserved — nothing to do.
        if slot_id in self._reserved_slots:
            return

        # Reservation only kicks in if the slot's prefix is hot AND we
        # have spare reservation slots.
        if current_count < self._reservation_threshold:
            return
        if len(self._reserved_slots) >= self._max_reserved_slots:
            # Already at capacity. Don't reserve this one.
            return

        # Reserve.
        self._reserved_slots[slot_id] = list(prefix_tokens)
        logger.info(
            "reserved slot %d after %d hits (max_reserved=%d)",
            slot_id, current_count, self._max_reserved_slots,
        )

    def _is_slot_reserved(self, slot_id: int) -> bool:
        return slot_id in self._reserved_slots

    def slot_state(self) -> dict:
        """Return current per-slot state for observability.

        Exposed via the daemon's `GET /v1/cache/state` endpoint. The web
        UI uses this to render a live slot-status panel.
        """
        self._init_slot_state()
        return {
            "slots": [
                {
                    "slot_id": sid,
                    "context_tokens": self._slot_context_tokens.get(sid, 0),
                    "last_prompt_tokens": len(self._slot_history.get(sid, [])),
                    "use_count": self._slot_use_count.get(sid, 0),
                    "reserved": sid in self._reserved_slots,
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
            "reserved_slots": sorted(self._reserved_slots.keys()),
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
        self._reserved_slots.clear()
        self._prefix_use_count.clear()
        self._spill_count = 0
