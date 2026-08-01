"""Persistent slot pool for hot prefixes.

The default recycling engine calls `?action=save` and `?action=restore`
on every request, which has overhead. For prefixes that are accessed
frequently (e.g. an agent's system prompt), this overhead exceeds the
prefill savings.

The persistent slot pool solves this: the top-N most-frequently-used
prefixes get dedicated slots that never clear. When a new request
matches one of these prefixes (by token-level prefix check), we skip the
save+restore cycle entirely — the slot already has the KV state in
memory.

We use the upstream llama-server endpoints to drive the pool:
  - `?action=save`    — explicit save (used for cache_dir persistence)
  - `?action=restore` — load a previously saved file into a slot
  - `/slots/:id` POST action=get — query current n_past for the slot

The pool maintains a priority list of prefixes and their corresponding
slot IDs. When a slot is requested and is not in use, it can be reused
without restore.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class PersistentSlot:
    """A slot that's been pre-loaded with a hot prefix's KV state."""

    slot_id: int
    cache_id: str  # filename used in --slot-save-path
    n_past: int
    last_used_at: float = field(default_factory=time.time)
    use_count: int = 0


class PersistentSlotPool:
    """LRU pool of persistent slots pre-loaded with hot prefixes.

    Capacity: `max_slots`. The pool allocates slots on first encounter
    with a new prefix and evicts the least-recently-used slot when full.
    """

    def __init__(
        self,
        *,
        max_slots: int = 4,
        llama_base_url: str = "http://127.0.0.1:8088",
        min_reuse_for_pooling: int = 50,
    ) -> None:
        self.max_slots = max_slots
        self.llama_base_url = llama_base_url.rstrip("/")
        self.min_reuse_for_pooling = min_reuse_for_pooling
        # map: cache_id -> PersistentSlot
        self._slots: OrderedDict[str, PersistentSlot] = OrderedDict()
        # map: cache_id -> use_count
        self._use_counts: dict[str, int] = {}

    def _evict_lru(self) -> None:
        """Evict the least-recently-used slot if over capacity."""
        while len(self._slots) >= self.max_slots:
            evicted_cache_id, evicted_slot = self._slots.popitem(last=False)
            logger.info(
                "evicting persistent slot cache_id=%s slot=%d (use_count=%d)",
                evicted_cache_id, evicted_slot.slot_id, evicted_slot.use_count,
            )

    def _allocate_slot(self) -> Optional[int]:
        """Find a free slot. llama-server exposes slot IDs via /slots.
        We use the first slot that's not currently processing."""
        try:
            r = httpx.get(f"{self.llama_base_url}/slots", timeout=2.0)
            r.raise_for_status()
            slots = r.json()
            for s in slots:
                if not s.get("is_processing", True):
                    return int(s["id"])
        except Exception as e:
            logger.warning("could not query slots: %s", e)
        return None

    def _save_to_slot(self, slot_id: int, cache_id: str, filename: str) -> dict:
        """Use ?action=save to persist the slot's KV to disk."""
        r = httpx.post(
            f"{self.llama_base_url}/slots/{slot_id}",
            json=[{"slot_id": slot_id, "action": "save", "filename": filename}],
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json()[0]

    def _restore_to_slot(self, slot_id: int, cache_id: str, filename: str) -> dict:
        """Use ?action=restore to load a previously saved KV state."""
        r = httpx.post(
            f"{self.llama_base_url}/slots/{slot_id}",
            json=[{"slot_id": slot_id, "action": "restore", "filename": filename}],
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json()[0]

    def acquire(self, cache_id: str, filename: str, n_past: int) -> Optional[PersistentSlot]:
        """Acquire a slot pre-loaded with the KV for `cache_id`.

        Returns the slot if poolable, None if the pool is full and this
        prefix doesn't deserve a slot (below `min_reuse_for_pooling`).
        """
        # Already pooled.
        if cache_id in self._slots:
            slot = self._slots[cache_id]
            slot.last_used_at = time.time()
            slot.use_count += 1
            self._slots.move_to_end(cache_id)
            self._use_counts[cache_id] = slot.use_count
            return slot

        # Not pooled yet. Check whether this prefix has earned its keep.
        if self._use_counts.get(cache_id, 0) < self.min_reuse_for_pooling:
            self._use_counts[cache_id] = self._use_counts.get(cache_id, 0) + 1
            return None

        # Evict if needed, then allocate.
        self._evict_lru()
        slot_id = self._allocate_slot()
        if slot_id is None:
            logger.warning("no free slot available for %s", cache_id)
            return None
        try:
            self._restore_to_slot(slot_id, cache_id, filename)
        except Exception as e:
            logger.warning("restore to slot %d failed: %s", slot_id, e)
            return None
        slot = PersistentSlot(
            slot_id=slot_id,
            cache_id=cache_id,
            n_past=n_past,
            use_count=self._use_counts[cache_id],
        )
        self._slots[cache_id] = slot
        logger.info(
            "persistent slot pool: cache_id=%s slot=%d n_past=%d",
            cache_id, slot_id, n_past,
        )
        return slot

    def stats(self) -> dict:
        return {
            "pooled_slots": len(self._slots),
            "max_slots": self.max_slots,
            "track_counts": dict(self._use_counts),
        }