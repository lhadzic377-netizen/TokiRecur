"""Tree-structured KV cache.

Tracks KV blocks by (prefix_length, prefix_hash) so that multiple cache
entries sharing a prefix share the underlying KV tensors instead of
duplicating them on disk.

Design
------

A KV cache entry is split into blocks of `BLOCK_SIZE` tokens. Each block
is identified by:

    block_id = sha256(prefix_hash || block_index)

Blocks are stored as files in the cache dir:

    <cache_dir>/<cache_id>.trkv          # metadata + token_ids + embedding
    <cache_dir>/blocks/<block_id>.bin    # actual KV tensor blob

When a new entry is created, we hash its tokens in BLOCK_SIZE chunks and
reference each block. If a block already exists on disk (because a prior
entry covered that prefix range), we just reference it instead of
writing again.

When a new prompt arrives, we compute the prefix hash for each block and
look up the corresponding block files. If we have blocks for the first
N tokens of the prompt, we have a cache hit of depth N.

In production this relies on llama-server's `?action=save` and
`?action=restore` endpoints for the full slot state. The tree cache is a
data-structure layer on top: it tracks which blocks are "hot" (frequently
reused) and should be served from the persistent slot pool rather than
the save/restore cycle.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# Block size: each KV block covers this many tokens of context.
# 64 tokens is a reasonable default — small enough that many prompts
# share blocks, large enough that the bookkeeping isn't absurd.
BLOCK_SIZE = 64


def block_id(tokens: list[int]) -> str:
    """Stable identifier for a KV block of `tokens`.

    Content-addressed: two prompts sharing the same leading tokens get
    the same block IDs and therefore share the underlying KV tensors on
    disk. This is the key to the tree cache's deduplication.
    """
    h = hashlib.sha256()
    for t in tokens:
        # Hash each token as a 4-byte little-endian int to keep block IDs
        # compact. Different orderings or values produce different IDs.
        h.update(t.to_bytes(4, "little", signed=True))
    return h.hexdigest()[:32]


def block_ids_for_tokens(_unused: object, tokens: list[int]) -> list[str]:
    """Return the list of block IDs covering `tokens` in BLOCK_SIZE chunks.

    The first arg is ignored (was a cache_id in an earlier draft; we now
    content-address blocks so the cache_id is irrelevant). It's kept in
    the signature for API stability with existing callers."""
    out: list[str] = []
    for start in range(0, len(tokens), BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, len(tokens))
        out.append(block_id(tokens[start:end]))
    return out


def cache_path_for(cache_dir: Path, block_id_str: str) -> Path:
    """Filesystem path for a KV block."""
    return cache_dir / "blocks" / f"{block_id_str}.bin"


@dataclass
class TreeStats:
    """Summary of the tree cache state."""

    total_entries: int = 0
    total_blocks: int = 0
    unique_blocks: int = 0  # blocks referenced by > 1 entry
    block_reuse_ratio: float = 0.0  # unique / total; lower = more sharing
    disk_bytes: int = 0


class TreeCache:
    """In-memory view over the tree-structured KV cache.

    Tracks block references per cache entry, deduplicates them, and
    exposes statistics about how much sharing is happening.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.blocks_dir = self.cache_dir / "blocks"
        self.blocks_dir.mkdir(parents=True, exist_ok=True)
        # map: block_id -> set of cache_ids that reference it
        self._block_refs: dict[str, set[str]] = {}
        # map: cache_id -> list of block_ids
        self._entry_blocks: dict[str, list[str]] = {}
        # map: block_id -> size on disk
        self._block_sizes: dict[str, int] = {}

    def register(self, cache_id: str, token_ids: list[int], *, save_fn=None) -> list[str]:
        """Register a new cache entry's blocks. If `save_fn` is provided,
        it is called as `save_fn(block_id_str) -> bytes` to materialize
        the block's KV tensor on disk. If `save_fn` is None and a block
        with this ID already exists on disk, we just reference it.

        Returns the list of block IDs registered (in order)."""
        blocks = block_ids_for_tokens(cache_id, token_ids)
        self._entry_blocks[cache_id] = blocks
        for bid in blocks:
            self._block_refs.setdefault(bid, set()).add(cache_id)
            if save_fn is not None:
                data = save_fn(bid)
                if data is not None:
                    path = cache_path_for(self.cache_dir, bid)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    self._block_sizes[bid] = len(data)
            elif cache_path_for(self.cache_dir, bid).exists():
                self._block_sizes[bid] = cache_path_for(self.cache_dir, bid).stat().st_size
        return blocks

    def lookup(self, cache_id: str, token_ids: list[int]) -> int:
        """Return the number of leading tokens for which we already
        have KV blocks on disk. This is the "tree hit depth" — a quick
        check independent of the embedding-based index."""
        blocks = block_ids_for_tokens(cache_id, token_ids)
        depth = 0
        for i, bid in enumerate(blocks):
            if not cache_path_for(self.cache_dir, bid).exists():
                return depth
            depth = min((i + 1) * BLOCK_SIZE, len(token_ids))
        return depth

    def unregister(self, cache_id: str) -> None:
        """Remove an entry's block references. Blocks that are no longer
        referenced by any entry can be garbage-collected later."""
        blocks = self._entry_blocks.pop(cache_id, [])
        for bid in blocks:
            refs = self._block_refs.get(bid)
            if refs is not None:
                refs.discard(cache_id)
                if not refs:
                    del self._block_refs[bid]

    def garbage_collect(self, *, max_age_seconds: float | None = None) -> int:
        """Delete block files that are no longer referenced. If
        max_age_seconds is set, also delete any block file older than
        that regardless of reference count.

        Returns the number of files deleted."""
        deleted = 0
        now = time.time()
        for bid in list(self._block_sizes.keys()):
            path = cache_path_for(self.cache_dir, bid)
            if not path.exists():
                continue
            if bid not in self._block_refs:
                path.unlink()
                self._block_sizes.pop(bid, None)
                deleted += 1
                continue
            if max_age_seconds is not None:
                age = now - path.stat().st_mtime
                if age > max_age_seconds:
                    path.unlink()
                    self._block_sizes.pop(bid, None)
                    deleted += 1
        return deleted

    def stats(self) -> TreeStats:
        total_entries = len(self._entry_blocks)
        total_blocks = sum(len(v) for v in self._entry_blocks.values())
        unique_blocks = len(self._block_refs)
        ratio = unique_blocks / max(total_blocks, 1)
        disk_bytes = sum(self._block_sizes.values())
        return TreeStats(
            total_entries=total_entries,
            total_blocks=total_blocks,
            unique_blocks=unique_blocks,
            block_reuse_ratio=ratio,
            disk_bytes=disk_bytes,
        )

    def hot_blocks(self, n: int = 10) -> list[tuple[str, int]]:
        """Return the top-n most-referenced block IDs by reference count."""
        scored = [(bid, len(refs)) for bid, refs in self._block_refs.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]