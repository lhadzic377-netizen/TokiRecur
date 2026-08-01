"""Token-bucket rate limiter for the daemon.

In-memory implementation. Keyed by an identifier (session_id or
client IP). Suitable for a single-host deployment; for multi-host,
swap in a Redis-backed implementation.

Defaults: 60 requests/minute burst, 10 req/sec sustained. Configurable
via env vars TOKI_RATE_LIMIT_BURST and TOKI_RATE_LIMIT_RATE.
"""

from __future__ import annotations

import os
import threading
import time


class TokenBucket:
    """A single token bucket."""

    def __init__(self, capacity: float, refill_rate: float) -> None:
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.tokens = capacity
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def try_consume(self, n: float = 1.0) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(
                self.capacity, self.tokens + elapsed * self.refill_rate
            )
            self.last_refill = now
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False


class RateLimiter:
    """Maps identifiers (session_id or IP) to TokenBucket instances."""

    def __init__(
        self,
        *,
        burst: float | None = None,
        refill_rate: float | None = None,
        max_keys: int = 4096,
    ) -> None:
        self.burst = burst or float(os.environ.get("TOKI_RATE_LIMIT_BURST", "60"))
        self.refill_rate = refill_rate or float(
            os.environ.get("TOKI_RATE_LIMIT_RATE", "1.0")  # 1 token / sec = 60/min
        )
        self.max_keys = max_keys
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> bool:
        """Return True if the request is allowed, False if rate-limited."""
        with self._lock:
            if key not in self._buckets:
                # Evict oldest if over capacity (simple LRU: pop arbitrary).
                if len(self._buckets) >= self.max_keys:
                    self._buckets.pop(next(iter(self._buckets)))
                self._buckets[key] = TokenBucket(self.burst, self.refill_rate)
            bucket = self._buckets[key]
        return bucket.try_consume(1.0)
