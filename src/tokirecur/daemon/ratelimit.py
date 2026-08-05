"""Token-bucket rate limiter for the daemon.

In-memory implementation. Keyed by an identifier (session_id or
client IP). Suitable for a single-host deployment; for multi-host,
swap in a Redis-backed implementation.

Defaults: 60 requests/minute burst, 10 req/sec sustained. Configurable
via env vars TOKI_RATE_LIMIT_BURST and TOKI_RATE_LIMIT_RATE. Malformed
env values fall back to the default with a warning instead of crashing
the daemon at startup.
"""

from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

_DEFAULT_BURST = 60
_DEFAULT_RATE = 1.0


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "ignoring invalid %s=%r (not a float); using default %s",
            name, raw, default,
        )
        return default


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
        # `is not None` (not `or ...`) so that burst=0 / refill_rate=0
        # are honoured as "disable burst / no refill" instead of being
        # silently overridden by the env-var default.
        if burst is None:
            burst = _env_float("TOKI_RATE_LIMIT_BURST", _DEFAULT_BURST)
        if refill_rate is None:
            refill_rate = _env_float("TOKI_RATE_LIMIT_RATE", _DEFAULT_RATE)
        self.burst = burst
        self.refill_rate = refill_rate
        self.max_keys = max_keys
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> bool:
        """Return True if the request is allowed, False if rate-limited."""
        with self._lock:
            if key not in self._buckets:
                # Insertion-order dict: the first-inserted key is the
                # one we saw longest ago. New clients are appended at
                # the end; evicting the front is therefore least-
                # recently-seen, which is the right eviction for a
                # rate-limiter memory bound (an active client keeps
                # reusing its bucket via the `bucket = ...` line below,
                # so it never gets re-inserted and never moves).
                if len(self._buckets) >= self.max_keys:
                    self._buckets.pop(next(iter(self._buckets)))
                self._buckets[key] = TokenBucket(self.burst, self.refill_rate)
            bucket = self._buckets[key]
        return bucket.try_consume(1.0)
