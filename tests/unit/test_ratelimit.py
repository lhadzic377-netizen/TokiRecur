"""Unit tests for the token-bucket rate limiter."""

import time

from tokirecur.daemon.ratelimit import RateLimiter


class TestRateLimiter:
    def test_burst_then_block(self):
        rl = RateLimiter(burst=3, refill_rate=0.001)
        assert rl.check("a") is True
        assert rl.check("a") is True
        assert rl.check("a") is True
        # Burst exhausted; refill is ~1 token / 1000s.
        assert rl.check("a") is False

    def test_keys_are_independent(self):
        rl = RateLimiter(burst=2, refill_rate=0.001)
        assert rl.check("a") is True
        assert rl.check("a") is True
        assert rl.check("a") is False
        # 'b' has its own bucket.
        assert rl.check("b") is True
        assert rl.check("b") is True
        assert rl.check("b") is False

    def test_refill(self):
        rl = RateLimiter(burst=1, refill_rate=1000.0)  # ~instant refill
        assert rl.check("x") is True
        assert rl.check("x") is False
        # Allow the bucket to refill.
        time.sleep(0.02)
        assert rl.check("x") is True

    def test_burst_zero_is_honoured(self):
        """burst=0 must NOT fall back to the env-var default."""
        rl = RateLimiter(burst=0, refill_rate=0.001)
        # No tokens at all — every request must be rejected.
        assert rl.check("a") is False
        assert rl.check("b") is False

    def test_eviction_when_over_max_keys(self):
        rl = RateLimiter(burst=1, refill_rate=0.001, max_keys=2)
        rl.check("a")
        rl.check("b")
        rl.check("c")  # triggers eviction of the oldest key ('a')
        # 'a' should now have a fresh bucket (1 token available).
        assert rl.check("a") is True

    def test_malformed_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("TOKI_RATE_LIMIT_BURST", "not-a-number")
        rl = RateLimiter()  # no explicit args → reads env
        assert rl.burst == 60  # default, not a crash

    def test_env_honoured_when_valid(self, monkeypatch):
        monkeypatch.setenv("TOKI_RATE_LIMIT_BURST", "5")
        monkeypatch.setenv("TOKI_RATE_LIMIT_RATE", "2.0")
        rl = RateLimiter()
        assert rl.burst == 5
        assert rl.refill_rate == 2.0
