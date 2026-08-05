"""Prometheus metrics for the Toki Recur daemon.

Centralises all metric definitions so different route handlers can
record without colliding on registry state. Exposes a single
`/metrics` endpoint that returns the standard text exposition format.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram


class DaemonMetrics:
    """All daemon-level Prometheus metrics."""

    def __init__(self) -> None:
        # Request counts
        self.requests_total = Counter(
            "toki_requests_total",
            "Total requests served",
            labelnames=("route", "status"),
        )
        self.request_duration = Histogram(
            "toki_request_duration_seconds",
            "End-to-end request duration",
            labelnames=("route",),
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
        )
        self.ttft_seconds = Histogram(
            "toki_ttft_seconds",
            "Time to first token (per completion)",
            buckets=(0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
        )

        # Cache effectiveness
        self.cache_hits = Counter(
            "toki_cache_hits_total",
            "Cache hit classifications",
            labelnames=("kind",),  # hit | soft_hit | miss
        )
        self.tokens_skipped = Counter(
            "toki_tokens_skipped_total",
            "Tokens served from cache instead of being prefilled",
        )
        self.prefill_tokens = Counter(
            "toki_prefill_tokens_total",
            "Tokens actually prefilled",
        )
        self.generated_tokens = Counter(
            "toki_generated_tokens_total",
            "Tokens generated (output)",
        )

        # Slot state
        self.active_slots = Gauge(
            "toki_active_slots",
            "Number of slots currently in use",
        )
        self.slot_context_tokens = Gauge(
            "toki_slot_context_tokens",
            "KV context size per slot",
            labelnames=("slot_id",),
        )
        self.spills_total = Counter(
            "toki_spills_total",
            "Slot spills (KV context cleared due to budget)",
        )

        # Auth
        self.auth_failures = Counter(
            "toki_auth_failures_total",
            "Requests rejected due to missing/invalid auth token",
        )

        # Rate limiting
        self.rate_limit_rejections = Counter(
            "toki_rate_limit_rejections_total",
            "Requests rejected due to rate limiting",
        )

        # Uptime (set on each scrape)
        self.uptime_seconds = Gauge(
            "toki_uptime_seconds",
            "Daemon uptime in seconds",
        )

    def observe_request(self, route: str, status: int, duration: float) -> None:
        self.requests_total.labels(route=route, status=str(status)).inc()
        self.request_duration.labels(route=route).observe(duration)

    def observe_completion(
        self,
        *,
        ttft_ms: float,
        prefill_tokens: int,
        generated_tokens: int,
        skipped_tokens: int,
        hit: bool,
        soft_hit: bool,
        slot_id: int,
        context_tokens: int,
        spilled: bool = False,
    ) -> None:
        if hit:
            self.cache_hits.labels(kind="hit").inc()
        elif soft_hit:
            self.cache_hits.labels(kind="soft_hit").inc()
        else:
            self.cache_hits.labels(kind="miss").inc()
        self.ttft_seconds.observe(ttft_ms / 1000.0)
        self.tokens_skipped.inc(skipped_tokens)
        self.prefill_tokens.inc(prefill_tokens)
        self.generated_tokens.inc(generated_tokens)
        self.slot_context_tokens.labels(slot_id=str(slot_id)).set(context_tokens)
        if spilled:
            self.spills_total.inc()


METRICS = DaemonMetrics()
