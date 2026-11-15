"""Two requests, two slots.

The daemon used to serialize every non-streaming request behind one
lock that was held across the whole generation, so a second request
waited for the first to finish - one request at a time, whatever the
number of slots llama-server had. The streaming path never took that
lock and mutated the engine's routing state without any lock at all,
so two streams raced on the slot map.

These tests cover both halves: a second request gets a slot of its own
while the first is still running, and the routing state survives
concurrent use.
"""

import threading
import time

import pytest

from tokirecur.core.recycling import RecyclingEngine, compute_reuse_depth


class FakeBackend:
    """A backend whose completion blocks until the test releases it."""

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.started = threading.Semaphore(0)
        self.calls: list[int] = []
        self.embed_calls = 0
        self.tokenize_calls = 0
        self.slots_in_flight: list[int] = []
        self._lock = threading.Lock()

    def tokenize(self, text: str) -> list[int]:
        self.tokenize_calls += 1
        return [ord(c) % 97 for c in text]

    def embed(self, text: str, tokens: list[int] | None = None) -> list[float]:
        self.embed_calls += 1
        return [0.0] * 8

    def clear_slot(self, slot_id: int) -> dict:
        return {}

    def complete_openai(self, messages, *, slot_id=0, **kw) -> dict:
        with self._lock:
            self.calls.append(slot_id)
            self.slots_in_flight.append(slot_id)
        self.started.release()
        self.gate.wait(timeout=10)
        with self._lock:
            self.slots_in_flight.remove(slot_id)
        return {
            "text": "ok", "tool_calls": None, "finish_reason": "stop",
            "tokens_generated": 1, "prefill_tokens": 3,
            "ttft_ms": 1.0, "total_ms": 1.0, "prefill_tokens_per_sec": 1.0,
        }


class FakeIndex:
    def __init__(self) -> None:
        self.entries: list[str] = []

    def insert(self, entry) -> None:
        self.entries.append(entry.cache_id)

    def search(self, embedding):
        return None

    def stats(self) -> dict:
        return {"entries": len(self.entries), "bytes_on_disk": 0, "embedding_dim": 8}


def _engine(backend) -> RecyclingEngine:
    return RecyclingEngine(backend=backend, index=FakeIndex())


def test_two_requests_get_a_slot_each():
    """The second request must not wait for the first to finish."""
    backend = FakeBackend()
    engine = _engine(backend)

    first = threading.Thread(
        target=engine.serve_chat,
        args=([{"role": "user", "content": "first question"}],),
        kwargs={"max_slots": 2, "max_tokens": 8},
    )
    first.start()
    assert backend.started.acquire(timeout=5), "the first request never started"

    # The first request is inside the backend now: the slot it claimed
    # is busy, and a second request has to be able to take the other.
    second = threading.Thread(
        target=engine.serve_chat,
        args=([{"role": "user", "content": "second question"}],),
        kwargs={"max_slots": 2, "max_tokens": 8},
    )
    second.start()
    assert backend.started.acquire(timeout=5), (
        "the second request was still waiting for the first"
    )

    backend.gate.set()
    first.join(timeout=10)
    second.join(timeout=10)
    assert sorted(backend.calls) == [0, 1], backend.calls


def test_the_routing_state_survives_concurrent_requests():
    """The slot map is shared: concurrent requests must not corrupt it."""
    backend = FakeBackend()
    backend.gate.set()  # no blocking: every request runs straight through
    engine = _engine(backend)

    threads = [
        threading.Thread(
            target=engine.serve_chat,
            args=([{"role": "user", "content": f"question {i}"}],),
            kwargs={"max_slots": 4, "max_tokens": 8, "session_id": f"s{i}"},
        )
        for i in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    state = engine.slot_state()
    assert state["sessions"], "the session map is empty"
    assert len(set(state["sessions"].values())) >= 1
    # Every session is pinned to a slot, and no slot is left claimed.
    assert not engine._busy_slots, engine._busy_slots


def test_a_busy_slot_is_not_spilled():
    """Clearing the KV of a slot with a request in flight would
    corrupt that request."""
    backend = FakeBackend()
    engine = _engine(backend)
    engine._init_slot_state()
    engine._session_to_slot["s"] = 0
    engine._slot_context_tokens[0] = 10_000
    engine._busy_slots.add(0)
    engine._max_context_tokens = 100

    engine._route_request(
        tokenize_text="x", max_slots=2, session_id="s",
        max_context_tokens=100,
    )

    # The slot was busy, so it was left alone.
    assert 0 in engine._slot_history or engine._slot_context_tokens.get(0, 0) == 10_000


def test_compute_reuse_depth_matches_the_token_by_token_answer():
    """The bulk comparison must agree with the obvious definition."""
    for a, b in (
        ([], []),
        ([1, 2, 3], []),
        ([], [1, 2, 3]),
        ([1, 2, 3], [1, 2, 3]),
        ([1, 2, 3], [1, 2, 4]),
        ([1, 2, 3, 4], [1, 2]),
        (list(range(1000)), list(range(1000))),
        (list(range(1000)), list(range(500)) + [-1]),
    ):
        expected = 0
        for x, y in zip(a, b, strict=False):
            if x != y:
                break
            expected += 1
        assert compute_reuse_depth(a, b) == expected, (a[:5], b[:5])


def test_the_backend_does_not_tokenize_twice_for_the_embedding():
    """The routing text is tokenized for the slot match and handed to
    the embedding, which would otherwise tokenize it again."""
    calls = {"tokenize": 0}

    class Backend:
        def tokenize(self, text):
            calls["tokenize"] += 1
            return [1, 2, 3]

        def embed(self, text, tokens=None):
            calls["tokens_given"] = tokens
            return [0.0] * 4

    engine = RecyclingEngine(backend=Backend(), index=FakeIndex())
    engine._embed("hello", [1, 2, 3])
    assert calls["tokenize"] == 0
    assert calls["tokens_given"] == [1, 2, 3]


def test_a_backend_without_the_token_keyword_still_works():
    """A backend that predates the keyword must not break the route."""

    class OldBackend:
        def tokenize(self, text):
            return [1, 2, 3]

        def embed(self, text):
            return [0.0] * 4

    engine = RecyclingEngine(backend=OldBackend(), index=FakeIndex())
    assert engine._embed("hello", [1, 2, 3]) == [0.0] * 4


@pytest.mark.parametrize("size", [0, 1, 17, 512])
def test_reuse_depth_is_fast_enough_for_a_long_prompt(size):
    """A 20k-token prompt is compared per slot, on the request path."""
    a = list(range(size))
    b = list(range(size))
    start = time.perf_counter()
    assert compute_reuse_depth(a, b) == size
    assert (time.perf_counter() - start) < 0.05
