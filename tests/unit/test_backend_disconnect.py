"""A client that walks away must not leave the GPU generating for nobody.

The upstream HTTP stream lives in a producer task. When the harness
disappears mid-turn (timeout, Ctrl+C, an aborted tool call), the producer
is cancelled and its httpx connection closes. llama-server reads that as
"cancel this task" and frees the slot — measured on the pinned build: the
slot is ``is_processing: false`` within a second of the socket closing.

The opposite policy (letting the producer finish) looked harmless, but a
32k-token turn at iGPU decode speed is most of an hour of work nobody
will read, and two abandoned turns starve every other harness behind the
saturated slots.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import socketserver
import threading
import time

import pytest

from tokirecur.backend import LlamaServerBackend
from tokirecur.core.recycling import RecyclingEngine


class _HangingBackend:
    """A backend whose stream yields one token and then never ends."""

    def tokenize(self, text: str) -> list[int]:
        return [1, 2, 3]

    def embed(self, text: str, tokens: list[int] | None = None) -> list[float]:
        return [0.0] * 4

    def cache_info(self, slot_id: int) -> dict:
        return {}

    def clear_slot(self, slot_id: int) -> dict:
        return {}

    async def complete_openai_stream(self, messages, **kwargs):
        slot_id = kwargs["slot_id"]
        yield {"kind": "delta", "delta": "hi"}
        await asyncio.Event().wait()
        yield {"kind": "result", "result": {"slot_id": slot_id}}


class _NullIndex:
    def search(self, embedding):
        return None

    def insert(self, entry) -> None:
        pass

    def touch(self, cache_id: str) -> None:
        pass

    def stats(self) -> dict:
        return {}


class _FailingBackend(_HangingBackend):
    def complete_openai(self, messages, **kwargs):
        raise RuntimeError("backend exploded")


class _StreamingHandler(http.server.BaseHTTPRequestHandler):
    """Writes N SSE chunks slowly and records what it managed to send."""

    chunks = 5
    gap = 0.12
    written: list = []
    payloads: list = []
    connection_closed = False

    def _check_client_close(self) -> None:
        self.connection.settimeout(0.5)
        try:
            _StreamingHandler.connection_closed = self.connection.recv(1) == b""
        except (TimeoutError, OSError):
            pass

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        try:
            _StreamingHandler.payloads.append(json.loads(raw))
        except ValueError:
            _StreamingHandler.payloads.append({})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(self.chunks):
            body = json.dumps({"choices": [{"delta": {"content": f"tok{i}"}}]})
            try:
                self.wfile.write(f"data: {body}\n\n".encode())
                self.wfile.flush()
            except ConnectionError:
                self.written.append(("broken", i))
                # Windows reports a closed peer as ConnectionAbortedError,
                # macOS/Linux as BrokenPipeError or ConnectionResetError.
                _StreamingHandler.connection_closed = True
                return
            self.written.append(i)
            time.sleep(self.gap)
        self._check_client_close()

    def log_message(self, *args) -> None:  # silence the test output
        return


def _serve() -> tuple[socketserver.TCPServer, int]:
    server = socketserver.TCPServer(("127.0.0.1", 0), _StreamingHandler)
    server.allow_reuse_address = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def test_upstream_stream_closes_when_client_disconnects():
    """Cancelling the consumer must close the upstream TCP connection.

    A pooled httpx client can stop reading without closing the socket;
    llama-server then keeps generating to ``max_tokens`` for nobody. The
    streaming path must use a client dedicated to this one request so its
    cancellation closes the socket and the slot is released.
    """
    _StreamingHandler.written = []
    _StreamingHandler.connection_closed = False
    server, port = _serve()
    backend = LlamaServerBackend(f"http://127.0.0.1:{port}")
    try:
        async def scenario() -> None:
            gen = backend.complete_openai_stream(
                [{"role": "user", "content": "hi"}], slot_id=0,
            )
            first = await gen.__anext__()
            assert first == {"kind": "delta", "delta": "tok0"}
            # The harness aborts the turn right after the first chunk.
            await asyncio.wait_for(gen.aclose(), timeout=5.0)
            # Give the server time to notice the closed socket.
            await asyncio.sleep(_StreamingHandler.chunks * _StreamingHandler.gap + 0.6)

        asyncio.run(scenario())
    finally:
        server.shutdown()
        server.server_close()

    written = _StreamingHandler.written
    sent = [w for w in written if isinstance(w, int)]
    # How many chunks the local server manages to push before the close
    # is timing-dependent; the socket actually being closed is the part
    # that matters to llama-server.
    assert 0 in sent, written
    assert _StreamingHandler.connection_closed, written


def test_stream_yields_result_item_after_deltas():
    """The refactor keeps the item contract: deltas then a result."""
    _StreamingHandler.written = []
    server, port = _serve()
    backend = LlamaServerBackend(f"http://127.0.0.1:{port}")
    try:
        async def scenario() -> list[dict]:
            items = []
            async for item in backend.complete_openai_stream(
                [{"role": "user", "content": "hi"}], slot_id=0,
            ):
                items.append(item)
            return items

        items = asyncio.run(scenario())
    finally:
        server.shutdown()
        server.server_close()

    kinds = [i["kind"] for i in items]
    assert kinds.count("delta") == _StreamingHandler.chunks
    assert kinds[-1] == "result"
    assert items[-1]["result"]["text"] == "".join(
        f"tok{i}" for i in range(_StreamingHandler.chunks)
    )


def test_clear_slot_uses_the_erase_action():
    """llama-server's slot reset action is `erase`, not `clear`."""
    backend = LlamaServerBackend("http://127.0.0.1:1")
    calls: list[str] = []

    def fake_post(path: str, payload: dict) -> dict:
        calls.append(path)
        return {"id_slot": 3, "n_erased": 7}

    backend._post = fake_post  # type: ignore[method-assign]
    result = backend.clear_slot(3)
    assert result["cleared"] is True
    assert calls == ["/slots/3?action=erase"]


def test_stream_routes_by_id_slot_on_the_wire():
    """llama-server reads `id_slot`; `slot_id` is silently ignored.

    Toki used to send `slot_id`, so the backend auto-assigned a slot and
    ignored the routing the engine had computed — and a later `erase`
    then targeted the wrong slot.
    """
    _StreamingHandler.written = []
    _StreamingHandler.payloads = []
    server, port = _serve()
    backend = LlamaServerBackend(f"http://127.0.0.1:{port}")
    try:
        async def scenario() -> None:
            async for _ in backend.complete_openai_stream(
                [{"role": "user", "content": "hi"}], slot_id=1,
            ):
                pass

        asyncio.run(scenario())
    finally:
        server.shutdown()
        server.server_close()

    payload = _StreamingHandler.payloads[-1]
    assert payload["id_slot"] == 1
    assert "slot_id" not in payload


def test_engine_releases_slot_when_stream_is_abandoned():
    """The slot claimed by _route_request must not stay busy forever.

    On a disconnect, _finalize_request — the normal release point — never
    runs. After two abandoned turns every later request would see every
    slot as claimed and pile up behind them.
    """
    engine = RecyclingEngine(_HangingBackend(), _NullIndex())

    async def scenario() -> None:
        gen = engine.serve_chat_stream(
            [{"role": "user", "content": "hi"}], max_tokens=10,
        )
        first = await gen.__anext__()
        assert first == {"kind": "delta", "delta": "hi"}
        assert engine._busy_slots == {0}
        await asyncio.wait_for(gen.aclose(), timeout=5.0)
        assert engine._busy_slots == set()

    asyncio.run(scenario())


def test_engine_releases_slot_when_backend_fails():
    """A failed call must not leave its slot claimed either."""
    engine = RecyclingEngine(_FailingBackend(), _NullIndex())

    with pytest.raises(RuntimeError, match="backend exploded"):
        engine.serve_chat(
            [{"role": "user", "content": "hi"}], max_tokens=10,
        )
    assert engine._busy_slots == set()


def test_streaming_response_closes_body_generator():
    """Starlette leaves the body generator open on client disconnect."""
    import tokirecur.api.app as app_mod

    closed = asyncio.Event()

    async def body():
        try:
            yield b"data: hi\n\n"
            yield b"data: there\n\n"
            await asyncio.Event().wait()
        finally:
            closed.set()

    async def scenario() -> None:
        disconnect = asyncio.Event()

        async def receive():
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message.get("body"):
                disconnect.set()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "method": "GET", "path": "/", "headers": [], "query_string": b"",
            "scheme": "http", "server": ("127.0.0.1", 80),
            "client": ("127.0.0.1", 1234), "root_path": "",
        }
        await app_mod.ClosingStreamingResponse(body())(scope, receive, send)
        await asyncio.wait_for(closed.wait(), timeout=2.0)

    asyncio.run(scenario())


def test_heartbeat_wrapper_closes_the_engine_stream():
    """Cancelling the SSE relay must close the stream it was reading.

    The relay keeps a pending ``__anext__`` across heartbeats. If that
    pending call were left behind, the engine would keep producing into
    a task nobody reads — the abandoned-generation bug again, one layer
    up.
    """
    import tokirecur.api.app as app_mod

    closed = asyncio.Event()

    async def source():
        try:
            yield {"kind": "delta", "delta": "hi"}
            await asyncio.Event().wait()
        finally:
            closed.set()

    async def scenario() -> None:
        async def consume() -> None:
            async for _ in app_mod._aiter_with_heartbeat(source(), interval=0.05):
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.2)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.wait_for(closed.wait(), timeout=2.0)

    asyncio.run(scenario())
