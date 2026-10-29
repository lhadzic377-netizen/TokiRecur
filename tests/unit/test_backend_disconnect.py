"""A client that walks away must not wedge the backend.

llama-server cancels a task when its HTTP client disconnects, and it
never releases the slot: the slot stays flagged ``is_processing`` with a
task id that will never advance. Two such cancellations wedge the whole
server — ``/health`` still answers 200, every completion queues forever,
and only restarting the stack recovers it.

So when the harness goes away mid-turn, the daemon has to keep the
upstream request alive long enough for llama-server to finish the task
and release the slot.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import socketserver
import threading
import time

from tokirecur.backend import LlamaServerBackend


class _StreamingHandler(http.server.BaseHTTPRequestHandler):
    """Writes N SSE chunks slowly and records what it managed to send."""

    chunks = 5
    gap = 0.12
    written: list = []

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(self.chunks):
            body = json.dumps({"choices": [{"delta": {"content": f"tok{i}"}}]})
            try:
                self.wfile.write(f"data: {body}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                self.written.append(("broken", i))
                return
            self.written.append(i)
            time.sleep(self.gap)

    def log_message(self, *args) -> None:  # silence the test output
        return


def _serve() -> tuple[socketserver.TCPServer, int]:
    server = socketserver.TCPServer(("127.0.0.1", 0), _StreamingHandler)
    server.allow_reuse_address = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def test_upstream_stream_survives_client_disconnect():
    """Cancelling the consumer must not close the upstream request."""
    _StreamingHandler.written = []
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
            await gen.aclose()
            # Give the server time to finish writing the rest.
            await asyncio.sleep(_StreamingHandler.chunks * _StreamingHandler.gap + 0.6)

        asyncio.run(scenario())
    finally:
        server.shutdown()
        server.server_close()

    written = _StreamingHandler.written
    # Every chunk made it out: the upstream connection was still open
    # after the client disappeared, so llama-server completes the task
    # and releases the slot.
    assert written == list(range(_StreamingHandler.chunks)), written


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
