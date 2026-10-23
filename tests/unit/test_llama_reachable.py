"""A port that answers is not the same as a model that is running.

The registry hands every model a llama port. Anything else on the machine
can hold that port - the desktop app's voice server sat on one of them and
answered /health with 200 - and a model that was not running then read as
running. The status line lied, and `use_model` tried to free the RAM of a
stack that was never up, failed to, and warned about a port it could not
stop.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from tokirecur import ctl


class _Server:
    """A server on a port that answers /health and lists a model."""

    def __init__(self, model_id: str | None):
        self.model_id = model_id
        self._httpd = None

    def __enter__(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path.rstrip("/") == "/health":
                    body = b'{"status":"ok"}'
                elif self.path.rstrip("/") == "/v1/models":
                    data = (
                        [{"id": outer.model_id, "object": "model"}]
                        if outer.model_id
                        else []
                    )
                    body = json.dumps({"object": "list", "data": data}).encode()
                else:
                    body = b"{}"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        thread.start()
        self.port = self._httpd.server_address[1]
        return self

    def __exit__(self, *exc):
        self._httpd.shutdown()
        self._httpd.server_close()


def _spec(port: int, gguf: str = "/models/mine/mine-Q4_K_M.gguf"):
    return SimpleNamespace(
        name="mine",
        label="Mine",
        model_id="mine",
        gguf=gguf,
        llama_port=port,
        daemon_port=port + 1,
        llama_url=f"http://127.0.0.1:{port}",
        daemon_url=f"http://127.0.0.1:{port + 1}",
        ctx_size=4096,
        kv="f16",
        parallel=1,
        extra_args=[],
        binary="",
        state_dir=lambda: None,
    )


def test_the_server_on_the_port_must_be_serving_this_model():
    """A model whose file is not what the port serves is not running."""
    with _Server("SomeOtherModel-Q4_K_M") as server:
        spec = _spec(server.port)
        assert ctl.llama_reachable(spec) is False


def test_the_model_serving_its_own_weights_is_running():
    with _Server("mine-Q4_K_M") as server:
        spec = _spec(server.port)
        assert ctl.llama_reachable(spec) is True


def test_a_server_that_will_not_say_is_given_the_benefit_of_the_doubt():
    """The point is to catch the impostor, not to start a second copy.

    A server that answers /health but cannot list its models is not
    proven to be somebody else's, so it is left alone.
    """
    with _Server(None) as server:
        spec = _spec(server.port)
        assert ctl.llama_reachable(spec) is True


def test_a_model_with_no_file_is_not_the_thing_on_the_port():
    """There is nothing for the port to be serving."""
    with _Server("SomethingElse") as server:
        spec = _spec(server.port, gguf="")
        assert ctl.llama_reachable(spec) is False


def test_a_sharded_model_is_matched_by_its_base_name():
    """A sharded file is named -00001-of-00003 on disk and reported without it."""
    with _Server("Qwen3.5-122B-A10B-Q4_K_M") as server:
        spec = _spec(
            server.port,
            gguf="/models/qwen/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf",
        )
        assert ctl.llama_reachable(spec) is True


def test_names_are_compared_without_case_or_separators():
    with _Server("Ornith-1.5-35B-Q4_K_M") as server:
        spec = _spec(server.port, gguf="/models/o/ornith_1_5_35b_q4_k_m.gguf")
        assert ctl.llama_reachable(spec) is True
