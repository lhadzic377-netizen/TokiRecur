"""The public-server tunnel: the URL, the credentials, and the cleanup.

The launcher offers a quick tunnel so a client that refuses 127.0.0.1 can
still reach the daemon. What matters here is that the hostname is read from
cloudflared's own log, that the credentials block names all three API
surfaces, and that a tunnel which never comes up does not leave a process
behind.
"""

import io

import pytest

from tokirecur import public_server as ps

URL = "https://blind-robbie-syndicate-bike.trycloudflare.com"


class FakeProc:
    """A Popen stand-in whose log is a list of lines."""

    def __init__(self, lines, exit_code=None):
        self.stdout = io.StringIO("".join(f"{line}\n" for line in lines))
        self._poll = exit_code
        self.terminated = False
        self.killed = False

    def poll(self):
        return self._poll

    def wait(self, timeout=None):
        return self._poll if self._poll is not None else 0

    def terminate(self):
        self.terminated = True
        self._poll = -15

    def kill(self):
        self.killed = True
        self._poll = -9


def test_quick_tunnel_hostname_is_read_from_the_log():
    proc = FakeProc([
        "INF precheck complete",
        f"Your quick Tunnel has been created! Visit it at {URL}",
        "INF Registered tunnel connection connIndex=0",
    ])
    captured = {}

    def popen(argv, **kwargs):
        captured["argv"] = argv
        return proc

    server = ps.start_public_server(
        8844, "k", "qwen", cloudflared="/usr/bin/cloudflared", popen=popen,
        timeout=5, registered_grace=1, warm=False, verify=False,
    )

    assert server.url == URL
    assert server.process is proc
    assert captured["argv"][:1] == ["/usr/bin/cloudflared"]
    assert "http://127.0.0.1:8844" in captured["argv"]


def test_the_three_api_surfaces_and_the_key_are_in_the_credentials():
    server = ps.PublicServer(
        url=URL, api_key="secret-key", model="qwen3.8",
        port=8844, process=None,
    )
    text = server.credentials_text()

    assert f"{URL}/v1" in text              # OpenAI
    assert f"Base URL (Anthropic)  {URL}" in text
    assert f"{URL}/v1beta" in text          # Google generative AI
    assert "secret-key" in text
    assert "qwen3.8" in text


def test_missing_cloudflared_explains_how_to_install_it(monkeypatch):
    monkeypatch.setattr(ps, "find_cloudflared", lambda: None)
    with pytest.raises(ps.PublicServerError) as err:
        ps.start_public_server(8844, "k", "m", timeout=1)
    assert "cloudflared is not installed" in str(err.value)


def test_a_tunnel_that_never_prints_a_url_is_stopped(monkeypatch):
    proc = FakeProc(["INF precheck complete", "INF still thinking"])
    with pytest.raises(ps.PublicServerError) as err:
        ps.start_public_server(
            8844, "k", "m", cloudflared="/usr/bin/cloudflared",
            popen=lambda argv, **kw: proc, timeout=0.4,
            warm=False, verify=False,
        )
    assert "did not come up" in str(err.value)
    assert proc.terminated


def test_stop_is_idempotent_and_kills_a_live_process():
    proc = FakeProc([])
    server = ps.PublicServer(url=URL, api_key="k", model="m", port=1, process=proc)

    server.stop()
    server.stop()

    assert proc.terminated
    assert not server.alive()


def test_tunnel_argv_targets_the_local_daemon_port():
    argv = ps.tunnel_argv("/usr/bin/cloudflared", 8844)
    assert argv[1] == "tunnel"
    assert argv[argv.index("--url") + 1] == "http://127.0.0.1:8844"
    assert "--no-autoupdate" in argv


def test_it_refuses_to_dial_when_nothing_listens_locally():
    """A tunnel to a dead port answers 502 on every request."""
    def popen(argv, **kwargs):
        pytest.fail("cloudflared must not be started for a dead origin")

    with pytest.raises(ps.PublicServerError) as exc:
        ps.start_public_server(
            9, "k", "m", cloudflared="/usr/bin/cloudflared", popen=popen,
            timeout=1, registered_grace=1, warm=False, verify=True,
        )
    assert "nothing is listening on 127.0.0.1:9" in str(exc.value)


def test_port_open_tells_the_truth():
    """The probe the CLI uses to notice a stack that went away."""
    assert ps.port_open(9) is False


def test_wait_until_live_reports_failure(monkeypatch):
    """A route that never answers must be reported as unverified."""
    import urllib.error
    import urllib.request

    def boom(*args, **kwargs):
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert ps._wait_until_live("https://example.invalid/v1/models", "k",
                               timeout=0.2) is False
