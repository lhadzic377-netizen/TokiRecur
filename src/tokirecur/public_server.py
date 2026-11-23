"""A public HTTPS origin for the running daemon.

Every app that takes a custom model endpoint (arkenOS, Cursor, ...) demands
a public HTTPS base URL on port 443 and refuses ``127.0.0.1``: the form is
written for cloud providers, and a loopback address fails its validation
before a single request is sent. The daemon, meanwhile, is the thing the
person actually wants to point at.

A ``cloudflared`` quick tunnel is exactly the missing piece: a public
hostname on 443, terminated at Cloudflare and forwarded to the local
daemon. The daemon's bearer token still authenticates every request, so
the hostname on its own is not access - and the hostname is random and
dies with the process, so nothing is left behind.

The tunnel is a child process, not a service: ``stop()`` terminates it and
the public route is gone.
"""

from __future__ import annotations

import os
import queue
import re
import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# cloudflared prints the hostname in its log as soon as the quick tunnel
# exists. Everything else in the log (prechecks, connection ids, QUIC
# probes) is noise here.
TUNNEL_PATTERN = re.compile(r"https://[a-z0-9][a-z0-9-]*\.trycloudflare\.com")

# A quick tunnel is registered a beat after the hostname is printed. Waiting
# for this line as well is what makes the first request from the other side
# succeed instead of racing the connector.
REGISTERED = "Registered tunnel connection"

# Where distro packages and the official installer put it. ``shutil.which``
# covers PATH; these cover a GUI-launched launcher whose PATH is thinner.
_KNOWN_PATHS = (
    "/usr/local/bin/cloudflared",
    "/usr/bin/cloudflared",
    "/opt/homebrew/bin/cloudflared",
)


class PublicServerError(RuntimeError):
    """The tunnel could not be started; the message is shown to the person."""


@dataclass(slots=True)
class PublicServer:
    """A live quick tunnel and the credentials a client needs to use it."""

    url: str
    api_key: str
    model: str
    port: int
    process: Any | None = None
    log: list[str] = field(default_factory=list)

    # -- the three API surfaces -------------------------------------------

    @property
    def openai_base(self) -> str:
        """OpenAI-compatible base: ``POST {base}/chat/completions``."""
        return f"{self.url}/v1"

    @property
    def anthropic_base(self) -> str:
        """Anthropic base: ``POST {base}/v1/messages``."""
        return self.url

    @property
    def google_base(self) -> str:
        """Google generative AI base: ``POST {base}/models/{model}:...``."""
        return f"{self.url}/v1beta"

    def credentials_text(self) -> str:
        """One block a person can paste into a chat or a config form."""
        return "\n".join((
            f"Base URL (OpenAI)     {self.openai_base}",
            f"Base URL (Anthropic)  {self.anthropic_base}",
            f"Base URL (Google)     {self.google_base}",
            f"API key               {self.api_key}",
            f"Model                 {self.model}",
        ))

    # -- lifecycle --------------------------------------------------------

    def alive(self) -> bool:
        proc = self.process
        return proc is not None and proc.poll() is None

    def stop(self, wait: float = 4.0) -> None:
        """Terminate the tunnel. Safe to call twice, or after it died."""
        proc = self.process
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=wait)
        except (OSError, subprocess.TimeoutExpired):
            try:
                proc.kill()
            except OSError:
                pass


def find_cloudflared() -> str | None:
    """The cloudflared binary, or None when the machine has none."""
    found = shutil.which("cloudflared")
    if found:
        return found
    for path in _KNOWN_PATHS:
        if os.path.isfile(path):
            return path
    return None


def tunnel_argv(binary: str, port: int) -> list[str]:
    return [
        binary, "tunnel",
        "--url", f"http://127.0.0.1:{port}",
        "--no-autoupdate",
    ]


def start_public_server(
    port: int,
    api_key: str,
    model: str,
    *,
    cloudflared: str | None = None,
    timeout: float = 45.0,
    registered_grace: float = 10.0,
    popen: Callable[..., Any] | None = None,
    warm: bool = True,
    verify: bool = True,
) -> PublicServer:
    """Run a quick tunnel to ``127.0.0.1:port`` and return its credentials.

    Blocks until cloudflared prints a hostname (and, if it does so within
    ``registered_grace`` seconds, until the connector registers). Raises
    ``PublicServerError`` with cloudflared's own last line when it cannot:
    "install it" and "Cloudflare refused" are different problems and the
    person should not have to guess which one they have.
    """
    binary = cloudflared or find_cloudflared()
    if not binary:
        raise PublicServerError(
            "cloudflared is not installed — install it with "
            "`sudo apt install cloudflared` (or the official binary from "
            "https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)"
        )

    factory = popen or subprocess.Popen
    try:
        proc = factory(
            tunnel_argv(binary, port),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
    except OSError as e:
        raise PublicServerError(f"cloudflared could not start: {e}") from e

    lines: queue.Queue[str | None] = queue.Queue()
    log: list[str] = []

    def pump() -> None:
        stream = getattr(proc, "stdout", None)
        if stream is None:
            lines.put(None)
            return
        try:
            for line in stream:
                lines.put(line)
        except (OSError, ValueError):
            pass
        finally:
            lines.put(None)

    threading.Thread(target=pump, daemon=True).start()

    def drain(until: float) -> None:
        """Read whatever the tunnel has said so far, up to ``until``."""
        while time.monotonic() < until:
            try:
                line = lines.get(timeout=0.2)
            except queue.Empty:
                if proc.poll() is not None:
                    return
                continue
            if line is None:
                return
            log.append(line.rstrip())

    deadline = time.monotonic() + timeout
    url: str | None = None
    while time.monotonic() < deadline:
        drain(min(deadline, time.monotonic() + 0.5))
        for raw in log:
            match = TUNNEL_PATTERN.search(raw)
            if match:
                url = match.group(0)
        if url:
            break
        if proc.poll() is not None:
            break
    if url is None:
        _terminate(proc)
        tail = next((line for line in reversed(log) if line.strip()), "")
        detail = f" — {tail}" if tail else ""
        raise PublicServerError(f"cloudflared did not come up within {timeout:.0f}s{detail}")

    # The connector registers a moment after the hostname is printed; give
    # it a short grace so the first request from the other side lands after
    # the route exists, then proceed whether or not the line showed up.
    grace_until = min(deadline, time.monotonic() + registered_grace)
    while time.monotonic() < grace_until:
        drain(min(grace_until, time.monotonic() + 0.5))
        if any(REGISTERED in line for line in log):
            break
        if proc.poll() is not None:
            break

    # A hostname looked up in the second it was created gets NXDOMAIN, and
    # systemd-resolved caches that answer: the client that is handed the
    # URL a minute later cannot resolve it even though the tunnel is live.
    # Resolving it here (retrying until the edge knows the name) leaves a
    # positive answer in the same cache the client will read.
    # Tests drive a fake process and must not touch DNS or the network;
    # production wants both, so the caller says which it is.
    if warm:
        _warm_dns(url)
    if verify:
        # A hostname that resolves is not yet a route: the edge needs a
        # moment to learn which connector owns the name, and a request in
        # that window hangs rather than fails. One GET through the tunnel
        # proves the credentials are usable before they are shown.
        _wait_until_live(f"{url}/v1/models", api_key)

    return PublicServer(
        url=url, api_key=api_key, model=model, port=port, process=proc, log=log,
    )


def _warm_dns(url: str, timeout: float = 15.0) -> None:
    """Resolve the quick-tunnel hostname, retrying while DNS catches up.

    The name did not exist a second ago, so the first lookup answers
    NXDOMAIN - and systemd-resolved caches that answer for its negative
    TTL. A single flush is not enough: a retry that lands before the edge
    publishes the name caches NXDOMAIN again. Flush, ask, and flush again
    a bounded number of times, so the cache the client will read holds a
    positive answer.
    """
    host = url.split("://", 1)[-1].split("/", 1)[0]

    def resolves() -> bool:
        try:
            socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
            return True
        except OSError:
            return False

    flush = shutil.which("resolvectl")

    def flush_cache() -> None:
        if not flush:
            return
        try:
            subprocess.run([flush, "flush-caches"], timeout=3, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError):
            pass

    deadline = time.monotonic() + timeout
    flushes = 0
    while time.monotonic() < deadline:
        if resolves():
            return
        if flushes < 3:
            flush_cache()
            flushes += 1
            time.sleep(2.5)
        else:
            time.sleep(1.0)


def _wait_until_live(endpoint: str, api_key: str, timeout: float = 12.0) -> None:
    """Poll ``endpoint`` until the public route answers, or give up quietly.

    Best effort by design: the tunnel process is up and the hostname is
    real, so a probe that times out is worth a slower first request, not
    an error that hides working credentials. A 401/403 is a success for
    this purpose - the edge reached the daemon, which only objected to
    the key.
    """
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        endpoint, headers={"Authorization": f"Bearer {api_key}"},
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(request, timeout=4) as response:
                if response.status < 500:
                    return
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                return
        except OSError:
            pass
        time.sleep(1.0)


def _terminate(proc: Any, wait: float = 3.0) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=wait)
    except (OSError, subprocess.TimeoutExpired):
        try:
            proc.kill()
        except OSError:
            pass
