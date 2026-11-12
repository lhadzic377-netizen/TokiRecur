"""Regressions for local-secret exposure and unsafe archive extraction
in the CLI download / launch paths.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest


class TestCurlTokenNotInArgv:
    def test_token_passed_via_0600_config_not_argv(self, tmp_path, monkeypatch):
        import tokirecur.cli as cli

        dest = tmp_path / "out.bin"
        captured: dict = {}

        def fake_run(cmd, **kw):
            captured["cmd"] = list(cmd)
            if "--config" in cmd:
                cfg = Path(cmd[cmd.index("--config") + 1])
                captured["cfg_path"] = cfg
                captured["cfg_content"] = cfg.read_text()
                captured["cfg_mode"] = cfg.stat().st_mode & 0o777
            dest.write_bytes(b"payload")
            return SimpleNamespace(returncode=0, stderr="")

        monkeypatch.setattr(cli.subprocess, "run", fake_run)
        cli._curl_download(
            "/usr/bin/curl", "http://x/y", dest, label="dl", token="SECRET-TOKEN",
        )
        # The token must never appear in argv (world-readable via /proc).
        assert all("SECRET-TOKEN" not in arg for arg in captured["cmd"])
        assert "--config" in captured["cmd"]
        assert "SECRET-TOKEN" in captured["cfg_content"]
        assert captured["cfg_mode"] == 0o600
        # The temporary config is removed after the run.
        assert not captured["cfg_path"].exists()

    def test_no_token_means_no_config(self, tmp_path, monkeypatch):
        import tokirecur.cli as cli

        dest = tmp_path / "out.bin"
        captured: dict = {}

        def fake_run(cmd, **kw):
            captured["cmd"] = list(cmd)
            dest.write_bytes(b"payload")
            return SimpleNamespace(returncode=0, stderr="")

        monkeypatch.setattr(cli.subprocess, "run", fake_run)
        cli._curl_download("/usr/bin/curl", "http://x/y", dest, label="dl", token=None)
        assert "--config" not in captured["cmd"]


class TestDshLogOwnerOnly:
    def test_log_created_0600(self, tmp_path, monkeypatch):
        import tokirecur.selector as sel

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        log_path = str(tmp_path / ".local/state/tokirecur/dsh-web.log")
        f = sel._open_dsh_log(log_path)
        try:
            f.write(b"http://127.0.0.1:3080/?token=abc\n")
        finally:
            f.close()
        assert Path(log_path).stat().st_mode & 0o777 == 0o600

    def test_existing_world_readable_log_forced_0600(self, tmp_path, monkeypatch):
        import tokirecur.selector as sel

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        log_path = tmp_path / ".local/state/tokirecur/dsh-web.log"
        log_path.parent.mkdir(parents=True)
        log_path.write_bytes(b"old token url\n")
        log_path.chmod(0o644)
        f = sel._open_dsh_log(str(log_path))
        f.close()
        assert log_path.stat().st_mode & 0o777 == 0o600


class TestTarExtractionFallback:
    """On stock LTS Python (< 3.10.12 / 3.11.4) filter='data' raises
    TypeError; the fallback must still refuse path traversal instead of
    running unfiltered extractall (CVE-2007-4559)."""

    def _evil_tar(self, path: Path) -> None:
        with tarfile.open(path, "w") as tf:
            info = tarfile.TarInfo("../escape.txt")
            data = b"pwned"
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

    def test_fallback_rejects_traversal(self, tmp_path, monkeypatch):
        import tokirecur.cli as cli

        archive = tmp_path / "evil.tar"
        self._evil_tar(archive)
        dest = tmp_path / "dest"
        dest.mkdir()

        def old_extractall(self, path=".", members=None, *, numeric_owner=False,
                           filter=None):
            if filter is not None:
                raise TypeError("unexpected keyword argument 'filter'")
            raise AssertionError("unfiltered extractall ran on an unsafe member")

        monkeypatch.setattr(tarfile.TarFile, "extractall", old_extractall)
        with tarfile.open(archive) as tf:
            with pytest.raises(RuntimeError, match="unsafe tar member"):
                cli._extract_tar_safe(tf, dest)
        assert not (tmp_path / "escape.txt").exists()

    def test_fallback_extracts_safe_member(self, tmp_path, monkeypatch):
        import tokirecur.cli as cli

        archive = tmp_path / "safe.tar"
        with tarfile.open(archive, "w") as tf:
            info = tarfile.TarInfo("bin/llama-server")
            data = b"binary"
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        dest = tmp_path / "dest"
        dest.mkdir()

        real = tarfile.TarFile.extractall

        def old_extractall(self, path=".", members=None, *, numeric_owner=False,
                           filter=None):
            if filter is not None:
                raise TypeError("unexpected keyword argument 'filter'")
            return real(self, path, members, numeric_owner=numeric_owner)

        monkeypatch.setattr(tarfile.TarFile, "extractall", old_extractall)
        with tarfile.open(archive) as tf:
            cli._extract_tar_safe(tf, dest)
        assert (dest / "bin" / "llama-server").read_bytes() == b"binary"
