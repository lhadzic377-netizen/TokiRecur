"""Tests that every runtime asset ships in the installed package.

pipx installs from the built wheel, so anything the app reads at
runtime (TUI stylesheet, web/dashboard CSS) must be declared in
[tool.setuptools.package-data]. A missing asset shows up on end-user
machines as "missing css file or variables" (TUI) or 404s on /static
(web) — this test fails locally and in CI before that can ship.
"""

import importlib.resources

import pytest

ASSETS = [
    "tui/theme.tcss",
    "web/static/style.css",
    "dashboard/static/style.css",
]


@pytest.mark.parametrize("asset", ASSETS)
def test_runtime_asset_is_packaged(asset):
    pkg, _, name = asset.rpartition("/")
    root = importlib.resources.files("tokirecur").joinpath(pkg)
    assert root is not None
    assert root.joinpath(name).is_file(), f"missing packaged asset: {asset}"


def test_installers_install_the_embeddings_extra_by_default():
    """The README, install.bat and the curl/irm one-liners all say the
    embeddings extra is installed by default; install.sh did not, so the
    benchmark's semantic similarity silently degraded to its proxy on
    the platform that reads the README.

    An empty TOKI_EXTRAS still means "no extras" (faster, no torch).
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    sh = (root / "install.sh").read_text(encoding="utf-8")
    bat = (root / "install.bat").read_text(encoding="utf-8")

    assert 'EXTRA=".[embeddings]"' in sh
    assert 'set "EXTRA=.[embeddings]"' in bat
    # The empty value has to be honoured as "no extras", which needs a
    # "is it set" test rather than a "is it non-empty" one.
    assert "TOKI_EXTRAS+set" in sh
    assert 'if "%TOKI_EXTRAS%"=="" set "EXTRA=."' in bat
