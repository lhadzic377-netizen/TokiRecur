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
