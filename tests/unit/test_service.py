"""Tests for systemd service management (service.py).

Focus: orphaned enablement cleanup. A model that was once marked
``boot: true`` but later lost the flag (or left the registry) must
stop auto-starting at boot — the historical bug where 9b/laguna
"started for no reason" on every login.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from tokirecur import ctl, service


def _seed_registry(env: Path, monkeypatch) -> None:
    reg = env / "models.yaml"
    reg.write_text(yaml.safe_dump({"models": [
        {"name": "m1", "gguf": "/tmp/m1.gguf", "llama_port": 1,
         "daemon_port": 2, "ctx_size": 100, "max_tokens": 90,
         "model_id": "m1", "boot": False},
        {"name": "m2", "gguf": "/tmp/m2.gguf", "llama_port": 3,
         "daemon_port": 4, "ctx_size": 100, "max_tokens": 90,
         "model_id": "m2", "boot": True},
    ]}, sort_keys=False))
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", reg)


def _fake_unit_dir(tmp_path: Path, monkeypatch) -> Path:
    unit_dir = tmp_path / "systemd" / "user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(service, "UNIT_DIR", unit_dir)
    return unit_dir


def _enable(unit_dir: Path, instance: str) -> None:
    wants = unit_dir / "default.target.wants"
    wants.mkdir(parents=True, exist_ok=True)
    (wants / f"toki-model@{instance}.service").write_text("# stub\n")


def test_enabled_instances_reads_wants_dir(tmp_path, monkeypatch):
    unit_dir = _fake_unit_dir(tmp_path, monkeypatch)
    assert service._enabled_instances() == []
    _enable(unit_dir, "9b")
    _enable(unit_dir, "laguna")
    # Non-toki links and the bare template must be ignored.
    (unit_dir / "default.target.wants" / "other.service").write_text("x")
    assert service._enabled_instances() == ["9b", "laguna"]


def test_drop_stale_enablement_removes_orphans_keeps_boot(
    tmp_path, monkeypatch,
):
    """Models no longer marked boot get disabled; current boot models
    keep their enablement."""
    unit_dir = _fake_unit_dir(tmp_path, monkeypatch)
    _seed_registry(tmp_path, monkeypatch)
    _enable(unit_dir, "9b")
    _enable(unit_dir, "laguna")
    _enable(unit_dir, "m2")  # currently boot-marked — must survive

    calls: list[list[str]] = []
    monkeypatch.setattr(
        service, "_systemctl",
        lambda *args: calls.append(list(args))
        or type("R", (), {"returncode": 0, "stdout": ""})(),
    )

    removed = service._drop_stale_enablement(keep={"m2"})
    assert removed == ["9b", "laguna"]
    disabled = [c for c in calls if c[:2] == ["disable", "--now"]]
    assert any("toki-model@9b.service" in c for c in disabled)
    assert not any("toki-model@m2.service" in c for c in disabled)
    wants = unit_dir / "default.target.wants"
    assert not (wants / "toki-model@9b.service").exists()
    assert (wants / "toki-model@m2.service").exists()


def test_uninstall_removes_every_enabled_instance_even_without_boot_models(
    tmp_path, monkeypatch, capsys,
):
    """The registry has zero boot models — uninstall must STILL clean
    up orphaned enablements. This is the '9b starts for no reason'
    regression: old uninstall only iterated boot-marked models and
    left phantom boot entries behind forever."""
    unit_dir = _fake_unit_dir(tmp_path, monkeypatch)
    _seed_registry(tmp_path, monkeypatch)
    _enable(unit_dir, "9b")
    _enable(unit_dir, "laguna")
    # Template + ordering drop-in from an earlier install.
    (unit_dir / service.UNIT_NAME).write_text("# template\n")
    d = unit_dir / "toki-model@laguna.service.d"
    d.mkdir()
    (d / "ordering.conf").write_text("[Unit]\nAfter=...\n")

    monkeypatch.setattr(
        service, "_systemctl",
        lambda *args: type("R", (), {"returncode": 0, "stdout": ""})(),
    )
    # The cleanup logic is under test, not the platform.
    monkeypatch.setattr(service, "_systemd_available", lambda: True)

    rc = service.uninstall()
    assert rc == 0
    out = capsys.readouterr().out
    assert "9b" in out and "laguna" in out
    wants = unit_dir / "default.target.wants"
    assert not list(wants.glob("toki-model@*"))
    assert not (unit_dir / service.UNIT_NAME).exists()
    assert not d.exists()


def test_install_cleans_stale_enablement_before_enabling(
    tmp_path, monkeypatch, capsys,
):
    """install() first drops enablements of models that lost their
    boot flag, then enables only the current set."""
    unit_dir = _fake_unit_dir(tmp_path, monkeypatch)
    _seed_registry(tmp_path, monkeypatch)
    _enable(unit_dir, "stale-old-model")

    monkeypatch.setenv("USER", os.environ.get("USER", "test"))
    sys_calls: list[list[str]] = []

    def fake_systemctl(*args):
        sys_calls.append(list(args))
        return type("R", (), {"returncode": 0, "stdout": ""})()

    monkeypatch.setattr(service, "_systemctl", fake_systemctl)
    # The unit logic is under test, not the platform: install() refuses
    # outright where systemd does not exist, so say it is available.
    monkeypatch.setattr(service, "_systemd_available", lambda: True)
    monkeypatch.setattr(
        service.subprocess, "run",
        lambda *a, **k: type("R", (), {"returncode": 0, "stdout": ""})(),
    )

    rc = service.install()
    assert rc == 0
    out = capsys.readouterr().out
    assert "disabled stale boot unit: toki-model@stale-old-model.service" in out
    wants = unit_dir / "default.target.wants"
    # Removal is done by our code directly (filesystem), enablement by
    # systemctl (stubbed → assert the call).
    assert not (wants / "toki-model@stale-old-model.service").exists()
    assert any(
        c[:2] == ["enable", "toki-model@m2.service"] for c in sys_calls
    )


def test_install_noop_when_no_boot_models_and_nothing_stale(
    tmp_path, monkeypatch,
):
    """No boot models AND no orphaned enablement → nothing to do."""
    _fake_unit_dir(tmp_path, monkeypatch)
    reg = tmp_path / "models.yaml"
    reg.write_text(yaml.safe_dump({"models": [
        {"name": "m1", "gguf": "/tmp/m1.gguf", "llama_port": 1,
         "daemon_port": 2, "ctx_size": 100, "max_tokens": 90,
         "model_id": "m1", "boot": False},
    ]}))
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", reg)
    rc = service.install()
    assert rc == 1
