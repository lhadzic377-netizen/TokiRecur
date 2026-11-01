"""Loading one model beside another must not fail silently on OOM."""

from __future__ import annotations

import logging

from tokirecur import ctl


def _spec(name: str, **kwargs) -> ctl.ModelSpec:
    return ctl.ModelSpec(
        name=name,
        label=name,
        gguf=f"/models/{name}.gguf",
        llama_port=1,
        daemon_port=2,
        ctx_size=1,
        max_tokens=1,
        model_id=name,
        parallel=1,
        kv="f16",
        **kwargs,
    )


def test_warns_when_another_model_is_still_running(monkeypatch, caplog):
    current = _spec("current")
    other = _spec("other")
    monkeypatch.setattr(ctl, "load_registry", lambda: [current, other])
    monkeypatch.setattr(ctl, "llama_reachable", lambda spec: spec.name == "other")
    monkeypatch.setattr(ctl, "daemon_reachable", lambda spec: False)
    with caplog.at_level(logging.WARNING, logger="tokirecur.ctl"):
        ctl.warn_other_models(current)
    assert "other" in caplog.text
    assert "still running" in caplog.text


def test_does_not_warn_for_self_or_always_on(monkeypatch, caplog):
    current = _spec("current")
    always = _spec("always", always_on=True)
    monkeypatch.setattr(ctl, "load_registry", lambda: [current, always])
    monkeypatch.setattr(ctl, "llama_reachable", lambda spec: True)
    monkeypatch.setattr(ctl, "daemon_reachable", lambda spec: False)
    with caplog.at_level(logging.WARNING, logger="tokirecur.ctl"):
        ctl.warn_other_models(current)
    assert caplog.text == ""
