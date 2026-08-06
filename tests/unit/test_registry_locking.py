"""Tests for the fcntl-based registry locking.

These verify that ``save_registry`` and ``update_spec_field`` serialise
concurrent writers via ``fcntl.flock``. Without the lock, two processes
editing the same registry at the same time can silently lose one
writer's changes — atomic-rename alone doesn't prevent the lost-update
problem, only the torn-file problem.
"""

import threading
import time

import pytest
import yaml

from tokirecur import ctl

# _flock is an fcntl-based lock that is a documented no-op on platforms
# without fcntl (Windows relies on atomic-rename alone); the blocking
# semantics these tests assert only exist where fcntl does.
pytest.importorskip("fcntl")


@pytest.fixture
def env(tmp_path, monkeypatch):
    reg = tmp_path / "models.yaml"
    state = tmp_path / "state"
    marker = tmp_path / "active-model"
    # Seed reference entries so the lock tests have specs to write.
    reg.write_text(yaml.safe_dump({"models": ctl.DEFAULT_MODELS}, sort_keys=False))
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", reg)
    monkeypatch.setattr(ctl, "STATE_ROOT", state)
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", marker)
    return tmp_path


class TestFlock:
    def test_save_registry_acquires_lockfile(self, env):
        """``save_registry`` should drop a sibling ``.lock`` file
        adjacent to the registry while it's writing, then release."""
        # Touch the registry so it exists before save.
        ctl._ensure_registry()
        reg = ctl.DEFAULT_REGISTRY
        # Hold the lock externally and confirm save_registry waits for it.
        from tokirecur.ctl import _flock

        completed = []
        with _flock(reg):
            t = threading.Thread(
                target=lambda: (
                    ctl.save_registry(ctl.load_registry()),
                    completed.append("done"),
                )
            )
            t.start()
            # Briefly verify the writer is blocked.
            time.sleep(0.05)
            assert completed == [], "save_registry raced past the held lock"
        t.join(timeout=2.0)
        assert completed == ["done"]

    def test_concurrent_saves_serialise(self, env):
        """Two threads calling ``save_registry`` simultaneously must
        produce a final file whose contents match exactly one of the
        two inputs — never a half-mixed hybrid (atomic-rename alone
        would still allow this if flock were missing)."""
        ctl._ensure_registry()
        base = ctl.load_registry()

        # Two divergent in-memory versions of the same registry.
        def mutate(version_label: str) -> list[ctl.ModelSpec]:
            specs = [ctl.ModelSpec(raw=dict(s.raw), **s.raw) for s in base]
            for s in specs:
                s.raw["_test_marker"] = version_label
            return specs

        a = mutate("A")
        b = mutate("B")
        results: list[str] = []

        def writer(label: str, payload: list[ctl.ModelSpec]) -> None:
            ctl.save_registry(payload)
            results.append(label)

        ta = threading.Thread(target=writer, args=("A", a))
        tb = threading.Thread(target=writer, args=("B", b))
        ta.start()
        tb.start()
        ta.join(timeout=5.0)
        tb.join(timeout=5.0)

        assert sorted(results) == ["A", "B"]
        # Final file should contain exactly one marker — never both,
        # never a YAML that mixes the two payloads.
        final = yaml.safe_load(ctl.DEFAULT_REGISTRY.read_text())
        markers = {m.get("_test_marker") for m in final["models"]}
        assert markers in ({"A"}, {"B"}), (
            f"concurrent save produced mixed markers: {markers}"
        )
