"""Unit tests for the benchmark dashboard (report listing + traversal guard)."""

import json

import pytest
from fastapi.testclient import TestClient

from tokirecur.dashboard.server import create_dashboard_app


def _write_report(reports_dir, name, data):
    path = reports_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


class TestDashboardReports:
    def test_list_reports(self, tmp_path):
        _write_report(tmp_path, "a.json", {"score": {"composite": 1.0}})
        _write_report(tmp_path, "b.json", {"score": {"composite": 2.0}})
        app = create_dashboard_app(reports_dir=tmp_path)
        with TestClient(app) as c:
            r = c.get("/api/reports")
        assert r.status_code == 200
        files = [item["file"] for item in r.json()]
        assert files == ["b.json", "a.json"]  # newest/most-recent first

    def test_get_report_roundtrip(self, tmp_path):
        _write_report(tmp_path, "a.json", {"score": {"composite": 1.5}})
        app = create_dashboard_app(reports_dir=tmp_path)
        with TestClient(app) as c:
            r = c.get("/api/reports/a.json")
        assert r.status_code == 200
        assert r.json()["score"]["composite"] == 1.5

    def test_missing_report_404(self, tmp_path):
        app = create_dashboard_app(reports_dir=tmp_path)
        with TestClient(app) as c:
            r = c.get("/api/reports/does-not-exist.json")
        assert r.status_code == 404


class TestDashboardTraversal:
    """Regression: /api/reports/{name} must not escape reports_dir.

    Previously `path = reports_dir / name` accepted "../secret.json",
    resolving outside the reports directory and reading arbitrary
    files. Now the resolved path must be a descendant of reports_dir.
    """

    @pytest.mark.parametrize("name", [
        "../secret.json",
        "..%2Fsecret.json",
        "../../etc/passwd",
        "..%2F..%2Fetc%2Fpasswd",
        "%2e%2e%2fsecret.json",
    ])
    def test_traversal_blocked(self, tmp_path, name):
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        secret = tmp_path / "secret.json"
        secret.write_text(json.dumps({"leak": True}))
        app = create_dashboard_app(reports_dir=reports_dir)
        with TestClient(app) as c:
            r = c.get(f"/api/reports/{name}")
        assert r.status_code == 404, (
            f"traversal {name!r} not blocked (status {r.status_code})"
        )


class TestDashboardMalformedReports:
    """A report is a JSON file someone else wrote.

    A null where a number was expected made the dashboard's first
    arithmetic call throw, and the whole table - and the summary - went
    blank. One bad report must not take the rest with it.
    """

    def test_null_numbers_become_numbers(self, tmp_path):
        _write_report(tmp_path, "a.json", {
            "score": {"composite": None, "grade": None,
                      "cache_hit_rate_pct": None, "speedup": None},
            "metadata": {"model": None, "timestamp": None},
            "workload": {"name": None, "prompt_count": None},
        })
        app = create_dashboard_app(reports_dir=tmp_path)
        with TestClient(app) as c:
            r = c.get("/api/reports")
        assert r.status_code == 200
        item = r.json()[0]
        assert item["composite"] == 0
        assert item["speedup"] == 1.0
        assert item["grade"] == "N/A"
        assert item["model"] == "unknown"
        assert item["workload"] == "unknown"

    def test_a_bad_report_does_not_hide_the_good_ones(self, tmp_path):
        _write_report(tmp_path, "good.json", {
            "score": {"composite": 42.0},
            "metadata": {"model": "m.gguf"},
        })
        _write_report(tmp_path, "bad.json", {"score": "not-a-dict"})
        app = create_dashboard_app(reports_dir=tmp_path)
        with TestClient(app) as c:
            r = c.get("/api/reports")
        assert r.status_code == 200
        names = [item["model"] for item in r.json()]
        assert "m.gguf" in names


class TestDashboardContentSecurityPolicy:
    """Report fields are rendered by the dashboard, so a report must
    not be able to run script in the dashboard's origin."""

    def test_html_response_carries_a_csp(self, tmp_path):
        app = create_dashboard_app(reports_dir=tmp_path)
        with TestClient(app) as c:
            r = c.get("/")
        csp = r.headers.get("Content-Security-Policy", "")
        assert "script-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp
