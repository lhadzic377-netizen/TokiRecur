"""Toki Recur benchmark dashboard.

Loads benchmark JSON reports from a directory and serves an HTML
table with side-by-side comparison.

Usage:
    toki dashboard --reports ~/toki-reports/ --port 7778
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).parent / "static"
TEMPLATE_DIR = Path(__file__).parent / "templates"


def _num(value, default: float = 0.0) -> float:
    """A number, whatever the report actually held.

    A report is a JSON file someone else wrote. A null where a number
    was expected made the dashboard's first arithmetic call throw, and
    the whole table - and the summary - went blank.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _text(value, default: str = "") -> str:
    """A string, whatever the report actually held."""
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)


def create_dashboard_app(*, reports_dir: Path) -> FastAPI:
    """Build a FastAPI app serving the dashboard."""
    app = FastAPI(title="Toki Recur Dashboard")
    app.state.reports_dir = reports_dir

    @app.middleware("http")
    async def content_security_policy(request, call_next):
        """The dashboard renders report fields, so it needs a policy.

        The values come from report JSON: the model is a GGUF file
        name, the workload is an arbitrary string from a shared
        workload file. A script in any of them must not run in the
        dashboard's origin.
        """
        response = await call_next(request)
        response.headers.setdefault(
            "Content-Security-Policy",
            (
                "default-src 'self'; "
                "script-src 'self'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "connect-src 'self'; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            ),
        )
        return response

    @app.get("/api/reports")
    async def list_reports() -> JSONResponse:
        """List all benchmark reports in the directory."""
        if not reports_dir.exists():
            return JSONResponse([])
        reports = []
        for path in sorted(reports_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(path.read_text())
                score = data.get("score", {})
                meta = data.get("metadata", {})
                workload = data.get("workload", {})
                recycled = data.get("recycled", {})
                baseline = data.get("baseline", {})
                reports.append({
                    "file": path.name,
                    "model": _text(meta.get("model"), "unknown"),
                    "workload": _text(workload.get("name"), "unknown"),
                    "prompts": _num(workload.get("prompt_count"), 0),
                    "composite": _num(score.get("composite"), 0),
                    "grade": _text(score.get("grade"), "N/A"),
                    "ttft_improvement_pct": _num(score.get("ttft_improvement_pct"), 0),
                    "prefill_skip_pct": _num(score.get("prefill_skip_pct"), 0),
                    "hit_rate_pct": _num(score.get("cache_hit_rate_pct"), 0),
                    "throughput_gain_pct": _num(score.get("throughput_gain_pct"), 0),
                    "speedup": _num(score.get("speedup"), 1.0),
                    "timestamp": _num(meta.get("timestamp"), 0),
                    "duration_baseline_sec": _num(baseline.get("duration_sec"), 0),
                    "duration_recycled_sec": _num(recycled.get("duration_sec"), 0),
                })
            except Exception:
                # Skip malformed reports.
                continue
        return JSONResponse(reports)

    @app.get("/api/reports/{name}")
    async def get_report(name: str) -> JSONResponse:
        """Return the full JSON for one report."""
        path = (reports_dir / name).resolve()
        root = reports_dir.resolve()
        if root not in path.parents or not path.is_file():
            raise HTTPException(status_code=404, detail="not found")
        try:
            return JSONResponse(json.loads(path.read_text()))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/", response_class=HTMLResponse)
    async def root() -> str:
        template = TEMPLATE_DIR / "index.html"
        if template.exists():
            return template.read_text()
        return "<h1>Toki Recur Dashboard</h1><p>Template not found.</p>"

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="toki dashboard",
        description="Launch the Toki Recur benchmark dashboard.",
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=Path.home() / "Downloads" / "toki-reports",
        help="Directory containing benchmark JSON reports",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7778)
    args = parser.parse_args(argv)

    args.reports.mkdir(parents=True, exist_ok=True)
    print("Toki Recur Dashboard")
    print(f"  Reports dir: {args.reports}")
    print(f"  URL:         http://{args.host}:{args.port}")
    print("  Ctrl+C to stop")

    app = create_dashboard_app(reports_dir=args.reports)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
