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


def create_dashboard_app(*, reports_dir: Path) -> FastAPI:
    """Build a FastAPI app serving the dashboard."""
    app = FastAPI(title="Toki Recur Dashboard")
    app.state.reports_dir = reports_dir

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
                    "model": meta.get("model", "unknown"),
                    "workload": workload.get("name", "unknown"),
                    "prompts": workload.get("prompt_count", 0),
                    "composite": score.get("composite", 0),
                    "grade": score.get("grade", "N/A"),
                    "ttft_improvement_pct": score.get("ttft_improvement_pct", 0),
                    "prefill_skip_pct": score.get("prefill_skip_pct", 0),
                    "hit_rate_pct": score.get("cache_hit_rate_pct", 0),
                    "throughput_gain_pct": score.get("throughput_gain_pct", 0),
                    "speedup": score.get("speedup", 1.0),
                    "timestamp": meta.get("timestamp", 0),
                    "duration_baseline_sec": baseline.get("duration_sec", 0),
                    "duration_recycled_sec": recycled.get("duration_sec", 0),
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
