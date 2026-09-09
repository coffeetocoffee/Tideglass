"""Minimal Web API for Tideglass — the v0.4 "product surface" (stdlib only).

Serves ``predict`` and ``advise`` over HTTP using :mod:`http.server`, so it runs
with zero extra dependencies. Point it at a ``--store`` directory populated by
``tideglass fit`` (or ``tideglass export``) and query it from any client.

    GET /predict?station=SF&date=2024-02-01&days=1
    GET /advise?station=SF&date=2024-02-01&days=1

Both return JSON. This is deliberately small; swap in Flask/FastAPI later
without touching the engine.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from tideglass.marea.model import TideModel
from tideglass.marine.advisor import TideAdvisor


def _build_times(date: str, days: int):
    day = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return [day + timedelta(hours=h) for h in range(max(1, int(days)) * 24)]


def _model(store: str, station: str) -> TideModel:
    path = os.path.join(store, f"{station}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no model for station {station!r} in {store!r}")
    return TideModel.load_harmonic(path)


class _Handler(BaseHTTPRequestHandler):
    store: str = ".tideglass"

    def _json(self, obj, code: int = 200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        try:
            if parsed.path == "/predict":
                self._json(self._predict(qs))
            elif parsed.path == "/advise":
                self._json(self._advise(qs))
            else:
                self._json({"error": "unknown route", "routes": ["/predict", "/advise"]}, 404)
        except (FileNotFoundError, ValueError) as exc:
            self._json({"error": str(exc)}, 400)

    def _predict(self, qs: dict) -> dict:
        station = qs.get("station", "")
        times = _build_times(qs.get("date", ""), int(qs.get("days", "1")))
        pred = _model(self.store, station).predict(times)
        return {
            "station": station,
            "time": [t.isoformat() for t in times],
            "height": [float(x) for x in pred.mean],
            "lower": [float(x) for x in pred.lower],
            "upper": [float(x) for x in pred.upper],
        }

    def _advise(self, qs: dict) -> dict:
        station = qs.get("station", "")
        times = _build_times(qs.get("date", ""), int(qs.get("days", "1")))
        advice = TideAdvisor(_model(self.store, station)).advise(times)
        peak = int(advice.rip.score.argmax())
        return {
            "station": station,
            "summary": advice.summary,
            "harvest_windows": [
                {"start": w.start.isoformat(), "end": w.end.isoformat(), "reason": w.reason}
                for w in advice.harvest
            ],
            "rip": {
                "peak_category": advice.rip.category[peak],
                "peak_score": float(advice.rip.score[peak]),
                "peak_time": advice.rip.times[peak].isoformat(),
            },
            "exposure": advice.exposure,
        }

    def log_message(self, *args):  # quiet by default
        return


def make_handler(store: str):
    """Return an HTTP request handler bound to ``store``."""
    class _Bound(_Handler):
        pass
    _Bound.store = store
    return _Bound


def run_server(store: str = ".tideglass", host: str = "127.0.0.1", port: int = 8000):
    """Start the predict/advise HTTP server (blocks)."""
    httpd = ThreadingHTTPServer((host, port), make_handler(store))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
