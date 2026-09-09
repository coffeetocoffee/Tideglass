"""Tests for v0.4 Web API (stdlib HTTP) and TUI dashboard."""

import json
import threading
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tideglass.cli import main as cli_main
from tideglass.marea import constituents as C
from tideglass.marea.model import TideModel
from tideglass.marea.solver import rad_per_hour
from tideglass.tui import build_dashboard
from tideglass.web import make_handler

UTC = timezone.utc
TRUE = {"M2": (1.0, 0.5), "S2": (0.30, -1.2), "K1": (0.15, 0.0), "O1": (0.10, 2.8)}
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _hours(times):
    return [(t - T0).total_seconds() / 3600.0 for t in times]


def _synthetic(times, seed=0):
    t = np.asarray(_hours(times), dtype=float)
    y = np.full(t.size, 0.7)
    for name, (amp, phi) in TRUE.items():
        w = float(rad_per_hour(C.speed(C.get(name))))
        y = y + amp * np.cos(t * w - phi)
    return y + np.random.default_rng(seed).normal(0, 0.01, size=t.size)


def _hourly(start, n):
    return [start + timedelta(hours=h) for h in range(n)]


@pytest.fixture
def store(tmp_path):
    import json
    import os
    train = _hourly(T0, 60 * 24)
    model = TideModel.fit(train, _synthetic(train, seed=3), alpha=1e-4, station="API")
    os.makedirs(tmp_path / ".tg", exist_ok=True)
    with open(tmp_path / ".tg" / "API.json", "w") as fh:
        json.dump(model.to_artifact(), fh)
    return str(tmp_path / ".tg")


def _run_server_in_thread(store):
    from http.server import ThreadingHTTPServer
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, port


def test_web_predict_and_advise(store):
    httpd, port = _run_server_in_thread(store)
    base = f"http://127.0.0.1:{port}"
    try:
        with urllib.request.urlopen(
            f"{base}/predict?station=API&date=2024-02-01&days=1"
        ) as r:
            pred = json.load(r)
        assert pred["station"] == "API"
        assert len(pred["height"]) == 24
        assert pred["lower"][0] <= pred["height"][0] <= pred["upper"][0]
        with urllib.request.urlopen(
            f"{base}/advise?station=API&date=2024-02-01&days=1"
        ) as r:
            adv = json.load(r)
        assert "harvest_windows" in adv and "rip" in adv
    finally:
        httpd.shutdown()


def test_tui_dashboard_renders(store):
    train = _hourly(T0, 60 * 24)
    model = TideModel.fit(train, _synthetic(train, seed=3), alpha=1e-4, station="API")
    dash = build_dashboard(model, _hourly(T0, 24))
    assert "station API" in dash
    assert "rip risk" in dash
    assert "harvest windows" in dash
    assert "species exposure" in dash


def test_cli_tui_runs(store, capsys):
    assert cli_main(["tui", "API", "2024-02-01", "--store", store]) == 0
    out = capsys.readouterr().out
    assert "station API" in out
