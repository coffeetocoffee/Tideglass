"""Tests for the v0.5 global harmonic-database coverage (public databases)."""

from __future__ import annotations

import os
import tempfile

import numpy as np

from tideglass.marea import harmonics_db as HDB
from tideglass.marea.harmonics_db import (
    add_harmonic_file,
    benchmark_region,
    coverage_report,
    load_station,
    stations_in_region,
)


def test_coverage_report_counts():
    rep = coverage_report()
    assert rep["n_regions"] >= 5
    assert rep["n_stations"] >= 10
    assert sum(rep["regions"].values()) == rep["n_stations"]


def test_list_and_stations():
    regions = HDB.list_regions()
    assert "US West Coast" in regions
    assert "San Francisco" in stations_in_region("US West Coast")


def test_load_station_predicts_physical_range():
    model = load_station("US West Coast", "San Francisco")
    from datetime import datetime, timedelta, timezone

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    times = [base + timedelta(hours=h) for h in range(24 * 30)]
    pred = model.predict(times)
    rng = float(np.ptp(pred.mean))
    # San Francisco's tides are ~1-2 m; a sane published-harmonic model
    # must reproduce a non-trivial, finite range.
    assert 0.5 < rng < 4.0
    assert pred.lower.shape == pred.mean.shape


def test_benchmark_region_rows():
    rows = benchmark_region("Gulf of Mexico")
    assert len(rows) == len(stations_in_region("Gulf of Mexico"))
    for r in rows:
        assert r["n_constituents"] == 8
        assert r["range_m"] > 0


def test_add_harmonic_file_extends_db():
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "extra.json")
        with open(f, "w") as fh:
            fh.write('{"station": "X", "mean": 0.5, '
                     '"constituents": [{"name": "M2", "amplitude": 0.4, '
                     '"phase": 30}]}')
        name = add_harmonic_file(f, "Pacific (Asia)", 1.0, 2.0, name="Extra")
        assert name == "Extra"
        assert "Extra" in stations_in_region("Pacific (Asia)")
