"""Tests for the v0.5 crowd-sourced gauge network (the network effect)."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np

from tideglass.marea.crowdsource import GaugeStore, read_csv
from tideglass.marea.spatial import harmonize


def _make_csv(path: str, seed: int, n: int = 240) -> None:
    rng = np.random.default_rng(seed)
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    times = [t0 + timedelta(hours=h) for h in range(n)]
    # Two nearby gauges share a dominant semidiurnal tide + different noise.
    base = 0.8 * np.cos(np.linspace(0, 4 * np.pi, n))
    heights = base + 0.15 * rng.standard_normal(n)
    with open(path, "w") as fh:
        fh.write("time,height\n")
        for t, h in zip(times, heights):
            fh.write(f"{t.isoformat()},{h:.4f}\n")


def test_read_csv_skips_header():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "g.csv")
        _make_csv(p, 1)
        times, heights = read_csv(p)
        assert len(times) == 240
        assert all(np.isfinite(h) for h in heights)


def test_store_add_and_count():
    with tempfile.TemporaryDirectory() as d:
        gs = GaugeStore(d)
        p1 = os.path.join(d, "a.csv")
        p2 = os.path.join(d, "b.csv")
        _make_csv(p1, 1)
        _make_csv(p2, 2)
        gs.add_csv(p1, "a", -122.34, 47.60)
        gs.add_csv(p2, "b", -122.31, 47.59)
        assert gs.count == 2
        names = {g.station for g in gs.stations()}
        assert names == {"a", "b"}


def test_network_effect_requires_two():
    with tempfile.TemporaryDirectory() as d:
        gs = GaugeStore(d)
        p = os.path.join(d, "a.csv")
        _make_csv(p, 1)
        gs.add_csv(p, "a", -122.34, 47.60)
        try:
            gs.network_effect()
            raise AssertionError("should require >=2 gauges")
        except ValueError:
            pass


def test_network_effect_runs():
    with tempfile.TemporaryDirectory() as d:
        gs = GaugeStore(d)
        for i, name in enumerate(("a", "b", "c")):
            p = os.path.join(d, f"{name}.csv")
            _make_csv(p, i + 1)
            gs.add_csv(p, name, -122.34 + i * 0.01, 47.60)
        eff = gs.network_effect()
        assert eff["n_total"] == 3
        assert len(eff["steps"]) == 2  # k=2, k=3
        # every prefix must explain a sensible fraction of variance
        assert all(s["total_explained"] >= 0.5 for s in eff["steps"])


def test_harmonize_common_grid():
    with tempfile.TemporaryDirectory() as d:
        gs = GaugeStore(d)
        for i, name in enumerate(("a", "b")):
            p = os.path.join(d, f"{name}.csv")
            _make_csv(p, i + 1)
            gs.add_csv(p, name, -122.34, 47.60)
        res = gs.harmonize()
        assert res.stations == ["a", "b"]
        assert "a" in res.reconstructed and "b" in res.reconstructed
