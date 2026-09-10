"""v1.0 tests: QC pipeline + federated refits for the crowd-sourced network."""

from __future__ import annotations

import csv
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tideglass import TideModel
from tideglass.marea import constituents as CON
from tideglass.marea import federation, qc

UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _hours(times):
    return np.array(
        [(t - T0).total_seconds() / 3600.0 for t in times], dtype=float
    )


def _tide(times, consts, noise=0.0, seed=None):
    h = np.zeros(len(times))
    for name, (amp, ph) in consts.items():
        c = CON.get(name)
        w = math.radians(CON.speed(c))
        h += amp * np.cos(w * _hours(times) + math.radians(ph))
    if noise > 0:
        rng = np.random.default_rng(seed)
        h += rng.normal(0.0, noise, len(times))
    return h


def _hourly(start, n):
    return [start + timedelta(hours=i) for i in range(n)]


def _write_csv(path, times, heights):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "height"])
        for t, hh in zip(times, heights):
            w.writerow([t.isoformat(), f"{hh:.4f}"])


def _consts(m2=1.0, s2=0.4, k1=0.6, ph=0.0):
    return {"M2": (m2, ph), "S2": (s2, ph), "K1": (k1, ph)}


# -- QC: spike ----------------------------------------------------------------

def test_qc_clean_passes():
    times = _hourly(T0, 240)
    h = _tide(times, _consts(), noise=0.02, seed=1)
    rep = qc.qc_check(times, h)
    assert rep.passed
    assert rep.spike_idx == []
    assert rep.datum_shift_m is not None
    assert abs(rep.datum_shift_m) < qc.QcConfig().datum_tol_m


def test_qc_detects_spikes():
    times = _hourly(T0, 240)
    h = _tide(times, _consts(), noise=0.02, seed=1)
    for i in (50, 100, 150, 200):
        h[i] += 5.0
    rep = qc.qc_check(times, h)
    assert len(rep.spike_idx) >= 3


def test_qc_rejects_many_spikes():
    times = _hourly(T0, 240)
    h = _tide(times, _consts(), noise=0.02, seed=1)
    rng = np.random.default_rng(2)
    idx = rng.choice(len(h), size=30, replace=False)  # > 5% of 240 = 12
    h[idx] += 5.0
    rep = qc.qc_check(times, h)
    assert rep.rejected
    assert any("spike fraction" in r for r in rep.reasons)


def test_qc_rejects_nonfinite():
    times = _hourly(T0, 240)
    h = _tide(times, _consts(), noise=0.02, seed=1)
    h[10] = float("nan")
    rep = qc.qc_check(times, h)
    assert rep.rejected
    assert any("non-finite" in r for r in rep.reasons)


def test_qc_clean_interpolates():
    times = _hourly(T0, 240)
    h = _tide(times, _consts(), noise=0.02, seed=1)
    h[100] += 5.0
    h[150] += 5.0
    rep = qc.qc_check(times, h)
    t2, h2 = qc.clean_series(times, h, rep)
    assert len(h2) == len(h)
    # spiked values are replaced by interpolation (no longer 5 m away)
    assert abs(h2[100] - h[100]) > 1.0
    assert abs(h2[150] - h[150]) > 1.0


# -- QC: datum / drift --------------------------------------------------------

def test_qc_detects_datum_shift():
    times = _hourly(T0, 240)
    h = _tide(times, _consts(), noise=0.01, seed=3)
    half = len(h) // 2
    h[half:] += 0.5  # sustained 0.5 m level jump in the second half
    rep = qc.qc_check(times, h)
    assert rep.datum_shift_m is not None
    assert rep.datum_shift_m > 0.3
    assert rep.rejected


def test_qc_detects_drift():
    times = _hourly(T0, 240)
    h = _tide(times, _consts(), noise=0.01, seed=4)
    h = h + np.linspace(0.0, 1.5, len(h))  # 0.15 m/day over the record
    rep = qc.qc_check(times, h)
    assert rep.drift_slope_m_per_day is not None
    assert rep.drift_slope_m_per_day > 0.1
    assert rep.rejected


def test_qc_reference_residual():
    times = _hourly(T0, 240)
    consts = _consts(ph=20.0)
    h = _tide(times, consts, noise=0.01, seed=5)
    model = TideModel.fit(times, h, alpha=0.05, station="ref")
    pred = model.predict(times).mean
    h = np.asarray(h) + 0.4  # 0.4 m sustained offset vs the model
    rep = qc.qc_check(times, h, reference=pred)
    assert rep.datum_shift_m is not None
    assert abs(rep.datum_shift_m) > 0.3


# -- Federation ---------------------------------------------------------------

def _seed_network(tmp_path):
    fed = federation.FederatedRefit(str(tmp_path), variance_threshold=0.95)
    for name, lon, lat, ph in (
        ("alpha", -122.34, 47.60, 10.0),
        ("bravo", -122.31, 47.59, 25.0),
    ):
        times = _hourly(T0, 240)
        h = _tide(times, _consts(ph=ph), noise=0.01, seed=hash(name) % 1000)
        fed.refit(name, lon, lat, times, h)
    return fed


def test_federation_ingest_dense_then_improve():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        fed = _seed_network(d)
        # the network now has two members; add a third sensor
        times = _hourly(T0, 72)
        h = _tide(times, _consts(ph=40.0), noise=0.05, seed=99)
        rep = fed.refit("charlie", -122.28, 47.58, times, h)
        assert rep.qc.passed
        assert rep.network_size == 3
        assert rep.improved
        assert rep.shrinkage, "pooled model should carry shrinkage diagnostics"
        # persisted improved artifact round-trips as a TideModel
        model = TideModel.load_harmonic(f"{d}/charlie.json")
        assert model.station == "charlie"
        # the pooled model tracks charlie's own observations (noise 0.05 m)
        pred = model.predict(times)
        rmse = float(np.sqrt(np.mean((pred.mean - np.asarray(h)) ** 2)))
        assert rmse < 0.3


def test_federation_network_effect_grows():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        fed = _seed_network(d)
        times = _hourly(T0, 240)
        h = _tide(times, _consts(ph=40.0), noise=0.01, seed=7)
        rep = fed.refit("charlie", -122.28, 47.58, times, h)
        assert rep.network_before is not None
        assert rep.network_after is not None
        assert rep.network_before["n_total"] == 2
        assert rep.network_after["n_total"] == 3
        # the per-round network-effect metric is computed both before and after
        # the new sensor joins (the v0.5 metric, reused for the feedback loop)
        assert "gain_total_explained" in rep.network_before
        assert "gain_total_explained" in rep.network_after


def test_federation_rejects_bad_upload():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        fed = _seed_network(d)
        times = _hourly(T0, 72)
        h = _tide(times, _consts(ph=40.0), noise=0.01, seed=11)
        h[0:40] = 0.0  # flatline the whole first half
        rep = fed.refit("bad", -122.28, 47.58, times, h)
        assert rep.qc.rejected


# -- CLI smoke ----------------------------------------------------------------

def test_cli_qc(tmp_path, capsys):
    from tideglass.cli import main

    times = _hourly(T0, 240)
    h = _tide(times, _consts(), noise=0.02, seed=1)
    p = tmp_path / "clean.csv"
    _write_csv(str(p), times, h)
    rc = main(["qc", str(p), "-122.34", "47.60", "--station", "clean"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "accepted" in out


def test_cli_federate(tmp_path, capsys):
    from tideglass.cli import main

    # pre-seed the network so the new sensor can be pooled
    seed = tmp_path / "seed"
    seed.mkdir()
    for name, lon, lat, ph in (
        ("alpha", -122.34, 47.60, 10.0),
        ("bravo", -122.31, 47.59, 25.0),
    ):
        ts = _hourly(T0, 240)
        hh = _tide(ts, _consts(ph=ph), noise=0.01, seed=hash(name) % 1000)
        _write_csv(str(seed / f"{name}.csv"), ts, hh)
    for name, lon, lat in (
        ("alpha", -122.34, 47.60),
        ("bravo", -122.31, 47.59),
    ):
        main(["contribute", str(seed / f"{name}.csv"), str(lon), str(lat),
              "--station", name, "--store", str(seed)])

    new = tmp_path / "charlie.csv"
    ts = _hourly(T0, 72)
    hh = _tide(ts, _consts(ph=40.0), noise=0.05, seed=99)
    _write_csv(str(new), ts, hh)
    rc = main(["federate", str(new), "-122.28", "47.58", "--station", "charlie",
               "--store", str(seed)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "federated" in out
    assert "network-effect gain" in out
    assert (seed / "charlie.json").exists()
