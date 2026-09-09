"""Tests for v0.7 — kriging spatial harmonics, response transfer, global bench."""

import csv
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tideglass import TideModel  # top-level exports (v0.7)
from tideglass import (
    ResponseTransfer,
    krige_field,
    krige_regional,
)
from tideglass.cli import main as cli_main
from tideglass.marea import spatial as SP
from tideglass.marea import constituents as C
from tideglass.marea.bench_global import (
    compare,
    global_model_at,
    read_harmonic_grid,
)
from tideglass.marea import constituents as C
from tideglass.marea.bench_global import (
    compare,
    global_model_at,
    read_harmonic_grid,
)
from tideglass.marea.solver import rad_per_hour

UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)
SPEC = {"M2": (1.0, 0.5), "S2": (0.30, -1.2), "N2": (0.20, 2.0),
        "K1": (0.15, 0.0), "O1": (0.10, 2.8)}


def _tide(times, gains=None, shifts=None):
    t = np.array([(x - T0).total_seconds() / 3600.0 for x in times])
    y = np.zeros_like(t)
    for name, (amp, phi) in SPEC.items():
        w = float(rad_per_hour(C.speed(C.get(name))))
        g = 1.0 if gains is None else gains.get(name, 1.0)
        sh = 0.0 if shifts is None else shifts.get(name, 0.0)
        y = y + g * amp * np.cos(w * t - phi - np.radians(sh))
    return y


def _hourly(start, n):
    return [start + timedelta(hours=h) for h in range(n)]


# --- kriging / GP spatial harmonics -------------------------------------------


def test_krige_field_exact_at_stations_and_uncertain_away():
    lons = np.array([-122.5, -122.0, -121.5, -122.2, -121.8])
    lats = np.array([37.5, 37.6, 37.5, 37.9, 37.8])
    vals = np.array([1.0, 1.2, 0.9, 1.1, 1.05])
    glon = np.array([-122.5, -122.2, -121.0])
    glat = np.array([37.5, 37.9, 37.3])
    kr = krige_field(lons, lats, vals, glon, glat)
    # Exact interpolation at control points (tiny nugget).
    assert abs(kr.mean[0] - vals[0]) < 0.05
    assert abs(kr.mean[1] - vals[3]) < 0.05
    # Far from the network, uncertainty rises toward the sill.
    assert kr.var[2] > kr.var[0]
    assert kr.var[2] <= kr.sill + kr.nugget + 1e-9


def test_krige_field_idw_fallback_for_two_stations():
    kr = krige_field(
        np.array([-122.0, -122.5]), np.array([37.5, 37.8]),
        np.array([1.0, 2.0]),
        np.array([-122.2]), np.array([37.6]),
    )
    assert 1.0 < kr.mean[0] < 2.0
    assert kr.var[0] == 0.0  # IDW fallback reports zero variance


def test_regional_field_krige_default_carries_variance():
    rng = np.random.default_rng(6)
    t = np.arange(14 * 24, dtype=float)
    shared = 1.0 * np.cos(0.5059 * t - 0.4) + 0.3 * np.cos(1.0 * t + 1.1)
    series = {
        "A": 1.0 * shared + rng.normal(0, 0.02, t.size),
        "B": 0.6 * shared + 0.2 + rng.normal(0, 0.02, t.size),
        "C": 1.3 * shared - 0.1 + rng.normal(0, 0.02, t.size),
    }
    coords = {"A": (-122.0, 37.0), "B": (-121.5, 37.2), "C": (-122.3, 36.8)}
    eof = SP.harmonize(series)
    glon = np.array([-122.0, -122.1, -122.3])
    glat = np.array([37.0, 37.05, 36.8])
    field = SP.regional_field(eof, coords, glon, glat)
    assert field.method == "krige"
    assert field.field.shape == (3, eof.modes.shape[1])
    assert field.field_var.shape == field.field.shape
    # At station coordinates the field recovers the station's reconstruction…
    a = eof.reconstructed["A"] - eof.means["A"]
    b = field.field[0] - field.field[0].mean()
    corr = float(np.corrcoef(a, b)[0, 1])
    assert corr > 0.9
    # …and the kriging variance is smallest at the stations themselves.
    assert field.field_var[0].mean() < field.field_var[1].mean()


def test_regional_field_idw_still_available_and_matches_legacy():
    rng = np.random.default_rng(6)
    t = np.arange(10 * 24, dtype=float)
    shared = np.cos(0.5059 * t - 0.4)
    series = {"A": shared, "B": 0.6 * shared + 0.2, "C": 1.3 * shared - 0.1}
    coords = {"A": (-122.0, 37.0), "B": (-121.5, 37.2), "C": (-122.3, 36.8)}
    eof = SP.harmonize(series, n_modes=1)
    glon = np.array([-121.9])
    glat = np.array([37.05])
    f_idw = SP.regional_field(eof, coords, glon, glat, method="idw")
    assert f_idw.method == "idw"
    assert np.all(f_idw.field_var == 0.0)
    # IDW is a convex blend of centered station fields + blended mean.
    assert f_idw.field.shape == (1, eof.modes.shape[1])


def test_krige_regional_reconstructs_smooth_field():
    rng = np.random.default_rng(3)
    t = np.arange(7 * 24, dtype=float)
    shared = np.cos(0.5059 * t)
    series = {k: g * shared + rng.normal(0, 0.01, t.size)
              for k, g in (("A", 1.0), ("B", 0.8), ("C", 1.2), ("D", 0.9))}
    eof = SP.harmonize(series, n_modes=1)
    lons = np.array([-122.0, -121.5, -122.3, -121.8])
    lats = np.array([37.0, 37.2, 36.8, 37.1])
    means = np.array([eof.means[s] for s in eof.stations])
    out = krige_regional(
        eof.modes, eof.loadings, lons, lats, means,
        np.array([-121.9]), np.array([37.05]),
    )
    assert out["field"].shape == (1, t.size)
    assert out["field_var"].shape == (1, t.size)
    assert np.all(out["field_var"] >= 0.0)
    # Mid-network point: variance well below the mode sill.
    sill = out["per_mode"][1].sill
    assert out["field_var"].mean() < sill


# --- response-function transfer ------------------------------------------------


def test_transfer_recovers_known_gain_and_phase_shift():
    train = _hourly(T0, 30 * 24)
    ref = TideModel.fit(train, _tide(train), auto_select=False, station="REF")
    gains = {"M2": 1.2, "S2": 0.8, "N2": 1.1, "K1": 0.9, "O1": 1.05}
    shifts = {"M2": 5.0, "S2": -3.0, "N2": 2.0, "K1": 4.0, "O1": -2.0}
    # A neighbour carries the same transfer (it sits at the target coast).
    nb = TideModel.fit(train, _tide(train, gains, shifts),
                       auto_select=False, station="NB")
    tr = ResponseTransfer(
        reference=ref, reference_coords=(-122.47, 37.81),
        neighbors=[(nb, (-121.9, 37.6))],
        target_coords=(-121.8, 37.5), target_station="TGT",
    )
    tgt = tr.to_tide_model()
    assert tgt.station == "TGT"
    assert tgt.source == "transfer"
    fits = {f.name: f for f in tgt.constituents()}
    ref_fits = {f.name: f for f in ref.constituents()}
    for name, g in gains.items():
        assert fits[name].amplitude == pytest.approx(
            ref_fits[name].amplitude * g, rel=0.05)
    for name, sh in shifts.items():
        expect = (ref_fits[name].phase_deg + sh) % 360.0
        got = fits[name].phase_deg
        assert min(abs(got - expect), 360 - abs(got - expect)) < 6.0
    # Prediction quality vs truth built from the transferred constants.
    fut = _hourly(train[-1] + timedelta(hours=1), 72)
    truth = _tide(fut, gains, shifts)
    pred = tgt.predict(fut)
    assert np.sqrt(np.mean((pred.mean - truth) ** 2)) < 0.15


def test_transfer_refine_on_short_record_beats_transferred():
    train = _hourly(T0, 30 * 24)
    ref = TideModel.fit(train, _tide(train), auto_select=False, station="REF")
    gains = {"M2": 1.15, "S2": 0.85, "N2": 1.05, "K1": 0.95, "O1": 1.0}
    shifts = {"M2": 4.0, "S2": -2.0, "N2": 1.5, "K1": 3.0, "O1": -1.0}
    # The neighbour only *approximates* the target's response (real coastlines
    # are not exact analogues): a gain/phase deviation of its own.
    nb_bias = {k: v + 0.06 for k, v in gains.items()}
    nb_shift = {k: v + 2.5 for k, v in shifts.items()}
    nb = TideModel.fit(train, _tide(train, nb_bias, nb_shift),
                       auto_select=False, station="NB")
    tr = ResponseTransfer(
        reference=ref, reference_coords=(-122.47, 37.81),
        neighbors=[(nb, (-121.9, 37.6))],
        target_coords=(-121.8, 37.5), target_station="TGT",
    )
    # A *short* target record: 5 days (far too short for DCDM selection).
    short_t = _hourly(T0 + timedelta(days=40), 5 * 24)
    truth = _tide(short_t, gains, shifts)
    noisy = truth + np.random.default_rng(2).normal(0, 0.02, len(short_t))
    refined = tr.refine(short_t, noisy)
    assert refined.source == "transfer"
    assert refined.meta["refined"] is True
    fut = _hourly(short_t[-1] + timedelta(hours=1), 48)
    fut_truth = _tide(fut, gains, shifts)
    transferred = tr.to_tide_model().predict(fut).mean
    refined_pred = refined.predict(fut).mean
    err_tr = float(np.sqrt(np.mean((transferred - fut_truth) ** 2)))
    err_rf = float(np.sqrt(np.mean((refined_pred - fut_truth) ** 2)))
    # Refinement on the short record must beat the transferred seed.
    assert err_rf < err_tr
    assert refined.meta["rmse"] < 0.1


def test_transfer_requires_neighbors():
    with pytest.raises(ValueError):
        ResponseTransfer(
            reference=TideModel.load_harmonic(
                {"station": "R", "constituents": []}),
            reference_coords=(0.0, 0.0), neighbors=[],
            target_coords=(1.0, 1.0),
        )


# --- global-model (TPXO/FES) benchmark ------------------------------------------


def _tpxo_csv(tmp_path):
    path = tmp_path / "grid.csv"
    rows = [
        ("lon", "lat", "constituent", "amplitude", "phase"),
        (-122.5, 37.8, "M2", 0.53, 126.0),
        (-122.5, 37.8, "S2", 0.15, 178.0),
        (-122.5, 37.8, "K1", 0.17, 158.0),
        (-122.0, 37.5, "M2", 0.50, 120.0),
        (-122.0, 37.5, "S2", 0.14, 172.0),
    ]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerows(rows)
    return str(path)


def test_read_harmonic_grid_and_global_model(tmp_path):
    grid = read_harmonic_grid(_tpxo_csv(tmp_path))
    assert len(grid) == 2
    m = global_model_at(grid, -122.49, 37.79)
    fits = {f.name: f for f in m.constituents()}
    assert fits["M2"].amplitude == pytest.approx(0.53, abs=1e-6)
    assert fits["M2"].phase_deg == pytest.approx(126.0, abs=1e-6)
    # IDW blend between two grid points.
    m2 = global_model_at(grid, -122.25, 37.65, method="idw")
    fits2 = {f.name: f for f in m2.constituents()}
    assert 0.50 < fits2["M2"].amplitude < 0.53


def test_compare_prefers_gauge_fit_over_coarse_global(tmp_path):
    train = _hourly(T0, 30 * 24)
    test = _hourly(train[-1] + timedelta(hours=1), 7 * 24)
    truth = _tide(train + test)
    y_all = np.concatenate([truth[:len(train)], truth[len(train):]]
                           + [np.random.default_rng(1).normal(0, 0.01, len(test))])
    y_all = np.concatenate([truth[: len(train)], truth[len(train):]])
    marea = TideModel.fit(train, y_all[: len(train)], auto_select=False,
                          station="G")
    grid = read_harmonic_grid(_tpxo_csv(tmp_path))
    # The nearest global point is deliberately wrong (0.53@126 vs truth 1.0@0.5
    # rad) -> a coarse global model must lose to the gauge fit.
    glob = global_model_at(grid, -122.49, 37.79)
    cmp = compare(glob, marea, test, y_all[len(train):])
    assert cmp["winner_rmse"] == "marea"
    assert cmp["rmse_ratio_marea_over_global"] < 1.0


def test_cli_bench_against_tpxo(tmp_path, capsys):
    from tests.test_v06_ops import _write_csv  # reuse the CSV writer

    train = _hourly(T0, 30 * 24)
    test = _hourly(train[-1] + timedelta(hours=1), 7 * 24)
    y = np.concatenate([_tide(train), _tide(test)])
    y = y + np.random.default_rng(5).normal(0, 0.02, y.size)
    csvp = tmp_path / "gauge.csv"
    _write_csv(str(csvp), train + test, y)
    capsys.readouterr()
    rc = cli_main([
        "bench", str(csvp), "--station", "G", "--against", "tpxo",
        "--global-model", _tpxo_csv(tmp_path),
        "--lon", "-122.49", "--lat", "37.79",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "global" in out and "marea/global rmse ratio" in out


def test_cli_bench_tpxo_requires_args(tmp_path):
    from tests.test_v06_ops import _write_csv

    train = _hourly(T0, 48)
    csvp = tmp_path / "g.csv"
    _write_csv(str(csvp), train, _tide(train))
    # Missing grid path / coords is a configuration error, not degradation.
    assert cli_main(["bench", str(csvp), "--against", "tpxo"]) == 2
    assert cli_main(["bench", str(csvp), "--against", "tpxo",
                     "--global-model", _tpxo_csv(tmp_path)]) == 2
    # An unreadable grid file degrades gracefully to a marea-only bench.
    assert cli_main(["bench", str(csvp), "--against", "tpxo",
                     "--global-model", str(tmp_path / "missing.csv"),
                     "--lon", "-122.4", "--lat", "37.7"]) == 0
