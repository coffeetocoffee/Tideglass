"""v3.0 — the living world model.

Fuses kriged regional fields with the federated peer network and a satellite
altimetry virtual peer into a continuous global tide+surge field that predicts at
any lon/lat with no local gauge.
"""

import csv
import json
from datetime import datetime, timedelta, timezone

import numpy as np

from tideglass.marea import constituents as CON
from tideglass.marea.federation import SurgeResponse
from tideglass.marea.model import Fit, TideModel
from tideglass.marea.world import (
    AltimetryTrack,
    WorldModel,
    WorldPrediction,
)

Z95 = 1.96


def _make_model(station, consts, H0, coef_map):
    """Build a TideModel with known (a, b) coefficients for a synthetic test."""
    consts = [CON.get(c) for c in consts]
    coef = np.zeros(1 + 2 * len(consts))
    coef[0] = H0
    fits = []
    for j, c in enumerate(consts):
        a = coef_map[c.name][0]
        b = coef_map[c.name][1]
        coef[1 + 2 * j] = a
        coef[2 + 2 * j] = b
        amp = float(np.hypot(a, b))
        phase = float(np.degrees(np.atan2(b, a)) % (2 * np.pi))
        fits.append(Fit(c.name, amp, phase, 0.0))
    cov = np.zeros((coef.size, coef.size))
    for j in range(len(consts)):
        cov[1 + 2 * j, 1 + 2 * j] = 1e-4
        cov[2 + 2 * j, 2 + 2 * j] = 1e-4
    return TideModel(consts, coef, cov, 1e-4, fits, station=station,
                    meta={"n_obs": 1000, "rmse": 0.01})


def _true_tide(coef_map, H0, consts, times):
    """Evaluate the synthetic harmonic field directly (ground truth)."""
    A = np.zeros((len(times), 1 + 2 * len(consts)))
    consts = [CON.get(c) for c in consts]
    from tideglass.marea.model import _basis_matrix

    A = _basis_matrix(consts, times)
    coef = np.zeros(1 + 2 * len(consts))
    coef[0] = H0
    for j, c in enumerate(consts):
        coef[1 + 2 * j] = coef_map[c.name][0]
        coef[2 + 2 * j] = coef_map[c.name][1]
    return A @ coef


def _stations():
    consts = ["M2", "S2", "K1", "O1"]
    # Coefficients vary linearly across space so kriging can interpolate them.
    stations = {
        "A": (-1.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.0, 1.0),
        "D": (0.0, -1.0),
    }
    models = {}
    coords = {}
    for name, (lon, lat) in stations.items():
        cm = {
            "M2": (1.0 + 0.1 * lon, 0.5 - 0.05 * lat),
            "S2": (0.4 - 0.05 * lat, 0.2 + 0.1 * lon),
            "K1": (0.3, 0.1 * lon),
            "O1": (0.2, -0.1 * lat),
        }
        H0 = 1.0 + 0.05 * lat
        models[name] = _make_model(name, consts, H0, cm)
        coords[name] = (lon, lat)
    return models, coords, consts


def _times(day="2024-01-01", hours=24):
    base = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return [base + timedelta(hours=h) for h in range(hours)]


def test_world_predict_matches_station_model():
    models, coords, consts = _stations()
    wm = WorldModel(models, coords)
    times = _times()
    # At station A's exact location the world field should reproduce A's own tide.
    pa = wm.predict(-1.0, 0.0, times)
    # A's coefficients are (lon=-1, lat=0) -> M2=(0.9,0.5) etc.
    cm = {"M2": (1.0 + 0.1 * -1.0, 0.5 - 0.05 * 0.0),
          "S2": (0.4 - 0.05 * 0.0, 0.2 + 0.1 * -1.0),
          "K1": (0.3, 0.1 * -1.0),
          "O1": (0.2, -0.1 * 0.0)}
    truth = _true_tide(cm, 1.0, consts, times)
    # At a gauge location the world field is a (near-exact) kriged blend of the
    # network, so it should coincide with the station's own tide to ~cm.
    assert np.allclose(pa.tide_mean, truth, atol=0.02)


def test_world_cold_start_interpolates_coefficients():
    models, coords, consts = _stations()
    wm = WorldModel(models, coords)
    times = _times()
    # Cold-start point (0, 0) is the centre: coefficients are the spatial mean.
    pc = wm.predict(0.0, 0.0, times)
    truth_cm = {"M2": (1.0, 0.5), "S2": (0.4, 0.2),
                "K1": (0.3, 0.0), "O1": (0.2, 0.0)}
    truth = _true_tide(truth_cm, 1.0, consts, times)
    # Kriging interpolates the spatial field; expect close to the centre truth.
    assert abs(float(np.mean(pc.tide_mean - truth))) < 0.05


def test_world_offshore_band_wider_than_at_station():
    models, coords, _ = _stations()
    wm = WorldModel(models, coords)
    times = _times()
    pa = wm.predict(-1.0, 0.0, times)
    po = wm.predict(5.0, 5.0, times)  # far offshore, no nearby gauge
    band_station = float(np.mean(pa.upper - pa.lower))
    band_offshore = float(np.mean(po.upper - po.lower))
    assert band_offshore > band_station


def test_world_predict_field_shape():
    models, coords, _ = _stations()
    wm = WorldModel(models, coords)
    lons = [-1.0, 0.0, 1.0]
    lats = [0.0, 0.5, -0.5]
    times = _times(hours=6)
    field = wm.predict_field(lons, lats, times)
    assert field.shape == (3, len(times))


def test_world_altimetry_virtual_peer_constrains_field():
    models, coords, consts = _stations()
    times = _times(hours=12)
    # An offshore point with a known constant non-tidal offset of +0.3 m.
    lo, la = 0.5, 0.5
    cm = {"M2": (1.05, 0.475), "S2": (0.375, 0.25),
          "K1": (0.3, 0.05), "O1": (0.2, -0.05)}
    truth = _true_tide(cm, 1.025, consts, times)
    lons = np.full(len(times), lo)
    lats = np.full(len(times), la)
    ssh = truth + 0.3  # constant non-tidal anomaly
    alt = AltimetryTrack(lons, lats, list(times), ssh)
    wm = WorldModel(models, coords, altimetry=alt)
    pred = wm.predict(lo, la, times)
    # The altimetry residual (ssh - tide) should be recovered as the correction.
    assert abs(float(pred.altimetry_correction[0]) - 0.3) < 0.15


def test_world_surge_forecast_from_blended_response():
    models, coords, _ = _stations()
    # A simple surge response: surge = intercept + stress_u * tau_u, no lag.
    sr = SurgeResponse(
        intercept=0.1, stress_u=0.5, stress_v=0.0, barometer=-0.01,
        lag_hours=0, sigma=0.05, r_squared=0.9, n=200, p_ref=1013.25)
    wm = WorldModel(models, coords, surges={"A": sr})
    times = _times(hours=6)
    # Constant eastward wind stress feature -> constant surge.
    ws = np.full(len(times), 5.0)
    wd = np.full(len(times), 270.0)  # from west -> u stress = -(w^2)*sin(270)= +w^2
    pr = np.full(len(times), 1013.25)
    # tau_u = -(w^2)*sin(270deg) = -(25)*(-1) = 25
    expected = sr.intercept + sr.stress_u * 25.0
    pred = wm.predict(0.0, 0.0, times, met=(ws, wd, pr))
    assert pred.surge_mean is not None
    assert np.allclose(pred.surge_mean, expected, atol=1e-6)
    assert np.allclose(pred.mean, pred.tide_mean + pred.surge_mean, atol=1e-9)


def test_world_no_surge_returns_none_mean():
    models, coords, _ = _stations()
    wm = WorldModel(models, coords)  # no surge responses
    times = _times()
    pred = wm.predict(0.0, 0.0, times)
    assert pred.surge_mean is None
    assert pred.surge_sigma is None


def test_world_from_artifacts_roundtrip(tmp_path):
    models, coords, _ = _stations()
    store = tmp_path / "store"
    store.mkdir()
    for name, m in models.items():
        (store / f"{name}.json").write_text(json.dumps(m.to_artifact()))
    coords_csv = store / "coords.csv"
    with open(coords_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["station", "lon", "lat"])
        for name, (lo, la) in coords.items():
            w.writerow([name, lo, la])
    wm = WorldModel.from_artifacts(str(store), coords=str(coords_csv))
    assert len(wm.models) == len(models)
    times = _times()
    pred = wm.predict(0.0, 0.0, times)
    assert isinstance(pred, WorldPrediction)
    assert pred.mean.shape == (len(times),)


def test_world_cli_smoke(tmp_path, capsys):
    from tideglass.cli import main

    models, coords, _ = _stations()
    store = tmp_path / "store"
    store.mkdir()
    for name, m in models.items():
        (store / f"{name}.json").write_text(json.dumps(m.to_artifact()))
    coords_csv = store / "coords.csv"
    with open(coords_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["station", "lon", "lat"])
        for name, (lo, la) in coords.items():
            w.writerow([name, lo, la])
    rc = main(["world", "0.0", "0.0", "2024-01-01",
               "--store", str(store), "--coords", str(coords_csv), "--days", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "world" in out or "lon=" in out
    assert "total_m" in out


def test_world_cli_with_met(tmp_path, capsys):
    from tideglass.cli import main

    models, coords, _ = _stations()
    store = tmp_path / "store"
    store.mkdir()
    for name, m in models.items():
        (store / f"{name}.json").write_text(json.dumps(m.to_artifact()))
    coords_csv = store / "coords.csv"
    with open(coords_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["station", "lon", "lat"])
        for name, (lo, la) in coords.items():
            w.writerow([name, lo, la])
    # A blended surge response so the CLI can produce a forecast.
    sr = SurgeResponse(intercept=0.0, stress_u=0.2, stress_v=0.0, barometer=-0.01,
                       lag_hours=0, sigma=0.05, r_squared=0.9, n=200, p_ref=1013.25)
    (store / "A.surge.json").write_text(json.dumps(sr.to_dict()))

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    met_csv = store / "met.csv"
    with open(met_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "wind_speed", "wind_dir", "pressure"])
        for h in range(24):
            t = base + timedelta(hours=h)
            w.writerow([t.isoformat(), "5.0", "270.0", "1013.25"])
    rc = main(["world", "0.0", "0.0", "2024-01-01",
               "--store", str(store), "--coords", str(coords_csv),
               "--met", str(met_csv)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "surge" in out
