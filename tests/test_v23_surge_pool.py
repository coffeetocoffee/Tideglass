"""v2.3 tests: shared surge response, estuarine river-discharge coupling, and
federated extremes (pooled GPD tail fits across deployments)."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np

from tideglass.marea import constituents as CON
from tideglass.marea import federation
from tideglass.marea.extremes import GPD
from tideglass.marea.federation import SurgeResponse
from tideglass.marea.met import (
    DischargeCoupling,
    MetResponse,
    _rolling_mean,
    learn_discharge_coupling,
    read_discharge_csv,
    stress_features,
    write_met_csv,
)
from tideglass.marea.surge import AR1

UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)


# --- shared helpers (mirrors the rest of the suite) --------------------------

def _hourly(start, n):
    return [start + timedelta(hours=i) for i in range(n)]


def _hours(times):
    return np.array([(t - T0).total_seconds() / 3600.0 for t in times], dtype=float)


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


def _met(n, seed=3):
    h = np.arange(n, dtype=float)
    rng = np.random.default_rng(seed)
    ws = np.clip(5.0 + 4.0 * np.sin(2 * np.pi * h / 72.0)
                 + 2.0 * np.sin(2 * np.pi * h / 13.0)
                 + rng.normal(0.0, 0.3, n), 0.1, None)
    wd = 200.0 + 50.0 * np.sin(2 * np.pi * h / 240.0) + rng.normal(0.0, 3.0, n)
    pr = 1013.0 + 15.0 * np.sin(2 * np.pi * h / 200.0 + 1.0) \
        + rng.normal(0.0, 0.5, n)
    return ws, wd, pr


def _truth_surge(ws, wd, pr, lag=2, c0=0.02, su=0.0041, sv=0.0030,
                 baro=-0.0099, noise=0.0, seed=11):
    su_v, sv_v = stress_features(ws, wd)
    dp = pr - 1013.25
    su_l = np.concatenate([np.full(lag, su_v[0]), su_v[:-lag]])
    sv_l = np.concatenate([np.full(lag, sv_v[0]), sv_v[:-lag]])
    dp_l = np.concatenate([np.full(lag, dp[0]), dp[:-lag]])
    s = c0 + su * su_l + sv * sv_l + baro * dp_l
    if noise > 0:
        s = s + np.random.default_rng(seed).normal(0.0, noise, s.size)
    return s


def _write_csv(path, times, heights):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "height"])
        for t, hh in zip(times, heights):
            w.writerow([t.isoformat(), f"{hh:.4f}"])


def _seed_store(root, stations):
    """Populate a GaugeStore with dense stations so models() fits them."""
    from tideglass.cli import main

    root.mkdir(parents=True, exist_ok=True)
    for name, lon, lat, ph in stations:
        ts = _hourly(T0, 240)
        hh = _tide(ts, {"M2": (1.0, ph), "K1": (0.6, ph)}, noise=0.01,
                  seed=hash(name) % 1000)
        _write_csv(str(root / f"{name}.csv"), ts, hh)
        main(["contribute", str(root / f"{name}.csv"), str(lon), str(lat),
              "--station", name, "--store", str(root)])


# --- river-discharge coupling ------------------------------------------------

def _discharge(n, seed=7):
    h = np.arange(n, dtype=float)
    rng = np.random.default_rng(seed)
    q = 2000.0 + 800.0 * np.sin(2 * np.pi * h / 240.0) \
        + 400.0 * np.sin(2 * np.pi * h / 37.0) + rng.normal(0.0, 20.0, n)
    return np.clip(q, 200.0, None)


def test_discharge_coupling_fit_recovers_truth(tmp_path):
    n = 24 * 30
    times = _hourly(T0, n)
    q = _discharge(n)
    q_ant = _rolling_mean(q, 48)
    alpha_true, beta_true = 0.004, 0.6
    resid = alpha_true * np.maximum(q_ant, 0.0) ** beta_true
    resid = resid + np.random.default_rng(5).normal(0.0, 1e-3, n)
    dc, diag = learn_discharge_coupling(
        times, resid, times, q, tau_hours=48)
    assert dc.alpha > 0
    assert abs(dc.alpha - alpha_true) / alpha_true < 0.1
    assert abs(dc.beta - beta_true) / beta_true < 0.1
    assert 0.0 < dc.r_squared < 1.0
    assert diag["r2"] > 0.99


def test_discharge_coupling_apply_reconstructs(tmp_path):
    n = 24 * 30
    times = _hourly(T0, n)
    q = _discharge(n)
    q_ant = _rolling_mean(q, 48)
    resid = 0.004 * np.maximum(q_ant, 0.0) ** 0.6
    dc, _ = learn_discharge_coupling(times, resid, times, q, tau_hours=48)
    recon = dc.apply(times, times, q)
    assert np.sqrt(np.mean((recon - resid) ** 2)) < 0.01


def test_discharge_coupling_roundtrip():
    dc = DischargeCoupling(alpha=0.004, beta=0.6, tau_hours=48, sigma=0.05,
                           r_squared=0.3, n=500)
    dc2 = DischargeCoupling.from_dict(dc.to_dict())
    assert (dc2.alpha, dc2.beta, dc2.tau_hours, dc2.n) == (
        dc.alpha, dc.beta, dc.tau_hours, dc.n)


def test_read_discharge_csv_roundtrip(tmp_path):
    path = tmp_path / "disc.csv"
    n = 48
    times = _hourly(T0, n)
    q = _discharge(n)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "discharge"])
        for t, v in zip(times, q):
            w.writerow([t.isoformat(), f"{v:.2f}"])
    rt, rdq = read_discharge_csv(str(path))
    assert len(rt) == n
    assert np.allclose(rdq, q, atol=0.01)


def test_discharge_coupling_rejects_short_record():
    times = _hourly(T0, 20)
    q = _discharge(20)
    resid = 0.004 * q ** 0.6
    try:
        learn_discharge_coupling(times, resid, times, q, tau_hours=48)
        assert False, "expected ValueError for a too-short record"
    except ValueError:
        pass


# --- shared surge response ---------------------------------------------------

def _sample_met_response():
    return MetResponse(intercept=0.1, stress_u=0.0041, stress_v=0.0030,
                      barometer=-0.0099, lag_hours=2, sigma=0.05,
                      r_squared=0.9, n=500, p_ref=1013.25)


def test_surge_response_roundtrip():
    ar = AR1(phi=0.7, sigma_eps=0.01, sigma=0.03)
    dc = DischargeCoupling(alpha=0.004, beta=0.6, tau_hours=48, sigma=0.05,
                           r_squared=0.3, n=500)
    sr = SurgeResponse.from_met(_sample_met_response(), ar=ar, discharge=dc)
    sr2 = SurgeResponse.from_dict(sr.to_dict())
    assert sr2.intercept == sr.intercept
    assert sr2.ar_phi == sr.ar_phi
    assert sr2.discharge["alpha"] == dc.alpha
    assert sr2.n == 500


def test_surge_response_forecast_matches_met():
    n = 24 * 10
    times = _hourly(T0, n)
    ws, wd, pr = _met(n)
    ar = AR1(phi=0.7, sigma_eps=0.01, sigma=0.03)
    resp = _sample_met_response()
    sr = SurgeResponse.from_met(resp, ar=ar)
    fc_sr = sr.forecast(times, ws, wd, pr, ar=ar)
    fc_mr = resp.forecast(times, ws, wd, pr, ar=ar)
    assert np.allclose(fc_sr.mean, fc_mr.mean, atol=1e-9)
    assert np.allclose(fc_sr.sigma, fc_mr.sigma, atol=1e-9)


def test_surge_response_forecast_without_ar():
    n = 24 * 10
    times = _hourly(T0, n)
    ws, wd, pr = _met(n)
    resp = _sample_met_response()
    sr = SurgeResponse.from_met(resp)  # no AR tracker carried
    fc_sr = sr.forecast(times, ws, wd, pr)
    fc_mr = resp.forecast(times, ws, wd, pr)
    assert np.allclose(fc_sr.mean, fc_mr.mean, atol=1e-9)


# --- federated extremes: pooled GPD ------------------------------------------

def test_pool_gpd_sums_peaks_and_weights():
    g1 = GPD(loc=0.5, scale=0.20, shape=0.10, n_peaks=50)
    g2 = GPD(loc=0.6, scale=0.25, shape=0.15, n_peaks=100)
    p = federation.pool_gpd([g1, g2])
    assert p.n_peaks == 150
    assert p.loc == 0.6
    assert abs(p.shape - (0.10 * 50 + 0.15 * 100) / 150) < 1e-9
    assert abs(p.scale - (0.20 * 50 + 0.25 * 100) / 150) < 1e-9


def test_pool_gpd_single_is_identity():
    g1 = GPD(loc=0.5, scale=0.20, shape=0.10, n_peaks=50)
    p = federation.pool_gpd([g1])
    assert p.n_peaks == 50
    assert p.loc == 0.5 and p.shape == 0.10 and p.scale == 0.20


def test_gpd_return_level_consistency():
    g1 = GPD(loc=0.5, scale=0.20, shape=0.10, n_peaks=50)
    d = {"loc": g1.loc, "scale": g1.scale, "shape": g1.shape}
    rate = 3.0
    for T in (1, 10, 100):
        pooled = federation._gpd_return_level(d, T, rate)
        direct = g1.return_level(T, rate)
        assert abs(pooled - direct) < 1e-9


def test_surge_blend_weights_by_n():
    a = SurgeResponse(intercept=0.0, stress_u=0.004, stress_v=0.003,
                      barometer=-0.0099, lag_hours=2, sigma=0.05,
                      r_squared=0.9, n=100, p_ref=1013.25)
    b = SurgeResponse(intercept=1.0, stress_u=0.004, stress_v=0.003,
                      barometer=-0.0099, lag_hours=2, sigma=0.05,
                      r_squared=0.9, n=300, p_ref=1013.25)
    blended = federation._surge_blend([a, b])
    # 75% weight on b: (0.0*100 + 1.0*300)/400 == 0.75
    assert abs(blended.intercept - 0.75) < 1e-9


# --- federation carries surge + extremes -------------------------------------

def _write_v23_artifacts(store, station):
    sr = SurgeResponse.from_met(_sample_met_response(),
                                ar=AR1(phi=0.7, sigma_eps=0.01, sigma=0.03))
    with open(store / f"{station}.surge.json", "w") as fh:
        json.dump(sr.to_dict(), fh, indent=2)
    gpd = GPD(loc=0.5, scale=0.2, shape=0.1, n_peaks=60)
    with open(store / f"{station}.gpd.json", "w") as fh:
        json.dump({"gpd": {"loc": gpd.loc, "scale": gpd.scale,
                           "shape": gpd.shape, "n_peaks": gpd.n_peaks},
                   "rate": 3.0}, fh, indent=2)


def test_peer_bundle_carries_surge_and_extremes(tmp_path):
    crowd = tmp_path / "crowd"
    _seed_store(crowd, [("alpha", -122.3, 37.8, 0.0),
                        ("beta", -122.4, 37.9, 0.2)])
    for s in ("alpha", "beta"):
        _write_v23_artifacts(crowd, s)
    bundle = federation.build_peer_bundle(str(crowd), "peerA")
    assert len(bundle.stations) == 2
    for c in bundle.stations.values():
        assert c.surge is not None
        assert c.extremes is not None
    # round-trip preserves the attachments
    roundtripped = federation.PeerBundle.from_dict(bundle.to_dict())
    for c in roundtripped.stations.values():
        assert c.surge is not None
        assert c.extremes is not None


def test_global_federation_surge_and_extremes_summary(tmp_path):
    crowd = tmp_path / "crowd"
    _seed_store(crowd, [("alpha", -122.3, 37.8, 0.0),
                        ("beta", -122.4, 37.9, 0.2)])
    for s in ("alpha", "beta"):
        _write_v23_artifacts(crowd, s)
    bundle = federation.build_peer_bundle(str(crowd), "peerA")

    fed = federation.GlobalFederation(str(tmp_path / "federation"))
    fed.ingest(bundle)

    ssum = fed.surge_summary()
    assert ssum is not None
    assert ssum["n_with_surge"] == 2
    assert "blended" in ssum
    assert SurgeResponse.from_dict(ssum["blended"]) is not None

    esum = fed.extremes_summary()
    assert esum is not None
    assert esum["n_peaks_total"] == 120  # 60 + 60
    assert esum["gpd"]["loc"] == 0.5
    assert esum["rate_per_year"] == 3.0


def test_extremes_summary_none_when_absent(tmp_path):
    crowd = tmp_path / "crowd"
    _seed_store(crowd, [("alpha", -122.3, 37.8, 0.0)])
    # no .gpd.json files -> nothing to pool
    bundle = federation.build_peer_bundle(str(crowd), "peerA")
    fed = federation.GlobalFederation(str(tmp_path / "federation"))
    fed.ingest(bundle)
    assert fed.surge_summary() is None
    assert fed.extremes_summary() is None


# --- CLI smoke ---------------------------------------------------------------

def test_cli_surge_writes_surge_and_discharge(tmp_path, capsys):
    from tideglass.cli import main

    n = 24 * 45
    times = _hourly(T0, n)
    ws, wd, pr = _met(n)
    obs = _tide(times, {"M2": (1.0, 0.0), "S2": (0.4, 0.0), "K1": (0.6, 0.0)}) \
        + _truth_surge(ws, wd, pr, noise=0.005, seed=21)
    obs_csv = tmp_path / "obs.csv"
    _write_csv(str(obs_csv), times, obs)
    met_csv = tmp_path / "met.csv"
    write_met_csv(list(zip(times, ws, wd, pr)), str(met_csv))

    # estuary discharge: correlated with the residual tail
    q = _discharge(n)
    disc_csv = tmp_path / "disc.csv"
    with open(disc_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "discharge"])
        for t, v in zip(times, q):
            w.writerow([t.isoformat(), f"{v:.2f}"])

    rc = main(["surge", str(obs_csv), str(met_csv),
               "--discharge", str(disc_csv), "--station", "sg",
               "--store", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "discharge coupling" in out
    assert (tmp_path / "sg.surge.json").exists()
    assert (tmp_path / "sg.met.json").exists()
    with open(tmp_path / "sg.surge.json") as fh:
        sr = SurgeResponse.from_dict(json.load(fh))
    assert sr.discharge is not None


def test_cli_extremes_writes_gpd_and_pool(tmp_path, capsys):
    from tideglass.cli import main

    # long record for the local GPD fit (needs enough declustered peaks)
    n = 24 * 400
    times = _hourly(T0, n)
    ws, wd, pr = _met(n)
    obs = _tide(times, {"M2": (1.0, 0.0), "S2": (0.4, 0.0), "K1": (0.6, 0.0)}) \
        + _truth_surge(ws, wd, pr, noise=0.005, seed=21)
    obs_csv = tmp_path / "obs.csv"
    _write_csv(str(obs_csv), times, obs)

    # federation store with a peer carrying extremes data
    crowd = tmp_path / "crowd"
    _seed_store(crowd, [("alpha", -122.3, 37.8, 0.0)])
    _write_v23_artifacts(crowd, "alpha")
    bundle = federation.build_peer_bundle(str(crowd), "peerA")
    fedstore = tmp_path / "federation"
    federation.GlobalFederation(str(fedstore)).ingest(bundle)

    rc = main(["extremes", str(obs_csv), "--station", "ext",
               "--store", str(tmp_path), "--pool",
               "--pool-store", str(fedstore)])
    out = capsys.readouterr().out
    assert rc == 0
    assert (tmp_path / "ext.gpd.json").exists()
    assert "pooled" in out
    assert "peaks total" in out
