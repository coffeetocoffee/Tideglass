"""Tests for v0.8 — extreme value analysis (skew surge, GPD, return levels)."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tideglass import (  # top-level exports (v0.8)
    TideModel,
    flood_probability,
)
from tideglass.cli import main as cli_main
from tideglass.marea import constituents as C
from tideglass.marea import extremes as EX
from tideglass.marea.solver import rad_per_hour

UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)
SPEC = {"M2": (1.0, 0.5), "S2": (0.30, -1.2), "K1": (0.15, 0.0), "O1": (0.10, 2.8)}


def _hourly(start, n):
    return [start + timedelta(hours=h) for h in range(n)]


def _tide(times):
    t = np.array([(x - T0).total_seconds() / 3600.0 for x in times])
    y = np.zeros_like(t)
    for name, (amp, phi) in SPEC.items():
        w = float(rad_per_hour(C.speed(C.get(name))))
        y = y + amp * np.cos(w * t - phi)
    return y


def _gpd_sample(u, sigma, xi, n, seed):
    rng = np.random.default_rng(seed)
    p = rng.random(n)
    if abs(xi) < 1e-12:
        return u - sigma * np.log(1.0 - p)
    return u + sigma / xi * ((1.0 - p) ** (-xi) - 1.0)


# --- skew surge ---------------------------------------------------------------


def test_skew_surge_recovers_constant_offset():
    times = _hourly(T0, 30 * 24)
    tide = _tide(times)
    obs = tide + 0.4  # a flat 0.4 m setup on top of the tide
    sk = EX.skew_surge(times, obs, tide)
    assert len(sk.times) > 40  # ~2 high waters a day for a month
    assert np.allclose(sk.skew, 0.4, atol=1e-9)
    assert np.allclose(sk.total - sk.tide, sk.skew)


def test_skew_surge_tracks_varying_surge_at_hw():
    times = _hourly(T0, 10 * 24)
    tide = _tide(times)
    h = np.array([(t - T0).total_seconds() / 3600.0 for t in times])
    surge = 0.3 * np.sin(2 * np.pi * h / (5 * 24))  # slow 5-day surge wave
    sk = EX.skew_surge(times, tide + surge, tide)
    # At each predicted HW the skew equals the surge sampled there.
    assert np.allclose(sk.skew, surge[[times.index(t) for t in sk.times]],
                       atol=1e-9)


def test_skew_surge_no_high_waters_raises():
    times = _hourly(T0, 48)
    with pytest.raises(ValueError):
        EX.skew_surge(times, np.linspace(0, 1, 48), np.linspace(0, 1, 48))


def test_skew_surge_length_mismatch_raises():
    times = _hourly(T0, 48)
    with pytest.raises(ValueError):
        EX.skew_surge(times, np.zeros(48), np.zeros(47))


# --- declustering -------------------------------------------------------------


def test_declustering_keeps_cluster_maxima():
    t0 = T0
    times = [t0 + timedelta(hours=h) for h in (0, 6, 12, 100, 106, 300)]
    vals = np.array([0.5, 0.9, 0.4, 0.7, 0.6, 1.2])
    kt, kv = EX.decluster(times, vals, gap_hours=72.0, threshold=0.3)
    assert list(kv) == [0.9, 0.7, 1.2]
    assert [t.hour for t in kt] == [6, 4, 12]


def test_declustering_default_threshold_is_high_quantile():
    rng = np.random.default_rng(4)
    times = _hourly(T0, 200 * 24)
    vals = np.abs(rng.normal(0, 0.05, len(times)))
    vals[1000] += 1.0  # one genuine storm
    _, kv = EX.decluster(times, vals, gap_hours=72.0)
    assert kv.max() == pytest.approx(vals[1000])
    assert len(kv) < 40  # a high threshold keeps only storm-like peaks


def test_declustering_no_exceedances_raises():
    with pytest.raises(ValueError):
        EX.decluster([T0, T0 + timedelta(hours=100)], [0.2, 0.2],
                     threshold=0.5)


def test_declustering_bad_gap_raises():
    with pytest.raises(ValueError):
        EX.decluster([T0, T0], [1.0, 2.0], gap_hours=0.0, threshold=0.0)


# --- GPD fit ------------------------------------------------------------------


def test_fit_gpd_recovers_known_tail():
    peaks = _gpd_sample(0.5, 0.30, 0.20, 4000, seed=7)
    g = EX.fit_gpd(peaks, threshold=0.5)
    assert g.loc == 0.5
    assert g.n_peaks == 4000
    assert abs(g.scale - 0.30) < 0.03
    assert abs(g.shape - 0.20) < 0.06


def test_fit_gpd_recovers_exponential_tail():
    peaks = _gpd_sample(0.2, 0.25, 0.0, 4000, seed=11)
    g = EX.fit_gpd(peaks, threshold=0.2)
    assert abs(g.scale - 0.25) < 0.03
    assert abs(g.shape) < 0.05


def test_fit_gpd_default_threshold_is_median():
    peaks = _gpd_sample(0.0, 0.2, 0.1, 2000, seed=3)
    g = EX.fit_gpd(peaks)
    assert g.loc == pytest.approx(float(np.median(peaks)))
    assert g.n_peaks == 1000


def test_fit_gpd_needs_enough_peaks_raises():
    with pytest.raises(ValueError):
        EX.fit_gpd(np.array([0.1, 0.2, 0.3]), threshold=0.0)


def test_gpd_survival_and_return_level():
    g = EX.GPD(loc=0.5, scale=0.3, shape=0.0, n_peaks=100)
    assert g.survival(0.5) == 1.0
    assert g.survival(0.4) == 1.0
    assert g.survival(0.5 + 0.3) == pytest.approx(np.exp(-1.0))
    # Exponential return level: u + σ·ln(λT).
    assert g.return_level(100.0, rate_per_year=2.0) == pytest.approx(
        0.5 + 0.3 * np.log(200.0))
    # Longer return periods give higher levels.
    assert g.return_level(100.0, 2.0) > g.return_level(10.0, 2.0)


def test_gpd_heavy_tail_return_level():
    g = EX.GPD(loc=0.0, scale=0.3, shape=0.5, n_peaks=100)
    assert g.return_level(50.0, 1.0) == pytest.approx(
        0.3 / 0.5 * (50.0**0.5 - 1.0))
    assert g.return_level(50.0, 1.0) > EX.GPD(
        0.0, 0.3, 0.0, 100).return_level(50.0, 1.0)


def test_annual_rate():
    times = _hourly(T0, 2 * 365 * 24)
    assert EX.annual_rate(times, 10) == pytest.approx(
        10 / ((len(times) - 1) / 24 / 365.25), rel=1e-9)


# --- joint exceedance / flood probability -------------------------------------


def test_joint_exceedance_probability():
    tide = np.zeros(1000)
    surge = np.linspace(0, 1, 1000)
    assert EX.joint_exceedance_probability(tide, surge, 0.9) == pytest.approx(0.1)
    levels = EX.joint_exceedance_probability(tide, surge, [0.5, 0.9])
    assert list(levels) == pytest.approx([0.5, 0.1])
    with pytest.raises(ValueError):
        EX.joint_exceedance_probability(tide, surge[:10], 0.5)


def test_flood_probability_sane_limits():
    train = _hourly(T0, 60 * 24)
    model = TideModel.fit(train, _tide(train), alpha=1e-4)
    held = _hourly(train[-1] + timedelta(hours=1), 48)
    pred = model.predict(held)
    hi = flood_probability(pred, -100.0)
    lo = flood_probability(pred, 100.0)
    assert np.all(hi > 1.0 - 1e-9)
    assert np.all(lo < 1e-9)
    # Threshold at the predicted mean is a coin flip (symmetric Gaussian).
    mid = flood_probability(pred, pred.mean)
    assert np.allclose(mid, 0.5, atol=1e-9)


def test_flood_probability_grows_with_surge_sigma():
    train = _hourly(T0, 60 * 24)
    model = TideModel.fit(train, _tide(train), alpha=1e-4)
    held = _hourly(train[-1] + timedelta(hours=1), 48)
    pred = model.predict(held)
    z = float(np.max(pred.mean)) + 0.5
    p0 = flood_probability(pred, z, surge_sigma=0.0)
    p1 = flood_probability(pred, z, surge_sigma=1.0)
    assert np.all(p1 >= p0)


def test_advisor_prices_probabilistic_flood_threshold():
    from tideglass.marine.advisor import TideAdvisor

    train = _hourly(T0, 60 * 24)
    model = TideModel.fit(train, _tide(train), alpha=1e-4, station="Cove")
    held = _hourly(train[-1] + timedelta(hours=1), 48)
    plain = TideAdvisor(model).advise(held)
    assert plain.flood is None
    z = float(np.max(_tide(held))) + 0.2
    adv = TideAdvisor(model).advise(held, flood_threshold_m=z)
    assert adv.flood is not None
    assert adv.flood.shape == (48,)
    assert np.all((adv.flood >= 0.0) & (adv.flood <= 1.0))
    assert f"P(level > {z:.2f} m)" in adv.summary
    # A higher threshold is everywhere less likely to be exceeded.
    adv2 = TideAdvisor(model).advise(held, flood_threshold_m=z + 1.0)
    assert np.all(adv2.flood <= adv.flood)


def test_cli_advise_flood_threshold(tmp_path, capsys):
    train = _hourly(T0, 60 * 24)
    csv = tmp_path / "cove.csv"
    with open(csv, "w") as fh:
        fh.write("time,height\n")
        fh.writelines(f"{t.isoformat()},{h:.4f}\n"
                      for t, h in zip(train, _tide(train)))
    store = str(tmp_path / "store")
    assert cli_main(["fit", str(csv), "--station", "Cove",
                     "--store", store]) == 0
    capsys.readouterr()
    z = float(np.max(_tide(train))) + 0.2
    assert cli_main(["advise", "Cove", "2024-03-02", "--store", store,
                     "--flood", str(z)]) == 0
    assert f"P(level > {z:.2f} m)" in capsys.readouterr().out


# --- end-to-end on synthetic storm surge --------------------------------------


def test_eva_end_to_end_on_synthetic_surge():
    rng = np.random.default_rng(5)
    times = _hourly(T0, 400 * 24)
    tide = _tide(times)
    # Sporadic storm surges: ~12 events of GPD(0.15, σ=0.25, ξ=0.15) over ~13 mo.
    surge = np.zeros_like(tide)
    for start in rng.choice(len(times) - 72, size=12, replace=False):
        peak = _gpd_sample(0.15, 0.25, 0.15, 1, seed=int(start))[0]
        surge[start:start + 72] = peak * np.sin(np.linspace(0, np.pi, 72)) ** 2
    obs = tide + surge + rng.normal(0, 0.01, size=tide.size)
    model = TideModel.fit(times, obs, alpha=1e-4)
    sk = EX.skew_surge(times, obs, model.predict(times).mean)
    _, kv = EX.decluster(sk.times, sk.skew, gap_hours=72.0, threshold=0.10)
    assert len(kv) >= 9  # most injected storms recovered as peaks
    g = EX.fit_gpd(kv, threshold=0.10)
    rate = EX.annual_rate(times, g.n_peaks)
    rl10 = g.return_level(10.0, rate)
    assert rl10 > np.quantile(kv, 0.9)  # 10-yr level exceeds typical peaks


def test_cli_extremes_end_to_end(tmp_path, capsys):
    rng = np.random.default_rng(9)
    times = _hourly(T0, 400 * 24)
    tide = _tide(times)
    surge = 0.25 * np.maximum(
        np.sin(2 * np.pi * np.arange(len(times)) / (30 * 24)), 0.0) ** 8
    obs = tide + surge + rng.normal(0, 0.01, size=tide.size)
    csv = tmp_path / "storm.csv"
    with open(csv, "w") as fh:
        fh.write("time,height\n")
        fh.writelines(f"{t.isoformat()},{h:.4f}\n" for t, h in zip(times, obs))
    rc = cli_main(["extremes", str(csv), "--station", "STORM",
                   "--return-periods", "2,10"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "return level" in out
    assert "skew surge" in out
