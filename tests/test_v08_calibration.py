"""Tests for v0.8 — graduated calibration: PIT, reliability, conformal."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tideglass import (  # top-level exports (v0.8)
    conformal_quantile,
    conformalize,
    pit_histogram,
    pit_values,
    reliability_curve,
)
from tideglass.cli import main as cli_main
from tideglass.marea import calibration as CAL
from tideglass.marea import constituents as C
from tideglass.marea.model import Prediction
from tideglass.marea.solver import rad_per_hour

UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)
TRUE = {"M2": (1.0, 0.5), "S2": (0.30, -1.2), "K1": (0.15, 0.0), "O1": (0.10, 2.8)}


def _hourly(start, n):
    return [start + timedelta(hours=h) for h in range(n)]


def _synthetic(times, noise=0.0, seed=0):
    t = np.array([(x - T0).total_seconds() / 3600.0 for x in times])
    y = np.full_like(t, 0.7)
    for name, (amp, phi) in TRUE.items():
        w = float(rad_per_hour(C.speed(C.get(name))))
        y = y + amp * np.cos(w * t - phi)
    if noise:
        y = y + np.random.default_rng(seed).normal(0.0, noise, size=t.size)
    return y


# --- PIT -----------------------------------------------------------------------


def test_pit_uniform_for_calibrated_forecast():
    rng = np.random.default_rng(21)
    mu = rng.normal(0, 1, 4000)
    sig = np.abs(rng.normal(1, 0.2, 4000)) + 0.1
    y = rng.normal(mu, sig)
    pit = pit_values(y, mu, sig)
    assert pit.shape == (4000,)
    assert abs(float(np.mean(pit)) - 0.5) < 0.02
    hist = pit_histogram(y, mu, sig, bins=10)
    assert np.all(np.abs(hist["heights"] - 1.0) < 0.15)
    assert hist["pit_mean"] == pytest.approx(float(np.mean(pit)))


def test_pit_detects_bias():
    rng = np.random.default_rng(22)
    mu = rng.normal(0, 1, 4000)
    y = mu + 0.5  # systematic +0.5σ miss
    assert float(np.mean(pit_values(y, mu, np.ones(4000)))) > 0.6


def test_pit_detects_overdispersion():
    rng = np.random.default_rng(23)
    mu = rng.normal(0, 1, 4000)
    y = rng.normal(mu, 0.2)  # forecast far too wide -> hump in the middle
    hist = pit_histogram(y, mu, np.ones(4000), bins=10)
    assert hist["heights"][4] > 3.0
    assert hist["heights"][5] > 3.0
    assert hist["heights"][0] == 0.0
    assert hist["heights"][-1] == 0.0


def test_pit_detects_underdispersion():
    rng = np.random.default_rng(27)
    mu = rng.normal(0, 1, 4000)
    y = rng.normal(mu, 3.0)  # forecast far too narrow -> U-shaped PIT
    hist = pit_histogram(y, mu, np.ones(4000), bins=10)
    edge = hist["heights"][0] + hist["heights"][-1]
    assert edge > 2.0 * hist["heights"][5]


def test_pit_shape_mismatch_raises():
    with pytest.raises(ValueError):
        pit_values(np.zeros(5), np.zeros(4), np.ones(5))


# --- inverse CDF + reliability -------------------------------------------------


def test_norm_ppf_matches_known_values():
    assert CAL._norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
    assert CAL._norm_ppf(0.975) == pytest.approx(1.959963986, rel=1e-6)
    assert CAL._norm_ppf(0.025) == pytest.approx(-1.959963986, rel=1e-6)
    # Tail regions (Acklam's lower/upper branches).
    assert CAL._norm_ppf(0.995) == pytest.approx(2.575829304, rel=1e-6)
    assert CAL._norm_ppf(0.005) == pytest.approx(-2.575829304, rel=1e-6)
    assert CAL._norm_ppf(0.999) == pytest.approx(3.090232306, rel=1e-6)
    p = np.array([0.1, 0.9])
    assert CAL._norm_ppf(p)[0] == pytest.approx(-CAL._norm_ppf(p)[1], rel=1e-9)
    with pytest.raises(ValueError):
        CAL._norm_ppf(0.0)


def test_reliability_curve_calibrated():
    rng = np.random.default_rng(24)
    mu = rng.normal(0, 1, 4000)
    sig = np.abs(rng.normal(1, 0.2, 4000)) + 0.1
    y = rng.normal(mu, sig)
    pred = Prediction(mean=mu, lower=mu - 1.96 * sig, upper=mu + 1.96 * sig,
                      se=np.zeros(4000))
    rel = reliability_curve(pred, y, levels=(0.5, 0.8, 0.9, 0.95, 0.99))
    assert np.all(np.abs(rel["empirical"] - rel["levels"]) < 0.03)


def test_reliability_curve_detects_miscalibration():
    rng = np.random.default_rng(25)
    mu = rng.normal(0, 1, 4000)
    y = rng.normal(mu, 2.0)  # truth twice as wide as the band claims
    pred = Prediction(mean=mu, lower=mu - 1.96, upper=mu + 1.96,
                      se=np.zeros(4000))
    rel = reliability_curve(pred, y, levels=(0.95,))
    assert rel["empirical"][0] < 0.80


# --- conformal -----------------------------------------------------------------


def test_conformal_quantile_finite_sample_correction():
    scores = np.arange(1, 101) / 100.0
    # ceil(101 · 0.95) = 96th order statistic -> 0.96.
    assert conformal_quantile(scores, 0.05) == pytest.approx(0.96)
    assert conformal_quantile(scores, 0.5) == pytest.approx(0.51)
    with pytest.raises(ValueError):
        conformal_quantile([], 0.05)
    with pytest.raises(ValueError):
        conformal_quantile(scores, 1.5)


def test_conformalize_hits_nominal_coverage():
    rng = np.random.default_rng(26)
    # Heteroscedastic truth; exchangeable calibration and test splits.
    sig_c = np.abs(rng.normal(1, 0.3, 500)) + 0.2
    y_c = rng.normal(0, sig_c)
    sig_t = np.abs(rng.normal(1, 0.3, 2000)) + 0.2
    y_t = rng.normal(0, sig_t)
    lo, hi, q = conformalize(np.zeros(2000), sig_t, np.zeros(500), sig_c,
                             y_c, alpha=0.05)
    assert q > 1.5  # normalized scores exceed the Gaussian 1.96 analogue
    cov = float(np.mean((lo <= y_t) & (y_t <= hi)))
    assert 0.94 <= cov <= 0.975


def test_conformalize_shape_mismatch_raises():
    # Calibration and test splits may legitimately differ in size ...
    lo, hi, _ = conformalize(np.zeros(3), np.ones(3), np.zeros(4), np.ones(4),
                             np.zeros(4))
    assert lo.shape == (3,) and hi.shape == (3,)
    # ... but inconsistent arrays within a split are an error.
    with pytest.raises(ValueError):
        conformalize(np.zeros(3), np.ones(2), np.zeros(4), np.ones(4),
                     np.zeros(4))


# --- CLI calibrate graduation --------------------------------------------------


def test_cli_calibrate_reports_suite(tmp_path, capsys):
    train = _hourly(T0, 60 * 24)
    y = _synthetic(train, noise=0.02, seed=3)
    csv = tmp_path / "gauge.csv"
    with open(csv, "w") as fh:
        fh.write("time,height\n")
        fh.writelines(f"{t.isoformat()},{h:.4f}\n" for t, h in zip(train, y))
    assert cli_main(["calibrate", str(csv), "--station", "G"]) == 0
    out = capsys.readouterr().out
    assert "pit_mean" in out
    assert "reliability" in out
    assert "conformal" in out
