"""Tests for uncertainty calibration (CRPS + per-constituent attribution)."""

from datetime import datetime, timedelta, timezone

import numpy as np

from tideglass.marea import calibration as CAL
from tideglass.marea import constituents as C
from tideglass.marea.model import TideModel
from tideglass.marea.solver import rad_per_hour

UTC = timezone.utc
TRUE = {"M2": (1.0, 0.5), "S2": (0.30, -1.2), "K1": (0.15, 0.0), "O1": (0.10, 2.8)}
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _hours(times):
    return np.array([(t - T0).total_seconds() / 3600.0 for t in times])


def _synthetic(times, noise=0.0, seed=0):
    t = _hours(times)
    y = np.full_like(t, 0.7)
    for name, (amp, phi) in TRUE.items():
        w = float(rad_per_hour(C.speed(C.get(name))))
        y = y + amp * np.cos(w * t - phi)
    if noise:
        y = y + np.random.default_rng(seed).normal(0.0, noise, size=t.size)
    return y


def _hourly(start, n):
    return [start + timedelta(hours=h) for h in range(n)]


def test_crps_zero_when_perfect():
    o = np.array([1.0, 2.0, 3.0])
    assert CAL.crps_gaussian(o, o, 0.0) < 1e-8
    assert CAL.crps_gaussian(o, o, 1e-9) < 1e-6


def test_crps_decreases_with_matching_sigma():
    # For a correctly-located forecast (mean == truth), a sharper spread scores
    # lower CRPS than an overly wide one (proper scoring rule rewards sharpness).
    o = np.array([0.0, 0.0, 0.0])
    wide = CAL.crps_gaussian(o, o, 5.0)
    tight = CAL.crps_gaussian(o, o, 0.5)
    assert tight < wide


def test_crps_interval_matches_gaussian():
    o = np.array([0.0, 1.0])
    lo = np.array([-1.0, 0.0])
    hi = np.array([1.0, 2.0])
    iv = CAL.crps_interval(o, lo, hi, z=1.96)
    ga = CAL.crps_gaussian(o, 0.5 * (lo + hi), (hi - lo) / (2 * 1.96))
    assert abs(iv - ga) < 1e-9


def test_evaluate_calibration_keys():
    train = _hourly(T0, 60 * 24)
    model = TideModel.fit(train, _synthetic(train, noise=0.02, seed=3), alpha=1e-4)
    held = _hourly(train[-1] + timedelta(hours=1), 30 * 24)
    pred = model.predict(held)
    noisy = _synthetic(held, noise=0.02, seed=99)
    cal = CAL.evaluate_calibration(pred, noisy)
    assert set(cal) >= {"crps", "rmse", "coverage_empirical", "coverage_nominal"}
    # Coverage should be in a sane neighbourhood of the nominal 95%.
    assert cal["coverage_empirical"] > 0.85


def test_constituent_attribution_sums_to_one():
    train = _hourly(T0, 60 * 24)
    model = TideModel.fit(train, _synthetic(train, noise=0.02, seed=3), alpha=1e-4)
    held = _hourly(train[-1] + timedelta(hours=1), 30 * 24)
    attr = CAL.constituent_attribution(model, held)
    assert len(attr["names"]) == len(model.constituents())
    assert abs(float(np.sum(attr["share"])) - 1.0) < 1e-9
    assert (attr["variance"] >= 0).all()
