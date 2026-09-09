"""Tests for the joint Kalman tide + surge + trend smoother (v0.3)."""

from datetime import datetime, timedelta, timezone

import numpy as np

from tideglass.marea import constituents as C
from tideglass.marea.kalman import JointModel
from tideglass.marea.solver import rad_per_hour

UTC = timezone.utc
TRUE = {"M2": (1.0, 0.5), "S2": (0.30, -1.2), "N2": (0.20, 2.0),
        "K1": (0.15, 0.0), "O1": (0.10, 2.8)}
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _hours(times):
    return np.array([(t - T0).total_seconds() / 3600.0 for t in times])


def _harmonic(times):
    t = _hours(times)
    y = np.full_like(t, 0.7)
    for name, (amp, phi) in TRUE.items():
        w = float(rad_per_hour(C.speed(C.get(name))))
        y = y + amp * np.cos(w * t - phi)
    return y


def _ar1(n, sigma, phi, seed):
    rng = np.random.default_rng(seed)
    x = [0.0]
    for _ in range(n - 1):
        x.append(phi * x[-1] + rng.normal(0.0, sigma))
    return np.array(x)


def _synthetic(times, noise=0.0, seed=0, trend_mm_yr=0.0, surge_sigma=0.0,
               surge_seed=0):
    y = _harmonic(times)
    if trend_mm_yr:
        slope = trend_mm_yr / 1000.0 / (24.0 * 365.25)  # m per hour
        y = y + slope * _hours(times)
    if surge_sigma:
        y = y + _ar1(len(times), surge_sigma, 0.95, surge_seed)
    if noise:
        y = y + np.random.default_rng(seed).normal(0.0, noise, size=y.size)
    return y


def _hourly(start, n):
    return [start + timedelta(hours=h) for h in range(n)]


def test_joint_recovers_harmonics():
    train = _hourly(T0, 60 * 24)
    y = _synthetic(train, noise=0.01, seed=3)
    model = JointModel.fit(train, y, alpha=1e-4, station="test")
    fit = model.fit_result
    for name, (amp, phi) in TRUE.items():
        assert name in fit.coefficients
        c = fit.coefficients[name]
        assert abs(c["amp"] - amp) / amp < 0.03
        assert abs(c["se_a"]) >= 0


def test_joint_recovers_secular_trend():
    # 15 mm/yr over a year with modest surge; separable from the AR(1) wander.
    train = _hourly(T0, 365 * 24)
    y = _synthetic(train, noise=0.005, seed=7, trend_mm_yr=15.0,
                   surge_sigma=0.005, surge_seed=11)
    model = JointModel.fit(train, y, auto_select=False, station="trend")
    fit = model.fit_result
    assert abs(fit.trend_mm_yr - 15.0) < 10.0
    assert fit.trend_mm_yr_se > 0


def test_joint_recovers_surge_correlation():
    train = _hourly(T0, 90 * 24)
    truth_surge = _ar1(len(train), 0.05, 0.95, 5)
    y = _harmonic(train) + truth_surge
    model = JointModel.fit(train, y, auto_select=False, station="surge")
    fit = model.fit_result
    corr = float(np.corrcoef(fit.surge, truth_surge)[0, 1])
    assert corr > 0.85


def test_joint_predict_returns_bands():
    train = _hourly(T0, 60 * 24)
    y = _synthetic(train, noise=0.01, seed=2)
    model = JointModel.fit(train, y, alpha=1e-4)
    future = _hourly(train[-1] + timedelta(hours=1), 24)
    pred = model.predict(future)
    assert pred.mean.shape == (24,)
    assert (pred.lower <= pred.mean + 1e-9).all()
    assert (pred.mean <= pred.upper + 1e-9).all()


def test_zero_trend_no_surge_is_near_zero():
    train = _hourly(T0, 60 * 24)
    y = _synthetic(train, seed=9)
    model = JointModel.fit(train, y, auto_select=False)
    assert abs(model.fit_result.trend_mm_yr) < 5.0
