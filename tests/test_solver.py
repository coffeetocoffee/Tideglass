"""Tests for the exact linear solver: synthetic-signal recovery."""

import math

import numpy as np
import pytest

from tideglass.marea import constituents as C
from tideglass.marea import solver as S

NAMES = ["M2", "S2", "N2", "K1", "O1"]
AMPS = {"M2": 1.0, "S2": 0.30, "N2": 0.20, "K1": 0.15, "O1": 0.10}
PHASES = {"M2": 0.5, "S2": -1.2, "N2": 2.0, "K1": 0.0, "O1": 2.8}
H0 = 0.7


def _synthetic(t, noise=0.0, seed=0):
    omegas = S.rad_per_hour([C.speed(C.get(n)) for n in NAMES])
    y = np.full_like(t, H0, dtype=float)
    for name, w in zip(NAMES, omegas):
        y = y + AMPS[name] * np.cos(w * t - PHASES[name])
    if noise:
        y = y + np.random.default_rng(seed).normal(0.0, noise, size=t.size)
    return y, omegas


def _phase_err(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def test_noise_free_recovery():
    t = np.arange(0.0, 30 * 24, 1.0)  # 30 days hourly
    y, omegas = _synthetic(t)
    sol = S.solve(t, y, omegas, NAMES)
    assert abs(sol.mean - H0) / abs(H0) < 1e-6
    for term in sol.terms:
        assert abs(term.amplitude - AMPS[term.name]) / AMPS[term.name] < 1e-6
        assert _phase_err(term.phase, PHASES[term.name]) < 1e-6
    assert sol.rmse < 1e-9


def test_term_evaluates_to_signal():
    t = np.arange(0.0, 30 * 24, 1.0)
    y, omegas = _synthetic(t)
    sol = S.solve(t, y, omegas, NAMES)
    pred = S.design_matrix(t, omegas) @ sol.coef
    assert np.max(np.abs(pred - y)) < 1e-9


def test_noisy_recovery_and_covariance():
    t = np.arange(0.0, 90 * 24, 1.0)
    y, omegas = _synthetic(t, noise=0.02)
    sol = S.solve(t, y, omegas, NAMES)
    # RMSE tracks the injected noise level.
    assert abs(sol.rmse - 0.02) < 0.005
    assert abs(sol.sigma2 - 0.02**2) < 0.2 * 0.02**2
    for term in sol.terms:
        assert abs(term.amplitude - AMPS[term.name]) / AMPS[term.name] < 0.05
    # Covariance is symmetric positive semi-definite.
    assert np.allclose(sol.covariance, sol.covariance.T)
    assert np.all(np.linalg.eigvalsh(sol.covariance) > 0)
    assert sol.se_mean > 0 and all(term.sigma_amp > 0 for term in sol.terms)


def test_mean_only_fit():
    t = np.arange(0.0, 48.0, 1.0)
    y = np.full_like(t, 2.5)
    sol = S.solve(t, y, [])
    assert sol.terms == []
    assert abs(sol.mean - 2.5) < 1e-12


def test_errors():
    t = np.arange(0.0, 10.0, 1.0)
    y = np.sin(t)
    w = S.rad_per_hour([C.speed(C.get("M2"))])
    with pytest.raises(ValueError):  # underdetermined: 10 obs, 11 params
        S.solve(t, y, list(w) * 5)
    with pytest.raises(ValueError):  # duplicated frequency
        S.solve(np.arange(0.0, 100.0), np.sin(np.arange(0.0, 100.0)), [0.5, 0.5])
    with pytest.raises(ValueError):  # length mismatch
        S.solve(t, y[:-1], w)
    with pytest.raises(ValueError):  # name count mismatch
        S.solve(np.arange(0.0, 100.0), np.sin(np.arange(0.0, 100.0)), w, ["a", "b"])
