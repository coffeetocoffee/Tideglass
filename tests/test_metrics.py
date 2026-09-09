"""Unit tests for benchmarking metrics (no external dependencies)."""

import math

import numpy as np
import pytest

from tideglass.marea import metrics as M
from tideglass.marea.model import Prediction


def test_rmse_mae_bias():
    pred = np.array([1.0, 2.0, 3.0])
    obs = np.array([1.0, 2.0, 4.0])
    assert M.rmse(pred, obs) == pytest.approx(math.sqrt(1 / 3))
    assert M.mae(pred, obs) == pytest.approx(1 / 3)
    assert M.bias(pred, obs) == pytest.approx(-1 / 3)
    assert M.rmse(obs, obs) == 0.0


def test_ci_coverage():
    lower = np.array([0.0, 0.0, 0.0, 0.0])
    upper = np.array([2.0, 2.0, 2.0, 2.0])
    assert M.ci_coverage(lower, upper, np.array([1.0, 0.0, 2.0, 5.0])) == 0.75


def test_peak_tide_error_at_extrema():
    # Observed highs at idx 1 (2.0) and lows at idx 3 (0.0).
    obs = np.array([1.0, 2.0, 1.0, 0.0, 1.0])
    pred = np.array([1.0, 1.8, 1.0, 0.2, 1.0])
    assert M.peak_tide_error(pred, obs) == pytest.approx(0.2)
    assert math.isnan(M.peak_tide_error(np.ones(4), np.ones(4)))  # flat -> nan
    assert math.isnan(M.peak_tide_error(np.ones(4), np.arange(4.0)))  # monotone


def test_evaluate_bundle():
    o = np.array([1.0, 2.0, 1.0, 0.0, 1.0])
    p = Prediction(mean=o.copy(), lower=o - 0.1, upper=o + 0.1,
                   se=np.zeros_like(o))
    out = M.evaluate(p, o)
    assert out["rmse"] == 0.0 and out["coverage"] == 1.0 and out["n"] == 5
    assert set(out) == {"n", "rmse", "mae", "bias", "peak_error", "coverage"}
