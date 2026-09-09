"""Benchmarking metrics for Marea Core — the proof of superiority.

All functions compare predicted series against reference gauge observations:

* ``rmse`` — root-mean-square error.
* ``mae`` / ``bias`` — mean absolute / signed error.
* ``peak_tide_error`` — mean absolute height error at observed high/low
  waters (the errors that matter for navigation and flooding). Extrema are
  interior local maxima/minima of the observed series; the prediction is
  scored at those same instants, so timing slips count as height errors.
* ``ci_coverage`` — fraction of observations inside the prediction band
  (``pytides`` has no bands, so it scores ``n/a`` there by construction).

``evaluate`` bundles everything for a :class:`Prediction` vs observations.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


def _asarray(x) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


def rmse(predicted, observed) -> float:
    p, o = _asarray(predicted), _asarray(observed)
    return float(math.sqrt(np.mean((p - o) ** 2)))


def mae(predicted, observed) -> float:
    p, o = _asarray(predicted), _asarray(observed)
    return float(np.mean(np.abs(p - o)))


def bias(predicted, observed) -> float:
    p, o = _asarray(predicted), _asarray(observed)
    return float(np.mean(p - o))


def ci_coverage(lower, upper, observed) -> float:
    lo, hi, o = _asarray(lower), _asarray(upper), _asarray(observed)
    return float(np.mean((lo <= o) & (o <= hi)))


def _extremum_indices(observed: np.ndarray) -> np.ndarray:
    o = _asarray(observed)
    d = np.diff(o)
    with np.errstate(invalid="ignore"):
        peak = (d[:-1] > 0) & (d[1:] < 0)
        trough = (d[:-1] < 0) & (d[1:] > 0)
    return np.nonzero(peak | trough)[0] + 1


def peak_tide_error(predicted, observed) -> float:
    """Mean absolute error at observed high/low waters (nan if none)."""
    p, o = _asarray(predicted), _asarray(observed)
    idx = _extremum_indices(o)
    if idx.size == 0:
        return float("nan")
    return float(np.mean(np.abs(p[idx] - o[idx])))


def evaluate(prediction, observed) -> Dict[str, float]:
    """Score a :class:`~tideglass.marea.model.Prediction` vs observations."""
    o = _asarray(observed)
    return {
        "n": int(o.size),
        "rmse": rmse(prediction.mean, o),
        "mae": mae(prediction.mean, o),
        "bias": bias(prediction.mean, o),
        "peak_error": peak_tide_error(prediction.mean, o),
        "coverage": ci_coverage(prediction.lower, prediction.upper, o),
    }
