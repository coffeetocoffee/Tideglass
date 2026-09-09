"""Residual / meteorological decomposition for Marea Core.

Splits the observed signal::

    observation = astronomical_tide + residual

The **residual** carries storm surge, setup, and weather-driven variability —
everything the harmonic engine cannot explain. This module:

* :func:`decompose` — residual, its scale, and surge flags
  (``|residual| > k·σ``),
* :func:`fit_ar1` / :func:`forecast_ar1` — a lightweight AR(1) tracker
  (the "ARIMA-lite" of the architecture doc) for short-range anomaly
  forecasts with growing uncertainty.

``pytides`` has no concept of any of this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurgeDecomposition:
    residual: np.ndarray  # observed − astronomical
    sigma: float  # std of the residual
    threshold: float  # k·σ surge threshold
    flags: np.ndarray  # bool array: surge events


@dataclass(frozen=True)
class AR1:
    phi: float  # lag-1 coefficient
    sigma_eps: float  # innovation std
    sigma: float  # marginal std of the series


def decompose(observed, predicted_mean, k: float = 3.0) -> SurgeDecomposition:
    """Split observations into astronomical tide + residual; flag surges."""
    o = np.asarray(observed, dtype=float).ravel()
    p = np.asarray(predicted_mean, dtype=float).ravel()
    if o.size != p.size:
        raise ValueError(f"{o.size} observations but {p.size} predictions")
    resid = o - p
    sigma = float(np.std(resid, ddof=1)) if resid.size > 1 else 0.0
    thresh = k * sigma
    return SurgeDecomposition(
        residual=resid, sigma=sigma, threshold=thresh,
        flags=np.abs(resid) > thresh,
    )


def fit_ar1(residual) -> AR1:
    """Fit x[t] = φ·x[t-1] + ε by ordinary least squares."""
    x = np.asarray(residual, dtype=float).ravel()
    if x.size < 3:
        raise ValueError("need at least 3 residual samples")
    x0, x1 = x[:-1], x[1:]
    denom = float(x0 @ x0)
    if denom == 0.0:
        raise ValueError("residual is identically zero")
    phi = float((x0 @ x1) / denom)
    eps = x1 - phi * x0
    return AR1(
        phi=phi,
        sigma_eps=float(math.sqrt(eps @ eps / (x.size - 2))),
        sigma=float(np.std(x, ddof=1)),
    )


def forecast_ar1(residual, ar: AR1, steps: int):
    """Iterate the AR(1) forward; return ``(mean, std)`` arrays.

    Uncertainty grows toward the marginal ``sigma`` with lead time.
    """
    x = np.asarray(residual, dtype=float).ravel()
    if steps < 1:
        raise ValueError("steps must be ≥ 1")
    last = float(x[-1])
    mean = np.array([ar.phi ** (h + 1) * last for h in range(steps)])
    if abs(ar.phi) < 1.0:
        var = ar.sigma_eps**2 * (1.0 - ar.phi ** (2 * (np.arange(steps) + 1))) \
            / (1.0 - ar.phi**2)
    else:  # non-stationary fallback: random-walk growth
        var = ar.sigma_eps**2 * (np.arange(steps) + 1)
    return mean, np.sqrt(np.maximum(var, 0.0))
