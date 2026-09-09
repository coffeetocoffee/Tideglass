"""Uncertainty calibration for Marea Core — the v0.3 scoring layer.

The prediction interval is only useful if it is *well calibrated*: a nominal
95% band should contain ~95% of held-out observations, and the predictive
distribution should score well on a proper scoring rule. This module provides:

* :func:`crps_gaussian` — the continuous ranked probability score for a Gaussian
  predictive distribution (a strictly proper scoring rule; lower is better).
* :func:`crps_interval` — CRPS for a symmetric interval ``[lower, upper]``
  (used when only the band, not the full σ, is available).
* :func:`evaluate_calibration` — CRPS + coverage/reliability vs a reference.
* :func:`constituent_attribution` — splits the parameter-uncertainty variance
  across fitted constituents (the "per-constituent error attribution" of the
  v0.3 plan), so we can say *which* harmonic dominates the prediction error
  budget rather than just reporting a single RMSE.

All functions work on plain arrays and on :class:`~tideglass.marea.model.Prediction`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

_SQRT_PI = math.sqrt(math.pi)
_Z95 = 1.96


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    # Standard normal CDF via the Abramowitz & Stegun 26.2.17 approximation
    # (vectorized, accurate to ~1e-7, no scipy required). Computes the upper
    # tail for |z| and uses symmetry.
    z = np.asarray(z, dtype=float)
    x = np.abs(z)
    t = 1.0 / (1.0 + 0.2316419 * x)
    poly = (
        0.319381530 * t
        - 0.356563782 * t**2
        + 1.781477937 * t**3
        - 1.821255978 * t**4
        + 1.330274429 * t**5
    )
    upper = np.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi) * poly
    return np.where(z >= 0.0, 1.0 - upper, upper)


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)


def crps_gaussian(observed, mean, sigma) -> float:
    """Continuous ranked probability score for N(``mean``, ``sigma²``).

    Closed form (Gneiting et al., 2005)::

        CRPS = σ [ z(2Φ(z) − 1) + 2φ(z) − 1/√π ],  z = (y − μ)/σ.

    A proper scoring rule: ``crps(obs, obs, 0) = 0``, and it rewards both
    sharpness and calibration. Returns the mean CRPS over the sample.
    """
    o = np.asarray(observed, dtype=float).ravel()
    mu = np.asarray(mean, dtype=float).ravel()
    sig = np.asarray(sigma, dtype=float).ravel()
    sig = np.maximum(sig, 1e-9)  # guard against zero-width forecasts
    z = (o - mu) / sig
    crps = sig * (z * (2.0 * _norm_cdf(z) - 1.0) + 2.0 * _norm_pdf(z) - 1.0 / _SQRT_PI)
    return float(np.mean(crps))


def crps_interval(observed, lower, upper, z: float = _Z95) -> float:
    """CRPS for a symmetric ``[lower, upper]`` band (Gaussian approximation).

    The band is treated as the ``mean ± z·σ`` interval of a Gaussian, so the
    implied ``σ = (upper − lower) / (2z)`` is fed to :func:`crps_gaussian`.
    """
    lo = np.asarray(lower, dtype=float).ravel()
    hi = np.asarray(upper, dtype=float).ravel()
    mean = 0.5 * (lo + hi)
    sigma = (hi - lo) / (2.0 * z)
    return crps_gaussian(observed, mean, sigma)


def evaluate_calibration(prediction, observed, z: float = _Z95) -> dict[str, float]:
    """Score a :class:`~tideglass.marea.model.Prediction` vs observations.

    Returns mean CRPS (interval form), empirical coverage of the nominal band,
    and the nominal coverage level — a reliability check (empirical ≈ nominal is
    "well calibrated"). Also returns RMSE for convenience.
    """
    o = np.asarray(observed, dtype=float).ravel()
    mean = np.asarray(prediction.mean, dtype=float).ravel()
    lo = np.asarray(prediction.lower, dtype=float).ravel()
    hi = np.asarray(prediction.upper, dtype=float).ravel()
    return {
        "crps": crps_interval(o, lo, hi, z),
        "rmse": float(math.sqrt(np.mean((mean - o) ** 2))),
        "coverage_empirical": float(np.mean((lo <= o) & (o <= hi))),
        "coverage_nominal": float(2.0 * _norm_cdf(z) - 1.0),
        "n": int(o.size),
    }


def constituent_attribution(model, times: Sequence, z: float = _Z95) -> dict:
    """Per-constituent share of the prediction uncertainty budget.

    For a fitted :class:`~tideglass.marea.model.TideModel`, the prediction
    variance at time ``t`` from *parameter* uncertainty is ``gᵀ C g`` where ``g``
    is the basis row and ``C`` the solution covariance. We split that per
    constituent into ``var_i(t) = g_iᵀ C_i g_i`` (its own 2×2 block), averaging
    over ``times`` to get each constituent's mean contribution and share of the
    total parameter-uncertainty variance.

    Returns a dict with ``names``, ``variance`` (mean var per constituent),
    ``share`` (fraction of total, sums to ~1), and ``amplitude_sigma`` (the
    fitted σ of each constituent's amplitude). This is the v0.3 "per-constituent
    error attribution": it answers *which harmonic is the weakest link*.
    """
    from tideglass.marea.model import _basis_matrix

    consts = model._constituents
    A = _basis_matrix(consts, list(times))
    C = model._covariance
    m = len(consts)
    variances = np.zeros(m)
    for j in range(m):
        cols = np.array([1 + 2 * j, 2 + 2 * j])
        g = A[:, cols]  # (n, 2)
        Cblock = C[np.ix_(cols, cols)]  # (2, 2)
        # diag(g C gᵀ) = row-wise quadratic form
        vars_t = np.einsum("ij,jk,ik->i", g, Cblock, g)
        variances[j] = float(np.mean(np.maximum(vars_t, 0.0)))
    total = float(variances.sum())
    shares = variances / total if total > 0 else np.zeros_like(variances)
    fits = model.constituents()
    return {
        "names": [c.name for c in consts],
        "variance": variances,
        "share": shares,
        "amplitude_sigma": np.array([f.sigma_amp for f in fits]),
        "total_variance": total,
    }
