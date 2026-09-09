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

The v0.8 graduation adds distribution-level diagnostics as first-class metrics:

* :func:`pit_values` / :func:`pit_histogram` — the probability integral
  transform should be uniform for a well-specified predictive distribution.
* :func:`reliability_curve` — empirical vs nominal coverage across levels
  (the reliability diagram data).
* :func:`conformal_quantile` / :func:`conformalize` — split-conformal bands
  with a finite-sample coverage guarantee, as a distribution-free complement
  to the Gaussian interval.

All functions work on plain arrays and on :class:`~tideglass.marea.model.Prediction`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

_SQRT_PI = math.sqrt(math.pi)
_Z95 = 1.96

_erf_vec = np.vectorize(math.erf)
_SQRT2 = math.sqrt(2.0)


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


# --- v0.8 graduation: PIT, reliability, conformal ------------------------------

# Acklam's coefficients for the inverse normal CDF (relative error < 1.2e-9).
_PPF_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
          1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
_PPF_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
          6.680131188771972e01, -1.328068155288572e01)
_PPF_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
          -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
_PPF_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
          3.754408661907416e00)


def _norm_cdf_exact(z) -> np.ndarray:
    """Standard normal CDF, exact to float precision via ``math.erf``."""
    return 0.5 * (1.0 + _erf_vec(np.asarray(z, dtype=float) / _SQRT2))


def _norm_ppf(p) -> np.ndarray:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    p = np.asarray(p, dtype=float).ravel()
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("probabilities must lie strictly inside (0, 1)")

    def _poly(c, x):
        out = np.zeros_like(x)
        for a in c:
            out = out * x + a
        return out

    q = p - 0.5
    r = q * q
    central = q * _poly(_PPF_A, r) / (_poly(_PPF_B, r) * r + 1.0)
    tail_q = np.sqrt(-2.0 * np.log(np.minimum(p, 1.0 - p)))
    tail = _poly(_PPF_C, tail_q) / (_poly(_PPF_D, tail_q) * tail_q + 1.0)
    # The tail rational function returns the signed lower-tail value, so the
    # lower region keeps it and the upper region negates it (Acklam's form).
    out = np.where((p < 0.02425) | (p > 1.0 - 0.02425),
                   np.where(p < 0.5, tail, -tail), central)
    return out


def pit_values(observed, mean, sigma) -> np.ndarray:
    """Probability integral transform of Gaussian predictive distributions.

    ``pit_i = Φ((y_i − μ_i) / σ_i)``. For a well-specified forecast the PIT
    values are uniform on ``[0, 1]``; humps or skew diagnose over/under-
    dispersion and bias. Returns one value per observation.
    """
    o = np.asarray(observed, dtype=float).ravel()
    mu = np.asarray(mean, dtype=float).ravel()
    sig = np.maximum(np.asarray(sigma, dtype=float).ravel(), 1e-12)
    if not (o.size == mu.size == sig.size):
        raise ValueError(
            f"{o.size} observations but {mu.size} means and {sig.size} sigmas")
    return _norm_cdf_exact((o - mu) / sig)


def pit_histogram(observed, mean, sigma, bins: int = 10) -> dict:
    """Histogram of the PIT values as a uniformity diagnostic.

    ``heights`` are densities (count / (n · width)), so a calibrated forecast
    shows heights ≈ 1.0 across all bins. Also returns the PIT mean (≈ 0.5 when
    unbiased) for a one-glance bias check.
    """
    if bins < 2:
        raise ValueError("bins must be ≥ 2")
    pit = pit_values(observed, mean, sigma)
    counts, edges = np.histogram(pit, bins=bins, range=(0.0, 1.0))
    widths = np.diff(edges)
    return {
        "heights": counts / (pit.size * widths),
        "edges": edges,
        "bins": int(bins),
        "n": int(pit.size),
        "pit_mean": float(np.mean(pit)),
    }


def _band_sigma(prediction) -> np.ndarray:
    """Recover the per-time predictive σ from a ``mean ± z·σ`` band."""
    lo = np.asarray(prediction.lower, dtype=float).ravel()
    hi = np.asarray(prediction.upper, dtype=float).ravel()
    return np.maximum((hi - lo) / (2.0 * _Z95), 1e-12)


def reliability_curve(prediction, observed,
                      levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95, 0.99)
                      ) -> dict:
    """Empirical vs nominal coverage across levels (reliability diagram data).

    For each nominal level ``ℓ`` the symmetric ``z = Φ⁻¹((1+ℓ)/2)`` band is
    scored on held-out observations. A calibrated forecast has
    ``empirical ≈ nominal`` at every level, not just at 95%.
    """
    lv = np.asarray(list(levels), dtype=float)
    if lv.size == 0:
        raise ValueError("need at least one level")
    if np.any((lv <= 0.0) | (lv >= 1.0)):
        raise ValueError("levels must lie strictly inside (0, 1)")
    o = np.asarray(observed, dtype=float).ravel()
    mu = np.asarray(prediction.mean, dtype=float).ravel()
    sig = _band_sigma(prediction)
    if o.size != mu.size:
        raise ValueError(f"{o.size} observations but {mu.size} predictions")
    z = _norm_ppf(0.5 * (1.0 + lv))
    emp = np.array([float(np.mean(np.abs(o - mu) <= zi * sig)) for zi in z])
    return {"levels": lv, "empirical": emp, "n": int(o.size)}


def conformal_quantile(scores, alpha: float = 0.05) -> float:
    """Finite-sample ``(1−α)`` quantile of nonconformity scores.

    Returns the ``⌈(n+1)(1−α)⌉``-th order statistic (``min(…, n)``), which is
    what gives split conformal its ``≥ 1−α`` coverage guarantee under
    exchangeability.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly inside (0, 1)")
    s = np.sort(np.asarray(scores, dtype=float).ravel())
    if s.size == 0:
        raise ValueError("no calibration scores")
    k = math.ceil((s.size + 1) * (1.0 - alpha))
    return float(s[min(k, s.size) - 1])


def conformalize(mean, sigma, calib_mean, calib_sigma, calib_observed,
                 alpha: float = 0.05):
    """Split-conformal band from normalized residuals on a calibration split.

    Scores are ``s_i = |y_i − μ_i| / σ_i`` on held-out calibration data; the
    test band is ``μ ± q·σ`` with ``q`` the finite-sample ``(1−α)`` score
    quantile. Distribution-free: coverage holds at ``≥ 1−α`` whenever the
    calibration and test residuals are exchangeable, even if the Gaussian
    interval is miscalibrated. Returns ``(lower, upper, q)``.
    """
    cm = np.asarray(calib_mean, dtype=float).ravel()
    cs = np.maximum(np.asarray(calib_sigma, dtype=float).ravel(), 1e-12)
    co = np.asarray(calib_observed, dtype=float).ravel()
    if not (cm.size == cs.size == co.size):
        raise ValueError(
            f"{cm.size} calibration means but {cs.size} sigmas and "
            f"{co.size} observations")
    q = conformal_quantile(np.abs(co - cm) / cs, alpha)
    m = np.asarray(mean, dtype=float).ravel()
    sg = np.maximum(np.asarray(sigma, dtype=float).ravel(), 1e-12)
    if m.size != sg.size:
        raise ValueError(f"{m.size} means but {sg.size} sigmas")
    half = q * sg
    return m - half, m + half, q
