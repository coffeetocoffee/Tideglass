"""Extreme value analysis for Marea Core — the v0.8 probabilistic tier.

Mean-tide prediction answers "how high will the water be?". Flood risk answers
a different question: "how high *could* it be, and how often?". This module
provides the tail machinery:

* :func:`skew_surge` — the skew-surge series: at every predicted high water,
  ``observed − predicted``. Skew surge is (nearly) independent of the tidal
  phase, so it is the right variable for surge extremes.
* :func:`decluster` — peaks-over-threshold declustering: one storm = one peak
  (cluster maxima separated by ``gap_hours``).
* :func:`fit_gpd` — peaks-over-threshold fit of the Generalised Pareto
  distribution by maximum likelihood (dependency-free Nelder–Mead; no scipy).
* :class:`GPD` — the fitted tail: ``survival`` and N-year ``return_level``
  (the level exceeded on average once every ``T`` years).
* :func:`joint_exceedance_probability` — empirical ``P(tide + surge > level)``
  from concurrent tide/surge samples.
* :func:`flood_probability` — per-time ``P(water level > threshold)`` combining
  the predictive tide Gaussian with a surge distribution. This is the
  *probabilistic threshold* that replaces the marine layer's heuristic cutoffs:
  instead of a fixed "alarm above 2 m", the advisor prices ``P(level > z)``.

All functions are numpy-only, like the rest of Marea Core.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

_erf_vec = np.vectorize(math.erf)
_SQRT2 = math.sqrt(2.0)
_Z95 = 1.96


def _norm_cdf(z) -> np.ndarray:
    """Standard normal CDF, exact to float precision via ``math.erf``."""
    return 0.5 * (1.0 + _erf_vec(np.asarray(z, dtype=float) / _SQRT2))


def _hours(times: Sequence) -> np.ndarray:
    """Elapsed hours since the first sample (datetimes or plain numbers)."""
    if times and isinstance(times[0], datetime):
        t0 = times[0]
        return np.array([(t - t0).total_seconds() / 3600.0 for t in times],
                        dtype=float)
    return np.asarray(list(times), dtype=float)


# --- skew surge ---------------------------------------------------------------


@dataclass(frozen=True)
class SkewSurge:
    """Skew-surge series sampled at predicted high waters."""

    times: list  # datetimes of predicted high water
    skew: np.ndarray  # observed − predicted at HW (m)
    tide: np.ndarray  # predicted HW height (m)
    total: np.ndarray  # observed height at HW (m)


def skew_surge(times: Sequence, observed, predicted) -> SkewSurge:
    """Extract the skew-surge series from concurrent series.

    High waters are interior local maxima of the *predicted* tide; the skew
    surge is the observed height minus the predicted height at those instants.
    Skew surge is (nearly) phase-independent, which is what makes it the
    standard variable for surge-extreme analysis.
    """
    t = list(times)
    o = np.asarray(observed, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()
    if not (len(t) == o.size == p.size):
        raise ValueError(
            f"{len(t)} times but {o.size} observations and {p.size} predictions")
    d = np.diff(p)
    idx = np.nonzero((d[:-1] > 0) & (d[1:] < 0))[0] + 1
    if idx.size == 0:
        raise ValueError("no predicted high waters in the series")
    return SkewSurge(
        times=[t[i] for i in idx],
        skew=o[idx] - p[idx],
        tide=p[idx],
        total=o[idx],
    )


# --- declustering -------------------------------------------------------------


def decluster(times: Sequence, values, gap_hours: float = 72.0,
              threshold: float | None = None):
    """Decluster a peak series: one storm, one peak (the runs method).

    Only samples above ``threshold`` (default: the 90th percentile of
    ``values``) can start a cluster; consecutive exceedances belong to the
    same storm unless separated by ``gap_hours`` of sub-threshold values.
    Only each cluster's maximum is kept. Returns ``(kept_times,
    kept_values)`` sorted in time. Storm surge decorrelates over ~2–3 days,
    hence the 72 h default.
    """
    t = list(times)
    v = np.asarray(values, dtype=float).ravel()
    if len(t) != v.size:
        raise ValueError(f"{len(t)} times but {v.size} values")
    if v.size == 0:
        raise ValueError("no values to decluster")
    if gap_hours <= 0:
        raise ValueError("gap_hours must be positive")
    u = float(np.quantile(v, 0.90)) if threshold is None else float(threshold)
    h = _hours(t)
    order = np.argsort(h, kind="stable")
    h, v = h[order], v[order]
    to = [t[i] for i in order]
    kept_t, kept_v = [], []
    cluster: list[int] = []

    def _flush():
        if cluster:
            j = max(cluster, key=lambda i: v[i])
            kept_t.append(to[j])
            kept_v.append(float(v[j]))
        cluster.clear()

    for k in range(v.size):
        if v[k] <= u:
            continue  # sub-threshold: only splits clusters via the gap below
        if cluster and h[k] - h[cluster[-1]] >= gap_hours:
            _flush()
        cluster.append(k)
    _flush()
    if not kept_v:
        raise ValueError(f"no exceedances over the {u:.4f} m threshold")
    return kept_t, np.asarray(kept_v, dtype=float)


# --- Generalised Pareto tail --------------------------------------------------


@dataclass(frozen=True)
class GPD:
    """Fitted Generalised Pareto tail over ``loc`` (the POT threshold).

    ``P(X > x | X > u) = (1 + ξ·(x−u)/σ)^(−1/ξ)`` (``ξ = 0``: the exponential
    ``exp(−(x−u)/σ)``).
    """

    loc: float  # POT threshold u (m)
    scale: float  # σ > 0 (m)
    shape: float  # ξ (dimensionless)
    n_peaks: int  # number of exceedances fitted

    def survival(self, x):
        """``P(X > x)`` conditional on exceeding the threshold (x ≥ loc)."""
        scalar = np.isscalar(x)
        xx = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.ones_like(xx)
        above = xx > self.loc
        y = xx[above] - self.loc
        if abs(self.shape) < 1e-8:
            out[above] = np.exp(-y / self.scale)
        else:
            out[above] = (1.0 + self.shape * y / self.scale) ** (-1.0 / self.shape)
        return float(out[0]) if scalar else out

    def return_level(self, period_years: float, rate_per_year: float) -> float:
        """Level exceeded on average once every ``period_years`` years.

        ``rate_per_year`` is the mean number of threshold exceedances per year
        (see :func:`annual_rate`). Only meaningful for
        ``period_years ≥ 1 / rate_per_year``.
        """
        if period_years <= 0:
            raise ValueError("period_years must be positive")
        if rate_per_year <= 0:
            raise ValueError("rate_per_year must be positive")
        lam_t = rate_per_year * period_years
        if abs(self.shape) < 1e-8:
            return float(self.loc + self.scale * math.log(lam_t))
        return float(self.loc + self.scale / self.shape * (lam_t**self.shape - 1.0))


def _gpd_nll(log_sigma: float, xi: float, y: np.ndarray) -> float:
    """Negative log-likelihood of GPD(σ = exp(logσ), ξ) on excesses ``y``."""
    sigma = math.exp(log_sigma)
    arg = 1.0 + xi * y / sigma
    if np.any(arg <= 0.0):
        return math.inf
    if abs(xi) < 1e-8:
        return float(y.size * log_sigma + y.sum() / sigma)
    return float(y.size * log_sigma + (1.0 + 1.0 / xi) * np.log(arg).sum())


def _nelder_mead(f, x0: np.ndarray, step: float = 0.1,
                 tol: float = 1e-8, maxiter: int = 2000) -> np.ndarray:
    """Dependency-free Nelder–Mead minimiser (2-D is all we need)."""
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    simplex = [x0] + [x0 + step * (np.arange(n) == j) for j in range(n)]
    vals = [f(*p) for p in simplex]
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    for _ in range(maxiter):
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if float(np.max(np.abs(simplex[0] - simplex[-1]))) < tol:
            break
        centroid = sum(simplex[:-1]) / n
        reflected = centroid + alpha * (centroid - simplex[-1])
        f_r = f(*reflected)
        if vals[0] <= f_r < vals[-2]:
            simplex[-1], vals[-1] = reflected, f_r
            continue
        if f_r < vals[0]:
            expanded = centroid + gamma * (reflected - centroid)
            f_e = f(*expanded)
            simplex[-1], vals[-1] = (expanded, f_e) if f_e < f_r else (reflected, f_r)
            continue
        contracted = centroid + rho * (simplex[-1] - centroid)
        f_c = f(*contracted)
        if f_c < vals[-1]:
            simplex[-1], vals[-1] = contracted, f_c
            continue
        simplex = [simplex[0]] + [simplex[0] + sigma * (p - simplex[0])
                                  for p in simplex[1:]]
        vals = [f(*p) for p in simplex]
    best = int(np.argmin(vals))
    return np.asarray(simplex[best], dtype=float)


def fit_gpd(values, threshold: float | None = None,
            min_peaks: int = 10) -> GPD:
    """Fit a Generalised Pareto tail to declustered peaks over ``threshold``.

    ``values`` are declustered peak heights (e.g. skew surges); only samples
    strictly above ``threshold`` (default: the median of ``values``) enter the
    fit as excesses ``y = x − u``. Maximum likelihood via Nelder–Mead.
    """
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        raise ValueError("no values to fit")
    u = float(np.median(v)) if threshold is None else float(threshold)
    y = v[v > u] - u
    if y.size < min_peaks:
        raise ValueError(
            f"only {y.size} exceedances over {u:.4f} m (need ≥ {min_peaks}); "
            "lower the threshold or collect more peaks")
    sigma0 = max(float(np.mean(y)), 1e-6)
    opt = _nelder_mead(
        lambda ls, xi: _gpd_nll(ls, xi, y),
        np.array([math.log(sigma0), 0.1]),
    )
    return GPD(loc=u, scale=float(math.exp(opt[0])), shape=float(opt[1]),
               n_peaks=int(y.size))


def annual_rate(times: Sequence, n_peaks: int) -> float:
    """Mean exceedances per year over the record span (365.25-day years)."""
    h = _hours(list(times))
    span_years = (float(h.max() - h.min()) / 24.0 / 365.25) if h.size > 1 else 0.0
    if span_years <= 0.0:
        raise ValueError("record span is zero; cannot annualise the rate")
    return float(n_peaks) / span_years


# --- joint exceedance / probabilistic thresholds ------------------------------


def joint_exceedance_probability(tide, surge, level) -> float | np.ndarray:
    """Empirical ``P(tide + surge > level)`` from concurrent samples.

    ``tide`` and ``surge`` are same-length series (e.g. predicted tide plus the
    AR(1) surge forecast, or observed decomposition). ``level`` may be a scalar
    or an array of thresholds.
    """
    t = np.asarray(tide, dtype=float).ravel()
    s = np.asarray(surge, dtype=float).ravel()
    if t.size != s.size:
        raise ValueError(f"{t.size} tide samples but {s.size} surge samples")
    if t.size == 0:
        raise ValueError("empty series")
    scalar = np.isscalar(level)
    lv = np.atleast_1d(np.asarray(level, dtype=float))
    total = t + s
    out = np.array([float(np.mean(total > z)) for z in lv])
    return float(out[0]) if scalar else out


def flood_probability(prediction, threshold,
                      surge_mean: float = 0.0, surge_sigma: float = 0.0
                      ) -> np.ndarray:
    """Per-time ``P(water level > threshold)`` — the probabilistic threshold.

    Combines the predictive tide Gaussian (mean plus the ``lower``/``upper``
    band) with a surge distribution ``N(surge_mean, surge_sigma²)`` (e.g. the
    AR(1) forecast std or the fitted surge marginal). ``threshold`` may be a
    scalar alarm level (a GPD return level, say) or a per-time array.
    """
    mu = np.asarray(prediction.mean, dtype=float).ravel() + float(surge_mean)
    lo = np.asarray(prediction.lower, dtype=float).ravel()
    hi = np.asarray(prediction.upper, dtype=float).ravel()
    z = np.asarray(threshold, dtype=float)
    sig_tide = (hi - lo) / (2.0 * _Z95)
    sig = np.sqrt(np.maximum(sig_tide**2 + float(surge_sigma) ** 2, 1e-12))
    return 1.0 - _norm_cdf((z - mu) / sig)
