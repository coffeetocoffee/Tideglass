"""Residual memory for Marea Core — the v1.1 learned bias layer.

Harmonics express the astronomical tide; they cannot express what the ocean
actually adds on top: seasonal mean-level biases, river-discharge offsets,
regional steric cycles. The v0.8 calibration suite proved the point — a
−0.6σ seasonal low-water bias sat invisible inside a nominally perfect band.
This module gives the engine a *memory* of its own residuals:

* :func:`learn_residual` — regress the post-harmonic residuals onto a
  day-of-year climatology (circular bins, robust medians, circular smoothing),
  then conformalize the corrected band on a held-out tail.
* :class:`ResidualModel` — the learned bias table. ``correction(times)`` gives
  the bias estimate anywhere in time; ``apply`` shifts the prediction mean and
  (optionally) rescales the band by the conformal quantile.

The layer is deliberately a *climatology, not a model*: it only re-expresses
what the record itself showed, so it cannot hallucinate structure the station
never exhibited. Harmonics stay the physics; the bias table is the asset that
accrues — and it is exactly what pytides/TPXO can never have, because no
residuals flow into them.

Round-trip: :meth:`TideModel.attach_residual` stores the table in the model's
meta, so ``to_artifact()`` → ``load_harmonic()`` preserves it and
:meth:`TideModel.predict` applies it automatically.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from tideglass.marea.calibration import conformal_quantile

_Z95 = 1.96
_DAYS_YEAR = 365.25


def _doy_fraction(t: datetime) -> float:
    """Fractional day of year in ``[0, year)`` (naive = UTC convention)."""
    tt = t.timetuple()
    hour = (tt.tm_hour + tt.tm_min / 60.0 + tt.tm_sec / 3600.0) / 24.0
    return tt.tm_yday - 1 + hour


@dataclass
class ResidualModel:
    """Learned day-of-year residual climatology (the station's bias table)."""

    bins: int
    biases: np.ndarray  # (bins,) robust mean residual per bin (m)
    counts: np.ndarray  # (bins,) samples per bin
    conformal_q: float | None = None  # band scale from held-out tail (None = shift only)
    alpha: float = 0.05

    def correction(self, times: Sequence) -> np.ndarray:
        """Bias estimate (m) at each time — circular interpolation between bins."""
        times = list(times)
        n = self.bins
        b = np.asarray(self.biases, dtype=float)
        # Position on the circle: bin i spans [i, i+1), center at i + 0.5.
        x = np.array([_doy_fraction(t) / _DAYS_YEAR * n for t in times],
                     dtype=float)
        idx = x - 0.5
        i0 = np.floor(idx).astype(int)
        frac = idx - i0
        lo = b[i0 % n]
        hi = b[(i0 + 1) % n]
        return (1.0 - frac) * lo + frac * hi

    def apply(self, prediction, times: Sequence):
        """Return a bias-corrected :class:`~tideglass.marea.model.Prediction`.

        The mean shifts by ``correction(times)``. With a conformal quantile the
        band is rebuilt as ``corrected ± q·σ`` (σ recovered from the incoming
        band); without one, the band shifts rigidly with the mean.
        """
        from tideglass.marea.model import Prediction

        corr = self.correction(times)
        mean = np.asarray(prediction.mean, dtype=float) + corr
        lo = np.asarray(prediction.lower, dtype=float)
        hi = np.asarray(prediction.upper, dtype=float)
        se = np.asarray(prediction.se, dtype=float)
        if self.conformal_q is not None:
            sigma = np.maximum((hi - lo) / (2.0 * _Z95), 0.0)
            half = self.conformal_q * sigma
            lo, hi = mean - half, mean + half
        else:
            lo = lo + corr
            hi = hi + corr
        return Prediction(mean=mean, lower=lo, upper=hi, se=se)

    def to_dict(self) -> dict:
        return {
            "bins": int(self.bins),
            "biases": [float(x) for x in self.biases],
            "counts": [int(x) for x in self.counts],
            "conformal_q": (None if self.conformal_q is None
                            else float(self.conformal_q)),
            "alpha": float(self.alpha),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ResidualModel":
        return cls(
            bins=int(d["bins"]),
            biases=np.asarray(d["biases"], dtype=float),
            counts=np.asarray(d["counts"], dtype=int),
            conformal_q=(None if d.get("conformal_q") is None
                         else float(d["conformal_q"])),
            alpha=float(d.get("alpha", 0.05)),
        )


def _fill_circular(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid entries from the nearest valid neighbour on the circle."""
    n = len(values)
    out = values.astype(float).copy()
    if not valid.any():
        return out
    for i in range(n):
        if valid[i]:
            continue
        for d in range(1, n):
            if valid[(i - d) % n]:
                out[i] = values[(i - d) % n]
                break
            if valid[(i + d) % n]:
                out[i] = values[(i + d) % n]
                break
    return out


def learn_residual(
    model,
    times: Sequence,
    observed,
    bins: int = 12,
    holdout: float = 0.25,
    conformal: bool = True,
    alpha: float = 0.05,
) -> tuple[ResidualModel, dict]:
    """Learn a station's residual climatology from its own record.

    Fits a day-of-year bias table to ``observed − model.predict()`` on the
    record head, conformalizes the corrected band on the chronological tail,
    and reports before/after diagnostics.

    :param model: a fitted :class:`~tideglass.marea.model.TideModel` (with
        meaningful prediction bands — loaded harmonics with collapsed bands
        skip the conformal step).
    :param bins: day-of-year bins (12 ≈ monthly resolution).
    :param holdout: chronological tail fraction reserved for conformal scoring.
    :returns: ``(ResidualModel, diagnostics)``.
    """
    if bins < 2:
        raise ValueError("bins must be ≥ 2")
    if not 0.0 < holdout < 1.0:
        raise ValueError("holdout must lie strictly inside (0, 1)")
    times = list(times)
    y = np.asarray(observed, dtype=float).ravel()
    if y.size != len(times):
        raise ValueError(f"{len(times)} times but {y.size} heights")
    if y.size < max(4 * bins, 48):
        raise ValueError(
            f"record too short to learn {bins} bins "
            f"(need ≥ {max(4 * bins, 48)} obs, got {y.size})")

    pred = model.predict(times)
    resid = y - np.asarray(pred.mean, dtype=float)
    doy = np.array([_doy_fraction(t) for t in times], dtype=float)
    bidx = np.floor(doy / _DAYS_YEAR * bins).astype(int) % bins

    biases = np.zeros(bins)
    counts = np.zeros(bins, dtype=int)
    for b in range(bins):
        m = bidx == b
        counts[b] = int(m.sum())
        if counts[b]:
            biases[b] = float(np.median(resid[m]))
    filled = _fill_circular(biases, counts > 0)
    # Circular smoothing: [1/4, 1/2, 1/4] — keeps the annual cycle continuous.
    biases = (0.25 * np.roll(filled, 1) + 0.5 * filled + 0.25 * np.roll(filled, -1))

    rm = ResidualModel(bins=bins, biases=biases, counts=counts, alpha=alpha)

    # -- conformal tail + diagnostics --------------------------------------
    n_test = max(24, round(y.size * holdout))
    head_t, tail_t = times[:-n_test], times[-n_test:]
    head_idx = slice(0, y.size - n_test)
    head_biases = np.zeros(bins)
    head_counts = np.zeros(bins, dtype=int)
    for b in range(bins):
        m = bidx[head_idx] == b
        head_counts[b] = int(m.sum())
        if head_counts[b]:
            head_biases[b] = float(np.median(resid[head_idx][m]))
    head_filled = _fill_circular(head_biases, head_counts > 0)
    head_biases = (0.25 * np.roll(head_filled, 1) + 0.5 * head_filled
                   + 0.25 * np.roll(head_filled, -1))
    rm_head = ResidualModel(bins=bins, biases=head_biases,
                            counts=head_counts, alpha=alpha)

    tail_pred = model.predict(tail_t)
    tail_mean = np.asarray(tail_pred.mean, dtype=float)
    tail_y = y[-n_test:]
    corr_all = rm_head.correction(times)
    rmse_before = float(np.sqrt(np.mean((tail_mean - tail_y) ** 2)))
    rmse_after = float(
        np.sqrt(np.mean((tail_mean + corr_all[-n_test:] - tail_y) ** 2)))

    sigma = np.maximum(
        (np.asarray(tail_pred.upper, dtype=float)
         - np.asarray(tail_pred.lower, dtype=float)) / (2.0 * _Z95), 0.0)
    coverage_before = float(np.mean(
        np.abs(tail_y - tail_mean) <= 1.96 * sigma))
    q = None
    if conformal and float(np.median(sigma)) > 1e-9:
        scores = np.abs(tail_y - (tail_mean + rm_head.correction(tail_t))) / sigma
        q = conformal_quantile(scores, alpha)
        rm.conformal_q = q
    coverage_after = None
    if q is not None:
        coverage_after = float(np.mean(
            np.abs(tail_y - (tail_mean + corr_all[-n_test:]))
            <= q * sigma))

    diag = {
        "n": int(y.size),
        "bins": int(bins),
        "max_abs_bias": float(np.max(np.abs(biases))),
        "rmse_before": rmse_before,
        "rmse_after": rmse_after,
        "coverage_before": coverage_before,
        "coverage_after": coverage_after,
        "conformal_q": q,
        "n_test": int(n_test),
    }
    return rm, diag
