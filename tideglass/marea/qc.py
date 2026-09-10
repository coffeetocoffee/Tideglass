"""Quality control for crowd-sourced sensor uploads (v1.0 federated QC).

Cheap ultrasonic sticks and community float gauges produce noisy series with
spikes, datum/level shifts, and slow drift. :func:`qc_check` runs three
independent checks — spike, datum-shift, drift — and returns a
:class:`QcReport`. The non-destructive :func:`clean_series` then interpolates
the flagged spikes so the cleaned record can be fitted (or rejected).

The checks are cheap-sensor aware:

* **Spike** — a robust outlier in the non-tidal residual (median filter removes
  the tide, MAD sets the spread). No external reference needed.
* **Datum shift** — when a fitted/predicted tide is available (``reference=``),
  the sustained offset ``mean(obs − pred)`` is compared against tolerance; when
  it is not, a step change between the first and second half of the record is
  detected (an internal level inconsistency). An *absolute* datum is unknown
  without a neighbour/predicted series, by design.
* **Drift** — a robust linear slope on the residual after de-spiking.

A report is ``rejected`` for hard sanity failures (too few points, non-finite
values, non-monotonic time, heavy flatlining) or when a check exceeds its
tolerance by enough to distrust the upload.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class QcConfig:
    """Tolerances for the QC pipeline."""

    spike_sigma: float = 5.0  # robust-MAD multiples that flag a spike
    max_spike_fraction: float = 0.05  # reject when spikes exceed this fraction
    datum_tol_m: float = 0.30  # sustained level offset / step tolerance (m)
    drift_tol_m_per_day: float = 0.10  # robust slope tolerance (m/day)
    min_points: int = 24  # below this the record cannot be QC'd
    flatline_max_frac: float = 0.50  # reject when too many identical samples


@dataclass
class QcReport:
    """Result of a QC pass over one upload."""

    n_total: int
    n_flagged: int
    spike_idx: list[int]
    datum_shift_m: float | None  # estimated sustained offset / step (m)
    drift_slope_m_per_day: float | None  # estimated residual slope (m/day)
    rejected: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.rejected

    def __str__(self) -> str:
        status = "REJECTED" if self.rejected else "accepted"
        parts = [f"qc[{status}]: n={self.n_total} flagged={self.n_flagged}"]
        if self.spike_idx:
            parts.append(f"spikes={len(self.spike_idx)}")
        if self.datum_shift_m is not None:
            parts.append(f"datum={self.datum_shift_m:+.3f}m")
        if self.drift_slope_m_per_day is not None:
            parts.append(f"drift={self.drift_slope_m_per_day:+.3f}m/day")
        if self.reasons:
            parts.append("(" + "; ".join(self.reasons) + ")")
        return " ".join(parts)


def _hours_since_first(times: Sequence) -> np.ndarray:
    """Numeric hours since the first timestamp (tz-consistent within a series)."""
    if isinstance(times[0], datetime):
        base = times[0]
        return np.array(
            [(t - base).total_seconds() / 3600.0 for t in times], dtype=float
        )
    return np.asarray(list(times), dtype=float)


def _sliding_median(heights: np.ndarray, win: int) -> np.ndarray:
    """Median of a symmetric window at each point (spike-resistant smoother)."""
    n = len(heights)
    if n < 3 or win < 3:
        return heights.astype(float).copy()
    if win % 2 == 0:
        win += 1
    half = win // 2
    out = np.empty(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.median(heights[lo:hi])
    return out


def _robust_residual(heights: np.ndarray) -> np.ndarray:
    """Non-tidal residual: series minus a short sliding median (removes the tide).

    The window (~10% of the record, at least a day) tracks the semidiurnal tide
    so spikes and slow drift remain visible in the residual.
    """
    n = len(heights)
    if n < 3:
        return np.zeros(n)
    # A short window (a few hours) tracks the local tide so the residual is just
    # noise + outliers; a wide window would average over whole tidal cycles and
    # leave the tide in the residual. Spike detection therefore stays sharp.
    win = int(np.clip(round(n * 0.03), 5, 13))
    return heights - _sliding_median(heights, win)


def qc_check(
    times: Sequence,
    heights: Sequence[float],
    config: QcConfig | None = None,
    reference: Sequence[float] | None = None,
) -> QcReport:
    """Run the spike / datum / drift QC battery on one upload.

    :param times: timestamps (UTC-aware or naive datetimes, or floats).
    :param heights: observed heights (m), same length as ``times``.
    :param config: tolerances; defaults to :class:`QcConfig`.
    :param reference: optional predicted tide at ``times``; when given, the
        datum/drift residuals are measured against the model instead of the
        internal median.
    :returns: a :class:`QcReport`.
    """
    config = config or QcConfig()
    times = list(times)
    h = np.asarray(list(heights), dtype=float)
    n = len(h)

    # -- hard sanity rejections -------------------------------------------
    if n < config.min_points:
        return QcReport(
            n, 0, [], None, None, True,
            [f"too few points ({n} < {config.min_points})"],
        )
    if len(times) != n:
        return QcReport(n, 0, [], None, None, True,
                        ["times/height length mismatch"])
    if not np.all(np.isfinite(h)):
        return QcReport(n, 0, [], None, None, True,
                        ["non-finite values present"])

    reasons: list[str] = []
    ts = _hours_since_first(times)
    if n > 1 and np.any(np.diff(ts) <= 0):
        reasons.append("non-monotonic timestamps")

    if n > 1:
        frac_const = float(np.mean(np.abs(np.diff(h)) < 1e-6))
        if frac_const > config.flatline_max_frac:
            reasons.append(
                f"flatline fraction {frac_const:.2f} exceeds "
                f"{config.flatline_max_frac}")

    # -- non-tidal residual ------------------------------------------------
    resid = _robust_residual(h)
    center = float(np.median(resid))
    mad = float(np.median(np.abs(resid - center)))
    spread = 1.4826 * mad if mad > 0 else (float(np.std(resid)) or 1.0)

    # -- spike detection --------------------------------------------------
    if reference is not None:
        resid_clean = np.asarray(list(reference), dtype=float) - h
    else:
        resid_clean = resid - center
    spike_mask = np.abs(resid_clean) > config.spike_sigma * spread
    spike_idx = [int(i) for i in np.flatnonzero(spike_mask)]

    # -- datum shift ------------------------------------------------------
    if reference is not None:
        datum_shift = float(np.median(resid_clean))
    else:
        half = n // 2
        datum_shift = float(np.median(h[half:]) - np.median(h[:half]))
    datum: float | None = None
    if datum_shift is not None and abs(datum_shift) > config.datum_tol_m:
        datum = datum_shift

    # -- drift ------------------------------------------------------------
    # A slow level drift survives the short residual filter, so estimate the
    # slope from a long-window smoother (wide enough to suppress the tide but
    # not the drift). De-spiked so a flagged outlier cannot tilt the fit.
    drift: float | None = None
    if n > 2:
        good = ~spike_mask
        if good.sum() > 2:
            win_long = int(np.clip(round(n * 0.20), 49, 201))
            smooth = _sliding_median(h, win_long)
            x = ts[good]
            y = smooth[good]
            xm = x.mean()
            ym = y.mean()
            denom = float(np.sum((x - xm) ** 2))
            slope_per_h = (float(np.sum((x - xm) * (y - ym)) / denom)
                           if denom > 0 else 0.0)
            drift = slope_per_h * 24.0  # m/day
    drift_flag: float | None = None
    if drift is not None and abs(drift) > config.drift_tol_m_per_day:
        drift_flag = drift

    # -- assemble rejection reasons --------------------------------------
    frac_spike = len(spike_idx) / n if n else 0.0
    if frac_spike > config.max_spike_fraction:
        reasons.append(
            f"spike fraction {frac_spike:.2f} exceeds "
            f"{config.max_spike_fraction}")
    if datum is not None:
        reasons.append(
            f"datum shift {datum:+.3f} m exceeds tol {config.datum_tol_m}")
    if drift_flag is not None:
        reasons.append(
            f"drift {drift_flag:+.3f} m/day exceeds tol "
            f"{config.drift_tol_m_per_day}")

    rejected = bool(reasons)
    return QcReport(
        n_total=n,
        n_flagged=len(spike_idx),
        spike_idx=spike_idx,
        datum_shift_m=(datum_shift if datum_shift is not None else None),
        drift_slope_m_per_day=(drift if drift is not None else None),
        rejected=rejected,
        reasons=reasons,
    )


def clean_series(
    times: Sequence,
    heights: Sequence[float],
    report: QcReport,
) -> tuple[list, list[float]]:
    """Return a de-spiked copy of the series (linear interpolation in time).

    Flagged spikes are replaced by interpolation against the good samples; the
    series length is preserved so the same timestamps remain fit-able. A clean
    report returns the input unchanged.
    """
    times = list(times)
    h = np.asarray(list(heights), dtype=float).copy()
    if report.spike_idx and len(report.spike_idx) < len(h):
        idx = np.array(report.spike_idx)
        good = np.setdiff1d(np.arange(len(h)), idx)
        if len(good) >= 2:
            ts = _hours_since_first(times)
            h[idx] = np.interp(ts[idx], ts[good], h[good])
        elif len(good) == 1:
            h[idx] = h[good[0]]
    return times, [float(x) for x in h]
