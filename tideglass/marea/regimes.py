"""Regime-conditional calibration — the v1.3 uncertainty upgrade.

A single conformal quantile is a blunt instrument: tidal prediction error is not
exchangeable across the fortnightly spring/neap cycle (large ranges carry larger
absolute error) nor across weather regimes (storms blow water where the harmonics
say it should not be). This module stratifies the split-conformal calibration by
regime, so each band uses the quantile of *its own* error distribution — the band
widens exactly when the evidence says it should.

Two regimes cross into four:

* **spring/neap** — from the local *predicted* tidal range (the fortnightly
  envelope of the M2/S2 beat). Large range ⇒ spring.
* **storm/non-storm** — from the magnitude of the residual (or, at forecast time,
  the v1.2 met-forced surge forecast): large ⇒ storm.

:func:`learn_regime_calibration` fits a per-regime conformal quantile;
:class:`RegimeCalibration` applies it (:meth:`apply`) so a prediction's band is
rescaled per regime. Persistence is a JSON artifact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from tideglass.marea.calibration import conformal_quantile
from tideglass.marea.model import Prediction, TideModel

_SPRING_PAD_H = 12  # half-window (hours) for the local tidal-range proxy
_REGIMES = ("neap", "spring", "neap_storm", "spring_storm")


@dataclass
class RegimeCalibration:
    """Per-regime conformal quantiles plus the thresholds that define the regimes."""

    alpha: float
    regimes: dict[str, float]  # regime label -> conformal q
    spring_range_median: float  # tidal-range threshold for spring/neap
    storm_resid_q: float  # |residual| threshold for storm/non-storm
    coverage: dict[str, float] = field(default_factory=dict)  # per-regime empirical coverage
    default_q: float = 1.96  # fallback when a regime is absent from the fit
    n: int = 0

    # -- classification ----------------------------------------------------------

    def classify(self, model: TideModel, times: Sequence[datetime],
                 surge=None, observed=None) -> list[str]:
        """Label each time with a regime (``spring``/``neap`` × ``storm``)."""
        times = list(times)
        ranges = _daily_range(model, times)
        spring = ranges >= self.spring_range_median
        if surge is not None:
            storm = np.abs(np.asarray(surge, dtype=float).ravel()) \
                >= self.storm_resid_q
        elif observed is not None:
            mean = np.asarray(model.predict(times).mean, dtype=float)
            storm = np.abs(np.asarray(observed, dtype=float).ravel() - mean) \
                >= self.storm_resid_q
        else:
            storm = np.zeros(len(times), dtype=bool)
        labels = []
        for s, st in zip(spring, storm):
            labels.append(("spring" if s else "neap") + ("_storm" if st else ""))
        return labels

    def quantile_for(self, model: TideModel, times: Sequence[datetime],
                     surge=None, observed=None) -> np.ndarray:
        labels = self.classify(model, times, surge=surge, observed=observed)
        return np.array([self.regimes.get(l, self.default_q) for l in labels],
                        dtype=float)

    # -- application -------------------------------------------------------------

    def apply(self, prediction: Prediction, times: Sequence[datetime],
              model: TideModel, surge=None, observed=None) -> Prediction:
        """Return the same prediction with per-regime-widened bands.

        The mean is untouched; the band uses each time's regime quantile instead
        of the global ``z = 1.96``.
        """
        q = self.quantile_for(model, times, surge=surge, observed=observed)
        mean = np.asarray(prediction.mean, dtype=float).ravel()
        lo = np.asarray(prediction.lower, dtype=float).ravel()
        hi = np.asarray(prediction.upper, dtype=float).ravel()
        sigma = np.maximum((hi - lo) / (2.0 * 1.96), 1e-12)
        half = q * sigma
        se = (np.asarray(prediction.se, dtype=float).ravel()
              if prediction.se is not None and np.asarray(prediction.se).size
              else np.full(mean.size, np.nan))
        return Prediction(mean=mean, lower=mean - half, upper=mean + half, se=se)

    # -- persistence -------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "alpha": float(self.alpha),
            "regimes": {k: float(v) for k, v in self.regimes.items()},
            "spring_range_median": float(self.spring_range_median),
            "storm_resid_q": float(self.storm_resid_q),
            "coverage": {k: float(v) for k, v in self.coverage.items()},
            "default_q": float(self.default_q),
            "n": int(self.n),
        }

    @classmethod
    def from_dict(cls, d: dict) -> RegimeCalibration:
        return cls(
            alpha=float(d["alpha"]),
            regimes={k: float(v) for k, v in d.get("regimes", {}).items()},
            spring_range_median=float(d["spring_range_median"]),
            storm_resid_q=float(d["storm_resid_q"]),
            coverage={k: float(v) for k, v in d.get("coverage", {}).items()},
            default_q=float(d.get("default_q", 1.96)),
            n=int(d.get("n", 0)),
        )


def _daily_range(model: TideModel, times: Sequence[datetime]) -> np.ndarray:
    """Local predicted tidal range (max − min) over a ±``_SPRING_PAD_H`` window.

    The fortnightly spring/neap envelope is overwhelmingly the M2/S2 beat, so
    this windowed range cleanly separates spring (large) from neap (small)
    regimes without scipy or a long record.
    """
    out = np.empty(len(times))
    for i, t in enumerate(times):
        grid = [t + timedelta(hours=h)
                for h in range(-_SPRING_PAD_H, _SPRING_PAD_H + 1)]
        pred = np.asarray(model.predict(grid).mean, dtype=float)
        out[i] = float(np.max(pred) - np.min(pred))
    return out


def learn_regime_calibration(model: TideModel, times: Sequence[datetime],
                             observed, alpha: float = 0.05, surge=None,
                             ) -> RegimeCalibration:
    """Fit per-regime conformal quantiles from a concurrent record.

    :param model: a fitted :class:`TideModel` (used both to score residuals and
        to derive the spring/neap range). It need not be the model fitted on
        ``times`` — any model that predicts the station suffices.
    :param times: timestamps of the concurrent ``observed`` record.
    :param observed: observed water levels (m).
    :param surge: optional concurrent surge (m) — e.g. the v1.2 met-forced
        forecast or a residual series — used to label the storm regime during
        the fit. If omitted, the fit labels storms from the observed residuals.
    :param alpha: conformal miscoverage level (band covers ``1 − alpha``).
    :returns: a :class:`RegimeCalibration`.
    """
    times = list(times)
    pred = model.predict(times)
    resid = np.asarray(observed, dtype=float).ravel() \
        - np.asarray(pred.mean, dtype=float).ravel()
    sigma = np.maximum(
        (np.asarray(pred.upper, dtype=float).ravel()
         - np.asarray(pred.lower, dtype=float).ravel()) / (2.0 * 1.96), 1e-12)
    scores = np.abs(resid) / sigma

    ranges = _daily_range(model, times)
    spring = ranges >= np.median(ranges)

    if surge is not None:
        storm_metric = np.abs(np.asarray(surge, dtype=float).ravel())
    else:
        storm_metric = np.abs(resid)
    storm = storm_metric >= np.quantile(storm_metric, 0.95) \
        if storm_metric.size else np.zeros(len(times), dtype=bool)

    regimes: dict[str, float] = {}
    coverage: dict[str, float] = {}
    default_q = float(conformal_quantile(scores, alpha))
    for label in _REGIMES:
        mask = _regime_mask(label, spring, storm)
        if mask.sum() < 1:
            continue
        q = float(conformal_quantile(scores[mask], alpha))
        regimes[label] = q
        coverage[label] = float(np.mean(np.abs(resid[mask]) <= q * sigma[mask]))

    return RegimeCalibration(
        alpha=float(alpha), regimes=regimes,
        spring_range_median=float(np.median(ranges)),
        storm_resid_q=float(np.quantile(storm_metric, 0.95))
        if storm_metric.size else 0.0,
        coverage=coverage, default_q=default_q, n=len(times))


def _regime_mask(label: str, spring: np.ndarray, storm: np.ndarray
                 ) -> np.ndarray:
    is_spring = label.startswith("spring")
    is_storm = label.endswith("storm")
    mask = np.ones(spring.shape, dtype=bool)
    mask &= (spring == is_spring)
    mask &= (storm == is_storm)
    return mask
