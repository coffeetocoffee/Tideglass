"""Rip-current screening from the tide curve (+ wave coupling in v0.4).

The score is a **screening heuristic, not a calibrated risk model**, and this
module now says so in its return value rather than only in prose. Rip-current
formation is driven primarily by wave breaking and bathymetry; a tide curve
alone is a weak predictor, so the returned :class:`RiskCurve` carries:

* the individual ``rate_term`` / ``range_term`` / ``wave_term`` contributions,
  so a consumer can see the decomposition instead of one opaque number;
* ``missing_inputs`` naming the dominant factors that were not supplied
  (``wave_height_m`` when waves are absent, plus bathymetry, which is never
  modelled);
* ``confidence``, which reports ``"screening"`` whenever the dominant wave
  input is missing and only ``"wave-informed"`` once it is supplied;
* ``sigma_m`` support, which propagates height uncertainty into the rate term
  instead of pretending the tide curve is exact.

Risk rises with the rate of tidal change (strong ebb flows) scaled by the
tidal range (spring tides push harder), and with wave energy when breaker
height/period are supplied. Bands are Low < 0.33 <= Moderate < 0.66 <= High.

This output must not be used for public-safety decisions; it is a tide-only
screening aid. Actual lifeguarding decisions require the local forecast, beach
state, and bathymetry.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from tideglass.marine.knowledge import RIP_DEFAULTS

SCREENING_DISCLAIMER = (
    "tide-only screening heuristic; not a calibrated rip forecast and not for "
    "public-safety decisions"
)


@dataclass(frozen=True)
class RiskCurve:
    times: list[datetime]
    score: np.ndarray  # 0..1
    category: list[str]  # Low / Moderate / High
    basis: str
    rate_term: np.ndarray = field(default_factory=lambda: np.zeros(0))
    range_term: np.ndarray = field(default_factory=lambda: np.zeros(0))
    wave_term: np.ndarray = field(default_factory=lambda: np.zeros(0))
    missing_inputs: tuple[str, ...] = ()
    confidence: str = "screening"

    @property
    def usable_for_safety(self) -> bool:
        """Always ``False``: waves and bathymetry dominate and are not modelled."""
        return False


def _categories(score: np.ndarray) -> list[str]:
    return ["Low" if s < 0.33 else "Moderate" if s < 0.66 else "High"
            for s in score]


def _rate_sigma(hours: np.ndarray, sigma_m: float) -> np.ndarray:
    """Std error of the ``np.gradient`` rate for iid height noise.

    The central difference ``(h[i+1] - h[i-1]) / (t[i+1] - t[i-1])`` propagates
    independent height errors to ``sigma * sqrt(1/dt_left^2 + 1/dt_right^2)``.
    One-sided ends use their single-sided coefficient. This lets the score widen
    when the tide prediction is itself uncertain, rather than treating a noisy
    curve as a confident signal.
    """
    n = hours.size
    if n < 2 or sigma_m <= 0.0:
        return np.zeros(n)
    dts = np.diff(hours)
    dts = np.where(dts > 0, dts, np.nan)
    var = np.zeros(n)
    if n == 2:
        var[:] = (sigma_m / dts[0]) ** 2
        return np.sqrt(var)
    var[1:-1] = (sigma_m**2) * (1.0 / dts[:-1] ** 2 + 1.0 / dts[1:] ** 2)
    var[0] = (sigma_m / dts[0]) ** 2
    var[-1] = (sigma_m / dts[-1]) ** 2
    return np.sqrt(var)


def risk(
    times: Sequence[datetime],
    heights,
    ref_rate: float = RIP_DEFAULTS["ref_rate_m_per_h"],
    ref_range: float = RIP_DEFAULTS["ref_range_m"],
    rate_weight: float = RIP_DEFAULTS["rate_weight"],
    range_weight: float = RIP_DEFAULTS["range_weight"],
    wave_weight: float = RIP_DEFAULTS["wave_weight"],
    ref_wave_height: float = RIP_DEFAULTS["ref_wave_height_m"],
    wave_height_m: float | None = None,
    wave_period_s: float | None = None,
    sigma_m: float | None = None,
    z: float = 1.96,
) -> RiskCurve:
    times = list(times)
    h = np.asarray(list(heights), dtype=float).ravel()
    if len(times) != h.size or h.size < 2:
        raise ValueError("need >= 2 matching times/heights")
    hours = np.array([t.timestamp() for t in times]) / 3600.0
    rate = np.abs(np.gradient(h, hours))  # m per hour
    if sigma_m is not None:
        rate = rate + z * _rate_sigma(hours, float(sigma_m))
    span = float(h.max() - h.min())
    norm_rate = np.clip(rate / ref_rate, 0, 1)
    rate_term = rate_weight * norm_rate
    range_term = range_weight * np.clip(span / ref_range, 0, 1) * norm_rate
    missing: list[str] = ["bathymetry"]
    if wave_weight and wave_height_m is not None:
        # Wave energy ~ H_b^2 · T (Iribarren-family proxy for rip forcing);
        # normalise by the reference height and keep it bounded in [0, 1].
        T = wave_period_s if wave_period_s else 8.0
        energy = (wave_height_m**2) * T
        ref_energy = (ref_wave_height**2) * 8.0
        wave = wave_weight * float(np.clip(energy / ref_energy, 0, 1))
        wave_term = np.full(h.size, wave, dtype=float)
        confidence = "wave-informed" if wave_weight > 0 else "screening"
        basis = (
            f"tidal rate x range x wave-energy screening heuristic "
            f"(rate_weight={rate_weight:g}, range_weight={range_weight:g}, "
            f"wave_weight={wave_weight:g}); {SCREENING_DISCLAIMER}"
        )
    else:
        wave_term = np.zeros(h.size, dtype=float)
        if not wave_height_m:
            missing.insert(0, "wave_height_m")
        confidence = "screening"
        basis = (
            f"tidal rate-of-change x range screening heuristic "
            f"(rate_weight={rate_weight:g}, range_weight={range_weight:g}); "
            f"waves not supplied; {SCREENING_DISCLAIMER}"
        )
    score = np.clip(rate_term + range_term + wave_term, 0, 1)
    return RiskCurve(
        times=times,
        score=score,
        category=_categories(score),
        basis=basis,
        rate_term=rate_term,
        range_term=range_term,
        wave_term=wave_term,
        missing_inputs=tuple(missing),
        confidence=confidence,
    )

