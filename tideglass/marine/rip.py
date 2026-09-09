"""Rip-current risk from the tide curve (+ wave coupling in v0.4).

Heuristic, documented as such: risk rises with the rate of tidal change
(strong ebb flows) scaled by the tidal range (spring tides push harder), and
now also with wave energy when breaker height/period are supplied
(``wave_height_m`` / ``wave_period_s``). Score blends a rate term, a range term,
and an optional wave term; bands are Low < 0.33 ≤ Moderate < 0.66 ≤ High.
Waves and morphology dominate real rip risk — tide is one ingredient, and the
score says so in ``basis``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from tideglass.marine.knowledge import RIP_DEFAULTS


@dataclass(frozen=True)
class RiskCurve:
    times: list[datetime]
    score: np.ndarray  # 0..1
    category: list[str]  # Low / Moderate / High
    basis: str = ("tidal rate-of-change × range heuristic "
                  "(waves + morphology not included)")


def _categories(score: np.ndarray) -> list[str]:
    return ["Low" if s < 0.33 else "Moderate" if s < 0.66 else "High"
            for s in score]


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
) -> RiskCurve:
    times = list(times)
    h = np.asarray(list(heights), dtype=float).ravel()
    if len(times) != h.size or h.size < 2:
        raise ValueError("need >= 2 matching times/heights")
    hours = np.array([t.timestamp() for t in times]) / 3600.0
    rate = np.abs(np.gradient(h, hours))  # m per hour
    span = float(h.max() - h.min())
    tidal = (
        rate_weight * np.clip(rate / ref_rate, 0, 1)
        + range_weight * np.clip(span / ref_range, 0, 1) * np.clip(rate / ref_rate, 0, 1)
    )
    if wave_weight and wave_height_m is not None:
        # Wave energy ~ H_b^2 · T (Iribarren-family proxy for rip forcing);
        # normalise by the reference height and keep it bounded in [0, 1].
        T = wave_period_s if wave_period_s else 8.0
        energy = (wave_height_m**2) * T
        ref_energy = (ref_wave_height**2) * 8.0
        wave = wave_weight * float(np.clip(energy / ref_energy, 0, 1))
        tidal = tidal + wave
        basis = ("tidal rate-of-change × range × wave-energy heuristic "
                 f"(wave_weight={wave_weight:g})")
    else:
        basis = ("tidal rate-of-change × range heuristic "
                 "(waves not supplied)")
    score = np.clip(tidal, 0, 1)
    return RiskCurve(times=times, score=score, category=_categories(score), basis=basis)

