"""Rip-current risk from the tide curve.

Heuristic, documented as such: risk rises with the rate of tidal change
(strong ebb flows) scaled by the tidal range (spring tides push harder).
Score blends a rate term and a range term; bands are Low < 0.33 ≤
Moderate < 0.66 ≤ High. Waves and morphology dominate real rip risk —
tide is one ingredient, and the score says so in ``basis``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Sequence

import numpy as np

from tideglass.marine.knowledge import RIP_DEFAULTS


@dataclass(frozen=True)
class RiskCurve:
    times: List[datetime]
    score: np.ndarray  # 0..1
    category: List[str]  # Low / Moderate / High
    basis: str = ("tidal rate-of-change × range heuristic "
                  "(waves + morphology not included)")


def _categories(score: np.ndarray) -> List[str]:
    return ["Low" if s < 0.33 else "Moderate" if s < 0.66 else "High"
            for s in score]


def risk(
    times: Sequence[datetime],
    heights,
    ref_rate: float = RIP_DEFAULTS["ref_rate_m_per_h"],
    ref_range: float = RIP_DEFAULTS["ref_range_m"],
    rate_weight: float = RIP_DEFAULTS["rate_weight"],
    range_weight: float = RIP_DEFAULTS["range_weight"],
) -> RiskCurve:
    times = list(times)
    h = np.asarray(list(heights), dtype=float).ravel()
    if len(times) != h.size or h.size < 2:
        raise ValueError("need ≥ 2 matching times/heights")
    hours = np.array([t.timestamp() for t in times]) / 3600.0
    rate = np.abs(np.gradient(h, hours))  # m per hour
    span = float(h.max() - h.min())
    score = np.clip(
        rate_weight * np.clip(rate / ref_rate, 0, 1)
        + range_weight * np.clip(span / ref_range, 0, 1) * np.clip(rate / ref_rate, 0, 1),
        0, 1,
    )
    return RiskCurve(times=times, score=score, category=_categories(score))
