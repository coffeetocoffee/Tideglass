"""Intertidal species metadata + exposure analysis.

Given a predicted tide curve, report when each species' zone is exposed
(tide below its threshold) and therefore active/collectable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class Species:
    name: str
    group: str  # e.g. bivalve, crustacean, algae
    zone: str  # upper / mid / lower intertidal
    exposure_threshold_m: float  # exposed when tide below this
    note: str


SPECIES: List[Species] = [
    Species("Pacific oyster", "bivalve", "mid", 0.4,
            "Filter-feeds when submerged; accessible on low-water exposure."),
    Species("Blue mussel", "bivalve", "mid", 0.5,
            "Dense beds mid-shore; collectable around low water."),
    Species("Manila clam", "bivalve", "lower", 0.2,
            "Buried in sediment; needs a good low tide to dig."),
    Species("Green crab", "crustacean", "lower", 0.3,
            "Forages on the flooding tide; most active near low-water turns."),
    Species("Kelp", "algae", "lower", 0.1,
            "Extreme low tides expose the fringe for survey."),
]

BY_NAME: Dict[str, Species] = {s.name: s for s in SPECIES}


def get(name: str) -> Species:
    return BY_NAME[name]


def exposure_windows(
    times: Sequence[datetime],
    heights,
    threshold_m: float,
    min_hours: float = 1.0,
) -> List[Tuple[datetime, datetime]]:
    """Contiguous runs with height below threshold, at least ``min_hours``."""
    import numpy as np

    times = list(times)
    h = np.asarray(list(heights), dtype=float).ravel()
    if len(times) != h.size or not times:
        raise ValueError("times/heights length mismatch or empty")
    dts = np.diff(np.array([t.timestamp() for t in times])) / 3600.0
    dt = float(np.median(dts)) if dts.size else 1.0
    below = h < threshold_m
    windows: List[Tuple[datetime, datetime]] = []
    start = None
    for i, b in enumerate(below):
        if b and start is None:
            start = i
        if not b and start is not None:
            if (i - start) * dt >= min_hours:
                windows.append((times[start], times[i - 1]))
            start = None
    if start is not None and (len(below) - start) * dt >= min_hours:
        windows.append((times[start], times[-1]))
    return windows


def exposure_fraction(times, heights, threshold_m: float) -> float:
    """Fraction of the period the zone is exposed."""
    import numpy as np

    h = np.asarray(list(heights), dtype=float).ravel()
    if h.size == 0:
        raise ValueError("empty heights")
    return float(np.mean(h < threshold_m))
