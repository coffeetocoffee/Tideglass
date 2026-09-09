"""Shellfish safe-window rules.

Safe hand-collection needs a workable low-water exposure window AND a
clean surge guard: no flagged residual/surge event in the preceding
``surge_guard_hours`` (runoff and resuspension risk).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Sequence

from tideglass.marine.knowledge import HARVEST_DEFAULTS
from tideglass.marine.species import exposure_windows


@dataclass(frozen=True)
class HarvestWindow:
    start: datetime
    end: datetime
    reason: str


def safe_windows(
    times: Sequence[datetime],
    heights,
    surge_flags=None,
    low_threshold_m: float = HARVEST_DEFAULTS["low_threshold_m"],
    min_hours: float = HARVEST_DEFAULTS["min_hours"],
    surge_guard_hours: float = HARVEST_DEFAULTS["surge_guard_hours"],
) -> List[HarvestWindow]:
    """Low-water work windows that pass the surge guard."""
    times = list(times)
    wins = exposure_windows(times, heights, low_threshold_m, min_hours)
    flags = list(surge_flags) if surge_flags is not None else [False] * len(times)
    if len(flags) != len(times):
        raise ValueError("surge_flags must match times")
    out: List[HarvestWindow] = []
    for start, end in wins:
        guard_from = start - timedelta(hours=surge_guard_hours)
        recent_surge = any(
            f and guard_from <= t <= end for t, f in zip(times, flags)
        )
        if recent_surge:
            continue
        out.append(HarvestWindow(
            start, end,
            f"low-water exposure ≥ {min_hours:g}h, no surge in prior "
            f"{surge_guard_hours:g}h",
        ))
    return out
