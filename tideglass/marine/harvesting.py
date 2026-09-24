"""Shellfish safe-window rules.

Safe hand-collection needs a workable low-water exposure window AND a
clean surge guard: no flagged residual/surge event in the preceding
``surge_guard_hours`` (runoff and resuspension risk).

This is an **exposure-timing screen, not a sanitation clearance**. A window
means the bed is uncovered for long enough to work and no surge was flagged
recently; it says nothing about microbiological water quality, biotoxins,
or local closures. Each returned :class:`HarvestWindow` carries that caveat in
its ``caution`` field.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from tideglass.marine.knowledge import HARVEST_DEFAULTS
from tideglass.marine.species import exposure_windows

HARVEST_CAUTION = (
    "exposure timing only; not a sanitation or biotoxin clearance, not a "
    "harvest licence, and not a substitute for local closures and advisories"
)


@dataclass(frozen=True)
class HarvestWindow:
    start: datetime
    end: datetime
    reason: str
    caution: str = HARVEST_CAUTION


def safe_windows(
    times: Sequence[datetime],
    heights,
    surge_flags=None,
    low_threshold_m: float = HARVEST_DEFAULTS["low_threshold_m"],
    min_hours: float = HARVEST_DEFAULTS["min_hours"],
    surge_guard_hours: float = HARVEST_DEFAULTS["surge_guard_hours"],
) -> list[HarvestWindow]:
    """Low-water work windows that pass the surge guard."""
    times = list(times)
    wins = exposure_windows(times, heights, low_threshold_m, min_hours)
    flags = list(surge_flags) if surge_flags is not None else [False] * len(times)
    if len(flags) != len(times):
        raise ValueError("surge_flags must match times")
    out: list[HarvestWindow] = []
    for start, end in wins:
        guard_from = start - timedelta(hours=surge_guard_hours)
        recent_surge = any(
            f and guard_from <= t <= end for t, f in zip(times, flags)
        )
        if recent_surge:
            continue
        out.append(HarvestWindow(
            start, end,
            f"low-water exposure >= {min_hours:g}h, no surge in prior "
            f"{surge_guard_hours:g}h",
        ))
    return out
