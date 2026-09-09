"""Surge-flag alerting for watched stations (v0.4 marine depth).

Operators watch a set of gauges and want to know *when* the non-tidal residual
crossed a danger threshold — a storm-surge event. This module consumes the
astronomical prediction and the observed level, isolates the residual via
:mod:`tideglass.marea.surge`, and groups flagged samples into discrete events
with their peak magnitude.

``TideAdvisor`` never reaches into this; alerting is a separate operational
consumer of the same ``predict()`` output, exactly like the harvesting/rip rules.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from tideglass.marea.model import TideModel
from tideglass.marea.surge import decompose


@dataclass(frozen=True)
class SurgeEvent:
    station: str
    start: datetime
    end: datetime
    peak_residual_m: float  # most extreme residual during the event
    threshold_m: float


@dataclass(frozen=True)
class StationWatch:
    """A watched station and its alerting threshold."""

    name: str
    threshold_m: float = 0.3  # |residual| above this raises an alert


def surge_events(
    observed,
    predicted_mean,
    times: Sequence[datetime],
    station: str = "?",
    k: float = 3.0,
    threshold_m: float | None = None,
) -> list[SurgeEvent]:
    """Group flagged surge residuals into discrete events.

    :param k: surge-flag multiple of the residual σ (see ``surge.decompose``);
        ignored if ``threshold_m`` is given directly.
    :param threshold_m: absolute ``|residual|`` trigger (m); overrides ``k``.
    """
    dec = decompose(np.asarray(observed, dtype=float).ravel(),
                   np.asarray(predicted_mean, dtype=float).ravel(),
                   k=k)
    thresh = float(threshold_m) if threshold_m is not None else dec.threshold
    flags = np.abs(dec.residual) > thresh
    times = list(times)
    resid = dec.residual
    events: list[SurgeEvent] = []
    start = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            seg = resid[start:i]
            events.append(SurgeEvent(
                station=station, start=times[start], end=times[i - 1],
                peak_residual_m=float(np.max(np.abs(seg))), threshold_m=thresh,
            ))
            start = None
    if start is not None:
        seg = resid[start:]
        events.append(SurgeEvent(
            station=station, start=times[start], end=times[-1],
            peak_residual_m=float(np.max(np.abs(seg))), threshold_m=thresh,
        ))
    return events


def watch(station: TideModel, times: Sequence[datetime], observed,
         watches: Sequence[StationWatch] | None = None) -> list[SurgeEvent]:
    """Evaluate surge alerts for watched stations against a fitted model."""
    times = list(times)
    pred = station.predict(times)
    watches = list(watches or [StationWatch(name=station.station or "?", threshold_m=0.3)])
    out: list[SurgeEvent] = []
    for w in watches:
        if w.name != station.station:
            continue
        out.extend(surge_events(observed, pred.mean, times,
                                station=w.name, threshold_m=w.threshold_m))
    return out
