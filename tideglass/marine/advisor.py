"""TideAdvisor — turns Marea Core predictions into marine advice.

Consumes ONLY ``TideModel.predict()`` output (plus an optional surge-flag
series): harvesting windows, rip-current risk, and species exposure. It
never reaches into signal math.

v0.4 adds region-specific rulesets (``region=``) and wave-coupled rip risk
(``wave_height_m`` / ``wave_period_s``), both flowing through
:mod:`tideglass.marine.knowledge` so the marine layer stays a thin consumer.

v0.8 adds the probabilistic threshold: pass a ``flood_threshold_m`` (e.g. a
GPD return level from :mod:`tideglass.marea.extremes`) and the advice prices
``P(level > threshold)`` per time from the prediction band instead of
tripping a fixed heuristic cutoff.

v1.0 makes the advice decision-theoretic: pass a ``cost``/``loss`` pair (the
economics of acting vs being hit) plus the event threshold and the advice
prices the act/wait policy per hour via :mod:`tideglass.marea.decision` —
expected utility over the CI instead of fixed thresholds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from tideglass.marea import decision as DC
from tideglass.marea.extremes import flood_probability
from tideglass.marea.model import Prediction, TideModel
from tideglass.marine import harvesting as HV
from tideglass.marine import rip as RIP
from tideglass.marine import species as SP
from tideglass.marine.knowledge import get_region


@dataclass(frozen=True)
class Advice:
    station: str | None
    region: str | None
    times: list[datetime]
    prediction: Prediction
    rip: RIP.RiskCurve
    harvest: list[HV.HarvestWindow]
    exposure: dict[str, float]  # species name -> exposed fraction
    summary: str
    flood: np.ndarray | None = None  # P(level > flood_threshold_m) per time
    flood_threshold_m: float | None = None
    decision: DC.DecisionCurve | None = None  # priced act/wait policy


class TideAdvisor:
    """Advise on harvesting, rip risk, and species for a fitted model."""

    def __init__(self, model: TideModel, species_names: Sequence[str] | None = None,
                 region: str | None = None):
        self.model = model
        self.region = region
        self.rules = get_region(region)
        self.species = [SP.get(n) for n in
                        (species_names or ["Pacific oyster", "Blue mussel", "Manila clam"])]

    def advise(self, times: Sequence[datetime], surge_flags=None, wave_height_m=None,
                wave_period_s=None, flood_threshold_m: float | None = None,
                surge_sigma_m: float = 0.0,
                decision_threshold_m: float | None = None,
                cost: float | None = None,
                loss: float | None = None) -> Advice:
        """Advise for ``times``.

        :param flood_threshold_m: when given (e.g. a GPD return level), price
            ``P(water level > threshold)`` per time from the prediction band
            plus a ``N(0, surge_sigma_m²)`` surge — the probabilistic
            replacement for a fixed heuristic cutoff.
        :param decision_threshold_m: event threshold (m) for the cost-loss
            policy; defaults to ``flood_threshold_m`` when omitted.
        :param cost: cost of the protective action per time step.
        :param loss: loss if the event hits during an unprepared time step.
            When ``cost`` and ``loss`` are both given, the advice also carries
            a priced act/wait :class:`~tideglass.marea.decision.DecisionCurve`.
        """
        times = list(times)
        pred = self.model.predict(times)
        mean = pred.mean
        rip_cfg = self.rules["rip"]
        rip = RIP.risk(
            times, mean,
            ref_rate=rip_cfg["ref_rate_m_per_h"],
            ref_range=rip_cfg["ref_range_m"],
            rate_weight=rip_cfg["rate_weight"],
            range_weight=rip_cfg["range_weight"],
            wave_weight=rip_cfg["wave_weight"],
            ref_wave_height=rip_cfg["ref_wave_height_m"],
            wave_height_m=wave_height_m, wave_period_s=wave_period_s,
        )
        hv_cfg = self.rules["harvest"]
        harvest = HV.safe_windows(
            times, mean, surge_flags,
            low_threshold_m=hv_cfg["low_threshold_m"],
            min_hours=hv_cfg["min_hours"],
            surge_guard_hours=hv_cfg["surge_guard_hours"],
        )
        exposure = {s.name: SP.exposure_fraction(times, mean, s.exposure_threshold_m)
                    for s in self.species}
        flood = None
        if flood_threshold_m is not None:
            flood = np.asarray(
                flood_probability(pred, flood_threshold_m,
                                  surge_sigma=surge_sigma_m),
                dtype=float)
        decision = None
        if cost is not None and loss is not None:
            threshold = (decision_threshold_m
                         if decision_threshold_m is not None
                         else flood_threshold_m)
            if threshold is None:
                raise ValueError(
                    "cost/loss decision requires an event threshold: pass "
                    "decision_threshold_m or flood_threshold_m")
            decision = DC.decision_curve(
                times, pred, threshold, cost, loss, surge_sigma=surge_sigma_m,
            )
        peak = int(rip.score.argmax())
        scope = f" (region={self.region})" if self.region else ""
        lines = [
            (
                f"station {self.model.station}{scope}: "
                f"{len(harvest)} safe harvest window(s), "
                f"peak rip risk {rip.category[peak]} ({rip.score[peak]:.2f}) "
                f"at {rip.times[peak].isoformat()}"
            )
        ]
        for w in harvest:
            lines.append(f"  harvest {w.start.isoformat()} -> {w.end.isoformat()} ({w.reason})")
        for s in self.species:
            lines.append(f"  {s.name}: exposed {exposure[s.name] * 100:.0f}% of period")
        if flood is not None:
            lines.append(f"  max P(level > {flood_threshold_m:.2f} m) = "
                         f"{float(np.max(flood)):.1%}")
        if decision is not None:
            lines.append(f"  {decision}")
        return Advice(
            station=self.model.station, region=self.region, times=times, prediction=pred,
            rip=rip, harvest=harvest, exposure=exposure,
            summary="\n".join(lines),
            flood=flood, flood_threshold_m=flood_threshold_m,
            decision=decision,
        )

