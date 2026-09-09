"""TideAdvisor — turns Marea Core predictions into marine advice.

Consumes ONLY ``TideModel.predict()`` output (plus an optional surge-flag
series): harvesting windows, rip-current risk, and species exposure. It
never reaches into signal math.

v0.4 adds region-specific rulesets (``region=``) and wave-coupled rip risk
(``wave_height_m`` / ``wave_period_s``), both flowing through
:mod:`tideglass.marine.knowledge` so the marine layer stays a thin consumer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

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
               wave_period_s=None) -> Advice:
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
        return Advice(
            station=self.model.station, region=self.region, times=times, prediction=pred,
            rip=rip, harvest=harvest, exposure=exposure,
            summary="\n".join(lines),
        )

