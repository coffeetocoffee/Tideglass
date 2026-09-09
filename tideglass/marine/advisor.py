"""TideAdvisor — turns Marea Core predictions into marine advice.

Consumes ONLY ``TideModel.predict()`` output (plus an optional surge-flag
series): harvesting windows, rip-current risk, and species exposure. It
never reaches into signal math.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Sequence

from tideglass.marea.model import Prediction, TideModel
from tideglass.marine import harvesting as HV
from tideglass.marine import rip as RIP
from tideglass.marine import species as SP


@dataclass(frozen=True)
class Advice:
    station: str | None
    times: List[datetime]
    prediction: Prediction
    rip: RIP.RiskCurve
    harvest: List[HV.HarvestWindow]
    exposure: Dict[str, float]  # species name -> exposed fraction
    summary: str


class TideAdvisor:
    """Advise on harvesting, rip risk, and species for a fitted model."""

    def __init__(self, model: TideModel, species_names: Sequence[str] | None = None):
        self.model = model
        self.species = [SP.get(n) for n in
                        (species_names or ["Pacific oyster", "Blue mussel", "Manila clam"])]

    def advise(self, times: Sequence[datetime], surge_flags=None) -> Advice:
        times = list(times)
        pred = self.model.predict(times)
        mean = pred.mean
        rip = RIP.risk(times, mean)
        harvest = HV.safe_windows(times, mean, surge_flags)
        exposure = {s.name: SP.exposure_fraction(times, mean, s.exposure_threshold_m)
                    for s in self.species}
        peak = int(rip.score.argmax())
        lines = [
            f"station {self.model.station}: "
            f"{len(harvest)} safe harvest window(s), "
            f"peak rip risk {rip.category[peak]} ({rip.score[peak]:.2f}) "
            f"at {rip.times[peak].isoformat()}",
        ]
        for w in harvest:
            lines.append(f"  harvest {w.start.isoformat()} → {w.end.isoformat()} ({w.reason})")
        for s in self.species:
            lines.append(f"  {s.name}: exposed {exposure[s.name] * 100:.0f}% of period")
        return Advice(
            station=self.model.station, times=times, prediction=pred,
            rip=rip, harvest=harvest, exposure=exposure,
            summary="\n".join(lines),
        )
