"""Per-station service-level objectives (v3.1).

The drift monitor (v0.6) answers *"is this model stale?"*; the SLO layer
answers *"does this model meet the service level its operator declared?"* — a
coverage floor, an RMSE ceiling, and a bias ceiling, per station. The autonomous
loop (v3.1) reports per-station SLO compliance (``met`` / ``degrading`` /
``breached``) alongside the health rules, so *"is the network healthy"* becomes
a query, not a judgment call.

SLO config is persisted per station as ``<store>/<station>.slo.json`` so the
operator's targets survive restarts and the loop enforces them without being
told each pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from tideglass.marea.drift import HealthReport


@dataclass
class SloConfig:
    """A station's declared service level."""

    station: str
    coverage_target: float = 0.90
    coverage_tolerance: float = 0.05
    rmse_target: float | None = None
    rmse_tolerance: float | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.coverage_target <= 1.0):
            raise ValueError(
                f"coverage_target must lie in [0,1], got {self.coverage_target}"
            )
        if self.coverage_tolerance < 0:
            raise ValueError(
                f"coverage_tolerance must be >= 0, got {self.coverage_tolerance}"
            )
        if self.rmse_target is not None and self.rmse_target <= 0:
            raise ValueError(f"rmse_target must be positive, got {self.rmse_target}")
        if self.rmse_tolerance is not None and self.rmse_tolerance < 0:
            raise ValueError(f"rmse_tolerance must be >= 0, got {self.rmse_tolerance}")

    def _coverage_status(self, cov: float, n: int) -> str:
        if n == 0:
            return "n/a"
        if cov >= self.coverage_target:
            return "met"
        if cov >= self.coverage_target - self.coverage_tolerance:
            return "degrading"
        return "breached"

    def _rmse_status(self, rmse: float, n: int) -> str:
        if n == 0 or self.rmse_target is None:
            return "n/a"
        tol = self.rmse_tolerance if self.rmse_tolerance is not None else 0.0
        if rmse <= self.rmse_target:
            return "met"
        if rmse <= self.rmse_target + tol:
            return "degrading"
        return "breached"

    def evaluate(self, health: HealthReport) -> SloReport:
        cov = float(health.coverage)
        rmse = float(health.rmse)
        cov_s = self._coverage_status(cov, health.n)
        rmse_s = self._rmse_status(rmse, health.n)
        rank = {"met": 0, "degrading": 1, "breached": 2, "n/a": 3}
        # ignore n/a for the overall worst-status computation
        candidates = [(s, rank.get(s, 3)) for s in (cov_s, rmse_s) if s != "n/a"]
        if not candidates:
            status = "n/a"
        else:
            status = max(candidates, key=lambda x: x[1])[0]
        reasons: list[str] = []
        if cov_s != "met":
            reasons.append(
                f"coverage {cov:.3f} {cov_s} (target {self.coverage_target:.2f})"
            )
        if rmse_s != "met" and rmse_s != "n/a":
            reasons.append(
                f"rmse {rmse:.4f} m {rmse_s} (ceiling {self.rmse_target:.4f} m)"
            )
        return SloReport(
            station=self.station,
            n=health.n,
            coverage=cov,
            rmse=rmse,
            coverage_status=cov_s,
            rmse_status=rmse_s,
            status=status,
            reasons=reasons,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SloConfig:
        return cls(
            station=str(d.get("station", "")),
            coverage_target=float(d.get("coverage_target", 0.90)),
            coverage_tolerance=float(d.get("coverage_tolerance", 0.05)),
            rmse_target=(
                None if d.get("rmse_target") is None else float(d["rmse_target"])
            ),
            rmse_tolerance=(
                None if d.get("rmse_tolerance") is None else float(d["rmse_tolerance"])
            ),
        )


@dataclass
class SloReport:
    """SLO compliance for one station on one pass."""

    station: str
    n: int
    coverage: float
    rmse: float
    coverage_status: str
    rmse_status: str
    status: str
    reasons: list[str] = field(default_factory=list)

    @property
    def coverage_gap(self) -> float:
        return max(0.0, 0.90 - self.coverage)

    def __str__(self) -> str:
        label = {
            "met": "met     ",
            "degrading": "degrading",
            "breached": "BREACHED",
            "n/a": "n/a     ",
        }[self.status]
        parts = [
            (
                f"{self.station}: SLO={label} coverage={self.coverage:.3f} "
                f"({self.coverage_status}) rmse={self.rmse:.4f} m "
                f"({self.rmse_status})"
            ),
        ]
        for r in self.reasons:
            parts.append(f"  - {r}")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SloReport:
        return cls(
            station=str(d["station"]),
            n=int(d.get("n", 0)),
            coverage=float(d.get("coverage", 0.0)),
            rmse=float(d.get("rmse", 0.0)),
            coverage_status=str(d.get("coverage_status", "n/a")),
            rmse_status=str(d.get("rmse_status", "n/a")),
            status=str(d.get("status", "n/a")),
            reasons=list(d.get("reasons", [])),
        )


@dataclass
class SloMonitor:
    """Evaluates rolling health against a station's SLO and keeps history.

    The monitor itself does not accumulate observations — it evaluates
    :class:`~tideglass.marea.drift.HealthReport` snapshots. A rolling history
    of :class:`SloReport` is kept so callers can inspect trends.
    """

    config: SloConfig
    history: int = 12

    def __post_init__(self) -> None:
        self.history = int(self.history)
        self._reports: list[SloReport] = []

    def evaluate(self, health: HealthReport) -> SloReport:
        rep = self.config.evaluate(health)
        self._reports.append(rep)
        if len(self._reports) > self.history:
            self._reports = self._reports[-self.history :]
        return rep

    @property
    def reports(self) -> list[SloReport]:
        return list(self._reports)

    def trend(self) -> str:
        """Compare first and second half of the stored history.

        Returns "improving", "stable", or "degrading" when enough reports
        exist; otherwise "insufficient".
        """
        if len(self._reports) < 4:
            return "insufficient"
        mid = len(self._reports) // 2
        first = self._reports[:mid]
        second = self._reports[mid:]
        rank = {"met": 0, "degrading": 1, "breached": 2, "n/a": 3}
        f_worst = max(first, key=lambda r: rank.get(r.status, 3)).status
        s_worst = max(second, key=lambda r: rank.get(r.status, 3)).status
        if rank.get(s_worst, 3) < rank.get(f_worst, 3):
            return "improving"
        if rank.get(s_worst, 3) > rank.get(f_worst, 3):
            return "degrading"
        return "stable"

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "history": self.history,
            "reports": [r.to_dict() for r in self._reports],
        }

    @classmethod
    def from_dict(cls, d: dict) -> SloMonitor:
        mon = cls(
            config=SloConfig.from_dict(d.get("config", {})),
            history=int(d.get("history", 12)),
        )
        mon._reports = [SloReport.from_dict(r) for r in d.get("reports", [])]
        return mon


def load_slo(store: str, station: str) -> SloConfig:
    """Load a station's SLO config from ``<store>/<station>.slo.json``."""
    import json
    import os

    path = os.path.join(store, f"{station}.slo.json")
    if os.path.exists(path):
        with open(path) as fh:
            return SloConfig.from_dict(json.load(fh))
    return SloConfig(station=station)


def save_slo(store: str, config: SloConfig) -> str:
    """Persist a station's SLO config; returns the path written."""
    import os

    os.makedirs(store, exist_ok=True)
    path = os.path.join(store, f"{config.station}.slo.json")
    with open(path, "w") as fh:
        import json

        json.dump(config.to_dict(), fh, indent=2)
    return path
