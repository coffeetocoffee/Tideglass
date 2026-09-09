"""Drift / staleness detection for deployed tide models (v0.6).

A fitted or assimilated model degrades silently as the gauge's local dynamics
shift (sensor drift, datum changes, morphological change). This monitor watches
the rolling window of recent observations vs the model's prediction interval and
flags when the model should be **refit**.

The rules (all configurable) combine the signals the engine already produces:

* **Coverage collapse** — empirical CI coverage on the last ``window``
  observations falls below ``min_coverage`` (nominal is 0.95, so 0.80 is a
  strong degradation signal).
* **RMSE blow-up** — rolling RMSE exceeds ``rmse_ratio`` × the model's own
  baseline RMSE (stored in ``meta`` at fit time).
* **Bias drift** — mean signed residual exceeds ``bias_tolerance`` (m), which
  points to a datum/level shift rather than pure scatter.

Any single trigger raises ``needs_refit``; the reasons are reported so the CLI
can tell the operator *why*.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from tideglass.marea.metrics import rmse as _rmse
from tideglass.marea.model import Prediction


@dataclass
class HealthReport:
    """Rolling-health snapshot of a model against recent observations."""

    n: int
    coverage: float
    rmse: float
    bias: float
    mean_abs_resid: float
    baseline_rmse: float | None
    needs_refit: bool
    reasons: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        flag = "REFIT" if self.needs_refit else "ok"
        lines = [
            f"health[{flag}]: n={self.n} coverage={self.coverage:.3f}"
            f" rmse={self.rmse:.4f} bias={self.bias:+.4f} m",
        ]
        if self.needs_refit:
            lines.append("  reasons:")
            for r in self.reasons:
                lines.append(f"    - {r}")
        return "\n".join(lines)


class HealthMonitor:
    """Rolling coverage / accuracy monitor with a refit trigger."""

    def __init__(
        self,
        window: int = 168,
        min_coverage: float = 0.80,
        rmse_ratio: float = 2.0,
        bias_tolerance: float = 0.15,
        baseline_rmse: float | None = None,
    ):
        """Configure the drift thresholds.

        :param window: number of most-recent observations to score (default
            168 = one week of hourly samples).
        :param min_coverage: flag if rolling CI coverage drops below this.
        :param rmse_ratio: flag if rolling RMSE exceeds this × ``baseline_rmse``
            (ignored when no baseline is known).
        :param bias_tolerance: flag if |mean residual| exceeds this (m).
        :param baseline_rmse: the model's own fit/assimilation RMSE; usually
            carried in ``meta`` so degradation is measured relative to itself.
        """
        self.window = int(window)
        self.min_coverage = float(min_coverage)
        self.rmse_ratio = float(rmse_ratio)
        self.bias_tolerance = float(bias_tolerance)
        self.baseline_rmse = baseline_rmse
        self._obs: list[float] = []
        self._mean: list[float] = []
        self._lo: list[float] = []
        self._hi: list[float] = []

    def update(
        self,
        observed,
        prediction: Prediction,
        times: Sequence | None = None,
    ) -> HealthReport:
        """Feed a batch of observations + predictions; return the report."""
        o = np.asarray(observed, dtype=float).ravel()
        m = np.asarray(prediction.mean, dtype=float).ravel()
        lo = np.asarray(prediction.lower, dtype=float).ravel()
        hi = np.asarray(prediction.upper, dtype=float).ravel()
        self._obs.extend(o.tolist())
        self._mean.extend(m.tolist())
        self._lo.extend(lo.tolist())
        self._hi.extend(hi.tolist())
        if len(self._obs) > self.window:
            self._obs = self._obs[-self.window:]
            self._mean = self._mean[-self.window:]
            self._lo = self._lo[-self.window:]
            self._hi = self._hi[-self.window:]
        return self.report()

    def report(self) -> HealthReport:
        """Score the current rolling window without consuming new data."""
        n = len(self._obs)
        reasons: list[str] = []
        coverage = rmse_ = bias_ = mar = 0.0
        if n:
            o = np.asarray(self._obs)
            m = np.asarray(self._mean)
            lo = np.asarray(self._lo)
            hi = np.asarray(self._hi)
            inside = np.mean((lo <= o) & (o <= hi))
            resid = o - m
            coverage = float(inside)
            rmse_ = float(_rmse(m, o))
            bias_ = float(np.mean(resid))
            mar = float(np.mean(np.abs(resid)))
            if coverage < self.min_coverage:
                reasons.append(
                    f"coverage {coverage:.3f} < min {self.min_coverage:.2f}")
            if (self.baseline_rmse is not None and self.baseline_rmse > 0
                    and rmse_ > self.rmse_ratio * self.baseline_rmse):
                reasons.append(
                    f"rmse {rmse_:.4f} > {self.rmse_ratio}×baseline "
                    f"{self.baseline_rmse:.4f}")
            if abs(bias_) > self.bias_tolerance:
                reasons.append(
                    f"bias {bias_:+.4f} m exceeds ±{self.bias_tolerance:.2f} m")
        return HealthReport(
            n=n, coverage=coverage, rmse=rmse_, bias=bias_,
            mean_abs_resid=mar, baseline_rmse=self.baseline_rmse,
            needs_refit=bool(reasons), reasons=reasons,
        )

    @property
    def needs_refit(self) -> bool:
        return self.report().needs_refit

    def to_dict(self) -> dict:
        return {
            "window": self.window,
            "min_coverage": self.min_coverage,
            "rmse_ratio": self.rmse_ratio,
            "bias_tolerance": self.bias_tolerance,
            "baseline_rmse": self.baseline_rmse,
            "n_stored": len(self._obs),
        }

    @classmethod
    def from_dict(cls, d: dict) -> HealthMonitor:
        mon = cls(
            window=d.get("window", 168),
            min_coverage=d.get("min_coverage", 0.80),
            rmse_ratio=d.get("rmse_ratio", 2.0),
            bias_tolerance=d.get("bias_tolerance", 0.15),
            baseline_rmse=d.get("baseline_rmse"),
        )
        mon._obs = list(d.get("obs", []))
        mon._mean = list(d.get("mean", []))
        mon._lo = list(d.get("lo", []))
        mon._hi = list(d.get("hi", []))
        return mon

    def state_for_save(self) -> dict:
        """Persisted rolling buffer so a monitor resumes across runs."""
        return {**self.to_dict(),
                "obs": self._obs, "mean": self._mean,
                "lo": self._lo, "hi": self._hi}
