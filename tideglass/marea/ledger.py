"""Decision ledger — the v1.3 accountability layer.

The v1.0 cost-loss model *prices* the act/wait decision per hour, but a price is
only trustworthy if it can be audited. :class:`DecisionLedger` logs every priced
decision (from a :class:`~tideglass.marea.decision.DecisionCurve`) against its
realized outcome and reports the *realized* cost versus the always-act and
never-act baselines — "the forecast earned X" in the operator's own currency.
That turns the advisor from a black box into an auditable, B2B-grade system.

Every entry carries the per-time probability, the action taken, the economics
(``cost``/``loss``), and — once the water level is known — the realized level,
whether the event actually occurred, and the realized cost of the action that
was taken. :meth:`DecisionLedger.report` reconciles the two.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np

from tideglass.marea.decision import DecisionCurve


def _isokey(t) -> str:
    return t.isoformat() if isinstance(t, datetime) else str(t)


@dataclass
class LedgerEntry:
    """One logged decision at one point in time."""

    time: str  # ISO of the decision
    threshold_m: float  # event threshold the decision prices (m)
    probability: float  # P(event) at decision time
    act: bool  # optimal action taken
    cost: float  # cost of acting (currency)
    loss: float  # loss if the event hits an unprepared step
    expected_cost: float  # min(cost, p·loss) — the priced value
    realized_level: float | None = None  # observed water level (m), once known
    realized_event: bool | None = None  # observed_level > threshold_m
    realized_cost: float | None = None  # cost of the *actual* action taken
    action: str = ""  # v3.1: autonomous action kind (refit/reseed/sync/observe)
    reason: str = ""  # v3.1: why the action was taken

    def reconcile(self) -> None:
        """Fill realized_event / realized_cost from a known realized_level."""
        if self.realized_level is None:
            return
        self.realized_event = bool(self.realized_level > self.threshold_m)
        self.realized_cost = (
            self.cost if self.act else (self.loss if self.realized_event else 0.0)
        )


@dataclass
class LedgerReport:
    """Reconciliation of logged decisions against realized outcomes."""

    n_decisions: int
    n_resolved: int
    total_realized: float
    total_always: float
    total_never: float
    earned_vs_always: float
    earned_vs_never: float
    recall: float | None  # fraction of actual events we acted on
    precision: float | None  # fraction of actions that met an event

    def __str__(self) -> str:
        if self.n_resolved == 0:
            return (
                f"ledger: {self.n_decisions} decision(s) logged, "
                f"0 resolved (awaiting realized levels)"
            )
        lines = [
            (
                f"ledger: {self.n_decisions} decision(s) logged, "
                f"{self.n_resolved} resolved"
            ),
            f"  realized cost:        {self.total_realized:.2f}",
            (
                f"  always-act baseline:  {self.total_always:.2f}  "
                f"(forecast earned {self.earned_vs_always:+.2f})"
            ),
            (
                f"  never-act baseline:   {self.total_never:.2f}  "
                f"(forecast earned {self.earned_vs_never:+.2f})"
            ),
        ]
        if self.recall is not None:
            lines.append(f"  event recall:    {self.recall:.2%}")
        if self.precision is not None:
            lines.append(f"  action precision:{self.precision:.2%}")
        return "\n".join(lines)


class DecisionLedger:
    """Append-only log of priced decisions and their realized outcomes."""

    def __init__(self, entries: Sequence[LedgerEntry] | None = None):
        self.entries: list[LedgerEntry] = list(entries or [])

    # -- logging ----------------------------------------------------------------

    def log(
        self,
        times: Sequence[datetime],
        probability,
        act,
        threshold_m: float,
        cost: float,
        loss: float,
        expected_cost=None,
    ) -> None:
        """Append one decision per time (arrays must align with ``times``)."""
        prob = np.asarray(probability, dtype=float).ravel()
        acts = np.asarray(act, dtype=bool).ravel()
        if not (len(times) == prob.size == acts.size):
            raise ValueError(
                f"{len(times)} times but {prob.size} probabilities and "
                f"{acts.size} actions"
            )
        for i, t in enumerate(times):
            exp = (
                float(min(cost, prob[i] * loss))
                if expected_cost is None
                else float(expected_cost[i])
            )
            self.entries.append(
                LedgerEntry(
                    time=_isokey(t),
                    threshold_m=float(threshold_m),
                    probability=float(prob[i]),
                    act=bool(acts[i]),
                    cost=float(cost),
                    loss=float(loss),
                    expected_cost=exp,
                )
            )

    def log_action(
        self,
        time,
        action: str,
        reason: str,
        cost: float,
        loss: float,
        slo_metric: float | None = None,
        slo_target: float | None = None,
        acted: bool = True,
        expected_cost: float | None = None,
    ) -> None:
        """Record an autonomous action as a priced, reconcilable entry (v3.1).

        The action is priced like a flood decision: acting costs ``cost``,
        and the loss of *not* acting is ``loss`` (the SLO-breach cost).
        ``slo_metric`` / ``slo_target`` become the realized outcome so the
        ledger can reconcile what the loop did against what it earned.
        """
        prob = 1.0 if acted else 0.0
        exp = (
            float(min(cost, prob * loss))
            if expected_cost is None
            else float(expected_cost)
        )
        e = LedgerEntry(
            time=_isokey(time),
            threshold_m=float(slo_target) if slo_target is not None else 0.0,
            probability=prob,
            act=bool(acted),
            cost=float(cost),
            loss=float(loss),
            expected_cost=exp,
            action=str(action),
            reason=str(reason),
            realized_level=(float(slo_metric) if slo_metric is not None else None),
        )
        e.reconcile()
        self.entries.append(e)

    def log_curve(
        self,
        curve: DecisionCurve,
        cost: float,
        loss: float,
        realized_levels: Sequence[float] | None = None,
    ) -> None:
        """Log a :class:`DecisionCurve` as a batch of decisions.

        ``cost``/``loss`` are the economics that produced the curve (the curve
        does not store them). ``realized_levels``, if given, must align with
        ``curve.times`` and are recorded immediately.
        """
        if cost is None or loss is None:
            raise ValueError("cost and loss are required to log a decision")
        self.log(
            curve.times,
            curve.probability,
            curve.act,
            curve.threshold_m,
            cost,
            loss,
            expected_cost=None,
        )
        if realized_levels is not None:
            self.record_outcomes(curve.times, realized_levels)

    # -- outcomes ---------------------------------------------------------------

    def record_outcome(self, time: datetime, level: float) -> None:
        for e in reversed(self.entries):
            if e.time == _isokey(time):
                e.realized_level = float(level)
                e.reconcile()
                return

    def record_outcomes(
        self, times: Sequence[datetime], levels: Sequence[float]
    ) -> None:
        if len(times) != len(levels):
            raise ValueError(f"{len(times)} times but {len(levels)} levels")
        for t, lvl in zip(times, levels):
            self.record_outcome(t, lvl)

    # -- reporting ---------------------------------------------------------------

    def report(self) -> LedgerReport:
        resolved = [e for e in self.entries if e.realized_level is not None]
        n_dec = len(self.entries)
        n_res = len(resolved)
        if n_res == 0:
            return LedgerReport(n_dec, 0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None)
        total_realized = float(sum(e.realized_cost or 0.0 for e in resolved))
        total_always = float(sum(e.cost for e in resolved))
        total_never = float(sum(e.loss if e.realized_event else 0.0 for e in resolved))
        events = [e for e in resolved if e.realized_event]
        acts = [e for e in resolved if e.act]
        warned_hits = sum(1 for e in resolved if e.act and e.realized_event)
        recall = (warned_hits / len(events)) if events else None
        precision = (warned_hits / len(acts)) if acts else None
        return LedgerReport(
            n_decisions=n_dec,
            n_resolved=n_res,
            total_realized=total_realized,
            total_always=total_always,
            total_never=total_never,
            earned_vs_always=total_always - total_realized,
            earned_vs_never=total_never - total_realized,
            recall=recall,
            precision=precision,
        )

    # -- persistence -------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"entries": [asdict(e) for e in self.entries]}

    @classmethod
    def from_dict(cls, d: dict) -> DecisionLedger:
        entries = [
            LedgerEntry(
                **{k: v for k, v in e.items() if k in LedgerEntry.__dataclass_fields__}
            )
            for e in d.get("entries", [])
        ]
        return cls(entries)

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> DecisionLedger:
        with open(path) as fh:
            return cls.from_dict(json.load(fh))
