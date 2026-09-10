"""Decision-theoretic pricing for Marea Core — the v1.0 uncertainty moat.

A prediction band answers "how uncertain is the water level?". Flood
probability (``marea.extremes``) answers "how likely is the damaging level?".
Neither tells the harbourmaster what to *do*. This module closes that loop with
the classical cost-loss decision model:

* acting (deploying a barrier, moving gear, cancelling a trip) costs ``cost``,
  paid whether or not the event happens;
* not acting costs ``loss`` if the event happens, nothing otherwise.

With ``p`` the event probability at one time, the expected costs are::

    E(act)   = cost
    E(wait)  = p · loss

so the rational policy acts exactly when ``p > cost/loss`` — the **break-even
probability**. The optimal per-time expected cost is ``min(cost, p·loss)``, and
the value of the forecast is the expected-cost saving against the two naive
policies that ignore the forecast: *always act* (``N · cost``) and *never act*
(``Σ p·loss``). No fixed thresholds anywhere — the CI is priced, in the
decision-maker's own currency, per hour of the forecast.

:func:`decision_curve` builds this per time step from a
:class:`~tideglass.marea.model.Prediction` (via
:func:`~tideglass.marea.extremes.flood_probability`), and
:class:`~tideglass.marine.advisor.TideAdvisor` consumes it as a thin consumer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from tideglass.marea.extremes import flood_probability
from tideglass.marea.model import Prediction


@dataclass(frozen=True)
class CostLoss:
    """The decision-maker's economics: protective action vs event loss.

    :param cost: cost of the protective action, paid unconditionally.
    :param loss: loss suffered if the event hits an unprepared operation.
        Both are in the same (any) currency; only their ratio matters for the
        policy, their magnitudes for the reported expected costs.
    """

    cost: float
    loss: float

    @property
    def ratio(self) -> float:
        """Break-even probability ``cost / loss`` — act iff ``p`` exceeds it."""
        return self.cost / self.loss


@dataclass
class DecisionCurve:
    """Per-time act/wait decisions priced from the prediction band."""

    times: list[datetime]
    probability: np.ndarray  # P(event) per time step
    threshold_m: float  # the event threshold (m)
    break_even: float  # cost/loss ratio — the act/wait cutoff on probability
    act: np.ndarray  # bool: optimal action per time step
    expected_cost: np.ndarray  # min(cost, p·loss) per time step
    always_cost: float  # total if acting every step, forecast ignored
    never_cost: float  # total expected loss if never acting
    optimal_cost: float  # total under the forecast-informed policy

    @property
    def value_vs_always(self) -> float:
        return self.always_cost - self.optimal_cost

    @property
    def value_vs_never(self) -> float:
        return self.never_cost - self.optimal_cost

    @property
    def act_hours(self) -> int:
        return int(np.count_nonzero(self.act))

    def __str__(self) -> str:
        return (
            f"decision[threshold={self.threshold_m:.2f} m, "
            f"break-even p={self.break_even:.2f}]: "
            f"act {self.act_hours}/{len(self.act)} h; "
            f"expected cost {self.optimal_cost:.2f} "
            f"(always-act {self.always_cost:.2f}, never-act {self.never_cost:.2f}); "
            f"value vs always {self.value_vs_always:+.2f}, "
            f"vs never {self.value_vs_never:+.2f}"
        )


def decision_curve(
    times: Sequence[datetime],
    prediction: Prediction,
    threshold_m: float,
    cost: float,
    loss: float,
    surge_mean: float = 0.0,
    surge_sigma: float = 0.0,
) -> DecisionCurve:
    """Price the act/wait decision at every time step of a prediction.

    :param times: timestamps matching ``prediction``.
    :param prediction: the predictive distribution (mean + band).
    :param threshold_m: event threshold (m) — e.g. a GPD return level from
        ``tideglass extremes`` or a flood alarm level.
    :param cost: cost of acting for one time step.
    :param loss: loss if the event hits during one unprepared time step.
    :param surge_mean: mean surge added to the predictive mean (m).
    :param surge_sigma: surge std (m) folded into the event probability.
    :returns: a :class:`DecisionCurve`.
    """
    if loss <= 0:
        raise ValueError(f"loss must be positive, got {loss}")
    if cost < 0:
        raise ValueError(f"cost must be non-negative, got {cost}")
    times = list(times)
    p = np.asarray(
        flood_probability(prediction, threshold_m,
                          surge_mean=surge_mean, surge_sigma=surge_sigma),
        dtype=float,
    ).ravel()
    if p.size == 0:
        raise ValueError("empty prediction")
    if len(times) != p.size:
        raise ValueError(
            f"{len(times)} times but {p.size} prediction steps")
    ratio = cost / loss
    act = p > ratio
    # Acting is optimal exactly when the protected loss beats the premium;
    # the expected cost of the optimal per-time policy is the cheaper branch.
    expected = np.minimum(cost, p * loss)
    return DecisionCurve(
        times=times,
        probability=p,
        threshold_m=float(threshold_m),
        break_even=ratio,
        act=act,
        expected_cost=expected,
        always_cost=float(cost * p.size),
        never_cost=float(np.sum(p * loss)),
        optimal_cost=float(np.sum(expected)),
    )
