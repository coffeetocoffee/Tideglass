"""v1.0 tests: decision-theoretic pricing (cost-loss over the prediction CI)."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tideglass import TideModel
from tideglass.marea import constituents as CON
from tideglass.marea.decision import CostLoss, decision_curve
from tideglass.marine.advisor import TideAdvisor

UTC = timezone.utc
T0 = datetime(2024, 3, 1, tzinfo=UTC)


def _hourly(n):
    return [T0 + timedelta(hours=i) for i in range(n)]


def _hours(times):
    return np.array([(t - T0).total_seconds() / 3600.0 for t in times],
                    dtype=float)


def _synthetic(times, noise=0.0, seed=0, mean=1.0):
    """Mean offset + M2 + K1 tide."""
    h = np.full(len(times), float(mean))
    for name, amp in (("M2", 1.0), ("K1", 0.3)):
        w = math.radians(CON.speed(CON.get(name)))
        h += amp * np.cos(w * _hours(times))
    if noise > 0:
        h += np.random.default_rng(seed).normal(0.0, noise, len(times))
    return h


@pytest.fixture()
def model():
    times = _hourly(24 * 30)
    h = _synthetic(times, noise=0.01, seed=1)
    return TideModel.fit(times, h, station="decide-test")


def test_cost_loss_ratio():
    cl = CostLoss(cost=0.2, loss=1.0)
    assert cl.ratio == pytest.approx(0.2)


def test_act_where_likely(model):
    times = _hourly(24)
    pred = model.predict(times)
    # threshold 1.5 m: exceeded near high water on a 1 m M2 + 1 m mean tide
    dc = decision_curve(times, pred, threshold_m=1.5, cost=0.2, loss=1.0)
    assert dc.break_even == pytest.approx(0.2)
    assert 0 < dc.act_hours < 24  # acts near HW only
    # the policy acts exactly where the event probability beats break-even
    assert np.array_equal(dc.act, dc.probability > dc.break_even)
    # high water probability is near-certain, low water near-impossible
    assert dc.probability.max() > 0.99
    assert dc.probability.min() < 0.01


def test_never_act_when_unreachable(model):
    times = _hourly(24)
    pred = model.predict(times)
    dc = decision_curve(times, pred, threshold_m=50.0, cost=0.2, loss=1.0)
    assert dc.act_hours == 0
    assert dc.optimal_cost == pytest.approx(0.0)
    assert dc.never_cost == pytest.approx(0.0)


def test_expected_cost_is_pointwise_min(model):
    times = _hourly(24)
    pred = model.predict(times)
    cost, loss = 0.2, 1.0
    dc = decision_curve(times, pred, threshold_m=1.5, cost=cost, loss=loss)
    assert np.allclose(dc.expected_cost, np.minimum(cost, dc.probability * loss))
    assert dc.always_cost == pytest.approx(cost * 24)
    assert dc.optimal_cost <= dc.always_cost + 1e-12
    assert dc.optimal_cost <= dc.never_cost + 1e-12
    assert dc.value_vs_always >= 0.0
    assert dc.value_vs_never >= 0.0


def test_invalid_economics(model):
    times = _hourly(24)
    pred = model.predict(times)
    with pytest.raises(ValueError):
        decision_curve(times, pred, 1.5, cost=0.2, loss=0.0)
    with pytest.raises(ValueError):
        decision_curve(times, pred, 1.5, cost=-1.0, loss=1.0)


def test_advisor_integration(model):
    times = _hourly(24)
    advice = TideAdvisor(model).advise(
        times, flood_threshold_m=1.5, cost=0.2, loss=1.0)
    assert advice.decision is not None
    assert advice.decision.threshold_m == pytest.approx(1.5)
    assert advice.decision.act_hours >= 1
    assert "decision[" in advice.summary
    # decision-only advice with an explicit threshold (no flood pricing)
    advice2 = TideAdvisor(model).advise(
        times, decision_threshold_m=1.5, cost=0.2, loss=1.0)
    assert advice2.decision is not None
    assert advice2.flood is None
    # cost/loss without any threshold is a clear error
    with pytest.raises(ValueError):
        TideAdvisor(model).advise(times, cost=0.2, loss=1.0)


def test_str_report(model):
    times = _hourly(24)
    dc = decision_curve(times, model.predict(times), 1.5, 0.2, 1.0)
    s = str(dc)
    assert "act " in s
    assert "break-even p=0.20" in s
    assert "value vs always" in s


# -- CLI -----------------------------------------------------------------------

def _write_store(tmp_path, model):
    path = tmp_path / "decide-test.json"
    with open(path, "w") as fh:
        json.dump(model.to_artifact(), fh, indent=2)
    return str(path)


def test_cli_advise_decision(model, tmp_path, capsys):
    from tideglass.cli import main

    _write_store(tmp_path, model)
    rc = main(["advise", "decide-test", "2024-03-01", "--store", str(tmp_path),
               "--flood", "1.5", "--cost", "0.2", "--loss", "1.0"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "decision[" in out


def test_cli_advise_cost_needs_flood(model, tmp_path, capsys):
    from tideglass.cli import main

    _write_store(tmp_path, model)
    rc = main(["advise", "decide-test", "2024-03-01", "--store", str(tmp_path),
               "--cost", "0.2", "--loss", "1.0"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "--flood" in err
