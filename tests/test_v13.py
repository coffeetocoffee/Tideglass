"""v1.3 tests: decision ledger (auditability) + regime-conditional calibration."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tideglass import TideModel
from tideglass.marea import constituents as CON
from tideglass.marea.decision import DecisionCurve
from tideglass.marea.ledger import DecisionLedger, LedgerEntry
from tideglass.marea.model import Prediction
from tideglass.marea.regimes import (
    RegimeCalibration,
    learn_regime_calibration,
)
from tideglass.marea.residual import learn_residual  # noqa: F401  (import sanity)

UTC = timezone.utc
T0 = datetime(2023, 1, 1, tzinfo=UTC)


def _hourly(n, start=T0):
    return [start + timedelta(hours=i) for i in range(n)]


def _hours(times):
    return np.array([(t - T0).total_seconds() / 3600.0 for t in times],
                    dtype=float)


def _tide(times, amps, noise=0.0, seed=0):
    h = np.zeros(len(times))
    for name, amp in amps.items():
        w = math.radians(CON.speed(CON.get(name)))
        h += amp * np.cos(w * _hours(times))
    if noise > 0:
        h += np.random.default_rng(seed).normal(0.0, noise, len(times))
    return h


def _pred(times, mean, sigma):
    mean = np.asarray(mean, dtype=float)
    lo = mean - 1.96 * sigma
    hi = mean + 1.96 * sigma
    return Prediction(mean=mean, lower=lo, upper=hi, se=np.full(mean.size, sigma))


# -- decision ledger --------------------------------------------------------------

def _curve(times, p, threshold=1.5, cost=0.05, loss=1.0):
    p = np.asarray(p, dtype=float).ravel()
    act = p > (cost / loss)
    exp = np.minimum(cost, p * loss)
    return DecisionCurve(
        times=list(times), probability=p, threshold_m=threshold,
        break_even=cost / loss, act=act, expected_cost=exp,
        always_cost=cost * p.size, never_cost=float(np.sum(p * loss)),
        optimal_cost=float(np.sum(exp)))


def test_ledger_log_report_no_outcomes():
    times = _hourly(12)
    p = np.linspace(0.0, 0.6, 12)
    curve = _curve(times, p)
    led = DecisionLedger()
    led.log_curve(curve, cost=0.05, loss=1.0)
    assert len(led.entries) == 12
    rep = led.report()
    assert rep.n_decisions == 12
    assert rep.n_resolved == 0
    assert rep.recall is None and rep.precision is None


def test_ledger_reconciles_realized_cost():
    times = _hourly(12)
    # first 6 events occur (level high), last 6 do not
    levels = [2.0] * 6 + [1.0] * 6
    p = np.linspace(0.0, 0.6, 12)
    curve = _curve(times, p, threshold=1.5, cost=0.05, loss=1.0)
    led = DecisionLedger()
    led.log_curve(curve, cost=0.05, loss=1.0)
    led.record_outcomes(times, levels)
    rep = led.report()
    assert rep.n_resolved == 12
    # 6 events; acting where p>0.05 (break-even) means acting on most: realized
    # cost = sum over acted (pay cost) + sum over waited&event (pay loss)
    assert rep.total_realized >= 0.0
    # always-act baseline = 12 * 0.05 = 0.60
    assert rep.total_always == pytest.approx(0.60)
    # never-act baseline = 6 events * 1.0 = 6.0
    assert rep.total_never == pytest.approx(6.0)
    # forecast earned vs always/never baselines
    assert rep.earned_vs_always == pytest.approx(rep.total_always - rep.total_realized)
    assert rep.earned_vs_never == pytest.approx(rep.total_never - rep.total_realized)
    # recall/precision defined when events and actions exist
    assert rep.recall is not None and rep.precision is not None


def test_ledger_roundtrip(tmp_path):
    times = _hourly(8)
    p = np.linspace(0.1, 0.7, 8)
    curve = _curve(times, p)
    led = DecisionLedger()
    led.log_curve(curve, cost=0.1, loss=2.0)
    led.record_outcomes(times, [1.0] * 4 + [0.5] * 4)
    path = tmp_path / "led.json"
    led.save(str(path))
    loaded = DecisionLedger.load(str(path))
    assert len(loaded.entries) == 8
    assert loaded.report().n_resolved == 8
    # entries survive exactly
    assert all(isinstance(e, LedgerEntry) for e in loaded.entries)
    assert loaded.report().total_realized == pytest.approx(led.report().total_realized)


def test_ledger_log_curve_requires_cost_loss():
    times = _hourly(4)
    curve = _curve(times, np.array([0.2, 0.3, 0.4, 0.5]))
    led = DecisionLedger()
    with pytest.raises(ValueError, match="cost and loss"):
        led.log_curve(curve, cost=None, loss=None)


def test_ledger_report_str():
    times = _hourly(4)
    led = DecisionLedger()
    led.log_curve(_curve(times, np.array([0.2, 0.3, 0.4, 0.5])),
                  cost=0.05, loss=1.0)
    s = str(led.report())
    assert "ledger:" in s


# -- regime-conditional calibration ----------------------------------------------

def _build_model_with_regimes():
    """Tide + a storm-driven residual so storm/non-storm residuals differ."""
    n = 24 * 25  # 25 days: spans a full spring/neap beat (14.77 d)
    times = _hourly(n)
    tide = _tide(times, {"M2": 1.2, "S2": 0.5, "K1": 0.6})
    # storms: a few multi-day windows of large positive surge + smaller noise
    rng = np.random.default_rng(4)
    resid = rng.normal(0.0, 0.03, n)
    storm_mask = np.zeros(n, dtype=bool)
    # centers span the spring/neap cycle (5 d, 12.5 d, 20 d) so storms hit both
    for center in (120, 300, 480):
        storm_mask[center - 12:center + 12] = True
    resid = np.where(storm_mask, 1.4 + rng.normal(0, 0.1, n), resid)
    observed = tide + resid
    model = TideModel.fit(times, observed, alpha=0.05, station="rg")
    surge = np.where(storm_mask, 1.5, 0.0)  # met-forced surge flag for storm labelling
    return times, observed, model, surge


def test_regime_classify_finds_spring_and_neap():
    times, _, model, _ = _build_model_with_regimes()
    # only spring/neap (no surge): storm arg omitted -> all non-storm
    from tideglass.marea.regimes import _daily_range

    ranges = _daily_range(model, times)
    # the M2/S2 envelope must vary across the beat
    assert ranges.max() - ranges.min() > 0.5
    rc = RegimeCalibration(
        alpha=0.05, regimes={"neap": 1.96, "spring": 2.0},
        spring_range_median=float(np.median(ranges)),
        storm_resid_q=1.0, coverage={}, default_q=1.96)
    labels = rc.classify(model, times)
    assert "spring" in labels and "neap" in labels
    assert not any("storm" in l for l in labels)


def test_regime_learn_and_apply():
    times, observed, model, surge = _build_model_with_regimes()
    rc = learn_regime_calibration(model, times, observed, alpha=0.05, surge=surge)
    # storm regime residuals are ~1.4 m; non-storm ~0.03 m => storm q much larger
    assert "neap_storm" in rc.regimes and "neap" in rc.regimes
    assert rc.regimes["neap_storm"] > rc.regimes["neap"]
    assert rc.coverage["neap_storm"] >= 1.0 - rc.alpha
    # apply: a forecast whose surge marks the first half storm, second half calm
    ftimes = _hourly(12)
    fmean = np.ones(12)
    pred = _pred(ftimes, fmean, 0.1)
    fc_surge = np.array([1.5] * 6 + [0.0] * 6)  # storm then calm
    out = rc.apply(pred, ftimes, model, surge=fc_surge)
    half = np.asarray(out.upper) - np.asarray(out.mean)
    # storm-band half is wider than the neap-band half at constant sigma
    assert half[:6].mean() > half[6:].mean()
    # mean untouched
    assert np.allclose(out.mean, fmean)


def test_regime_storm_classified_from_surge_at_forecast():
    times, observed, model, surge = _build_model_with_regimes()
    rc = learn_regime_calibration(model, times, observed, alpha=0.05, surge=surge)
    ftimes = _hourly(8)
    # surge absent -> all non-storm
    labels_calm = rc.classify(model, ftimes)
    assert not any("storm" in l for l in labels_calm)
    # surge present -> storm labelled
    labels_storm = rc.classify(model, ftimes, surge=np.full(8, 1.5))
    assert all("storm" in l for l in labels_storm)


def test_regime_roundtrip():
    times, observed, model, surge = _build_model_with_regimes()
    rc = learn_regime_calibration(model, times, observed, alpha=0.05, surge=surge)
    rc2 = RegimeCalibration.from_dict(rc.to_dict())
    assert rc2.regimes == rc.regimes
    assert rc2.spring_range_median == pytest.approx(rc.spring_range_median)
    assert rc2.default_q == pytest.approx(rc.default_q)


# -- CLI integration --------------------------------------------------------------

def _write_obs_csv(path, times, observed):
    with open(path, "w") as fh:
        fh.write("time,height\n")
        fh.writelines(f"{t.isoformat()},{o:.4f}\n" for t, o in zip(times, observed))


def test_calibrate_regimes_cli(tmp_path, capsys):
    from tideglass.cli import main

    n = 24 * 25
    times = _hourly(n)
    observed = _tide(times, {"M2": 1.2, "S2": 0.5, "K1": 0.6}) \
        + np.concatenate([np.full(12, 1.4), np.zeros(n - 12)]) \
        + np.random.default_rng(2).normal(0, 0.03, n)
    csv = tmp_path / "r.csv"
    _write_obs_csv(csv, times, observed)
    rc = main(["calibrate", str(csv), "--station", "rg", "--store",
               str(tmp_path), "--regimes"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "regime-conditional calibration" in out
    assert "neap" in out and "spring" in out
    import os

    assert os.path.exists(tmp_path / "rg.regimes.json")
    saved = json.loads((tmp_path / "rg.regimes.json").read_text())
    assert "regimes" in saved


def test_ledger_cli_roundtrip(tmp_path, capsys):
    from tideglass.cli import main

    n = 24 * 30
    times = _hourly(n)
    observed = _tide(times, {"M2": 1.0, "S2": 0.4, "K1": 0.6})
    csv = tmp_path / "g.csv"
    _write_obs_csv(csv, times, observed)
    # fit a model into the store
    assert main(["fit", str(csv), "--station", "g1", "--store",
                 str(tmp_path)]) == 0

    # advise with a priced decision, logged to the ledger file
    ledger_path = tmp_path / "g1.ledger.json"
    rc = main(["advise", "g1", "2023-01-15", "--store", str(tmp_path),
               "--days", "2", "--flood", "1.6", "--cost", "0.05",
               "--loss", "1.0", "--ledger", str(ledger_path)])
    assert rc == 0
    assert ledger_path.exists()
    led = DecisionLedger.load(str(ledger_path))
    assert led.report().n_decisions == 48

    # reconcile against realized outcomes (all high -> events everywhere here)
    out_csv = tmp_path / "out.csv"
    _write_obs_csv(out_csv, _hourly(48, start=datetime(2023, 1, 15, tzinfo=UTC)),
                   np.full(48, 2.0))
    rc = main(["ledger", "g1", "--store", str(tmp_path),
               "--outcome", str(out_csv)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "realized cost" in out

    # pure report call
    rc = main(["ledger", "g1", "--store", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0 and "ledger:" in out
