"""v1.1 tests: residual memory (learned bias layer) + trust-weighted federation."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tideglass import TideModel
from tideglass.marea import constituents as CON
from tideglass.marea.federation import TrustLedger, trust_score
from tideglass.marea.pooling import HierarchicalPool
from tideglass.marea.qc import QcReport
from tideglass.marea.residual import learn_residual

UTC = timezone.utc
T0 = datetime(2023, 1, 1, tzinfo=UTC)


def _hourly(n, start=T0):
    return [start + timedelta(hours=i) for i in range(n)]


def _hours(times):
    return np.array([(t - T0).total_seconds() / 3600.0 for t in times],
                    dtype=float)


def _tide(times, amps, noise=0.0, seed=0, bias=None):
    h = np.zeros(len(times))
    for name, amp in amps.items():
        w = math.radians(CON.speed(CON.get(name)))
        h += amp * np.cos(w * _hours(times))
    if noise > 0:
        h += np.random.default_rng(seed).normal(0.0, noise, len(times))
    if bias is not None:
        h += bias
    return h


def _seasonal_bias(times, amp=0.15):
    doy = np.array([t.timetuple().tm_yday - 1 for t in times], dtype=float)
    return amp * np.sin(2.0 * np.pi * doy / 365.25)


# -- residual climatology ------------------------------------------------------

@pytest.fixture(scope="module")
def seasonal():
    times = _hourly(24 * 360)  # nearly a full year
    truth = _tide(times, {"M2": 1.0, "S2": 0.4, "K1": 0.6})
    h = truth + _seasonal_bias(times) + \
        np.random.default_rng(1).normal(0.0, 0.005, len(times))
    model = TideModel.fit(times, h, auto_select=False,
                          candidates=CON.principal(), station="seas")
    return times, truth, h, model


def test_learn_seasonal_bias(seasonal):
    times, truth, h, model = seasonal
    rm, diag = learn_residual(model, times, h, bins=12)
    # the learned table recovers the annual bias the principal-8 cannot fit
    assert 0.10 < diag["max_abs_bias"] < 0.20
    assert diag["rmse_after"] < 0.4 * diag["rmse_before"]


def test_attach_changes_predict(seasonal):
    times, truth, h, model = seasonal
    bias = _seasonal_bias(times)
    rm, diag = learn_residual(model, times, h, bins=12)
    raw = model.predict(times)
    model.attach_residual(rm)
    assert model.residual is rm
    pred = model.predict(times)
    # predict() now applies the correction: mean = raw mean + bias(t)
    assert np.allclose(pred.mean, raw.mean + rm.correction(times))
    # the corrected prediction tracks the real water level (tide + seasonal
    # bias) — the raw harmonics alone cannot express the seasonal structure
    rmse = float(np.sqrt(np.mean((pred.mean - (truth + bias)) ** 2)))
    assert rmse < 0.03
    rmse_raw = float(np.sqrt(np.mean((raw.mean - (truth + bias)) ** 2)))
    assert rmse < 0.4 * rmse_raw


def test_conformal_band(seasonal):
    times, truth, h, model = seasonal
    rm, diag = learn_residual(model, times, h, bins=12)
    assert diag["conformal_q"] is not None
    assert 0.5 < diag["conformal_q"] < 3.0
    assert diag["coverage_after"] >= 0.90


def test_residual_roundtrip(seasonal):
    times, truth, h, model = seasonal
    rm, _ = learn_residual(model, times, h, bins=12)
    model.attach_residual(rm)
    restored = TideModel.load_harmonic(
        json.loads(json.dumps(model.to_artifact())))
    assert restored.residual is not None
    assert np.allclose(restored.residual.biases, rm.biases)
    p1 = model.predict(times)
    p2 = restored.predict(times)
    assert np.allclose(p1.mean, p2.mean)


def test_learn_rejects_short_record(seasonal):
    times, _, _, model = seasonal
    with pytest.raises(ValueError):
        learn_residual(model, times[:40], np.zeros(40), bins=12)


# -- trust ---------------------------------------------------------------------

def test_trust_score_clean_and_bad():
    assert trust_score(0.0, 0.0, 0.0) == pytest.approx(1.0)
    # 0.5 penalty exactly at each tolerance
    assert trust_score(0.0, 0.30, 0.0) == pytest.approx(0.5)
    assert trust_score(0.0, 0.0, 0.10) == pytest.approx(0.5)
    assert trust_score(1.0, 0.0, 0.0) == pytest.approx(1e-3)


def test_trust_ledger_accumulates(tmp_path):
    led = TrustLedger(str(tmp_path))
    clean = QcReport(240, 0, [], 0.01, 0.001, False, [])
    bad = QcReport(240, 30, list(range(30)), 0.8, 0.2, True,
                   ["datum shift +0.800 m exceeds tol 0.3"])
    t1 = led.record("s1", clean)
    # tiny datum/drift signals keep it essentially perfect
    assert t1 == pytest.approx(1.0, abs=1e-2)
    t2 = led.record("s1", bad)
    assert t2 < t1  # a rejection + datum/drift flags cost reputation
    t3 = led.record("s1", bad)
    assert t3 < t2  # repeated failures keep decaying
    assert led.get("s1") == pytest.approx(t3)
    assert led.get("unknown") == 1.0
    assert led.all()["s1"] == pytest.approx(t3)
    # persistence round-trips
    led2 = TrustLedger(str(tmp_path))
    assert led2.get("s1") == pytest.approx(t3)


# -- trust-weighted pooling ----------------------------------------------------

def _member(station, amp, seed):
    times = _hourly(240)
    h = _tide(times, {"M2": amp}, noise=0.005, seed=seed)
    return TideModel.fit(times, h, auto_select=False,
                         candidates=[CON.get("M2")], station=station)


def test_pool_trust_downweights_neighbour():
    ma = _member("a", 1.0, 1)
    mb = _member("b", 2.0, 2)
    trusted = HierarchicalPool({"a": ma, "b": mb},
                               trust={"a": 1.0, "b": 1.0}).pool("a")
    distrusted = HierarchicalPool({"a": ma, "b": mb},
                                  trust={"a": 1.0, "b": 0.01}).pool("a")
    amp = lambda m: {f.name: f.amplitude
                     for f in m.constituents()}["M2"]  # noqa: E731
    # with b trusted, a's M2 is pulled toward b's 2.0; with b distrusted it
    # stays at a's own 1.0
    assert amp(distrusted) < amp(trusted)
    assert abs(amp(distrusted) - 1.0) < abs(amp(trusted) - 1.0)


def test_pool_rejects_bad_trust():
    ma = _member("a", 1.0, 1)
    mb = _member("b", 1.0, 2)
    with pytest.raises(ValueError):
        HierarchicalPool({"a": ma, "b": mb}, trust={"a": 2.0})
    with pytest.raises(ValueError):
        HierarchicalPool({"a": ma, "b": mb}, trust={"zz": 0.5})


def test_trust_recorded_in_pooled_meta():
    ma = _member("a", 1.0, 1)
    mb = _member("b", 1.5, 2)
    pooled = HierarchicalPool({"a": ma, "b": mb},
                              trust={"a": 1.0, "b": 0.2}).pool("a")
    assert pooled.meta["trust"] == {"a": 1.0, "b": 0.2}


# -- CLI -----------------------------------------------------------------------

def test_cli_correct(tmp_path, capsys):
    from tideglass.cli import main

    times = _hourly(240)
    h = _tide(times, {"M2": 1.0, "S2": 0.4, "K1": 0.6}, noise=0.01, seed=5)
    csv = tmp_path / "rec.csv"
    with open(csv, "w") as fh:
        fh.write("time,height\n")
        for t, hh in zip(times, h):
            fh.write(f"{t.isoformat()},{hh:.4f}\n")
    assert main(["fit", str(csv), "--station", "c1", "--store",
                 str(tmp_path)]) == 0
    capsys.readouterr()
    rc = main(["correct", str(csv), "--station", "c1", "--store",
               str(tmp_path), "--bins", "4"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "rmse:" in out
    # the artifact now applies the bias layer in every predict
    model = TideModel.load_harmonic(str(tmp_path / "c1.json"))
    assert model.residual is not None
    assert model.predict(times).mean.shape == (len(times),)
