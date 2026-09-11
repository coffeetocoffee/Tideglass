"""Tests for v3.1 — autonomous closed-loop + per-station SLOs."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tideglass import TideModel
from tideglass.cli import main as cli_main
from tideglass.marea import constituents as C
from tideglass.marea.drift import HealthReport
from tideglass.marea.ledger import DecisionLedger
from tideglass.marea.ops import (
    AutonomousLoop,
    AutonomousReport,
)
from tideglass.marea.slo import SloConfig, SloMonitor, load_slo, save_slo
from tideglass.marea.solver import rad_per_hour

UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)
TRUE = {
    "M2": (1.0, 0.5),
    "S2": (0.30, -1.2),
    "N2": (0.20, 2.0),
    "K1": (0.15, 0.0),
    "O1": (0.10, 2.8),
}


def _hours(times):
    return np.array([(t - T0).total_seconds() / 3600.0 for t in times])


def _tide(times, noise=0.0, seed=0, shift=0.0):
    t = _hours(times)
    y = np.full_like(t, 0.7)
    for name, (amp, phi) in TRUE.items():
        w = float(rad_per_hour(C.speed(C.get(name))))
        y = y + amp * np.cos(w * t - phi)
    if noise:
        y = y + np.random.default_rng(seed).normal(0.0, noise, size=t.size)
    return y + shift


def _hourly(start, n):
    return [start + timedelta(hours=h) for h in range(n)]


def _write_csv(path, times, heights):
    with open(path, "w") as fh:
        fh.write("time,height\n")
        fh.writelines(f"{t.isoformat()},{h:.4f}\n" for t, h in zip(times, heights))


def _seed_store(store, station="OP", n=30 * 24, noise=0.01, seed=5):
    train = _hourly(T0, n)
    model = TideModel.fit(
        train,
        _tide(train, noise=noise, seed=seed),
        auto_select=False,
        station=station,
        source=f"csv:{station}.csv",
    )
    os.makedirs(store, exist_ok=True)
    with open(f"{store}/{station}.json", "w") as fh:
        json.dump(model.to_artifact(), fh)
    return train


def _make_fetcher(rows):
    def fetcher(station, begin, end, datum="MLLW", **kw):
        b = datetime.fromisoformat(begin).replace(tzinfo=UTC)
        e = datetime.fromisoformat(end).replace(tzinfo=UTC)
        return [(t, h) for t, h in rows if b <= t <= e]

    return fetcher


# --- SLO config ------------------------------------------------------------


def test_slo_defaults():
    cfg = SloConfig(station="X")
    assert cfg.coverage_target == 0.90
    assert cfg.coverage_tolerance == 0.05
    assert cfg.rmse_target is None
    rep = cfg.evaluate(
        HealthReport(
            n=168,
            coverage=0.95,
            rmse=0.05,
            bias=0.01,
            mean_abs_resid=0.03,
            baseline_rmse=0.05,
            needs_refit=False,
        )
    )
    assert rep.status == "met"
    assert rep.coverage_status == "met"
    assert rep.rmse_status == "n/a"


def test_slo_coverage_degrading():
    cfg = SloConfig(station="X", coverage_target=0.90, coverage_tolerance=0.05)
    rep = cfg.evaluate(
        HealthReport(
            n=100,
            coverage=0.87,
            rmse=0.10,
            bias=0.01,
            mean_abs_resid=0.05,
            baseline_rmse=0.10,
            needs_refit=False,
        )
    )
    assert rep.status == "degrading"
    assert rep.coverage_status == "degrading"


def test_slo_coverage_breached():
    cfg = SloConfig(station="X", coverage_target=0.90, coverage_tolerance=0.05)
    rep = cfg.evaluate(
        HealthReport(
            n=100,
            coverage=0.82,
            rmse=0.10,
            bias=0.01,
            mean_abs_resid=0.05,
            baseline_rmse=0.10,
            needs_refit=False,
        )
    )
    assert rep.status == "breached"
    assert rep.coverage_status == "breached"


def test_slo_rmse_degrading():
    cfg = SloConfig(station="X", rmse_target=0.10, rmse_tolerance=0.05)
    rep = cfg.evaluate(
        HealthReport(
            n=100,
            coverage=0.95,
            rmse=0.13,
            bias=0.01,
            mean_abs_resid=0.05,
            baseline_rmse=0.10,
            needs_refit=False,
        )
    )
    assert rep.status == "degrading"
    assert rep.rmse_status == "degrading"


def test_slo_rmse_breached():
    cfg = SloConfig(station="X", rmse_target=0.10, rmse_tolerance=0.05)
    rep = cfg.evaluate(
        HealthReport(
            n=100,
            coverage=0.95,
            rmse=0.18,
            bias=0.01,
            mean_abs_resid=0.05,
            baseline_rmse=0.10,
            needs_refit=False,
        )
    )
    assert rep.status == "breached"
    assert rep.rmse_status == "breached"


def test_slo_overall_worst_wins():
    cfg = SloConfig(
        station="X",
        coverage_target=0.80,
        coverage_tolerance=0.05,
        rmse_target=0.10,
        rmse_tolerance=0.05,
    )
    rep = cfg.evaluate(
        HealthReport(
            n=100,
            coverage=0.95,
            rmse=0.20,
            bias=0.01,
            mean_abs_resid=0.05,
            baseline_rmse=0.10,
            needs_refit=False,
        )
    )
    assert rep.status == "breached"
    assert any("rmse" in r for r in rep.reasons)


def test_slo_roundtrip():
    cfg = SloConfig(station="Z", coverage_target=0.92, rmse_target=0.15)
    d = cfg.to_dict()
    back = SloConfig.from_dict(d)
    assert back.coverage_target == 0.92
    assert back.rmse_target == pytest.approx(0.15)
    assert back.station == "Z"


def test_slo_monitor_trend():
    cfg = SloConfig(station="X", coverage_target=0.90, coverage_tolerance=0.05)
    mon = SloMonitor(cfg)
    for cov in [0.97, 0.96, 0.93, 0.88, 0.82, 0.78]:
        mon.evaluate(
            HealthReport(
                n=100,
                coverage=cov,
                rmse=0.05,
                bias=0.01,
                mean_abs_resid=0.03,
                baseline_rmse=0.05,
                needs_refit=False,
            )
        )
    assert mon.trend() == "degrading"


def test_slo_persistence(tmp_path):
    cfg = SloConfig(station="S1", coverage_target=0.92, rmse_target=0.12)
    save_slo(str(tmp_path), cfg)
    loaded = load_slo(str(tmp_path), "S1")
    assert loaded.coverage_target == 0.92
    assert loaded.rmse_target == pytest.approx(0.12)


# --- AutonomousLoop --------------------------------------------------------


def _setup_two_station_fleet(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    _seed_store(str(store), station="alpha", n=30 * 24, noise=0.01, seed=5)
    _seed_store(str(store), station="bravo", n=30 * 24, noise=0.01, seed=6)
    save_slo(str(store), SloConfig(station="alpha"))
    save_slo(str(store), SloConfig(station="bravo"))
    return str(store)


def test_loop_step_assimilates_and_reports(tmp_path):
    store = _setup_two_station_fleet(tmp_path)
    train_t = _hourly(T0, 30 * 24)
    rows = [(t, float(h)) for t, h in zip(train_t, _tide(train_t, noise=0.01, seed=5))]
    fetcher = _make_fetcher(rows)
    loop = AutonomousLoop(store, stations=["alpha"], fetcher=fetcher)
    rep = loop.step(end=train_t[-1])
    assert isinstance(rep, AutonomousReport)
    assert rep.stations == 1
    assert rep.pass_no == 1
    assert isinstance(rep.slo, dict)
    assert "alpha" in rep.slo
    print(rep)


def test_loop_refits_on_drift(tmp_path):
    store = _setup_two_station_fleet(tmp_path)
    # feed with a +50 cm datum shift starting after the training window
    feed_t = _hourly(T0 + timedelta(hours=30 * 24), 72)
    rows = [
        (t, float(h)) for t, h in zip(feed_t, _tide(feed_t, noise=0.01, seed=99) + 0.5)
    ]
    fetcher = _make_fetcher(rows)
    loop = AutonomousLoop(store, stations=["alpha"], auto_refit=True, fetcher=fetcher)
    rep = loop.step(end=feed_t[-1])
    assert rep.n_refit >= 1
    refit_actions = [a for a in rep.actions if a.kind == "refit"]
    assert len(refit_actions) == 1
    assert refit_actions[0].station == "alpha"
    print(rep)


def test_loop_skips_refit_when_disabled(tmp_path):
    store = _setup_two_station_fleet(tmp_path)
    feed_t = _hourly(T0 + timedelta(hours=30 * 24), 72)
    rows = [
        (t, float(h)) for t, h in zip(feed_t, _tide(feed_t, noise=0.01, seed=99) + 0.5)
    ]
    fetcher = _make_fetcher(rows)
    loop = AutonomousLoop(store, stations=["alpha"], auto_refit=False, fetcher=fetcher)
    rep = loop.step(end=feed_t[-1])
    assert rep.n_refit == 0
    refit_actions = [a for a in rep.actions if a.kind == "refit"]
    assert len(refit_actions) == 0


def test_loop_slo_breached_reports(tmp_path):
    store = _setup_two_station_fleet(tmp_path)
    # set a tight RMSE target so the 0.5 m datum shift breaches it
    save_slo(
        store,
        SloConfig(
            station="alpha",
            coverage_target=0.90,
            coverage_tolerance=0.05,
            rmse_target=0.15,
            rmse_tolerance=0.05,
        ),
    )
    feed_t = _hourly(T0 + timedelta(hours=30 * 24), 72)
    rows = [
        (t, float(h)) for t, h in zip(feed_t, _tide(feed_t, noise=0.01, seed=99) + 0.5)
    ]
    fetcher = _make_fetcher(rows)
    loop = AutonomousLoop(store, stations=["alpha"], auto_refit=False, fetcher=fetcher)
    rep = loop.step(end=feed_t[-1])
    slo = rep.slo.get("alpha")
    assert slo is not None
    assert slo.n > 0
    assert slo.status == "breached"
    print(rep)


def test_loop_ledger_audit_trail(tmp_path):
    store = _setup_two_station_fleet(tmp_path)
    feed_t = _hourly(T0 + timedelta(hours=30 * 24), 72)
    rows = [
        (t, float(h)) for t, h in zip(feed_t, _tide(feed_t, noise=0.01, seed=99) + 0.5)
    ]
    fetcher = _make_fetcher(rows)
    ledger_dir = tmp_path / "ledger"
    loop = AutonomousLoop(
        store, stations=["alpha"], ledger_root=str(ledger_dir), fetcher=fetcher
    )
    loop.step(end=feed_t[-1])
    ledger_path = ledger_dir / "alpha.ledger.json"
    assert ledger_path.exists()
    ledger = DecisionLedger.load(str(ledger_path))
    assert len(ledger.entries) >= 1
    actions = [e for e in ledger.entries if e.action]
    assert len(actions) >= 1
    print(ledger)


def test_loop_run_max_passes(tmp_path):
    store = _setup_two_station_fleet(tmp_path)
    train_t = _hourly(T0, 30 * 24)
    rows = [(t, float(h)) for t, h in zip(train_t, _tide(train_t, noise=0.01, seed=5))]
    fetcher = _make_fetcher(rows)
    loop = AutonomousLoop(
        store, stations=["alpha"], max_passes=2, sleep_s=0.0, fetcher=fetcher
    )
    rc = loop.run()
    assert rc == 0


def test_loop_sync_skips_unchanged_after_first(tmp_path):
    store = _setup_two_station_fleet(tmp_path)
    loop = AutonomousLoop(store, stations=["alpha", "bravo"])
    # first pass: no feed -> no assimilation, but initial manifest is written
    rep1 = loop.step()
    assert rep1.synced is True
    # second pass: same state -> unchanged
    rep2 = loop.step()
    assert rep2.synced is False


def test_loop_coverage_slo_met(tmp_path):
    store = _setup_two_station_fleet(tmp_path)
    feed_t = _hourly(T0 + timedelta(hours=30 * 24), 24)
    rows = [(t, float(h)) for t, h in zip(feed_t, _tide(feed_t, noise=0.01, seed=99))]
    fetcher = _make_fetcher(rows)
    loop = AutonomousLoop(store, stations=["alpha"], auto_refit=False, fetcher=fetcher)
    rep = loop.step(end=feed_t[-1])
    slo = rep.slo.get("alpha")
    assert slo is not None
    assert slo.n > 0
    assert slo.status == "met"


# --- CLI smoke -------------------------------------------------------------


def test_cli_slo_query(tmp_path, capsys):
    store = _setup_two_station_fleet(tmp_path)
    rc = cli_main(["slo", "--store", str(store)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "alpha" in out
    assert "bravo" in out


def test_cli_slo_json(tmp_path, capsys):
    store = _setup_two_station_fleet(tmp_path)
    rc = cli_main(["slo", "--store", str(store), "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert "alpha" in data
    assert "coverage" in data["alpha"]


def test_cli_loop_max_passes(tmp_path, capsys):
    store = _setup_two_station_fleet(tmp_path)
    train_t = _hourly(T0, 30 * 24)
    rows = [(t, float(h)) for t, h in zip(train_t, _tide(train_t, noise=0.01, seed=5))]
    fetcher = _make_fetcher(rows)
    loop = AutonomousLoop(
        store, stations=["alpha"], max_passes=1, sleep_s=0.0, fetcher=fetcher
    )
    rc = loop.run()
    assert rc == 0
