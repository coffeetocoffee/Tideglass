"""Tests for v0.6 — operational core: provenance, nowcast, drift, ops."""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tideglass import TideModel  # top-level exports (v0.6)
from tideglass import HealthMonitor, NowcastEngine
from tideglass.cli import main as cli_main
from tideglass.marea import constituents as C
from tideglass.marea import ops as OPS
from tideglass.marea.drift import HealthReport
from tideglass.marea.model import _basis_matrix
from tideglass.marea.nowcast import load_state, save_state
from tideglass.marea.provenance import canonical_digest, provenance
from tideglass.marea.solver import rad_per_hour

UTC = timezone.utc
TRUE = {"M2": (1.0, 0.5), "S2": (0.30, -1.2), "N2": (0.20, 2.0),
        "K1": (0.15, 0.0), "O1": (0.10, 2.8)}
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _hours(times):
    return np.array([(t - T0).total_seconds() / 3600.0 for t in times])


def _synthetic(times, noise=0.0, seed=0, shift=0.0):
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


# --- provenance ------------------------------------------------------------


def test_digest_is_order_and_format_invariant():
    times = _hourly(T0, 48)
    y = _synthetic(times, noise=0.01, seed=1)
    d1 = canonical_digest(times, y)
    rev = canonical_digest(list(reversed(times)), list(reversed(y)))
    assert d1 == rev
    assert d1 != canonical_digest(times, y + 0.5)


def test_provenance_block_pins_window_and_source():
    times = _hourly(T0, 48)
    y = _synthetic(times)
    meta = provenance(times, y, source="noaa-coops:9414290", station="SF")
    assert meta["source"] == "noaa-coops:9414290"
    assert meta["station"] == "SF"
    assert meta["obs_start"] == times[0].isoformat()
    assert meta["obs_end"] == times[-1].isoformat()
    assert meta["n_obs"] == 48
    assert len(meta["data_sha256"]) == 64
    assert meta["tideglass_version"]
    assert meta["fitted_at"]
    with pytest.raises(ValueError):
        provenance([], [])


def test_fit_embeds_provenance_and_roundtrips():
    train = _hourly(T0, 30 * 24)
    model = TideModel.fit(train, _synthetic(train, noise=0.01, seed=5),
                          auto_select=False, station="P",
                          source="csv:train.csv")
    for key in ("source", "obs_start", "obs_end", "n_obs",
                "data_sha256", "tideglass_version", "fitted_at", "rmse"):
        assert key in model.meta, key
    assert model.meta["source"] == "csv:train.csv"
    blob = json.dumps(model.to_artifact())
    restored = TideModel.load_harmonic(blob)
    assert restored.meta["data_sha256"] == model.meta["data_sha256"]
    assert restored.meta["obs_start"] == model.meta["obs_start"]


# --- nowcast engine ----------------------------------------------------------


def test_engine_assimilates_and_tracks_time():
    train = _hourly(T0, 30 * 24)
    model = TideModel.fit(train, _synthetic(train, noise=0.01, seed=5),
                          auto_select=False)
    eng = NowcastEngine(model)
    assert eng.last_time is None
    feed = _hourly(train[-1] + timedelta(hours=1), 72)
    log = eng.update(feed, _synthetic(feed, noise=0.01, seed=6))
    assert log.n_obs == 72
    assert eng.last_time == feed[-1]
    assert log.rms_innovation < 0.05  # good model, small innovations


def test_engine_state_roundtrip(tmp_path):
    train = _hourly(T0, 30 * 24)
    model = TideModel.fit(train, _synthetic(train, noise=0.01, seed=5),
                          auto_select=False)
    eng = NowcastEngine(model)
    feed = _hourly(train[-1] + timedelta(hours=1), 24)
    eng.update(feed, _synthetic(feed, seed=6))
    path = str(tmp_path / "eng.json")
    save_state(eng, path)
    eng2 = load_state(model, path)
    assert eng2.last_time == eng.last_time
    assert eng2._n_updates == eng._n_updates
    fut = _hourly(feed[-1] + timedelta(hours=1), 12)
    p1, p2 = eng.predict(fut), eng2.predict(fut)
    assert np.max(np.abs(p1.mean - p2.mean)) == 0.0


def test_engine_predict_bands_ordered_and_surge_decays():
    train = _hourly(T0, 30 * 24)
    model = TideModel.fit(train, _synthetic(train, noise=0.01, seed=5),
                          auto_select=False)
    eng = NowcastEngine(model, phi=0.9)
    feed = _hourly(train[-1] + timedelta(hours=1), 24)
    # Perturb the last observations so the surge tracker has something to hold.
    eng.update(feed, _synthetic(feed, seed=6) + 0.2)
    fut = _hourly(feed[-1] + timedelta(hours=1), 48)
    pred = eng.predict(fut)
    assert (pred.lower <= pred.mean + 1e-9).all()
    assert (pred.mean <= pred.upper + 1e-9).all()
    # The surge nudge is strongest at short lead and decays away.
    plain = _basis_matrix(eng._constituents, fut) @ eng._coef
    nudge = pred.mean - plain
    assert abs(nudge[0]) > 0.01
    assert abs(nudge[-1]) < abs(nudge[0])


def test_engine_phi_zero_disables_surge():
    train = _hourly(T0, 30 * 24)
    model = TideModel.fit(train, _synthetic(train, noise=0.01, seed=5),
                          auto_select=False)
    eng = NowcastEngine(model, phi=0.0)
    feed = _hourly(train[-1] + timedelta(hours=1), 24)
    log = eng.update(feed, _synthetic(feed, seed=6) + 0.2)
    assert log.last_surge == 0.0


def test_engine_auto_refit_from_history():
    train = _hourly(T0, 30 * 24)
    model = TideModel.fit(train, _synthetic(train, noise=0.01, seed=5),
                          auto_select=False, station="R",
                          source="csv:train.csv")
    eng = NowcastEngine(model)
    feed = _hourly(train[-1] + timedelta(hours=1), 72)
    eng.update(feed, _synthetic(feed, noise=0.01, seed=6))
    fresh_model, fresh_eng = eng.auto_refit(source="nowcast-refit:R")
    assert fresh_model.meta["source"] == "nowcast-refit:R"
    assert fresh_model.meta["data_sha256"]  # new window pinned
    assert len(fresh_eng.history_data()[0]) == 72  # history carried over


def test_engine_from_loaded_harmonic_seeds_prior():
    loaded = TideModel.load_harmonic({
        "station": "X", "mean": 0.7,
        "constituents": [{"name": n, "amplitude": a, "phase": 0.0}
                         for n, (a, _) in TRUE.items()],
    })
    eng = NowcastEngine(loaded)
    assert np.all(np.diag(eng._P) > 0)
    feed = _hourly(T0, 24)
    eng.update(feed, _synthetic(feed, seed=3))
    assert eng._n_updates == 24


# --- drift monitor -----------------------------------------------------------


def _monitor(**kw):
    kw.setdefault("window", 72)
    return HealthMonitor(**kw)


def test_monitor_healthy_on_good_model():
    train = _hourly(T0, 30 * 24)
    model = TideModel.fit(train, _synthetic(train, noise=0.01, seed=5),
                          auto_select=False)
    feed = _hourly(train[-1] + timedelta(hours=1), 72)
    y = _synthetic(feed, noise=0.01, seed=9)
    mon = _monitor(baseline_rmse=model.meta["rmse"])
    rep = mon.update(y, model.predict(feed))
    assert isinstance(rep, HealthReport)
    assert not rep.needs_refit
    assert rep.coverage > 0.80


def test_monitor_flags_datum_shift():
    train = _hourly(T0, 30 * 24)
    model = TideModel.fit(train, _synthetic(train, noise=0.01, seed=5),
                          auto_select=False)
    feed = _hourly(train[-1] + timedelta(hours=1), 72)
    y = _synthetic(feed, noise=0.01, seed=9) + 0.5  # +50 cm datum jump
    mon = _monitor(baseline_rmse=model.meta["rmse"])
    rep = mon.update(y, model.predict(feed))
    assert rep.needs_refit
    assert any("bias" in r or "coverage" in r or "rmse" in r for r in rep.reasons)
    assert mon.needs_refit


def test_monitor_buffer_roundtrip():
    train = _hourly(T0, 30 * 24)
    model = TideModel.fit(train, _synthetic(train, noise=0.01, seed=5),
                          auto_select=False)
    feed = _hourly(train[-1] + timedelta(hours=1), 72)
    mon = _monitor()
    mon.update(_synthetic(feed, seed=9), model.predict(feed))
    mon2 = HealthMonitor.from_dict(mon.state_for_save())
    r1, r2 = mon.report(), mon2.report()
    assert (r1.coverage, r1.rmse, r1.n) == (r2.coverage, r2.rmse, r2.n)


# --- ops flywheel ------------------------------------------------------------


def _seed_store(store, station="OP"):
    train = _hourly(T0, 30 * 24)
    model = TideModel.fit(train, _synthetic(train, noise=0.01, seed=5),
                          auto_select=False, station=station,
                          source="csv:train.csv")
    import os as _os

    _os.makedirs(store, exist_ok=True)
    with open(f"{store}/{station}.json", "w") as fh:
        json.dump(model.to_artifact(), fh)
    return train


def test_rerun_assimilates_and_skips_replays(tmp_path):
    store, station = str(tmp_path / "s"), "OP"
    train = _seed_store(store, station)
    feed_t = _hourly(train[-1] + timedelta(hours=1), 72)
    feed_csv = str(tmp_path / "feed.csv")
    _write_csv(feed_csv, feed_t, _synthetic(feed_t, noise=0.01, seed=6))

    rep1 = OPS.rerun(station, feed_csv, store=store)
    assert rep1.n_new == 72 and rep1.n_skipped == 0
    assert not rep1.health.needs_refit
    assert not rep1.refit_done

    rep2 = OPS.rerun(station, feed_csv, store=store)  # same feed again
    assert rep2.n_new == 0 and rep2.n_skipped == 72
    assert rep2.log is None


def test_rerun_auto_refits_on_drift(tmp_path):
    store, station = str(tmp_path / "s"), "OP"
    train = _seed_store(store, station)
    feed_t = _hourly(train[-1] + timedelta(hours=1), 72)
    feed_csv = str(tmp_path / "feed.csv")
    _write_csv(feed_csv, feed_t, _synthetic(feed_t, noise=0.01, seed=6) + 0.5)

    probe = OPS.rerun(station, feed_csv, store=store, auto_refit=False)
    assert probe.health.needs_refit and not probe.refit_done

    import os as _os

    _os.remove(f"{store}/{station}.nowcast.json")
    _os.remove(f"{store}/{station}.health.json")
    rep = OPS.rerun(station, feed_csv, store=store, auto_refit=True)
    assert rep.refit_done
    # The fresh model is healthy by construction on its own training window.
    assert not rep.health.needs_refit
    # ... and the next pass stays quiet (no refit loop on the same data).
    rep2 = OPS.rerun(station, feed_csv, store=store, auto_refit=True)
    assert rep2.n_new == 0 and not rep2.health.needs_refit
    assert not rep2.refit_done
    with open(f"{store}/{station}.json") as fh:
        artifact = json.load(fh)
    assert artifact["meta"]["source"] == "nowcast-refit:OP"
    assert artifact["meta"]["refit_history"]  # lineage recorded
    assert artifact["meta"]["obs_start"]  # original window kept


def test_cold_start_and_poll_with_fake_fetcher(tmp_path):
    store, station = str(tmp_path / "s"), "CS"
    rows = [(t, float(h)) for t, h in
            zip(_hourly(T0, 30 * 24), _synthetic(_hourly(T0, 30 * 24), seed=1))]

    def fake_fetch(station, begin, end, datum="MLLW", **kw):
        return list(rows)

    model = OPS.cold_start(station, "2024-01-01", "2024-01-30", store=store,
                           fetcher=fake_fetch)
    assert model.station == station
    import os as _os

    for suffix in (".json", ".nowcast.json", ".health.json", ".ops.json"):
        assert _os.path.exists(f"{store}/{station}{suffix}"), suffix

    # Nothing newer than the cursor: no new rows.
    assert OPS.poll(station, store=store,
                    fetcher=lambda *a, **k: list(rows[:100])) is None

    # New rows after the cursor get assimilated; replays are skipped.
    extra_t = _hourly(T0 + timedelta(hours=30 * 24), 48)
    extra = [(t, float(h)) for t, h in
             zip(extra_t, _synthetic(extra_t, seed=2))]
    rep = OPS.poll(station, store=store,
                   fetcher=lambda *a, **k: extra, auto_refit=False)
    assert rep is not None and rep.n_new == len(extra)
    rep2 = OPS.poll(station, store=store,
                    fetcher=lambda *a, **k: extra, auto_refit=False)
    assert rep2 is not None and rep2.n_new == 0


# --- CLI ---------------------------------------------------------------------


def test_cli_fit_pins_source_and_nowcast_runs(tmp_path, capsys):
    train = _hourly(T0, 30 * 24)
    csv = tmp_path / "obs.csv"
    _write_csv(str(csv), train, _synthetic(train, noise=0.02, seed=11))
    store = str(tmp_path / "store")
    assert cli_main(["fit", str(csv), "--station", "T", "--store", store,
                     "--alpha", "1e-4", "--source", "csv:obs.csv",
                     "--no-select"]) == 0
    out = capsys.readouterr().out
    assert "csv:obs.csv" in out and "data_sha256" in out

    feed_t = _hourly(train[-1] + timedelta(hours=1), 72)
    feed = tmp_path / "feed.csv"
    _write_csv(str(feed), feed_t, _synthetic(feed_t, noise=0.02, seed=12))
    capsys.readouterr()
    assert cli_main(["nowcast", "T", str(feed), "--store", store,
                     "--no-auto-refit", "--hours", "6"]) == 0
    out = capsys.readouterr().out
    assert "assimilated" in out and "health[ok]" in out
    assert len([ln for ln in out.splitlines()
                if ln and not ln.startswith(("#", "station:", "assimilated:",
                                             "innovation:", "health"))]) == 6


def test_cli_nowcast_flags_stale_without_refit(tmp_path, capsys):
    train = _hourly(T0, 30 * 24)
    csv = tmp_path / "obs.csv"
    _write_csv(str(csv), train, _synthetic(train, noise=0.01, seed=11))
    store = str(tmp_path / "store")
    assert cli_main(["fit", str(csv), "--station", "T", "--store", store,
                     "--no-select"]) == 0
    feed_t = _hourly(train[-1] + timedelta(hours=1), 72)
    feed = tmp_path / "feed.csv"
    _write_csv(str(feed), feed_t, _synthetic(feed_t, seed=12) + 0.5)
    capsys.readouterr()
    assert cli_main(["nowcast", "T", str(feed), "--store", store,
                     "--no-auto-refit", "--hours", "0"]) == 0
    out = capsys.readouterr().out
    assert "REFIT" in out


def test_cli_nowcast_missing_station(tmp_path):
    assert cli_main(["nowcast", "NOPE", str(tmp_path / "f.csv"),
                     "--store", str(tmp_path)]) == 2
