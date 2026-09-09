"""Operational flywheel for Marea Core (v0.6).

This is the "runs itself" layer: fetch fresh gauge observations from a source
(NOAA CO-OPS by default), assimilate them into the deployed model's
:class:`~tideglass.marea.nowcast.NowcastEngine`, score the rolling window with
:class:`~tideglass.marea.drift.HealthMonitor`, and auto-refit when the monitor
fires. Per-station state lives next to the artifacts in the store::

    <store>/<station>.json          model artifact (with provenance)
    <store>/<station>.nowcast.json  engine state (coefficients + surge + history)
    <store>/<station>.health.json   monitor rolling buffer + thresholds
    <store>/<station>.ops.json      last poll cursor (feed window bookkeeping)

Entry points:

* :func:`cold_start` — fit the first model for a station from a fetched window.
* :func:`rerun` — one assimilation + health + (optional) refit pass over a CSV.
* :func:`poll` — one live pass: fetch the newest observations, then :func:`rerun`.

All three are plain functions (no threads, no daemons); the CLI repeats
:func:`poll` on a sleep loop, and tests inject a fake ``fetcher``.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from tideglass.fetch import fetch_noaa
from tideglass.marea.drift import HealthMonitor, HealthReport
from tideglass.marea.model import TideModel
from tideglass.marea.nowcast import (
    NowcastEngine,
    UpdateLog,
    load_state,
    save_state,
)

_Fetcher = Callable[..., list[tuple[datetime, float]]]

_DEFAULT_WINDOW = 168
_DEFAULT_MIN_COV = 0.80
_DEFAULT_RMSE_RATIO = 2.0
_DEFAULT_BIAS_TOL = 0.15


@dataclass
class OpsReport:
    """Everything one :func:`rerun` / :func:`poll` pass produced."""

    station: str
    n_new: int  # observations assimilated this pass
    n_skipped: int  # feed rows already seen (≤ last assimilated time)
    log: UpdateLog | None
    health: HealthReport
    refit_done: bool
    feed_start: str | None = None
    feed_end: str | None = None


def _parse_time(s: str) -> datetime:
    s = s.strip()
    if s.endswith(("Z", "z")):
        s = s[:-1] + "+00:00"
    t = datetime.fromisoformat(s)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t


def read_feed_rows(path: str) -> list[tuple[datetime, float]]:
    """Read ``time,height`` rows from a CSV (header-tolerant, sorted)."""
    rows: list[tuple[datetime, float]] = []
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if not row or not row[0].strip():
                continue
            try:
                t = _parse_time(row[0])
            except ValueError:
                continue  # header line
            rows.append((t, float(row[1])))
    rows.sort(key=lambda r: r[0])
    return rows


def _paths(store: str, station: str) -> dict[str, str]:
    return {
        "model": os.path.join(store, f"{station}.json"),
        "nowcast": os.path.join(store, f"{station}.nowcast.json"),
        "health": os.path.join(store, f"{station}.health.json"),
        "ops": os.path.join(store, f"{station}.ops.json"),
    }


def _load_model(store: str, station: str) -> TideModel:
    path = _paths(store, station)["model"]
    if not os.path.exists(path):
        raise ValueError(
            f"no model artifact for station {station!r} in {store!r} "
            f"(run `tideglass fit` or `cold_start` first)")
    return TideModel.load_harmonic(path)


def _load_monitor(store: str, station: str,
                  baseline_rmse: float | None) -> HealthMonitor:
    path = _paths(store, station)["health"]
    if os.path.exists(path):
        with open(path) as fh:
            mon = HealthMonitor.from_dict(json.load(fh))
        if mon.baseline_rmse is None:
            mon.baseline_rmse = baseline_rmse
        return mon
    return HealthMonitor(
        window=_DEFAULT_WINDOW, min_coverage=_DEFAULT_MIN_COV,
        rmse_ratio=_DEFAULT_RMSE_RATIO, bias_tolerance=_DEFAULT_BIAS_TOL,
        baseline_rmse=baseline_rmse,
    )


def _seat_engine(model: TideModel, store: str, station: str,
                 q_mean: float = 1e-8, q_coef: float = 1e-10,
                 r: float | None = None, phi: float = 0.9) -> NowcastEngine:
    """Resume the persisted engine, re-seating when the base model changed."""
    paths = _paths(store, station)
    digest = str(model.meta.get("data_sha256", "unknown"))
    if os.path.exists(paths["nowcast"]):
        eng = load_state(model, paths["nowcast"])
        if eng.state().base_sha256 == digest:
            # Adopt the persisted tunables (they may differ from defaults).
            return eng
    return NowcastEngine(model, q_mean=q_mean, q_coef=q_coef, r=r, phi=phi,
                         base_sha256=digest)


def rerun(
    station: str,
    feed: Sequence[tuple[datetime, float]] | str,
    store: str = ".tideglass",
    alpha: float = 0.05,
    auto_refit: bool = True,
    min_history: int = 48,
) -> OpsReport:
    """Assimilate a feed, score health, and refit when the monitor fires.

    :param feed: ``[(time, height), ...]`` rows or a CSV path readable by
        :func:`read_feed_rows`.
    :param auto_refit: when the monitor reports stale, fit a fresh model from
        the engine's accumulated history, replace the artifact (lineage goes
        to ``refit_history``), and re-seat the engine.
    :param min_history: minimum accumulated observations before a refit is
        attempted (a refit on a handful of rows would be worse than the
        drift itself).
    """
    rows = read_feed_rows(feed) if isinstance(feed, str) else sorted(feed)
    model = _load_model(store, station)
    engine = _seat_engine(model, store, station)
    monitor = _load_monitor(store, station, engine.baseline_rmse)

    # Only genuinely new observations move the filter; replays are skipped so
    # a repeated feed can never regress the state.
    last = engine.last_time
    new_rows = [row for row in rows if last is None or row[0] > last]
    n_skipped = len(rows) - len(new_rows)

    log: UpdateLog | None = None
    if new_rows:
        times = [t for t, _ in new_rows]
        heights = np.array([h for _, h in new_rows], dtype=float)
        prior = engine.predict(times)
        monitor.update(heights, prior)
        log = engine.update(times, heights)
        feed_start, feed_end = times[0].isoformat(), times[-1].isoformat()
    else:
        feed_start = feed_end = None

    health = monitor.report()
    refit_done = False
    # A refit needs fresh evidence: with no new observations the stale flag
    # just carries over, and re-fitting the same history would loop forever.
    if (auto_refit and len(new_rows) > 0 and health.needs_refit
            and len(engine.history_data()[0]) >= min_history):
        fresh_model, engine = engine.auto_refit(
            alpha=alpha, source=f"nowcast-refit:{station}")
        # Lineage: keep the original window, append this refit.
        prev = dict(model.meta)
        new_meta = dict(fresh_model.meta)
        hist = list(prev.get("refit_history", []))
        hist.append({
            "source": f"nowcast-refit:{station}",
            "obs_start": new_meta.get("obs_start"),
            "obs_end": new_meta.get("obs_end"),
            "n_obs": new_meta.get("n_obs"),
            "data_sha256": new_meta.get("data_sha256"),
            "fitted_at": new_meta.get("fitted_at"),
        })
        new_meta["refit_history"] = hist
        if prev.get("obs_start"):
            new_meta["obs_start"] = prev["obs_start"]
        fresh_model.meta.update(new_meta)
        with open(_paths(store, station)["model"], "w") as fh:
            json.dump(fresh_model.to_artifact(), fh, indent=2)
        # The new model is healthy by construction on its training window, so
        # reset the rolling buffer (which still holds pre-refit residuals) and
        # seed it by scoring the fresh model on the refit window's tail.
        monitor = HealthMonitor(
            window=monitor.window, min_coverage=monitor.min_coverage,
            rmse_ratio=monitor.rmse_ratio,
            bias_tolerance=monitor.bias_tolerance,
            baseline_rmse=fresh_model.meta.get("rmse"),
        )
        hist_t, hist_y = engine.history_data()
        tail = slice(max(0, len(hist_t) - monitor.window), len(hist_t))
        monitor.update(np.asarray(hist_y[tail]),
                       engine.predict(list(hist_t[tail])))
        health = monitor.report()
        refit_done = True
        model = fresh_model

    save_state(engine, _paths(store, station)["nowcast"])
    with open(_paths(store, station)["health"], "w") as fh:
        json.dump(monitor.state_for_save(), fh, indent=2)

    return OpsReport(
        station=station, n_new=len(new_rows), n_skipped=n_skipped,
        log=log, health=health, refit_done=refit_done,
        feed_start=feed_start, feed_end=feed_end,
    )


def cold_start(
    station: str,
    begin: str,
    end: str,
    store: str = ".tideglass",
    alpha: float = 0.05,
    datum: str = "MLLW",
    fetcher: _Fetcher | None = None,
) -> TideModel:
    """Fit the first model for a station from a fetched window and persist it.

    Dates are ``YYYY-MM-DD`` (UTC). Returns the fitted model; also seeds empty
    engine/health state so the first :func:`poll` continues from ``end``.
    """
    get = fetcher or fetch_noaa
    rows = get(station, begin, end, datum=datum)
    if not rows:
        raise ValueError(f"no data for station {station!r} in {begin}..{end}")
    rows.sort(key=lambda r: r[0])
    os.makedirs(store, exist_ok=True)
    model = TideModel.fit(
        [t for t, _ in rows], [h for _, h in rows],
        alpha=alpha, station=station, source=f"noaa-coops:{station}",
    )
    paths = _paths(store, station)
    with open(paths["model"], "w") as fh:
        json.dump(model.to_artifact(), fh, indent=2)
    with open(paths["ops"], "w") as fh:
        json.dump({"station": station, "last_end": rows[-1][0].isoformat(),
                   "datum": datum}, fh, indent=2)
    monitor = _load_monitor(store, station, model.meta.get("rmse"))
    with open(paths["health"], "w") as fh:
        json.dump(monitor.state_for_save(), fh, indent=2)
    save_state(_seat_engine(model, store, station), paths["nowcast"])
    return model


def poll(
    station: str,
    store: str = ".tideglass",
    lookback_hours: int = 72,
    alpha: float = 0.05,
    datum: str = "MLLW",
    fetcher: _Fetcher | None = None,
    auto_refit: bool = True,
    end: datetime | None = None,
) -> OpsReport | None:
    """One live pass: fetch new observations since the last cursor, then rerun.

    The window starts at the persisted ``last_end`` cursor (minus a 1-hour
    overlap for safety) or, on first run, ``end - lookback_hours``. Returns
    ``None`` when the source has no new rows.
    """
    get = fetcher or fetch_noaa
    paths = _paths(store, station)
    end = end or datetime.now(timezone.utc)
    last_end: datetime | None = None
    if os.path.exists(paths["ops"]):
        with open(paths["ops"]) as fh:
            raw = json.load(fh).get("last_end")
        if raw:
            last_end = _parse_time(raw)
    begin = ((last_end - timedelta(hours=1)) if last_end
             else (end - timedelta(hours=lookback_hours)))
    if begin >= end:
        return None
    rows = get(station, begin.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
               datum=datum)
    rows = sorted(rows)
    if last_end is not None:
        rows = [row for row in rows if row[0] > last_end - timedelta(hours=1)]
    if not rows:
        return None
    report = rerun(station, rows, store=store, alpha=alpha,
                   auto_refit=auto_refit)
    with open(paths["ops"], "w") as fh:
        json.dump({"station": station,
                   "last_end": rows[-1][0].isoformat(), "datum": datum}, fh, indent=2)
    return report
