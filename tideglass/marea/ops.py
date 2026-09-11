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
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from tideglass.fetch import fetch_noaa_range
from tideglass.marea.drift import HealthMonitor, HealthReport
from tideglass.marea.federation import (
    GlobalFederation,
    PeerBundle,
    _contribution_from_model,
)
from tideglass.marea.ledger import DecisionLedger
from tideglass.marea.model import TideModel
from tideglass.marea.nowcast import (
    NowcastEngine,
    UpdateLog,
    load_state,
    save_state,
)
from tideglass.marea.pooling import HierarchicalPool
from tideglass.marea.slo import SloMonitor, SloReport, load_slo

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
            f"(run `tideglass fit` or `cold_start` first)"
        )
    return TideModel.load_harmonic(path)


def _load_monitor(
    store: str, station: str, baseline_rmse: float | None
) -> HealthMonitor:
    path = _paths(store, station)["health"]
    if os.path.exists(path):
        with open(path) as fh:
            mon = HealthMonitor.from_dict(json.load(fh))
        if mon.baseline_rmse is None:
            mon.baseline_rmse = baseline_rmse
        return mon
    return HealthMonitor(
        window=_DEFAULT_WINDOW,
        min_coverage=_DEFAULT_MIN_COV,
        rmse_ratio=_DEFAULT_RMSE_RATIO,
        bias_tolerance=_DEFAULT_BIAS_TOL,
        baseline_rmse=baseline_rmse,
    )


def _seat_engine(
    model: TideModel,
    store: str,
    station: str,
    q_mean: float = 1e-8,
    q_coef: float = 1e-10,
    r: float | None = None,
    phi: float = 0.9,
) -> NowcastEngine:
    """Resume the persisted engine, re-seating when the base model changed."""
    paths = _paths(store, station)
    digest = str(model.meta.get("data_sha256", "unknown"))
    if os.path.exists(paths["nowcast"]):
        eng = load_state(model, paths["nowcast"])
        if eng.state().base_sha256 == digest:
            # Adopt the persisted tunables (they may differ from defaults).
            return eng
    return NowcastEngine(
        model, q_mean=q_mean, q_coef=q_coef, r=r, phi=phi, base_sha256=digest
    )


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
    if (
        auto_refit
        and len(new_rows) > 0
        and health.needs_refit
        and len(engine.history_data()[0]) >= min_history
    ):
        fresh_model, engine = engine.auto_refit(
            alpha=alpha, source=f"nowcast-refit:{station}"
        )
        # Lineage: keep the original window, append this refit.
        prev = dict(model.meta)
        new_meta = dict(fresh_model.meta)
        hist = list(prev.get("refit_history", []))
        hist.append(
            {
                "source": f"nowcast-refit:{station}",
                "obs_start": new_meta.get("obs_start"),
                "obs_end": new_meta.get("obs_end"),
                "n_obs": new_meta.get("n_obs"),
                "data_sha256": new_meta.get("data_sha256"),
                "fitted_at": new_meta.get("fitted_at"),
            }
        )
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
            window=monitor.window,
            min_coverage=monitor.min_coverage,
            rmse_ratio=monitor.rmse_ratio,
            bias_tolerance=monitor.bias_tolerance,
            baseline_rmse=fresh_model.meta.get("rmse"),
        )
        hist_t, hist_y = engine.history_data()
        tail = slice(max(0, len(hist_t) - monitor.window), len(hist_t))
        monitor.update(np.asarray(hist_y[tail]), engine.predict(list(hist_t[tail])))
        health = monitor.report()
        refit_done = True
        model = fresh_model

    save_state(engine, _paths(store, station)["nowcast"])
    with open(_paths(store, station)["health"], "w") as fh:
        json.dump(monitor.state_for_save(), fh, indent=2)

    return OpsReport(
        station=station,
        n_new=len(new_rows),
        n_skipped=n_skipped,
        log=log,
        health=health,
        refit_done=refit_done,
        feed_start=feed_start,
        feed_end=feed_end,
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
    get = fetcher or fetch_noaa_range
    rows = get(station, begin, end, datum=datum)
    if not rows:
        raise ValueError(f"no data for station {station!r} in {begin}..{end}")
    rows.sort(key=lambda r: r[0])
    os.makedirs(store, exist_ok=True)
    model = TideModel.fit(
        [t for t, _ in rows],
        [h for _, h in rows],
        alpha=alpha,
        station=station,
        source=f"noaa-coops:{station}",
    )
    paths = _paths(store, station)
    with open(paths["model"], "w") as fh:
        json.dump(model.to_artifact(), fh, indent=2)
    with open(paths["ops"], "w") as fh:
        json.dump(
            {"station": station, "last_end": rows[-1][0].isoformat(), "datum": datum},
            fh,
            indent=2,
        )
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
    ``None`` when the source has no new rows. Fetches through
    :func:`~tideglass.fetch.fetch_noaa_range`, which chunks at NOAA's 31-day
    per-request limit, so any ``lookback_hours`` works.
    """
    get = fetcher or fetch_noaa_range
    paths = _paths(store, station)
    end = end or datetime.now(timezone.utc)
    last_end: datetime | None = None
    if os.path.exists(paths["ops"]):
        with open(paths["ops"]) as fh:
            raw = json.load(fh).get("last_end")
        if raw:
            last_end = _parse_time(raw)
    begin = (
        (last_end - timedelta(hours=1))
        if last_end
        else (end - timedelta(hours=lookback_hours))
    )
    if begin >= end:
        return None
    rows = get(
        station, begin.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), datum=datum
    )
    rows = sorted(rows)
    if last_end is not None:
        rows = [row for row in rows if row[0] > last_end - timedelta(hours=1)]
    if not rows:
        return None
    report = rerun(station, rows, store=store, alpha=alpha, auto_refit=auto_refit)
    with open(paths["ops"], "w") as fh:
        json.dump(
            {"station": station, "last_end": rows[-1][0].isoformat(), "datum": datum},
            fh,
            indent=2,
        )
    return report


# ---------------------------------------------------------------------------
# v3.1 — autonomous closed-loop + per-station SLOs
# ---------------------------------------------------------------------------


@dataclass
class AutonomousAction:
    """One autonomous action taken by the loop on one station."""

    station: str
    at: str
    kind: str
    reason: str
    cost: float
    loss: float
    slo_before: str | None
    slo_after: str | None
    health_before: dict | None = None
    health_after: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> AutonomousAction:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AutonomousReport:
    """One standing-loop pass over the fleet."""

    pass_no: int
    end: str
    stations: int
    actions: list[AutonomousAction]
    slo: dict[str, SloReport]
    synced: bool
    bundle_changed: bool
    bundle_digest: str | None

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    @property
    def n_refit(self) -> int:
        return sum(1 for a in self.actions if a.kind == "refit")

    @property
    def n_reseed(self) -> int:
        return sum(1 for a in self.actions if a.kind == "reseed")

    @property
    def n_sync(self) -> int:
        return sum(1 for a in self.actions if a.kind == "sync")

    @property
    def slo_met(self) -> int:
        return sum(1 for r in self.slo.values() if r.status == "met")

    @property
    def slo_degrading(self) -> int:
        return sum(1 for r in self.slo.values() if r.status == "degrading")

    @property
    def slo_breached(self) -> int:
        return sum(1 for r in self.slo.values() if r.status == "breached")

    def __str__(self) -> str:
        lines = [
            (
                f"loop pass {self.pass_no}: {self.stations} station(s)  "
                f"actions={self.n_actions} (refit={self.n_refit} reseed="
                f"{self.n_reseed} sync={self.n_sync})  synced={self.synced}"
            ),
            (
                f"SLO: met={self.slo_met} degrading={self.slo_degrading} "
                f"breached={self.slo_breached}"
            ),
        ]
        for a in self.actions:
            lines.append(f"  {a.kind:8s} {a.station}: {a.reason}")
        for s, r in sorted(self.slo.items()):
            lines.append(
                f"  SLO {s}: {r.status} coverage={r.coverage:.3f} rmse={r.rmse:.4f} m"
            )
        return "\n".join(lines)


_NON_MODEL_BASE = {"loop", "peer_bundle"}


def _is_model_artifact(path: str) -> bool:
    """Cheap check: a model artifact has a 'constituents' key."""
    try:
        with open(path) as fh:
            head = fh.read(8192)
        if not head.lstrip().startswith("{"):
            return False
        import json as _json

        data = _json.loads(head) if len(head) < 8192 else None
        if data is None:
            with open(path) as fh:
                data = _json.load(fh)
        return "constituents" in data
    except (OSError, ValueError):
        return False


def _fleet_stations(store: str) -> list[str]:
    """Return station ids with deployed harmonic artifacts in ``store``."""
    out: list[str] = []
    if not os.path.isdir(store):
        return out
    for fn in sorted(os.listdir(store)):
        if not fn.endswith(".json") or fn.startswith("."):
            continue
        name = fn[:-5]
        if name in _NON_MODEL_BASE:
            continue
        if not _is_model_artifact(os.path.join(store, fn)):
            continue
        out.append(name)
    return out


def _load_coords(coords) -> dict[str, tuple[float, float]] | None:
    """Accept a dict or a ``station,lon,lat`` CSV path and return coords."""
    if coords is None:
        return None
    if isinstance(coords, dict):
        return {k: (float(v[0]), float(v[1])) for k, v in coords.items()}
    path = str(coords)
    out: dict[str, tuple[float, float]] = {}
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if not row or not row[0].strip() or row[0].strip().startswith("#"):
                continue
            parts = [p.strip() for p in row[:3]]
            if len(parts) < 3:
                continue
            out[parts[0]] = (float(parts[1]), float(parts[2]))
    return out or None


def _artifact_bundle(store: str, peer_id: str) -> PeerBundle:
    """Build a :class:`PeerBundle` from deployed harmonic artifacts in ``store``.

    Unlike :func:`build_peer_bundle`, this reads ``<station>.json`` model
    artifacts directly (not crowd CSVs), so the loop can sync the deployed
    fleet without a separate crowd store.
    """
    stations: dict[str, Any] = {}
    for name in _fleet_stations(store):
        try:
            model = TideModel.load_harmonic(os.path.join(store, f"{name}.json"))
        except (OSError, ValueError, KeyError):
            continue
        alias = _alias(peer_id, name)
        c = _contribution_from_model(model, alias)
        # attach surge / extremes if present (v2.3)
        surge_p = os.path.join(store, f"{name}.surge.json")
        if os.path.exists(surge_p):
            try:
                with open(surge_p) as fh:
                    c.surge = json.load(fh)
            except (OSError, ValueError):
                pass
        gpd_p = os.path.join(store, f"{name}.gpd.json")
        if os.path.exists(gpd_p):
            try:
                with open(gpd_p) as fh:
                    c.extremes = json.load(fh)
            except (OSError, ValueError):
                pass
        stations[alias] = c
    gen = datetime.now(timezone.utc).isoformat()
    return PeerBundle(
        peer_id=peer_id,
        generated_at=gen,
        tideglass_version=_pkg_version(),
        stations=stations,
        meta={"store": os.path.abspath(store)},
    )


def _pkg_version() -> str:
    try:
        from importlib.metadata import version

        return version("tideglass")
    except (ImportError, OSError):
        return "3.0.0"


def _alias(peer_id: str, name: str) -> str:
    import hashlib

    return hashlib.sha256(f"{peer_id}|{name}".encode()).hexdigest()[:12]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AutonomousLoop:
    """Standing operator-free loop over a deployed fleet (v3.1).

    Each :meth:`step` watches every station's :class:`HealthMonitor`, refits
    when ``needs_refit`` fires, re-seeds short records through the pool, and
    re-syncs the peer bundle only when constants actually changed (v2.1
    deltas keep it cheap). Every autonomous action is logged into a
    :class:`~tideglass.marea.ledger.DecisionLedger` as the audit trail.

    The loop can run once (``step``) or block with ``run``.
    """

    def __init__(
        self,
        store: str,
        stations: Sequence[str] | None = None,
        *,
        crowd_store: str | None = None,
        federation_root: str | None = None,
        coords=None,
        ledger_root: str | None = None,
        peer_id: str = "local",
        sleep_s: float = 600.0,
        max_passes: int | None = None,
        fetcher: _Fetcher | None = None,
        datum: str = "MLLW",
        alpha: float = 0.05,
        auto_refit: bool = True,
        min_history: int = 48,
        seed_short_min: int = 48,
        lookback_hours: int = 72,
        action_cost: float = 0.0,
        breach_loss: float = 0.0,
    ):
        self.store = os.path.abspath(store)
        self.crowd_store = os.path.abspath(crowd_store) if crowd_store else None
        self.federation_root = (
            os.path.abspath(os.path.join(federation_root, "federation"))
            if federation_root
            else None
        )
        self.coords = _load_coords(coords)
        self.ledger_root = (
            os.path.abspath(ledger_root)
            if ledger_root
            else os.path.join(self.store, "ledger")
        )
        self.peer_id = peer_id
        self.sleep_s = float(sleep_s)
        self.max_passes = max_passes
        self.fetcher = fetcher
        self.datum = datum
        self.alpha = float(alpha)
        self.auto_refit = bool(auto_refit)
        self.min_history = int(min_history)
        self.seed_short_min = int(seed_short_min)
        self.lookback_hours = int(lookback_hours)
        self.action_cost = float(action_cost)
        self.breach_loss = float(breach_loss)
        self._stations = list(stations) if stations else _fleet_stations(store)
        self._pass = 0
        self._last_manifest: dict | None = None
        self._load_loop_state()

    # -- state persistence --------------------------------------------------

    def _loop_state_path(self) -> str:
        return os.path.join(self.store, "loop.json")

    def _load_loop_state(self) -> None:
        p = self._loop_state_path()
        if os.path.exists(p):
            try:
                with open(p) as fh:
                    d = json.load(fh)
                self._last_manifest = d.get("last_manifest")
            except (OSError, ValueError):
                self._last_manifest = None

    def _save_loop_state(self, manifest: dict | None) -> None:
        p = self._loop_state_path()
        with open(p, "w") as fh:
            json.dump(
                {"last_pass": _now_iso(), "last_manifest": manifest}, fh, indent=2
            )

    # -- internal helpers ---------------------------------------------------

    def _slo_for(self, station: str) -> SloMonitor:
        cfg = load_slo(self.store, station)
        return SloMonitor(config=cfg)

    def _federation(self) -> GlobalFederation | None:
        if self.federation_root and os.path.isdir(self.federation_root):
            return GlobalFederation(self.federation_root)
        return None

    def _local_pool(self) -> HierarchicalPool | None:
        models: dict[str, TideModel] = {}
        for name in _fleet_stations(self.store):
            if name == "__pycache__":
                continue
            try:
                models[name] = TideModel.load_harmonic(
                    os.path.join(self.store, f"{name}.json")
                )
            except (OSError, ValueError, KeyError):
                continue
        if len(models) < 2:
            return None
        return HierarchicalPool(models, coords=self.coords)

    def _reseed_station(
        self,
        station: str,
        times: list[datetime],
        heights: np.ndarray,
    ) -> TideModel | None:
        """Pool a short record against the federated prior (or local pool)."""
        fed = self._federation()
        if fed is not None:
            coords = None
            if self.coords and station in self.coords:
                coords = self.coords[station]
            try:
                model = fed.seed(list(times), heights, station=station, coords=coords)
            except (ValueError, OSError):
                model = None
            if model is not None:
                return model
        pool = self._local_pool()
        if pool is not None:
            coords = None
            if self.coords and station in self.coords:
                coords = self.coords[station]
            try:
                return pool.seed_short(list(times), heights, station, coords)
            except (ValueError, OSError):
                return None
        return None

    def _persist_model(self, station: str, model: TideModel) -> None:
        path = os.path.join(self.store, f"{station}.json")
        with open(path, "w") as fh:
            json.dump(model.to_artifact(), fh, indent=2)
        # reset the nowcast engine + monitor so the pooled model starts clean
        _seat_engine(model, self.store, station)
        mon = HealthMonitor(
            window=168,
            min_coverage=0.80,
            rmse_ratio=2.0,
            bias_tolerance=0.15,
            baseline_rmse=model.meta.get("rmse"),
        )
        with open(_paths(self.store, station)["health"], "w") as fh:
            json.dump(mon.state_for_save(), fh, indent=2)

    def _ledger_path(self, station: str) -> str:
        os.makedirs(self.ledger_root, exist_ok=True)
        return os.path.join(self.ledger_root, f"{station}.ledger.json")

    def _log_action(
        self,
        station: str,
        kind: str,
        reason: str,
        slo_metric: float | None,
        slo_target: float | None,
        acted: bool = True,
        cost: float | None = None,
        loss: float | None = None,
    ) -> None:
        p = self._ledger_path(station)
        if os.path.exists(p):
            ledger = DecisionLedger.load(p)
        else:
            ledger = DecisionLedger()
        c = cost if cost is not None else self.action_cost
        l = loss if loss is not None else self.breach_loss
        ledger.log_action(
            time=_now_iso(),
            action=kind,
            reason=reason,
            cost=c,
            loss=l,
            slo_metric=slo_metric,
            slo_target=slo_target,
            acted=acted,
        )
        ledger.save(p)

    def _maybe_sync(self) -> tuple[bool, dict | None]:
        """Sync the fleet bundle when constants actually changed.

        Returns ``(synced, manifest_or_None)``.
        """
        if not self._stations:
            return False, None
        try:
            bundle = _artifact_bundle(self.store, self.peer_id)
        except (OSError, ValueError):
            return False, None
        manifest = bundle.manifest()
        if self._last_manifest is not None and manifest.get(
            "bundle_sha256"
        ) == self._last_manifest.get("bundle_sha256"):
            return False, manifest
        # write the bundle so peers can ingest it
        out_dir = self.crowd_store or self.store
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "peer_bundle.json")
        try:
            bundle.save(out)
        except OSError:
            return False, manifest
        # merge into local federation if present
        fed = self._federation()
        if fed is not None:
            try:
                fed.ingest(bundle)
            except (ValueError, OSError):
                pass
        self._last_manifest = manifest
        self._save_loop_state(manifest)
        return True, manifest

    # -- public API ----------------------------------------------------------

    def step(self, end: datetime | None = None) -> AutonomousReport:
        """One pass over the fleet. Returns the pass report."""
        self._pass += 1
        end = end or datetime.now(timezone.utc)
        actions: list[AutonomousAction] = []
        slo_reports: dict[str, SloReport] = {}

        # 1. watch every station: assimilate + health (+ auto-refit on long records)
        for station in self._stations:
            rep = None
            try:
                rep = poll(
                    station,
                    self.store,
                    lookback_hours=self.lookback_hours,
                    alpha=self.alpha,
                    datum=self.datum,
                    fetcher=self.fetcher,
                    auto_refit=self.auto_refit,
                    end=end,
                )
            except (OSError, ValueError):
                pass
            mon = _load_monitor(self.store, station, None)
            health = mon.report()
            mon_slo = self._slo_for(station)
            slo_rep = mon_slo.evaluate(health)
            slo_reports[station] = slo_rep
            health_dict = {
                "n": health.n,
                "coverage": health.coverage,
                "rmse": health.rmse,
                "bias": health.bias,
                "needs_refit": health.needs_refit,
                "reasons": health.reasons,
            }
            cfg = load_slo(self.store, station)
            target = cfg.coverage_target
            # record the pass in the ledger even when nothing happened
            kind = "none"
            reason = ""
            if rep is not None and rep.refit_done:
                kind = "refit"
                reason = (
                    "; ".join(rep.health.reasons)
                    if rep.health.reasons
                    else "auto_refit"
                )
                actions.append(
                    AutonomousAction(
                        station=station,
                        at=_now_iso(),
                        kind=kind,
                        reason=reason,
                        cost=self.action_cost,
                        loss=self.breach_loss,
                        slo_before=None,
                        slo_after=slo_rep.status,
                        health_before=health_dict,
                        health_after=health_dict,
                    )
                )
            # reseed short records when the SLO is breached and we have a pool
            if slo_rep.status in ("degrading", "breached") and (
                rep is None or not rep.refit_done
            ):
                try:
                    model = TideModel.load_harmonic(
                        os.path.join(self.store, f"{station}.json")
                    )
                except (OSError, ValueError, KeyError):
                    model = None
                if model is not None:
                    n_obs = int(model.meta.get("n_obs", 0) or 0)
                    if n_obs < self.seed_short_min:
                        # try to build a short history from the engine
                        try:
                            eng = _seat_engine(model, self.store, station)
                            hist_t, hist_y = eng.history_data()
                        except (OSError, ValueError):
                            hist_t, hist_y = [], []
                        if len(hist_t) >= self.min_history:
                            heights_arr = np.asarray(hist_y, dtype=float)
                            pooled = self._reseed_station(station, hist_t, heights_arr)
                            if pooled is not None:
                                self._persist_model(station, pooled)
                                kind = "reseed"
                                reason = (
                                    f"short record reseeded via pool (n_obs={n_obs})"
                                )
                                actions.append(
                                    AutonomousAction(
                                        station=station,
                                        at=_now_iso(),
                                        kind=kind,
                                        reason=reason,
                                        cost=self.action_cost,
                                        loss=self.breach_loss,
                                        slo_before=slo_rep.status,
                                        slo_after=slo_rep.status,
                                        health_before=health_dict,
                                        health_after=health_dict,
                                    )
                                )
            # always log the pass into the ledger (the audit trail)
            self._log_action(
                station,
                kind,
                reason
                if reason
                else (f"SLO={slo_rep.status} (coverage={slo_rep.coverage:.3f})"),
                slo_metric=slo_rep.coverage,
                slo_target=target,
                acted=(kind != "none"),
            )

        # 2. re-sync the peer bundle only when constants actually changed
        synced, manifest = self._maybe_sync()
        if synced:
            actions.append(
                AutonomousAction(
                    station="*",
                    at=_now_iso(),
                    kind="sync",
                    reason="constants changed; bundle re-synced to peers",
                    cost=self.action_cost,
                    loss=self.breach_loss,
                    slo_before=None,
                    slo_after=None,
                )
            )

        report = AutonomousReport(
            pass_no=self._pass,
            end=end.isoformat(),
            stations=len(self._stations),
            actions=actions,
            slo=slo_reports,
            synced=synced,
            bundle_changed=bool(synced or manifest is not None),
            bundle_digest=manifest.get("bundle_sha256") if manifest else None,
        )
        return report

    def run(self) -> int:
        """Blocking standing loop. Returns 0 on clean exit, 130 on KeyboardInterrupt."""
        import time as _time

        passes = 0
        while True:
            try:
                self.step()
            except KeyboardInterrupt:
                return 130
            except (OSError, ValueError):
                pass
            passes += 1
            if self.max_passes is not None and passes >= self.max_passes:
                break
            try:
                _time.sleep(self.sleep_s)
            except KeyboardInterrupt:
                return 130
        return 0

    def slo_status(self) -> dict[str, SloReport]:
        """Return current SLO compliance for every watched station."""
        out: dict[str, SloReport] = {}
        for station in self._stations:
            try:
                poll(
                    station,
                    self.store,
                    lookback_hours=self.lookback_hours,
                    alpha=self.alpha,
                    datum=self.datum,
                    fetcher=self.fetcher,
                )
            except (OSError, ValueError):
                pass
            mon = _load_monitor(self.store, station, None)
            health = mon.report()
            mon_slo = self._slo_for(station)
            out[station] = mon_slo.evaluate(health)
        return out

    def slo_status_str(self) -> str:
        status = self.slo_status()
        lines = [f"# fleet SLO ({len(status)} station(s))"]
        met = degrading = breached = 0
        for s, r in sorted(status.items()):
            flag = {
                "met": "met",
                "degrading": "degrading",
                "breached": "BREACHED",
                "n/a": "n/a",
            }[r.status]
            cov = f"{r.coverage:.3f}" if r.n else "n/a"
            rmse = f"{r.rmse:.4f} m" if r.n else "n/a"
            lines.append(f"  {s:<16} coverage={cov} rmse={rmse} status={flag}")
            if r.status == "met":
                met += 1
            elif r.status == "degrading":
                degrading += 1
            elif r.status == "breached":
                breached += 1
        lines.append(f"  met={met}  degrading={degrading}  breached={breached}")
        return "\n".join(lines)
