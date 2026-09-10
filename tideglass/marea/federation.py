"""Federated refits for the crowd-sourced network (v1.0).

Each cheap sensor upload is QC-checked, fitted locally, and then a single round
of partial pooling (see :mod:`tideglass.marea.pooling`) folds every station's
harmonic constants into everyone's model. Adding a sensor therefore improves the
whole network — the v0.5 network effect, now closed into a feedback loop — while
the new sensor itself borrows strength from its neighbours via
:meth:`HierarchicalPool.seed_short`.

The loop is deliberately dependency-free and local, like the rest of the
crowd layer: a :class:`FederatedRefit` wraps a :class:`GaugeStore` and reuses
its EOF network-effect metric so the per-round gain is directly comparable to
the v0.5 numbers.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from tideglass.marea import constituents as CON
from tideglass.marea.crowdsource import GaugeStore
from tideglass.marea.model import Fit, TideModel
from tideglass.marea.pooling import HierarchicalPool
from tideglass.marea.qc import QcConfig, QcReport, clean_series, qc_check


def trust_score(reject_rate: float, datum_m: float,
                drift_m_day: float, config: QcConfig | None = None) -> float:
    """Per-sensor reputation in ``(0, 1]`` from its QC history (v1.1).

    Combines the three QC signals multiplicatively — ``acceptance · datum ·
    drift`` — where each physics term decays as ``exp(-ln2 · (x/tol)²)``:
    1.0 at zero signal, 0.5 exactly at its QC tolerance, near 0 at twice the
    tolerance. A sensor with a clean history scores ≈ 1; one that constantly
    trips checks scores near 0 (and pools as pure prior).
    """
    cfg = config or QcConfig()
    accept = 1.0 - float(np.clip(reject_rate, 0.0, 1.0))
    if accept <= 0.0:
        return 1e-3

    def decay(x: float, tol: float) -> float:
        return math.exp(-math.log(2.0) * (abs(x) / tol) ** 2)

    return float(max(
        accept * decay(datum_m, cfg.datum_tol_m)
        * decay(drift_m_day, cfg.drift_tol_m_per_day), 1e-3))


class TrustLedger:
    """Per-sensor QC history → reputation (v1.1 the network's immune system).

    Every QC pass on a station is recorded; the reputation is recomputed from
    the accumulated rates via :func:`trust_score` and stored as
    ``<root>/<station>.trust.json``. :meth:`FederatedRefit.refit` feeds the
    ledger's weights into :class:`HierarchicalPool` so unreliable sensors
    borrow strength from their neighbours instead of contaminating them.
    """

    def __init__(self, root: str):
        self.root = root

    def _path(self, station: str) -> str:
        return os.path.join(self.root, f"{station}.trust.json")

    def record(self, station: str, report: QcReport,
               config: QcConfig | None = None) -> float:
        """Fold one QC report into the history and return the new trust."""
        d = {"station": station, "n_total": 0, "n_rejected": 0,
             "datum_mean": 0.0, "drift_mean": 0.0}
        if os.path.exists(self._path(station)):
            with open(self._path(station)) as fh:
                d.update(json.load(fh))
        n0 = int(d["n_total"])
        d["n_total"] = n0 + 1
        d["n_rejected"] = int(d["n_rejected"]) + (1 if report.rejected else 0)
        if report.datum_shift_m is not None:
            dm = abs(float(report.datum_shift_m))
            d["datum_mean"] = (float(d["datum_mean"]) * n0 + dm) / (n0 + 1)
        if report.drift_slope_m_per_day is not None:
            dr = abs(float(report.drift_slope_m_per_day))
            d["drift_mean"] = (float(d["drift_mean"]) * n0 + dr) / (n0 + 1)
        d["trust"] = trust_score(
            d["n_rejected"] / d["n_total"],
            float(d["datum_mean"]), float(d["drift_mean"]), config)
        os.makedirs(self.root, exist_ok=True)
        with open(self._path(station), "w") as fh:
            json.dump(d, fh, indent=2)
        return float(d["trust"])

    def get(self, station: str, default: float = 1.0) -> float:
        if not os.path.exists(self._path(station)):
            return default
        with open(self._path(station)) as fh:
            return float(json.load(fh).get("trust", default))

    def all(self) -> dict[str, float]:
        out = {}
        if not os.path.isdir(self.root):
            return out
        for fn in sorted(os.listdir(self.root)):
            if fn.endswith(".trust.json"):
                with open(os.path.join(self.root, fn)) as fh:
                    d = json.load(fh)
                out[d["station"]] = float(d["trust"])
        return out


@dataclass
class FederatedReport:
    """Outcome of one federated refit round for a single uploaded sensor."""

    station: str
    qc: QcReport
    network_size: int
    network_before: dict | None
    network_after: dict | None
    shrinkage: dict = field(default_factory=dict)
    improved: bool = False
    trust: float | None = None  # per-sensor reputation after this round (v1.1)
    global_network_size: int | None = None  # installed-base stations (v2.0)

    def __str__(self) -> str:
        lines = [
            f"federated: station={self.station} network={self.network_size}",
            f"  {self.qc}",
        ]
        if self.trust is not None:
            lines.append(f"  trust: {self.trust:.3f}")
        if self.global_network_size is not None:
            lines.append(f"  global federation: {self.global_network_size} "
                         "stations across the installed base")
        if self.network_before and self.network_after:
            before = self.network_before["gain_total_explained"]
            after = self.network_after["gain_total_explained"]
            lines.append(
                f"  network-effect gain: {after - before:+.4f} explained variance")
        if self.shrinkage:
            mean_shr = sum(self.shrinkage.values()) / len(self.shrinkage)
            lines.append(f"  mean_shrinkage: {mean_shr:.4f} "
                         "(0 = own data, 1 = pure prior)")
        return "\n".join(lines)


def _net_effect(store: GaugeStore, threshold: float) -> dict | None:
    if store.count < 2:
        return None
    try:
        return store.network_effect(variance_threshold=threshold)
    except ValueError:
        return None


def _save_model(root: str, station: str, model: TideModel) -> None:
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, f"{station}.json"), "w") as fh:
        json.dump(model.to_artifact(), fh, indent=2)


class FederatedRefit:
    """QC + local fit + network pooling for one crowd-sourced sensor at a time."""

    def __init__(self, store_root: str = ".tideglass/crowd",
                 variance_threshold: float = 0.95,
                 use_trust: bool = True,
                 global_root: str | None = None):
        self.store = GaugeStore(store_root)
        self.threshold = variance_threshold
        self.use_trust = use_trust
        self.ledger = TrustLedger(store_root)
        self.global_fed = (
            GlobalFederation(global_root) if global_root else None)

    def coords(self) -> dict[str, tuple[float, float]]:
        return {g.station: (g.lon, g.lat) for g in self.store.stations()}

    def local_models(self, alpha: float = 0.05) -> dict[str, TideModel]:
        """Fit every well-sampled stored gauge (>= 48 obs) as a network member."""
        return self.store.models()

    def aggregate(self, alpha: float = 0.05) -> dict[str, TideModel]:
        """Recompute the partially-pooled model for every network member.

        When a global federation is configured, the pooled prior is computed over
        the union of the local network and the installed base, so even local-only
        members borrow strength from every peer deployment (v2.0).
        """
        models = self.local_models(alpha)
        pool_models = dict(self._global_models())
        pool_models.update(models)
        if len(pool_models) < 2:
            return models
        coords = self.coords()
        trust = self.ledger.all() if self.use_trust else None
        pool = HierarchicalPool(pool_models, coords, trust=trust)
        return {s: pool.pool(s, coords.get(s)) for s in models}

    def _global_models(self) -> dict[str, TideModel]:
        if self.global_fed is None:
            return {}
        try:
            return self.global_fed.models()
        except Exception:
            return {}

    def refit(
        self,
        station: str,
        lon: float,
        lat: float,
        times: Sequence,
        heights: Sequence[float],
        source: str = "crowd",
        config: QcConfig | None = None,
        alpha: float = 0.05,
    ) -> FederatedReport:
        """Ingest one sensor, QC it, and fold it into the network.

        The upload is QC-checked (rejected outright if the report is rejected),
        added to the store, and — once the network has at least two members —
        a new pooled model is produced for the station (``seed_short`` for short
        records, ``pool`` for dense ones) and persisted as ``<store>/<station>.json``.
        The network-effect gain before/after the round is reported so the
        "every sensor helps everyone" property is measurable.
        """
        rep = qc_check(times, heights, config)
        trust = self.ledger.record(station, rep) if self.use_trust else None
        before = _net_effect(self.store, self.threshold)
        if rep.rejected:
            return FederatedReport(station, rep, self.store.count, before, None,
                                   trust=trust)

        t, h = clean_series(times, heights, rep)
        self.store.add(station, lon, lat, t, h, source)

        models = self.local_models(alpha)
        shrinkage: dict = {}
        improved = False
        g = self._global_models()
        if len(models) >= 2 or (g and len(models) + len(g) >= 2):
            coords = self.coords()
            try:
                trust_w = self.ledger.all() if self.use_trust else None
                pool_models = dict(g)
                pool_models.update(models)
                pool = HierarchicalPool(pool_models, coords, trust=trust_w)
                if station in models:
                    model = pool.pool(station, coords.get(station))
                else:
                    model = pool.seed_short(t, h, station, coords.get(station))
                shrinkage = dict(model.meta.get("shrinkage", {}))
                improved = True
                _save_model(self.store.root, station, model)
            except (ValueError, KeyError):
                improved = False

        after = _net_effect(self.store, self.threshold)
        gsize = self.global_fed.n_stations if self.global_fed is not None else None
        return FederatedReport(
            station, rep, self.store.count, before, after, shrinkage, improved,
            trust=trust, global_network_size=gsize)


# ---------------------------------------------------------------------------
# v2.0 — the federation protocol
# ---------------------------------------------------------------------------
# The v1.0 loop is *local*: every sensor improves one store's network. v2.0 turns
# that into a protocol between independent Tideglass *installations* (peers). Each
# peer exports an anonymized :class:`PeerBundle` — its stations' harmonic
# constants plus residual statistics, with station identities replaced by
# one-way aliases — and merges its peers' bundles into a :class:`GlobalFederation`.
# Every peer's stations then pool against the *entire installed base*, so the
# network effect compounds across deployments, not just stations.

_PEER_SCHEMA = "tideglass.federated.peer/v1"


def _pkg_version() -> str:
    """Best-effort Tideglass version without importing the top-level package."""
    try:
        from importlib.metadata import version

        return version("tideglass")
    except Exception:
        return "1.0.0"


def _alias(peer_id: str, name: str) -> str:
    """One-way anonymized alias for a station within a peer.

    The original station id never leaves the generating installation: only the
    12-hex alias is exported in the bundle.
    """
    return hashlib.sha256(f"{peer_id}|{name}".encode()).hexdigest()[:12]


@dataclass
class StationContribution:
    """One station's anonymized export: harmonic constants + residual stats."""

    alias: str
    constituents: list[str]
    coef: list[float]                       # [H0, a1, b1, ...] (datum-dependent H0)
    covariance: list[list[float]]           # (a, b) covariance in the same layout
    sigma2: float
    n_obs: int
    residual_stats: dict                    # e.g. {"rmse":..., "n_obs":...}
    residual_bias: dict | None = None       # learned DoY bias table, if attached


@dataclass
class PeerBundle:
    """An installation's anonymized contribution to the federation protocol.

    Stations are keyed by their one-way :func:`_alias`; the mapping back to real
    station names is deliberately *not* serialized, so a shared bundle carries no
    station identities, coordinates, or raw observations — only the learned
    harmonic constants and residual statistics.
    """

    peer_id: str
    generated_at: str
    tideglass_version: str
    stations: dict[str, StationContribution]
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema": _PEER_SCHEMA,
            "peer_id": self.peer_id,
            "generated_at": self.generated_at,
            "tideglass_version": self.tideglass_version,
            "stations": {
                a: {
                    "alias": c.alias,
                    "constituents": c.constituents,
                    "coef": c.coef,
                    "covariance": c.covariance,
                    "sigma2": c.sigma2,
                    "n_obs": c.n_obs,
                    "residual_stats": c.residual_stats,
                    "residual_bias": c.residual_bias,
                }
                for a, c in self.stations.items()
            },
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PeerBundle":
        stations: dict[str, StationContribution] = {}
        for a, c in d.get("stations", {}).items():
            stations[a] = StationContribution(
                alias=c["alias"],
                constituents=list(c["constituents"]),
                coef=list(c["coef"]),
                covariance=[list(r) for r in c["covariance"]],
                sigma2=float(c["sigma2"]),
                n_obs=int(c["n_obs"]),
                residual_stats=dict(c.get("residual_stats", {})),
                residual_bias=c.get("residual_bias"),
            )
        return cls(
            peer_id=d["peer_id"],
            generated_at=d["generated_at"],
            tideglass_version=d.get("tideglass_version", "unknown"),
            stations=stations,
            meta=dict(d.get("meta", {})),
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "PeerBundle":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


def _contribution_from_model(model: TideModel, alias: str) -> StationContribution:
    names = [f.name for f in model.constituents()]
    resid = {}
    if model.meta.get("residual_stats"):
        resid.update(model.meta["residual_stats"])
    if model.meta.get("rmse") is not None:
        resid.setdefault("rmse", float(model.meta["rmse"]))
    if model.meta.get("n_obs") is not None:
        resid.setdefault("n_obs", int(model.meta["n_obs"]))
    return StationContribution(
        alias=alias,
        constituents=names,
        coef=model._coef.tolist(),
        covariance=model._covariance.tolist(),
        sigma2=float(model._sigma2),
        n_obs=int(model.meta.get("n_obs", 0) or 0),
        residual_stats={
            k: (None if v is None else float(v))
            for k, v in resid.items() if v is not None
        },
        residual_bias=model.meta.get("residual_bias"),
    )


def _model_from_contribution(c: StationContribution, station: str) -> TideModel:
    consts = [CON.get(n) for n in c.constituents]
    coef = np.asarray(c.coef, dtype=float)
    cov = np.asarray(c.covariance, dtype=float)
    fits: list[Fit] = []
    for j, n in enumerate(c.constituents):
        a = float(coef[1 + 2 * j])
        b = float(coef[2 + 2 * j])
        amp = math.hypot(a, b)
        phase = math.degrees(math.atan2(b, a)) % 360.0
        Caa = float(cov[1 + 2 * j, 1 + 2 * j])
        Cbb = float(cov[2 + 2 * j, 2 + 2 * j])
        var = (a * a * Caa + b * b * Cbb) / amp**2 if amp > 0 else 0.5 * (Caa + Cbb)
        fits.append(Fit(n, amp, phase, math.sqrt(max(var, 0.0))))
    meta = {"source": "federated", "n_obs": c.n_obs,
            "residual_stats": c.residual_stats}
    if c.residual_bias:
        meta["residual_bias"] = c.residual_bias
    return TideModel(consts, coef, cov, float(c.sigma2), fits,
                     station=station, source="federated", meta=meta)


def build_peer_bundle(store_root: str, peer_id: str,
                      generated_at: str | None = None) -> PeerBundle:
    """Aggregate a local crowd store into an anonymized :class:`PeerBundle`.

    Only the well-sampled network members (>= 48 obs) are exported, and each is
    replaced by its one-way alias so the bundle leaks no station identity.
    """
    gs = GaugeStore(store_root)
    models = gs.models()
    stations: dict[str, StationContribution] = {}
    for name, m in models.items():
        alias = _alias(peer_id, name)
        stations[alias] = _contribution_from_model(m, alias)
    if generated_at is None:
        generated_at = datetime.now(timezone.utc).isoformat()
    return PeerBundle(
        peer_id=peer_id,
        generated_at=generated_at,
        tideglass_version=_pkg_version(),
        stations=stations,
        meta={"store": os.path.abspath(store_root)},
    )


class GlobalFederation:
    """A peer's merged view of the installed base (v2.0).

    Each ingested :class:`PeerBundle` is persisted under ``<root>/peers`` and the
    union of all peers' anonymized stations forms one :class:`HierarchicalPool`.
    A new local station therefore seeds against the *global* regional prior —
    borrowing strength from every installation, not just the local network — and
    the network effect compounds across deployments.
    """

    def __init__(self, root: str):
        self.root = root
        self.peers_dir = os.path.join(root, "peers")
        os.makedirs(self.peers_dir, exist_ok=True)
        self._bundles: dict[str, PeerBundle] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.isdir(self.peers_dir):
            return
        for fn in sorted(os.listdir(self.peers_dir)):
            if not fn.endswith(".json"):
                continue
            try:
                b = PeerBundle.load(os.path.join(self.peers_dir, fn))
            except (OSError, ValueError, KeyError):
                continue
            self._bundles[b.peer_id] = b

    def ingest(self, bundle: PeerBundle) -> None:
        """Fold one peer's bundle into the federation and persist it."""
        if not bundle.stations:
            raise ValueError("cannot ingest an empty peer bundle")
        self._bundles[bundle.peer_id] = bundle
        os.makedirs(self.peers_dir, exist_ok=True)
        bundle.save(os.path.join(self.peers_dir, f"{bundle.peer_id}.json"))

    @property
    def n_peers(self) -> int:
        return len(self._bundles)

    @property
    def n_stations(self) -> int:
        return sum(len(b.stations) for b in self._bundles.values())

    def peers(self) -> list[str]:
        return sorted(self._bundles)

    def models(self) -> dict[str, TideModel]:
        """Every anonymized station from every peer, as plain TideModels."""
        out: dict[str, TideModel] = {}
        for b in self._bundles.values():
            for alias, c in b.stations.items():
                key = f"{b.peer_id}:{alias}"
                out[key] = _model_from_contribution(c, key)
        return out

    def global_pool(self, coords=None, trust=None) -> HierarchicalPool:
        models = self.models()
        if len(models) < 2:
            raise ValueError(
                f"need >= 2 federated stations to pool, have {len(models)}")
        return HierarchicalPool(models, coords=coords, trust=trust)

    def seed(self, station: str, times: Sequence, heights: Sequence[float],
             coords: tuple[float, float] | None = None) -> TideModel:
        """Seed a (typically short) new station against the global prior."""
        models = self.models()
        if len(models) < 2:
            # No federated neighbours yet: fall back to a standalone local fit.
            return TideModel.fit(list(times), heights, station=station,
                                 source="local")
        pool = self.global_pool(coords={station: coords} if coords else None)
        return pool.seed_short(list(times), heights, station, coords)

    def residual_summary(self) -> dict | None:
        """Aggregate of residual statistics across the installed base."""
        rmses = []
        for b in self._bundles.values():
            for c in b.stations.values():
                rs = c.residual_stats or {}
                if rs.get("rmse") is not None:
                    rmses.append(float(rs["rmse"]))
        if not rmses:
            return None
        return {
            "n": len(rmses),
            "mean_rmse": float(np.mean(rmses)),
            "max_rmse": float(np.max(rmses)),
        }

    def report(self) -> dict:
        """Federation status: peer/station counts + per-constituent prior spread.

        ``tau`` is the between-station standard deviation of the global regional
        prior. As more deployments join with consistent constants, ``tau``
        tightens and a new station's seeded model leans harder on the prior.
        """
        out: dict = {
            "n_peers": self.n_peers,
            "n_stations": self.n_stations,
            "constituents": {},
            "residual": self.residual_summary(),
        }
        models = self.models()
        if len(models) >= 2:
            pool = self.global_pool()
            for c in pool.union:
                try:
                    pr = pool._prior(c.name, "", None)
                except (KeyError, ValueError):
                    continue
                out["constituents"][c.name] = {
                    "tau_a": pr[0][1], "tau_b": pr[1][1], "n": pr[0][2]
                }
        return out


def federate(
    store_root: str,
    station: str,
    lon: float,
    lat: float,
    times: Sequence,
    heights: Sequence[float],
    source: str = "crowd",
    config: QcConfig | None = None,
    alpha: float = 0.05,
    variance_threshold: float = 0.95,
) -> FederatedReport:
    """One-shot federated refit (convenience wrapper around :class:`FederatedRefit`)."""
    fed = FederatedRefit(store_root, variance_threshold=variance_threshold)
    return fed.refit(station, lon, lat, times, heights,
                     source=source, config=config, alpha=alpha)
