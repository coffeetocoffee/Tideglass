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

import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from tideglass.marea.crowdsource import GaugeStore
from tideglass.marea.model import TideModel
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

    def __str__(self) -> str:
        lines = [
            f"federated: station={self.station} network={self.network_size}",
            f"  {self.qc}",
        ]
        if self.trust is not None:
            lines.append(f"  trust: {self.trust:.3f}")
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
                 use_trust: bool = True):
        self.store = GaugeStore(store_root)
        self.threshold = variance_threshold
        self.use_trust = use_trust
        self.ledger = TrustLedger(store_root)

    def coords(self) -> dict[str, tuple[float, float]]:
        return {g.station: (g.lon, g.lat) for g in self.store.stations()}

    def local_models(self, alpha: float = 0.05) -> dict[str, TideModel]:
        """Fit every well-sampled stored gauge (>= 48 obs) as a network member."""
        return self.store.models()

    def aggregate(self, alpha: float = 0.05) -> dict[str, TideModel]:
        """Recompute the partially-pooled model for every network member."""
        models = self.local_models(alpha)
        if len(models) < 2:
            return models
        coords = self.coords()
        trust = self.ledger.all() if self.use_trust else None
        pool = HierarchicalPool(models, coords, trust=trust)
        return {s: pool.pool(s, coords.get(s)) for s in models}

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
        if len(models) >= 2:
            coords = self.coords()
            try:
                trust_w = self.ledger.all() if self.use_trust else None
                pool = HierarchicalPool(models, coords, trust=trust_w)
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
        return FederatedReport(
            station, rep, self.store.count, before, after, shrinkage, improved,
            trust=trust)


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
