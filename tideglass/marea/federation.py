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
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from tideglass.marea.crowdsource import GaugeStore
from tideglass.marea.model import TideModel
from tideglass.marea.pooling import HierarchicalPool
from tideglass.marea.qc import QcConfig, QcReport, clean_series, qc_check


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

    def __str__(self) -> str:
        lines = [
            f"federated: station={self.station} network={self.network_size}",
            f"  {self.qc}",
        ]
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
                 variance_threshold: float = 0.95):
        self.store = GaugeStore(store_root)
        self.threshold = variance_threshold

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
        pool = HierarchicalPool(models, coords)
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
        before = _net_effect(self.store, self.threshold)
        if rep.rejected:
            return FederatedReport(station, rep, self.store.count, before, None)

        t, h = clean_series(times, heights, rep)
        self.store.add(station, lon, lat, t, h, source)

        models = self.local_models(alpha)
        shrinkage: dict = {}
        improved = False
        if len(models) >= 2:
            coords = self.coords()
            try:
                pool = HierarchicalPool(models, coords)
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
            station, rep, self.store.count, before, after, shrinkage, improved)


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
