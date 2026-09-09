"""Benchmark Marea Core against global hydrodynamic models (TPXO / FES) — v0.7.

pytides is a soft target; the real heavyweights are the global *data-assimilating*
tide models — TPXO (OSU) and FES (LEGOS/CNES) — which publish harmonic constants
on a regular lat/lon grid. Those constants are exactly the schema Marea Core
already consumes (:meth:`~tideglass.marea.model.TideModel.load_harmonic`): a
``(lon, lat, constituent, amplitude, phase)`` table per grid point.

This module loads such a grid and evaluates Marea Core against it on held-out
gauge observations, using the same ``metrics`` the pytides bench uses — so the
comparison is apples-to-apples. The bundled ``data/grids/tpxo_sample.csv`` is a
small, clearly-labelled *representative* stand-in (derived from NOAA published
constants for the San Francisco region with a coarse grid and a few constituents
dropped, emulating a low-resolution global product); drop a real TPXO/FES
extraction in the same schema and the same command benchmarks the genuine model.

Only harmonic math is touched; no network calls.
"""

from __future__ import annotations

import csv as _csv
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import numpy as np

from tideglass.marea.model import TideModel
from tideglass.marea import constituents as CON
from tideglass.marea.metrics import evaluate

_GRID_KEY = tuple[float, float]


def read_harmonic_grid(path: str) -> dict[_GRID_KEY, dict[str, tuple[float, float]]]:
    """Read a TPXO/FES-style harmonic-constants grid.

    Expected CSV columns: ``lon,lat,constituent,amplitude,phase`` (phase in
    degrees, NOAA kappa convention). Returns ``{(lon, lat): {c: (amp, phase)}}``.
    """
    grid: dict[_GRID_KEY, dict[str, tuple[float, float]]] = {}
    with open(path, newline="") as fh:
        for row in _csv.DictReader(fh):
            key = (round(float(row["lon"]), 6), round(float(row["lat"]), 6))
            c = row["constituent"].strip()
            grid.setdefault(key, {})[c] = (float(row["amplitude"]), float(row["phase"]))
    if not grid:
        raise ValueError(f"no harmonic rows parsed from {path!r}")
    return grid


def _haversine_km(lon1, lat1, lon2, lat2) -> float:
    import math

    lon1, lat1, lon2, lat2 = map(math.radians, (lon1, lat1, lon2, lat2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2.0 * math.asin(min(1.0, math.sqrt(a)))


def global_model_at(
    grid: dict[_GRID_KEY, dict[str, tuple[float, float]]],
    lon: float,
    lat: float,
    method: str = "nearest",
) -> TideModel:
    """Build a :class:`TideModel` for ``(lon, lat)`` from the global grid.

    :param method: ``"nearest"`` (default) uses the closest grid point; ``"idw"``
        blends the nearest few points by inverse-distance. Loaded models carry no
        covariance, so prediction bands collapse to the mean curve.
    """
    pts = list(grid)
    if method == "idw":
        d = np.array([_haversine_km(lon, lat, pl, pa) for (pl, pa) in pts])
        order = np.argsort(d)[: min(4, len(pts))]
        w = 1.0 / (d[order] + 1e-3) ** 2
        w = w / w.sum()
        merged: dict[str, list[tuple[float, float]]] = {}
        for o, wt in zip(order, w):
            for c, (a, p) in grid[pts[int(o)]].items():
                merged.setdefault(c, []).append((a * wt, p))
        consts: dict[str, tuple[float, float]] = {}
        for c, vals in merged.items():
            amp = float(sum(v[0] for v in vals))
            # Circular mean of the phases (weighted).
            r = sum(v[0] * np.cos(np.radians(v[1])) for v in vals)
            im = sum(v[0] * np.sin(np.radians(v[1])) for v in vals)
            consts[c] = (amp, float(np.degrees(np.arctan2(im, r)) % 360.0))
        return _model_from_consts(consts, station="global")
    # nearest
    best = min(pts, key=lambda p: _haversine_km(lon, lat, p[0], p[1]))
    return _model_from_consts(grid[best], station="global")


def _model_from_consts(
    consts: dict[str, tuple[float, float]], station: str = "global"
) -> TideModel:
    entries = [{"name": n, "amplitude": a, "phase": p} for n, (a, p) in consts.items()]
    return TideModel.load_harmonic(
        {"station": station, "mean": 0.0, "constituents": entries},
        station=station,
    )


def compare(
    global_model: TideModel,
    marea_model: TideModel,
    times: Sequence[datetime],
    heights,
) -> dict[str, Any]:
    """Score both models on held-out gauge observations (apples-to-apples)."""
    times = list(times)
    y = np.asarray(heights, dtype=float).ravel()
    g = evaluate(global_model.predict(times), y)
    m = evaluate(marea_model.predict(times), y)
    winner = "marea" if m["rmse"] <= g["rmse"] else "global"
    return {
        "global": {k: g[k] for k in ("rmse", "mae", "bias", "peak_error", "coverage")},
        "marea": {k: m[k] for k in ("rmse", "mae", "bias", "peak_error", "coverage")},
        "winner_rmse": winner,
        "rmse_ratio_marea_over_global": float(m["rmse"] / max(g["rmse"], 1e-12)),
    }
