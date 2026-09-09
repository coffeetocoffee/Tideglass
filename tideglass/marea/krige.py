"""Kriging / Gaussian-process spatial harmonics — v0.7 spatial moat.

:mod:`tideglass.marea.spatial` builds a *shared* EOF basis across stations and
then drops to inverse-distance weighting (IDW) to paint the loadings onto a
grid. IDW is interpolation, not modeling: it has no notion of uncertainty and
leaks station artefacts between neighbours. This module replaces that last step
with **ordinary kriging** (the GP-special-case of a Gaussian process with a
stationary covariance), so the regional field is itself a *random field* —
every grid point carries a predicted loading **and** a kriging variance.

For each EOF mode ``k`` we treat the per-station loadings ``w_k`` as scattered
samples of a smooth spatial process. Given great-circle distances between
stations we fit an exponential covariance ``sill·exp(−h/range) + nugget``
(``range`` set to a fraction of the network extent, ``nugget`` a small fraction
of the sill) and solve the ordinary-kriging system per grid point. The field is
then rebuilt as ``Σ_k s_k·ŵ_k(grid) ⊗ mode_k`` and its variance as
``Σ_k (s_k)²·σ²_k(grid)·mode_k²`` — a genuine *gridded regional field with
uncertainty*, not a per-station series.

Dependency-free (numpy only). Falls back to IDW when fewer than three stations
are present, so it never breaks the single/dual-station cases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _haversine_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance (km) between coordinate arrays (broadcastable)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _exp_cov(dist_km, range_km, sill, nugget) -> np.ndarray:
    """Exponential covariance matrix / vector from a distance array.

    The nugget (a multiple of the identity) is added only when requested —
    prediction vectors against grid points must not receive it.
    """
    c = sill * np.exp(-np.asarray(dist_km, dtype=float) / max(range_km, 1e-9))
    if c.ndim == 2 and nugget > 0.0:
        c = c + nugget * np.eye(c.shape[0])
    return c


@dataclass
class KrigeField:
    """One kriged spatial field (e.g. a single EOF mode's loadings)."""

    grid_lons: np.ndarray
    grid_lats: np.ndarray
    mean: np.ndarray  # (n_grid,) predicted loading at each grid point
    var: np.ndarray  # (n_grid,) kriging variance at each grid point
    range_km: float
    sill: float
    nugget: float


def krige_field(
    station_lons: np.ndarray,
    station_lats: np.ndarray,
    station_values: np.ndarray,
    grid_lons: np.ndarray,
    grid_lats: np.ndarray,
    range_km: float | None = None,
    nugget_frac: float = 0.05,
) -> KrigeField:
    """Ordinary kriging of scattered station values onto a grid.

    :param station_lons/lats: ``(n_stations,)`` control-point coordinates.
    :param station_values: ``(n_stations,)`` observations (one EOF-mode loading).
    :param grid_lons/lats: ``(n_grid,)`` target coordinates.
    :param range_km: covariance decay length; defaults to ``0.6×`` the max
        pairwise station distance (a robust fraction of the network extent).
    :param nugget_frac: nugget as a fraction of the sample variance.
    """
    slon = np.asarray(station_lons, dtype=float).ravel()
    slat = np.asarray(station_lats, dtype=float).ravel()
    y = np.asarray(station_values, dtype=float).ravel()
    glon = np.asarray(grid_lons, dtype=float).ravel()
    glat = np.asarray(grid_lats, dtype=float).ravel()
    n = slon.size

    if n < 3:
        # Not enough control points for a stable GP; defer to IDW (see spatial).
        return _idw_field(slon, slat, y, glon, glat)

    D = _haversine_km(slon[:, None], slat[:, None], slon[None, :], slat[None, :])
    if range_km is None:
        off_diag = D[~np.eye(n, dtype=bool)]  # exclude the zero self-distances
        range_km = 0.6 * float(off_diag.max())
    sill = float(np.var(y))
    if sill <= 0.0:
        sill = 1e-6
    nugget = nugget_frac * sill

    # Ordinary-kriging matrix: [K 1; 1ᵀ 0].
    K = _exp_cov(D, range_km, sill, nugget)
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = K
    A[:n, n] = 1.0
    A[n, :n] = 1.0
    Ainv = np.linalg.inv(A)

    gdist = _haversine_km(
        glon[None, :], glat[None, :], slon[:, None], slat[:, None])  # (n × n_grid)
    kvec = _exp_cov(gdist, range_km, sill, 0.0)  # (n × n_grid), no nugget
    rhs = np.vstack([kvec, np.ones((1, gdist.shape[1]))])  # (n+1 × n_grid)
    lam = Ainv @ rhs  # (n+1 × n_grid); last row is the Lagrange multiplier μ
    weights = lam[:n, :]
    mu = lam[n, :]
    pred = weights.T @ y
    # Kriging variance = C(0) − λᵀ k + μ ; C(0) = sill + nugget.
    krig_var = (sill + nugget) - np.einsum("ij,ji->i", weights.T, kvec) + mu
    return KrigeField(
        grid_lons=glon, grid_lats=glat,
        mean=np.asarray(pred, dtype=float),
        var=np.maximum(np.asarray(krig_var, dtype=float), 0.0),
        range_km=range_km, sill=sill, nugget=nugget,
    )


def _idw_field(slon, slat, y, glon, glat, power=2.0, eps=1e-6):
    """IDW fallback used when kriging is under-determined."""
    glon = np.asarray(glon, dtype=float).ravel()
    glat = np.asarray(glat, dtype=float).ravel()
    n_grid = glon.size
    pred = np.empty(n_grid)
    var = np.zeros(n_grid)
    for g in range(n_grid):
        d = _haversine_km(np.full_like(slon, glon[g]), np.full_like(slat, glat[g]),
                          slon, slat)
        w = 1.0 / (d + eps) ** power
        wsum = w.sum()
        if wsum <= 0:
            w = np.ones_like(w) / w.size
        else:
            w = w / wsum
        pred[g] = float(w @ y)
    return KrigeField(
        grid_lons=glon, grid_lats=glat, mean=pred, var=var,
        range_km=float("nan"), sill=float(np.var(y)), nugget=0.0,
    )


def _pairwise_max_km(lons, lats) -> float:
    n = len(lons)
    if n < 2:
        return 1.0
    d = _haversine_km(np.asarray(lons)[:, None], np.asarray(lats)[:, None],
                      np.asarray(lons)[None, :], np.asarray(lats)[None, :])
    return float(np.nanmax(d))


def krige_regional(
    eof_modes: np.ndarray,        # (n_modes × n_time)
    station_loadings: np.ndarray,  # (n_stations × n_modes)
    station_lons: np.ndarray,
    station_lats: np.ndarray,
    station_means: np.ndarray,     # (n_stations,)
    grid_lons: np.ndarray,
    grid_lats: np.ndarray,
    scale: np.ndarray | None = None,  # (n_modes,) SVD singular values (optional)
    range_km: float | None = None,
    nugget_frac: float = 0.05,
) -> dict:
    """Krige every EOF-mode loading onto a grid and rebuild the field + variance.

    Returns ``{"field": (n_grid × n_time), "field_var": (n_grid × n_time),
    "mean_field": (n_grid,), "mean_var": (n_grid,), "per_mode": [KrigeField...]}``.
    """
    modes = np.asarray(eof_modes, dtype=float)
    load = np.asarray(station_loadings, dtype=float)
    n_modes = modes.shape[0]
    n_time = modes.shape[1]
    if scale is None:
        scale = np.ones(n_modes)
    glon = np.asarray(grid_lons, dtype=float).ravel()
    glat = np.asarray(grid_lats, dtype=float).ravel()
    n_grid = glon.size

    field = np.zeros((n_grid, n_time))
    field_var = np.zeros((n_grid, n_time))
    mean_field, mean_var = np.zeros(n_grid), np.zeros(n_grid)
    per_mode: list[KrigeField] = []

    # Mean level is itself a spatial field (a degenerate mode with constant 1).
    mkr = krige_field(station_lons, station_lats, station_means,
                      glon, glat, range_km=range_km, nugget_frac=nugget_frac)
    mean_field = mkr.mean
    mean_var = mkr.var
    per_mode.append(mkr)

    for k in range(n_modes):
        kr = krige_field(station_lons, station_lats, load[:, k],
                         glon, glat, range_km=range_km, nugget_frac=nugget_frac)
        per_mode.append(kr)
        contrib = (scale[k] * kr.mean)[:, None] * modes[k][None, :]
        field = field + contrib
        var_contrib = (scale[k] ** 2 * kr.var)[:, None] * (modes[k][None, :] ** 2)
        field_var = field_var + var_contrib
    field = field + mean_field[:, None]
    field_var = field_var + mean_var[:, None]
    return {
        "field": field,
        "field_var": field_var,
        "mean_field": mean_field,
        "mean_var": mean_var,
        "per_mode": per_mode,
    }
