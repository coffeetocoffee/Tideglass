"""Multi-station harmonization for Marea Core — the "wow".

Single-station analysis cannot borrow strength from neighbours. Given
stations on a common time grid, we fit the region jointly in a reduced
Empirical Orthogonal Function (EOF) basis: each station's series is
projected onto the leading shared modes, so a sparse or noisy station is
denoised by the regional field::

    stations ──▶ shared EOF basis ──▶ consistent regional field

Modes come from the SVD of the (per-station-centered) data matrix; the
truncation keeps enough modes to explain ``variance_threshold`` of the
total variance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tideglass.marea.krige import krige_regional


@dataclass(frozen=True)
class EOFResult:
    stations: list[str]
    modes: np.ndarray  # (n_modes × n_time) orthonormal temporal modes
    loadings: np.ndarray  # (n_stations × n_modes) spatial weights per mode
    explained: np.ndarray  # variance fraction per kept mode
    total_explained: float
    reconstructed: dict[str, np.ndarray]
    means: dict[str, float]


def harmonize(
    series: dict[str, object],
    n_modes: int | None = None,
    variance_threshold: float = 0.95,
) -> EOFResult:
    """Joint EOF reconstruction of station series on a common grid."""
    names = list(series)
    if len(names) < 2:
        raise ValueError("need at least 2 stations to harmonize")
    data = {k: np.asarray(series[k], dtype=float).ravel() for k in names}
    lengths = {v.size for v in data.values()}
    if len(lengths) != 1:
        raise ValueError(f"stations must share a time grid, got lengths {lengths}")
    n_time = lengths.pop()
    if n_time < 2:
        raise ValueError("need at least 2 time samples")

    means = {k: float(v.mean()) for k, v in data.items()}
    X = np.stack([data[k] - means[k] for k in names])  # stations × time
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    energy = s**2
    total = float(energy.sum())
    if total == 0.0:
        raise ValueError("all series are constant")
    frac = energy / total
    if n_modes is None:
        n_modes = int(np.searchsorted(np.cumsum(frac), variance_threshold) + 1)
    n_modes = max(1, min(int(n_modes), len(names), n_time))
    Uk = (X @ Vt[:n_modes].T) / s[:n_modes]  # spatial loadings (stations × modes)
    recon = (Uk * s[:n_modes]) @ Vt[:n_modes]  # low-rank field
    return EOFResult(
        stations=names,
        modes=Vt[:n_modes],
        loadings=Uk,
        explained=frac[:n_modes],
        total_explained=float(frac[:n_modes].sum()),
        reconstructed={k: recon[i] + means[k] for i, k in enumerate(names)},
        means=means,
    )


def _haversine(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance (km) between coordinate arrays (vectorized)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(a))


@dataclass(frozen=True)
class RegionalField:
    grid_lons: np.ndarray  # (n_grid,) target longitudes
    grid_lats: np.ndarray  # (n_grid,) target latitudes
    field: np.ndarray  # (n_grid × n_time) continuous reconstructed field
    field_var: np.ndarray  # (n_grid × n_time) field variance (kriging/GP)
    modes: np.ndarray  # (n_modes × n_time) temporal EOFs
    explained: np.ndarray  # variance fraction per mode
    total_explained: float
    method: str = "krige"  # "krige" (default, v0.7) or "idw"


def regional_field(
    eof: EOFResult,
    stations_coords: dict[str, tuple[float, float]],
    grid_lons: np.ndarray,
    grid_lats: np.ndarray,
    power: float = 2.0,
    eps_km: float = 1e-6,
    method: str = "krige",
    range_km: float | None = None,
    nugget_frac: float = 0.05,
) -> RegionalField:
    """Interpolate the EOF loadings onto a continuous spatial grid.

    ``harmonize`` produces *per-station* loadings ``Uk`` (each station's weight
    on every shared mode). Here we treat those loadings as scattered samples of a
    continuous spatial field and interpolate them across ``(grid_lons,
    grid_lats)`` to rebuild the field as ``Σ_k s_k · w_k(grid) ⊗ mode_k`` — a
    genuine *gridded* regional field rather than a per-station series.

    :param stations_coords: ``{name: (lon, lat)}`` for the stations in ``eof``.
    :param grid_lons, grid_lats: 1-D target coordinate arrays (a regular mesh is
        typical, but any points are fine).
    :param power: IDW exponent ``p`` (used only when ``method="idw"``).
    :param method: ``"krige"`` (default, v0.7) — ordinary kriging / GP spatial
        harmonics. The returned field carries a **variance** field
        (``field_var``): the regional field is itself a random field. Falls back
        to ``"idw"`` for < 3 stations.
    :param range_km: kriging covariance decay length (auto if ``None``).
    :param nugget_frac: kriging nugget as a fraction of the mode variance.
    """
    missing = [s for s in eof.stations if s not in stations_coords]
    if missing:
        raise ValueError(f"missing coordinates for stations: {missing}")
    names = eof.stations
    slon = np.array([stations_coords[s][0] for s in names], dtype=float)
    slat = np.array([stations_coords[s][1] for s in names], dtype=float)
    glon = np.asarray(grid_lons, dtype=float).ravel()
    glat = np.asarray(grid_lats, dtype=float).ravel()
    modes = eof.modes  # (n_modes × n_time)
    n_modes = modes.shape[0]
    n_time = modes.shape[1]
    # Recompute the same scaling (s[:n_modes]) that harmonize applied.
    X = np.stack([eof.reconstructed[s] - eof.means[s] for s in names])
    s_full = np.linalg.svd(X, compute_uv=False)
    scale = s_full[:n_modes]
    means = np.array([eof.means[s] for s in names], dtype=float)

    if method == "idw" or len(names) < 3:
        field = np.zeros((glon.size, n_time))
        for g in range(glon.size):
            dist = _haversine(
                np.full_like(slon, glon[g]), np.full_like(slat, glat[g]), slon, slat
            )
            w = 1.0 / (dist + eps_km) ** power
            wsum = w.sum()
            if wsum <= 0:
                w = np.ones_like(w) / w.size
            else:
                w = w / wsum
            load = eof.loadings.T @ w  # (n_modes,)
            field[g] = (scale * load) @ modes + w @ means
        return RegionalField(
            grid_lons=glon, grid_lats=glat, field=field,
            field_var=np.zeros_like(field), modes=modes,
            explained=eof.explained, total_explained=eof.total_explained,
            method="idw",
        )

    out = krige_regional(
        modes, eof.loadings, slon, slat, means, glon, glat,
        scale=scale, range_km=range_km, nugget_frac=nugget_frac,
    )
    return RegionalField(
        grid_lons=glon, grid_lats=glat,
        field=out["field"], field_var=out["field_var"], modes=modes,
        explained=eof.explained, total_explained=eof.total_explained,
        method="krige",
    )
