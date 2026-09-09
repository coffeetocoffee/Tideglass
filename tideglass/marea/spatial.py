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
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class EOFResult:
    stations: List[str]
    modes: np.ndarray  # (n_modes × n_time) orthonormal temporal modes
    explained: np.ndarray  # variance fraction per kept mode
    total_explained: float
    reconstructed: Dict[str, np.ndarray]
    means: Dict[str, float]


def harmonize(
    series: Dict[str, object],
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
        explained=frac[:n_modes],
        total_explained=float(frac[:n_modes].sum()),
        reconstructed={k: recon[i] + means[k] for i, k in enumerate(names)},
        means=means,
    )
