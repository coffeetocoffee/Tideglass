"""Exact linear least-squares solver for Marea Core.

The tidal signal is **linear in amplitude/phase** once frequencies are fixed::

    h(t) = H0 + Σ_i [ a_i·cos(ω_i t) + b_i·sin(ω_i t) ] + ε

We build the design matrix ``A`` (rows = observations, columns =
``[1, cos, sin, …]``) and solve via the thin SVD of ``A`` directly — the
stable equivalent of the Normal Equations ``x = (AᵀA)⁺ Aᵀy`` (no squaring of
the condition number)::

    A = U S Vᵀ            (thin SVD)
    x = V (Uᵀy / s)
    C = σ² · V diag(1/s²) Vᵀ      with  σ² = RSS / (n − p)

From ``x`` we recover amplitude ``A_i = √(a_i² + b_i²)`` and phase
``φ_i = atan2(b_i, a_i)``, so each term reads ``A_i·cos(ω_i t − φ_i)``,
with standard errors propagated from ``C``.

Units: ``t`` in hours relative to any fixed origin (the caller must use the
same origin for fitting and prediction); ``speeds`` in radians per hour
(use :func:`rad_per_hour` to convert catalog speeds in °/h).

This module is pure linear algebra: it takes frequencies, not constituents,
so it stays independently testable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TermFit:
    """Fitted amplitude/phase of one harmonic term.

    The term contributes ``amplitude·cos(ω·t − phase)`` to the signal.
    """

    name: str
    amplitude: float
    phase: float  # radians, in (−π, π]
    sigma_amp: float
    sigma_phase: float
    a: float  # cos coefficient
    b: float  # sin coefficient
    se_a: float
    se_b: float


@dataclass(frozen=True)
class Solution:
    mean: float  # H0, the time-mean level
    se_mean: float
    terms: list[TermFit]
    coef: np.ndarray  # full solution vector [H0, a1, b1, …]
    covariance: np.ndarray  # C = σ²·(AᵀA)⁺
    sigma2: float  # residual variance estimate
    rmse: float  # in-sample root-mean-square residual
    n_obs: int


def rad_per_hour(deg_per_hour) -> np.ndarray:
    """Convert catalog speeds (°/h) to angular frequencies (rad/h)."""
    return np.asarray(deg_per_hour, dtype=float) * (math.pi / 180.0)


def design_matrix(t_hours, speeds_rad_per_h: Sequence[float]) -> np.ndarray:
    """Build ``[1, cos(ω₁t), sin(ω₁t), …]`` (shape ``n_obs × (1 + 2·m)``)."""
    t = np.asarray(t_hours, dtype=float).ravel()
    speeds = np.asarray(list(speeds_rad_per_h), dtype=float).ravel()
    n, m = t.size, speeds.size
    A = np.empty((n, 1 + 2 * m))
    A[:, 0] = 1.0
    for j, w in enumerate(speeds):
        A[:, 1 + 2 * j] = np.cos(w * t)
        A[:, 2 + 2 * j] = np.sin(w * t)
    return A


def solve_matrix(A, y, names: Sequence[str] | None = None) -> Solution:
    """Solve a prebuilt ``[1, cos, sin, …]`` design matrix via SVD.

    Same closed-form solution and covariance as :func:`solve`, for callers
    (like :mod:`tideglass.marea.model`) that build a modulated basis —
    e.g. nodal-corrected ``f(t)·cos(V(t)+u(t))`` columns.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("design matrix must be 2-D")
    y = np.asarray(y, dtype=float).ravel()
    n, p = A.shape
    if n != y.size:
        raise ValueError(f"{n} rows but {y.size} heights")
    if (p - 1) % 2:
        raise ValueError("columns must be [1, cos, sin, …]")
    m = (p - 1) // 2
    if names is None:
        labels = [f"c{j}" for j in range(m)]
    else:
        labels = list(names)
        if len(labels) != m:
            raise ValueError(f"{m} terms but {len(labels)} names")
    if n <= p:
        raise ValueError(f"underdetermined: {n} observations, {p} parameters")

    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    if np.linalg.matrix_rank(A) < p:
        raise ValueError("rank-deficient design matrix (aliased frequencies?)")

    x = (Vt.T / s) @ (U.T @ y)
    resid = y - A @ x
    rss = float(resid @ resid)
    dof = n - p
    sigma2 = rss / dof
    # (AᵀA)⁺ = V diag(1/s²) Vᵀ — covariance of the solution.
    AtA_pinv = (Vt.T / s**2) @ Vt
    C = sigma2 * AtA_pinv

    terms: list[TermFit] = []
    for j, name in enumerate(labels):
        ia, ib = 1 + 2 * j, 2 + 2 * j
        a, b = float(x[ia]), float(x[ib])
        va, vb, cov = C[ia, ia], C[ib, ib], C[ia, ib]
        amp = math.hypot(a, b)
        phi = math.atan2(b, a)
        if amp > 0.0:
            var_amp = (a * a * va + b * b * vb + 2 * a * b * cov) / amp**2
            var_phi = (b * b * va + a * a * vb - 2 * a * b * cov) / amp**4
        else:  # degenerate zero-amplitude term
            var_amp = 0.5 * (va + vb)
            var_phi = math.inf
        terms.append(
            TermFit(
                name=name,
                amplitude=amp,
                phase=phi,
                sigma_amp=math.sqrt(max(var_amp, 0.0)),
                sigma_phase=math.sqrt(var_phi) if math.isfinite(var_phi) else math.inf,
                a=a,
                b=b,
                se_a=math.sqrt(max(va, 0.0)),
                se_b=math.sqrt(max(vb, 0.0)),
            )
        )

    return Solution(
        mean=float(x[0]),
        se_mean=math.sqrt(max(C[0, 0], 0.0)),
        terms=terms,
        coef=x,
        covariance=C,
        sigma2=sigma2,
        rmse=math.sqrt(rss / n),
        n_obs=n,
    )


def solve(
    t_hours,
    heights,
    speeds_rad_per_h: Sequence[float],
    names: Sequence[str] | None = None,
) -> Solution:
    """Fit ``h(t) = H0 + Σ[a·cos(ωt) + b·sin(ωt)]`` exactly via SVD.

    Raises ``ValueError`` when the system is underdetermined (``n ≤ p``) or
    rank-deficient (e.g. duplicate/aliased frequencies).
    """
    t = np.asarray(t_hours, dtype=float).ravel()
    y = np.asarray(heights, dtype=float).ravel()
    speeds = np.asarray(list(speeds_rad_per_h), dtype=float).ravel()
    if t.size != y.size:
        raise ValueError(f"{t.size} times but {y.size} heights")
    m = speeds.size
    if names is not None and len(list(names)) != m:
        raise ValueError(f"{m} speeds but {len(list(names))} names")
    return solve_matrix(design_matrix(t, speeds), y, names)
