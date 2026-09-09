"""DCDM automatic constituent inference for Marea Core.

``pytides`` explicitly refuses to choose constituents; we implement the
Darwin–Cartwright–Doodson method as greedy forward selection with orthogonal
least squares:

1. Start with the dominant constituent (M2) plus the mean level.
2. Orthogonalize every remaining candidate's ``[cos, sin]`` block against the
   already-selected basis (modified Gram–Schmidt in observation space).
3. Add the candidate that most reduces residual variance.
4. **Stop** when the addition fails an **F-test** at the chosen significance
   level (guards against fitting noise).

An optional Rayleigh separability gate (default on) drops candidates a record
is too short to resolve: candidates are processed in input order (pass
dominant constituents first — the catalog already is), and any candidate
that completes fewer than ``min_separation_cycles`` beat cycles against an
earlier-kept candidate over the record span is excluded up front. This is
the standard treatment (cf. Foreman/t_tide): e.g. K1/P1 need ~182 days to
separate, so on shorter records only K1 is solved for. The F-distribution
tail is computed in-house via the incomplete beta function, so this module
needs only ``numpy`` + stdlib.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# --- F-distribution tail via the regularized incomplete beta function --------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz)."""
    maxit, eps, fpmin = 200, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x < (a + 1.0) / (a + b + 2.0):
        front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                         + a * math.log(x) + b * math.log1p(-x))
        return front * _betacf(a, b, x) / a
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def f_sf(f_stat: float, d1: int, d2: int) -> float:
    """Survival function P(F(d1, d2) > f_stat)."""
    if f_stat <= 0.0:
        return 1.0
    x = d2 / (d2 + d1 * f_stat)
    return _betai(d2 / 2.0, d1 / 2.0, x)


# --- DCDM selection ------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    name: str
    speed: float  # rad/h


@dataclass(frozen=True)
class SelectionResult:
    selected: list[str]  # names in selection order
    excluded: list[str]  # names dropped by the Rayleigh gate (unresolvable)
    rss: list[float]  # residual sum of squares after each accepted addition
    f_stats: list[float]  # F statistic of each accepted addition
    p_values: list[float]
    n_obs: int
    n_params: int  # 1 + 2·len(selected)


def _as_candidate(c) -> Candidate:
    if isinstance(c, Candidate):
        return c
    name, speed = c
    return Candidate(str(name), float(speed))


def _orthonormalize_block(Q: np.ndarray, G: np.ndarray, tol: float) -> np.ndarray | None:
    """Orthonormalize the n×2 block G against basis Q (MGS, twice).

    Returns an n×2 orthonormal basis for the new directions, or ``None``
    when a direction is numerically collinear (aliased candidate).
    """
    W = G - Q @ (Q.T @ G)
    W = W - Q @ (Q.T @ W)  # reorthogonalization
    out = np.empty_like(W)
    for j in range(W.shape[1]):
        v = W[:, j].copy()
        for k in range(j):  # against previously accepted directions of G
            v -= out[:, k] * (out[:, k] @ v)
        nrm = float(np.sqrt(v @ v))
        if nrm < tol:
            return None
        out[:, j] = v / nrm
    return out


def select(
    t_hours,
    heights,
    candidates: Sequence[Candidate],
    alpha: float = 0.05,
    start: Sequence[str] = ("M2",),
    min_separation_cycles: float = 1.0,
) -> SelectionResult:
    """Greedily select constituents; stop when the F-test fails.

    :param alpha: significance level — a candidate is accepted only when its
        F-test p-value is below ``alpha``.
    :param start: constituents pre-selected without a test (the dominant
        M2 by default); entries absent from ``candidates`` are ignored.
    :param min_separation_cycles: Rayleigh gate — candidates are considered
        in input order (dominant first) and one is excluded when it completes
        fewer than this many beat cycles against an earlier-kept candidate
        over the record span. Set to 0 to disable.
    """
    t = np.asarray(t_hours, dtype=float).ravel()
    y = np.asarray(heights, dtype=float).ravel()
    if t.size != y.size:
        raise ValueError(f"{t.size} times but {y.size} heights")
    cands = [_as_candidate(c) for c in candidates]
    if len({c.name for c in cands}) != len(cands):
        raise ValueError("duplicate candidate names")
    by_name = {c.name: c for c in cands}
    n = t.size
    span = float(t.max() - t.min()) if n > 1 else 0.0

    # Rayleigh pre-filter: keep dominant-first, drop unresolvable followers.
    # The mean level counts as an implicit member: a candidate must also
    # complete enough cycles itself, so Ssa/Sa (periods of months) are
    # excluded from short records instead of aliasing storms into fake
    # long-period cycles.
    excluded: list[str] = []
    if min_separation_cycles > 0 and span > 0:
        kept: list[Candidate] = []
        for c in cands:
            own_cycles = abs(c.speed) * span / (2 * math.pi)
            if own_cycles < min_separation_cycles or any(
                abs(c.speed - k.speed) * span / (2 * math.pi) < min_separation_cycles
                for k in kept
            ):
                excluded.append(c.name)
            else:
                kept.append(c)
        cands = kept
        by_name = {c.name: c for c in cands}

    Q = np.full((n, 1), 1.0 / math.sqrt(n))  # mean level is always in the model
    proj = Q.T @ y
    r = y - Q @ proj
    rss = float(r @ r)

    selected: list[str] = []
    f_stats: list[float] = []
    p_values: list[float] = []
    rss_path = [rss]

    def absorb(name: str) -> None:
        """Add a constituent's block to the basis (no test)."""
        nonlocal Q, r, rss
        w = by_name[name].speed
        G = np.column_stack([np.cos(w * t), np.sin(w * t)])
        Qg = _orthonormalize_block(Q, G, tol=1e-8 * math.sqrt(n))
        if Qg is None:
            return
        Q = np.column_stack([Q, Qg])
        r = r - Qg @ (Qg.T @ r)
        rss = float(r @ r)

    for name in start:
        if name in by_name:
            absorb(name)
            selected.append(name)
    rss_path = [rss]

    remaining = [c for c in cands if c.name not in selected]
    while remaining:
        best = None  # (reduction, candidate, Qg, rss_new)
        for c in remaining:
            G = np.column_stack([np.cos(c.speed * t), np.sin(c.speed * t)])
            Qg = _orthonormalize_block(Q, G, tol=1e-8 * math.sqrt(n))
            if Qg is None:  # numerically collinear: unresolvable, skip
                continue
            coef = Qg.T @ r
            reduction = float(coef @ coef)
            rss_new = rss - reduction
            if reduction <= 0 or rss_new < 0:
                continue
            if best is None or reduction > best[0]:
                best = (reduction, c, Qg, rss_new)
        if best is None:
            break
        _, c, Qg, rss_new = best
        p_new = Q.shape[1] + 2
        dof = n - p_new
        if dof <= 0:
            break
        f_stat = ((rss - rss_new) / 2.0) / (rss_new / dof)
        p_value = f_sf(f_stat, 2, dof)
        if p_value >= alpha:
            break
        Q = np.column_stack([Q, Qg])
        r = r - Qg @ (Qg.T @ r)
        rss = rss_new
        selected.append(c.name)
        f_stats.append(f_stat)
        p_values.append(p_value)
        rss_path.append(rss)
        remaining = [x for x in remaining if x.name != c.name]

    return SelectionResult(
        selected=selected,
        excluded=excluded,
        rss=rss_path,
        f_stats=f_stats,
        p_values=p_values,
        n_obs=n,
        n_params=Q.shape[1],
    )
