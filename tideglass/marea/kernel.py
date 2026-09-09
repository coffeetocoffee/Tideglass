"""Optional accelerated numeric kernel for Marea Core — v0.9.

The tidal design matrix is the engine's hot loop: for every ``(constituent,
time)`` we evaluate the nodal-modulated equilibrium argument and its ``cos`` /
``sin`` pair. The reference path (:func:`tideglass.marea.model._basis_matrix`)
computes this in vectorised numpy. This module adds an *optional* **Numba** JIT
kernel that fuses the whole computation (Julian date → mean longitudes →
Schureman node factors → design rows) into a single typed loop.

Numba is **never** a hard dependency. The project still installs with
``pip install tideglass`` (numpy only); the accelerated path is selected only
when Numba happens to be present, or when the caller passes ``kernel="numba"``.
All backends produce the same design matrix (verified by the test suite), so the
choice is purely a speed/dependency trade-off — the math is identical.

Public entry point: :func:`basis_matrix(constituents, times, backend="auto")`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from tideglass.marea import constituents as CON
from tideglass.marea.astronomy import D2R

# Polynomial coefficients of the mean-longitude series (Meeus), kept here so the
# numba kernel is self-contained (no round-trip through astronomy.py). Each is
# (constant, linear, quadratic, ...) in Julian centuries T from J2000.
_S_COEFFS = np.array(
    (218.3164591, 481267.88134236, -0.0013268, 1 / 538841.0 - 1 / 65194000.0),
    dtype=float)
_H_COEFFS = np.array((280.46645, 36000.76983, 0.0003032), dtype=float)
_P_COEFFS = np.array(
    (83.3532430, 4069.0137111, -0.0103238, -1 / 80053.0 + 1 / 18999000.0),
    dtype=float)
_N_COEFFS = np.array(
    (125.0445550, -1934.1361849, 0.0020762, 1 / 467410.0 - 1 / 60616000.0),
    dtype=float)
_PP_COEFFS = np.array((-77.06265, 1.71902, 0.0004591, 0.00000048), dtype=float)
_OBL_COEFFS = np.array([
    c * (1e-2) ** i
    for i, c in enumerate((
        23.0 + 26.0 / 60.0 + 21.448 / 3600.0,
        -(0.0 + 0.0 / 60.0 + 4680.93 / 3600.0),
        -(0.0 + 0.0 / 60.0 + 1.55 / 3600.0),
        0.0 + 0.0 / 60.0 + 1999.25 / 3600.0,
        -(0.0 + 0.0 / 60.0 + 51.38 / 3600.0),
        -(0.0 + 0.0 / 60.0 + 249.67 / 3600.0),
        -(0.0 + 0.0 / 60.0 + 39.05 / 3600.0),
        0.0 + 0.0 / 60.0 + 7.12 / 3600.0,
        0.0 + 0.0 / 60.0 + 27.87 / 3600.0,
        0.0 + 0.0 / 60.0 + 5.79 / 3600.0,
        0.0 + 0.0 / 60.0 + 2.45 / 3600.0,
    ))
], dtype=float)

_J2000_JD = 2451545.0
_LUNAR_INCLINATION = 5.145  # degrees, effectively constant

# Constituent `.nodal` family -> integer code used by the numba dispatcher.
NODAL_CODE = {
    "unity": 0, "M2": 1, "O1": 2, "K1": 3, "K2": 4, "Mf": 5, "Mm": 6,
}


def _nodal_code(c: CON.Constituent) -> int:
    return NODAL_CODE.get(c.nodal, 0)


try:
    import numba  # noqa: F401

    _HAS_NUMBA = True
except (ImportError, ModuleNotFoundError):  # numba is optional
    _HAS_NUMBA = False


def available_backends() -> list[str]:
    """Backends this install can run: always ``numpy``; ``numba`` if installed."""
    return ["numpy", "numba"] if _HAS_NUMBA else ["numpy"]


# --- numpy reference path ------------------------------------------------------


def _poly(coeffs: np.ndarray, x: float) -> float:
    out = 0.0
    for i in range(coeffs.shape[0]):
        out = out + coeffs[i] * (x ** i)
    return out


def _inclination(N, i, omega):
    n, ii, o = D2R * N, D2R * i, D2R * omega
    return math.degrees(math.acos(max(-1.0, min(1.0,
        math.cos(ii) * math.cos(o) - math.sin(ii) * math.sin(o) * math.cos(n)))))


def _xi(N, i, omega):
    n, ii, o = D2R * N, D2R * i, D2R * omega
    t = math.tan(0.5 * n)
    e1 = math.atan(math.cos(0.5 * (o - ii)) / math.cos(0.5 * (o + ii)) * t) - 0.5 * n
    e2 = math.atan(math.sin(0.5 * (o - ii)) / math.sin(0.5 * (o + ii)) * t) - 0.5 * n
    return -math.degrees(e1 + e2)


def _nu(N, i, omega):
    n, ii, o = D2R * N, D2R * i, D2R * omega
    t = math.tan(0.5 * n)
    e1 = math.atan(math.cos(0.5 * (o - ii)) / math.cos(0.5 * (o + ii)) * t) - 0.5 * n
    e2 = math.atan(math.sin(0.5 * (o - ii)) / math.sin(0.5 * (o + ii)) * t) - 0.5 * n
    return math.degrees(e1 - e2)


def _nup(N, i, omega):
    inc = D2R * _inclination(N, i, omega)
    nu = D2R * _nu(N, i, omega)
    return math.degrees(math.atan2(
        math.sin(2 * inc) * math.sin(nu),
        math.sin(2 * inc) * math.cos(nu) + 0.3347))


def _nupp(N, i, omega):
    inc = D2R * _inclination(N, i, omega)
    nu = D2R * _nu(N, i, omega)
    return math.degrees(0.5 * math.atan2(
        math.sin(inc) ** 2 * math.sin(2 * nu),
        math.sin(inc) ** 2 * math.cos(2 * nu) + 0.0727))


def _f_factor(code, omega, i, I, nu):
    o, ii, Ii, nuv = (D2R * x for x in (omega, i, I, nu))
    if code == 0:
        return 1.0
    if code == 1:  # M2
        return math.cos(0.5 * Ii) ** 4 / (
            math.cos(0.5 * o) ** 4 * math.cos(0.5 * ii) ** 4)
    if code == 2:  # O1
        return (math.sin(Ii) * math.cos(0.5 * Ii) ** 2) / (
            math.sin(o) * math.cos(0.5 * o) ** 2 * math.cos(0.5 * ii) ** 4)
    if code == 3:  # K1
        mean = 0.5023 * math.sin(2 * o) * (1 - 1.5 * math.sin(ii) ** 2) + 0.1681
        return math.sqrt(0.2523 * math.sin(2 * Ii) ** 2
                         + 0.1689 * math.sin(2 * Ii) * math.cos(nuv)
                         + 0.0283) / mean
    if code == 4:  # K2
        mean = 0.5023 * math.sin(o) ** 2 * (1 - 1.5 * math.sin(ii) ** 2) + 0.0365
        return math.sqrt(0.2523 * math.sin(Ii) ** 4
                         + 0.0367 * math.sin(Ii) ** 2 * math.cos(2 * nuv)
                         + 0.0013) / mean
    if code == 5:  # Mf
        return math.sin(Ii) ** 2 / (math.sin(o) ** 2 * math.cos(0.5 * ii) ** 4)
    # Mm
    return (2.0 / 3.0 - math.sin(Ii) ** 2) / (
        (2.0 / 3.0 - math.sin(o) ** 2) * (1 - 1.5 * math.sin(ii) ** 2))


def _u_factor(code, xi, nu, nup, nupp):
    if code == 0:
        return 0.0
    if code == 1:  # M2
        return 2.0 * xi - 2.0 * nu
    if code == 2:  # O1
        return 2.0 * xi - nu
    if code == 3:  # K1
        return -nup
    if code == 4:  # K2
        return -2.0 * nupp
    if code == 5:  # Mf
        return -2.0 * xi
    return 0.0  # Mm


def _basis_matrix_numpy(constituents: Sequence[CON.Constituent],
                       times: Sequence) -> np.ndarray:
    """Vectorised numpy reference (mirrors the former model._basis_matrix)."""
    times = list(times)
    n, m = len(times), len(constituents)
    A = np.empty((n, 1 + 2 * m))
    A[:, 0] = 1.0
    if n and m:
        Y = np.array([t.year for t in times], dtype=float)
        Mo = np.array([t.month for t in times], dtype=float)
        D = np.array([
            t.day + t.hour / 24.0 + t.minute / 1440.0 + t.second / 86400.0
            + t.microsecond / 86400.0e6 for t in times
        ])
        Mo2 = np.where(Mo <= 2, Mo + 12.0, Mo)
        Y2 = np.where(Mo <= 2, Y - 1.0, Y)
        a = np.floor(Y2 / 100.0)
        b = 2.0 - a + np.floor(a / 4.0)
        jd = (np.floor(365.25 * (Y2 + 4716.0)) + np.floor(30.6001 * (Mo2 + 1.0))
              + D + b - 1524.5)
        T = (jd - _J2000_JD) / 36525.0
        s = _poly(_S_COEFFS, T) % 360.0
        h = _poly(_H_COEFFS, T) % 360.0
        p = _poly(_P_COEFFS, T) % 360.0
        N = _poly(_N_COEFFS, T) % 360.0
        pp = _poly(_PP_COEFFS, T) % 360.0
        omega = _poly(_OBL_COEFFS, T) % 360.0
        i = _LUNAR_INCLINATION
        inc = np.array([_inclination(nn, i, om) for nn, om in zip(N, omega)])
        xi = np.array([_xi(nn, i, om) for nn, om in zip(N, omega)]) % 360.0
        nu = np.array([_nu(nn, i, om) for nn, om in zip(N, omega)]) % 360.0
        nup = np.array([_nup(nn, i, om) for nn, om in zip(N, omega)]) % 360.0
        nupp = np.array([_nupp(nn, i, om) for nn, om in zip(N, omega)]) % 360.0
        hour_angle = (jd - np.floor(jd)) * 360.0
        tau = (hour_angle + h - s) % 360.0
        args = np.vstack([tau, s, h, p, N, pp])  # (6, n)
        for j, c in enumerate(constituents):
            dood = np.array(c.doodson, dtype=float)
            V = (dood @ args + c.phase0) % 360.0
            code = _nodal_code(c)
            f = np.array([_f_factor(code, om, i, ic, nv)
                          for om, ic, nv in
                          zip(omega, inc, nu)]) ** c.f_power
            u = np.array([_u_factor(code, x, nv, npv, nppv)
                          for x, nv, npv, nppv in
                          zip(xi, nu, nup, nupp)]) * c.u_power
            theta = (V + u) * D2R
            A[:, 1 + 2 * j] = f * np.cos(theta)
            A[:, 2 + 2 * j] = f * np.sin(theta)
    return A


# --- numba accelerated path ---------------------------------------------------


def _build_numba():
    """Compile the fused numba kernel once and return it (or None if no numba)."""
    if not _HAS_NUMBA:
        return None
    from numba import njit

    S = _S_COEFFS
    H = _H_COEFFS
    P = _P_COEFFS
    Nn = _N_COEFFS
    PP = _PP_COEFFS
    OBL = _OBL_COEFFS
    J2000 = _J2000_JD
    INCL = _LUNAR_INCLINATION
    D2RAD = math.pi / 180.0

    @njit(cache=True)
    def _poly(c, x):
        out = 0.0
        for i in range(c.shape[0]):
            out = out + c[i] * (x ** i)
        return out

    @njit(cache=True)
    def _incl(N, i, omega):
        n = D2RAD * N
        ii = D2RAD * i
        o = D2RAD * omega
        arg = math.cos(ii) * math.cos(o) - math.sin(ii) * math.sin(o) * math.cos(n)
        if arg > 1.0:
            arg = 1.0
        elif arg < -1.0:
            arg = -1.0
        return math.degrees(math.acos(arg))

    @njit(cache=True)
    def _xi(N, i, omega):
        n = D2RAD * N
        ii = D2RAD * i
        o = D2RAD * omega
        t = math.tan(0.5 * n)
        e1 = math.atan(math.cos(0.5 * (o - ii)) / math.cos(0.5 * (o + ii)) * t) - 0.5 * n
        e2 = math.atan(math.sin(0.5 * (o - ii)) / math.sin(0.5 * (o + ii)) * t) - 0.5 * n
        return -math.degrees(e1 + e2)

    @njit(cache=True)
    def _nu(N, i, omega):
        n = D2RAD * N
        ii = D2RAD * i
        o = D2RAD * omega
        t = math.tan(0.5 * n)
        e1 = math.atan(math.cos(0.5 * (o - ii)) / math.cos(0.5 * (o + ii)) * t) - 0.5 * n
        e2 = math.atan(math.sin(0.5 * (o - ii)) / math.sin(0.5 * (o + ii)) * t) - 0.5 * n
        return math.degrees(e1 - e2)

    @njit(cache=True)
    def _nup(N, i, omega):
        inc = D2RAD * _incl(N, i, omega)
        nu = D2RAD * _nu(N, i, omega)
        return math.degrees(math.atan2(
            math.sin(2.0 * inc) * math.sin(nu),
            math.sin(2.0 * inc) * math.cos(nu) + 0.3347))

    @njit(cache=True)
    def _nupp(N, i, omega):
        inc = D2RAD * _incl(N, i, omega)
        nu = D2RAD * _nu(N, i, omega)
        return math.degrees(0.5 * math.atan2(
            math.sin(inc) ** 2 * math.sin(2.0 * nu),
            math.sin(inc) ** 2 * math.cos(2.0 * nu) + 0.0727))

    @njit(cache=True)
    def _f(code, omega, i, I, nu):
        o = D2RAD * omega
        ii = D2RAD * i
        Ii = D2RAD * I
        nuv = D2RAD * nu
        if code == 0:
            return 1.0
        if code == 1:
            return math.cos(0.5 * Ii) ** 4 / (
                math.cos(0.5 * o) ** 4 * math.cos(0.5 * ii) ** 4)
        if code == 2:
            return (math.sin(Ii) * math.cos(0.5 * Ii) ** 2) / (
                math.sin(o) * math.cos(0.5 * o) ** 2 * math.cos(0.5 * ii) ** 4)
        if code == 3:
            mean = 0.5023 * math.sin(2.0 * o) * (1.0 - 1.5 * math.sin(ii) ** 2) + 0.1681
            return math.sqrt(0.2523 * math.sin(2.0 * Ii) ** 2
                             + 0.1689 * math.sin(2.0 * Ii) * math.cos(nuv)
                             + 0.0283) / mean
        if code == 4:
            mean = 0.5023 * math.sin(o) ** 2 * (1.0 - 1.5 * math.sin(ii) ** 2) + 0.0365
            return math.sqrt(0.2523 * math.sin(Ii) ** 4
                             + 0.0367 * math.sin(Ii) ** 2 * math.cos(2.0 * nuv)
                             + 0.0013) / mean
        if code == 5:
            return math.sin(Ii) ** 2 / (math.sin(o) ** 2 * math.cos(0.5 * ii) ** 4)
        return (2.0 / 3.0 - math.sin(Ii) ** 2) / (
            (2.0 / 3.0 - math.sin(o) ** 2) * (1.0 - 1.5 * math.sin(ii) ** 2))

    @njit(cache=True)
    def _u(code, xi, nu, nup, nupp):
        if code == 0:
            return 0.0
        if code == 1:
            return 2.0 * xi - 2.0 * nu
        if code == 2:
            return 2.0 * xi - nu
        if code == 3:
            return -nup
        if code == 4:
            return -2.0 * nupp
        if code == 5:
            return -2.0 * xi
        return 0.0

    @njit(cache=True)
    def _build(years, months, dayfrac, doodson, phase0, ncode, fpow, upow,
               S_c, H_c, P_c, N_c, PP_c, OBL_c, J2000, INCL, D2RAD):
        n = years.shape[0]
        m = doodson.shape[0]
        A = np.empty((n, 1 + 2 * m))
        for t in range(n):
            Y = years[t]
            Mo = months[t]
            D = dayfrac[t]
            if Mo <= 2:
                Mo2 = Mo + 12.0
                Y2 = Y - 1.0
            else:
                Mo2 = Mo
                Y2 = Y
            a = math.floor(Y2 / 100.0)
            b = 2.0 - a + math.floor(a / 4.0)
            jd = (math.floor(365.25 * (Y2 + 4716.0))
                  + math.floor(30.6001 * (Mo2 + 1.0)) + D + b - 1524.5)
            T = (jd - J2000) / 36525.0
            s = _poly(S_c, T) % 360.0
            h = _poly(H_c, T) % 360.0
            p = _poly(P_c, T) % 360.0
            N = _poly(N_c, T) % 360.0
            pp = _poly(PP_c, T) % 360.0
            omega = _poly(OBL_c, T) % 360.0
            i = INCL
            I = _incl(N, i, omega)
            xi = _xi(N, i, omega) % 360.0
            nu = _nu(N, i, omega) % 360.0
            nup = _nup(N, i, omega) % 360.0
            nupp = _nupp(N, i, omega) % 360.0
            hour_angle = (jd - math.floor(jd)) * 360.0
            tau = (hour_angle + h - s) % 360.0
            args = np.empty(6)
            args[0] = tau
            args[1] = s
            args[2] = h
            args[3] = p
            args[4] = N
            args[5] = pp
            for c in range(m):
                V = (doodson[c, 0] * args[0] + doodson[c, 1] * args[1]
                     + doodson[c, 2] * args[2] + doodson[c, 3] * args[3]
                     + doodson[c, 4] * args[4] + doodson[c, 5] * args[5]
                     + phase0[c]) % 360.0
                code = ncode[c]
                f = _f(code, omega, i, I, nu) ** fpow[c]
                u = _u(code, xi, nu, nup, nupp) * upow[c]
                theta = (V + u) * D2RAD
                A[t, 0] = 1.0
                A[t, 1 + 2 * c] = f * math.cos(theta)
                A[t, 2 + 2 * c] = f * math.sin(theta)
        return A

    def _call(constituents, times):
        years = np.array([t.year for t in times], dtype=np.float64)
        months = np.array([t.month for t in times], dtype=np.float64)
        dayfrac = np.array([
            t.day + t.hour / 24.0 + t.minute / 1440.0 + t.second / 86400.0
            + t.microsecond / 86400.0e6 for t in times
        ], dtype=np.float64)
        doodson = np.array([list(c.doodson) for c in constituents], dtype=np.float64)
        phase0 = np.array([float(c.phase0) for c in constituents], dtype=np.float64)
        ncode = np.array([_nodal_code(c) for c in constituents], dtype=np.int64)
        fpow = np.array([float(c.f_power) for c in constituents], dtype=np.float64)
        upow = np.array([float(c.u_power) for c in constituents], dtype=np.float64)
        return _build(years, months, dayfrac, doodson, phase0, ncode, fpow, upow,
                      S, H, P, Nn, PP, OBL, J2000, INCL, D2RAD)

    return _call


_NUMBA_CALL = None


def _basis_matrix_numba(constituents, times):
    """Fused Numba JIT path; raises ``RuntimeError`` if Numba is unavailable."""
    global _NUMBA_CALL
    if not _HAS_NUMBA:
        raise RuntimeError(
            "numba backend requested but numba is not installed "
            "(pip install numba, or use backend='numpy')")
    if _NUMBA_CALL is None:
        _NUMBA_CALL = _build_numba()
    return _NUMBA_CALL(list(constituents), list(times))


def basis_matrix(constituents: Sequence[CON.Constituent], times: Sequence,
                 backend: str = "auto") -> np.ndarray:
    """Tidal design matrix ``[1, cos, sin, …]`` for the given constituents/times.

    :param backend: ``"auto"`` (numba when available, else numpy), ``"numpy"``,
        or ``"numba"``. All backends are numerically equivalent.
    """
    if backend not in ("auto", "numpy", "numba"):
        raise ValueError(f"unknown backend {backend!r}")
    if backend == "numba" or (backend == "auto" and _HAS_NUMBA):
        return _basis_matrix_numba(constituents, times)
    return _basis_matrix_numpy(constituents, times)
