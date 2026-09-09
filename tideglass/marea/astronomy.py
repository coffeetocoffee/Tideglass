"""Dynamic astronomical kernel for Marea Core.

Computes, for any datetime ``t``, the slow astronomical variables that drive
every tidal constituent, plus the per-constituent nodal corrections.

Basis: polynomial mean longitudes from Meeus, *Astronomical Algorithms*
(the same family of approximations behind IERS-style kernels), evaluated in
Julian centuries from J2000 — so the engine is valid at any epoch and the
constituent catalog only needs integer Doodson multipliers.

Nodal corrections (``f``, ``u``) follow Schureman, *Special Publication 98*
(node-factor equations 65–78, 195–235 and Table 2).

Conventions
-----------
* Naive datetimes are treated as UTC.
* ``doodson_args`` returns ``(tau, s, h, p, Np, p1)`` in degrees, where
  ``tau`` is the mean lunar time angle (hour angle of the mean moon),
  ``Np`` the longitude of the Moon's ascending node, and ``p1`` the mean
  longitude of solar perigee.
* ``RATES`` holds the mean hourly rates (°/h) of those six arguments,
  derived from the polynomial coefficients — never hardcoded magic numbers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timezone

import numpy as np

D2R = math.pi / 180.0
R2D = 180.0 / math.pi

_J2000_JD = 2451545.0  # Julian date of J2000.0 (2000-01-01 12:00 TT)
# Julian centuries elapsed per hour of wall time.
_CENTURIES_PER_HOUR = 1.0 / (24.0 * 365.25 * 100.0)

ARG_ORDER = ("tau", "s", "h", "p", "Np", "p1")


def _s2d(degrees: float, arcmins: float = 0.0, arcsecs: float = 0.0) -> float:
    """Sexagesimal angle to decimal degrees (Meeus-style helper)."""
    return degrees + arcmins / 60.0 + arcsecs / 3600.0


def _poly(coeffs: tuple[float, ...], x: float) -> float:
    return sum(c * x**i for i, c in enumerate(coeffs))


# --- Mean-longitude polynomials in T = Julian centuries from J2000 ---------
# (Meeus, Astronomical Algorithms: lunar longitude 45.1, lunar node 45.7,
# lunar perigee, solar longitude 24.2, terrestrial obliquity 21.3,
# solar perigee from the 24.2/24.3 difference.)

_S_COEFFS = (
    218.3164591,
    481267.88134236,
    -0.0013268,
    1 / 538841.0 - 1 / 65194000.0,
)
_H_COEFFS = (280.46645, 36000.76983, 0.0003032)
_P_COEFFS = (
    83.3532430,
    4069.0137111,
    -0.0103238,
    -1 / 80053.0 + 1 / 18999000.0,
)
_N_COEFFS = (
    125.0445550,
    -1934.1361849,
    0.0020762,
    1 / 467410.0 - 1 / 60616000.0,
)
_PP_COEFFS = (-77.06265, 1.71902, 0.0004591, 0.00000048)

_OBL_COEFFS = tuple(
    c * (1e-2) ** i
    for i, c in enumerate(
        (
            _s2d(23, 26, 21.448),
            -_s2d(0, 0, 4680.93),
            -_s2d(0, 0, 1.55),
            _s2d(0, 0, 1999.25),
            -_s2d(0, 0, 51.38),
            -_s2d(0, 0, 249.67),
            -_s2d(0, 0, 39.05),
            _s2d(0, 0, 7.12),
            _s2d(0, 0, 27.87),
            _s2d(0, 0, 5.79),
            _s2d(0, 0, 2.45),
        )
    )
)

_LUNAR_INCLINATION = 5.145  # degrees; essentially constant (JPL Horizons)


def _mean_rate(coeffs: tuple[float, ...]) -> float:
    """Mean hourly rate (°/h) = leading polynomial term × centuries/hour."""
    return coeffs[1] * _CENTURIES_PER_HOUR


_S_RATE = _mean_rate(_S_COEFFS)
_H_RATE = _mean_rate(_H_COEFFS)
_P_RATE = _mean_rate(_P_COEFFS)
_N_RATE = _mean_rate(_N_COEFFS)
_PP_RATE = _mean_rate(_PP_COEFFS)
# Mean lunar time advances with solar time (15°/h) corrected by the
# Sun–Moon longitude drift: tau = T + h − s.
_TAU_RATE = 15.0 + _H_RATE - _S_RATE

RATES: dict[str, float] = {
    "tau": _TAU_RATE,  # == 15.0 + ε with ε = rate(h) − rate(s) ≈ −0.5079
    "s": _S_RATE,
    "h": _H_RATE,
    "p": _P_RATE,
    "Np": _N_RATE,
    "p1": _PP_RATE,
}


# --- Time helpers ------------------------------------------------------------


def _as_utc(t: datetime) -> datetime:
    if t.tzinfo is None:
        return t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc)


def julian_day(t: datetime) -> float:
    """Julian date (Meeus formula 7.1)."""
    t = _as_utc(t)
    year, month = t.year, t.month
    day = (
        t.day
        + t.hour / 24.0
        + t.minute / 1440.0
        + t.second / 86400.0
        + t.microsecond / 86400.0e6
    )
    if month <= 2:
        year -= 1
        month += 12
    a = math.floor(year / 100.0)
    b = 2 - a + math.floor(a / 4.0)
    return (
        math.floor(365.25 * (year + 4716))
        + math.floor(30.6001 * (month + 1))
        + day
        + b
        - 1524.5
    )


def _centuries(t: datetime) -> float:
    return (julian_day(t) - _J2000_JD) / 36525.0


# --- Schureman auxiliary angles (functions of N, i, omega) -------------------


def _inclination(N, i, omega):
    """Mean inclination of the lunar orbit (Schureman Table 6, ``I``).

    Array-safe: accepts scalars or numpy arrays for ``N``/``omega``.
    """
    n, ii, o = D2R * N, D2R * i, D2R * omega
    arg = np.cos(ii) * np.cos(o) - np.sin(ii) * np.sin(o) * np.cos(n)
    return R2D * np.arccos(np.clip(arg, -1.0, 1.0))


def _xi(N, i, omega):
    n, ii, o = D2R * N, D2R * i, D2R * omega
    e1 = np.arctan(np.cos(0.5 * (o - ii)) / np.cos(0.5 * (o + ii)) * np.tan(0.5 * n)) - 0.5 * n
    e2 = np.arctan(np.sin(0.5 * (o - ii)) / np.sin(0.5 * (o + ii)) * np.tan(0.5 * n)) - 0.5 * n
    return -(e1 + e2) * R2D


def _nu(N, i, omega):
    n, ii, o = D2R * N, D2R * i, D2R * omega
    e1 = np.arctan(np.cos(0.5 * (o - ii)) / np.cos(0.5 * (o + ii)) * np.tan(0.5 * n)) - 0.5 * n
    e2 = np.arctan(np.sin(0.5 * (o - ii)) / np.sin(0.5 * (o + ii)) * np.tan(0.5 * n)) - 0.5 * n
    return (e1 - e2) * R2D


def _nup(N, i, omega):
    """Schureman eq. 224 (K1 phase correction argument). Array-safe."""
    inc = D2R * _inclination(N, i, omega)
    nu = D2R * _nu(N, i, omega)
    return R2D * np.arctan2(
        np.sin(2 * inc) * np.sin(nu),
        np.sin(2 * inc) * np.cos(nu) + 0.3347,
    )


def _nupp(N, i, omega):
    """Schureman eq. 232 (K2 phase correction argument). Array-safe."""
    inc = D2R * _inclination(N, i, omega)
    nu = D2R * _nu(N, i, omega)
    return R2D * 0.5 * np.arctan2(
        np.sin(inc) ** 2 * np.sin(2 * nu),
        np.sin(inc) ** 2 * np.cos(2 * nu) + 0.0727,
    )


def _astro_state_array(times: Sequence[datetime]) -> dict[str, np.ndarray]:
    """Full astronomical state for many times at once (arrays, degrees).

    Vectorized over ``times`` so the design matrix can be built without a
    per-(constituent, time) Python loop. ``_astro_state`` (single time) is a
    scalar view over this same code path.
    """
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
    jd = (
        np.floor(365.25 * (Y2 + 4716.0))
        + np.floor(30.6001 * (Mo2 + 1.0))
        + D + b - 1524.5
    )
    T = (jd - _J2000_JD) / 36525.0

    s = _poly(_S_COEFFS, T) % 360.0
    h = _poly(_H_COEFFS, T) % 360.0
    p = _poly(_P_COEFFS, T) % 360.0
    N = _poly(_N_COEFFS, T) % 360.0
    pp = _poly(_PP_COEFFS, T) % 360.0
    omega = _poly(_OBL_COEFFS, T) % 360.0
    i = _LUNAR_INCLINATION
    inc = _inclination(N, i, omega)
    xi = _xi(N, i, omega) % 360.0
    nu = _nu(N, i, omega) % 360.0
    nup = _nup(N, i, omega) % 360.0
    nupp = _nupp(N, i, omega) % 360.0
    hour_angle = (jd - np.floor(jd)) * 360.0
    tau = (hour_angle + h - s) % 360.0
    return {
        "tau": tau,
        "s": s,
        "h": h,
        "p": p,
        "N": N,
        "pp": pp,
        "omega": omega,
        "i": i,
        "I": inc,
        "xi": xi,
        "nu": nu,
        "nup": nup,
        "nupp": nupp,
    }


def _astro_state(t: datetime) -> dict[str, float]:
    """Full astronomical state at time ``t`` (all values in degrees)."""
    st = _astro_state_array([t])
    return {k: float(v[0]) if np.ndim(v) else float(v) for k, v in st.items()}


# --- Public kernel API ---------------------------------------------------------


def mean_longitudes(t: datetime) -> dict[str, float]:
    """Mean longitudes ``s, h, p, N, p1`` (degrees) from epoch J2000."""
    st = _astro_state(t)
    return {"s": st["s"], "h": st["h"], "p": st["p"], "N": st["N"], "p1": st["pp"]}


def doodson_args(t: datetime) -> tuple[float, ...]:
    """The six Doodson arguments ``(tau, s, h, p, Np, p1)`` in degrees."""
    st = _astro_state(t)
    return (st["tau"], st["s"], st["h"], st["p"], st["N"], st["pp"])


# --- Nodal corrections (Schureman) --------------------------------------------


def _f_unity(a: dict) -> float:
    return 1.0


def _f_Mm(a: dict) -> float:
    o, i, I = D2R * a["omega"], D2R * a["i"], D2R * a["I"]
    mean = (2 / 3.0 - np.sin(o) ** 2) * (1 - 1.5 * np.sin(i) ** 2)
    return (2 / 3.0 - np.sin(I) ** 2) / mean


def _f_Mf(a: dict) -> float:
    o, i, I = D2R * a["omega"], D2R * a["i"], D2R * a["I"]
    mean = np.sin(o) ** 2 * np.cos(0.5 * i) ** 4
    return np.sin(I) ** 2 / mean


def _f_O1(a: dict) -> float:
    o, i, I = D2R * a["omega"], D2R * a["i"], D2R * a["I"]
    mean = np.sin(o) * np.cos(0.5 * o) ** 2 * np.cos(0.5 * i) ** 4
    return (np.sin(I) * np.cos(0.5 * I) ** 2) / mean


def _f_M2(a: dict) -> float:
    o, i, I = D2R * a["omega"], D2R * a["i"], D2R * a["I"]
    mean = np.cos(0.5 * o) ** 4 * np.cos(0.5 * i) ** 4
    return np.cos(0.5 * I) ** 4 / mean


def _f_K1(a: dict) -> float:
    o, i, I = D2R * a["omega"], D2R * a["i"], D2R * a["I"]
    nu = D2R * a["nu"]
    mean = 0.5023 * np.sin(2 * o) * (1 - 1.5 * np.sin(i) ** 2) + 0.1681
    return np.sqrt(
        0.2523 * np.sin(2 * I) ** 2 + 0.1689 * np.sin(2 * I) * np.cos(nu) + 0.0283
    ) / mean


def _f_K2(a: dict) -> float:
    o, i, I = D2R * a["omega"], D2R * a["i"], D2R * a["I"]
    nu = D2R * a["nu"]
    mean = 0.5023 * np.sin(o) ** 2 * (1 - 1.5 * np.sin(i) ** 2) + 0.0365
    return np.sqrt(
        0.2523 * np.sin(I) ** 4 + 0.0367 * np.sin(I) ** 2 * np.cos(2 * nu) + 0.0013
    ) / mean


def _u_zero(a: dict[str, float]) -> float:
    return 0.0


def _u_Mf(a: dict[str, float]) -> float:
    return -2.0 * a["xi"]


def _u_O1(a: dict[str, float]) -> float:
    return 2.0 * a["xi"] - a["nu"]


def _u_M2(a: dict[str, float]) -> float:
    return 2.0 * a["xi"] - 2.0 * a["nu"]


def _u_K1(a: dict[str, float]) -> float:
    return -a["nup"]


def _u_K2(a: dict[str, float]) -> float:
    return -2.0 * a["nupp"]


_F_DISPATCH = {
    "unity": _f_unity,
    "M2": _f_M2,
    "O1": _f_O1,
    "K1": _f_K1,
    "K2": _f_K2,
    "Mf": _f_Mf,
    "Mm": _f_Mm,
}
_U_DISPATCH = {
    "unity": _u_zero,
    "M2": _u_M2,
    "O1": _u_O1,
    "K1": _u_K1,
    "K2": _u_K2,
    "Mf": _u_Mf,
    "Mm": _u_zero,
}


def _wrap_deg(x: float) -> float:
    """Wrap angle to [-180, 180)."""
    return (x + 180.0) % 360.0 - 180.0


def nodal_factor(constituent, t: datetime) -> tuple[float, float]:
    """Node factor ``f`` and equilibrium correction ``u`` (degrees).

    ``constituent`` is a :class:`tideglass.marea.constituents.Constituent`;
    only its ``nodal``/``f_power``/``u_power`` fields are used, so shallow-water
    multiples (M4 = 2×M2, M6 = 3×M2, …) compose correctly.
    """
    a = _astro_state(t)
    f = _F_DISPATCH[constituent.nodal](a) ** constituent.f_power
    u = _U_DISPATCH[constituent.nodal](a) * constituent.u_power
    return (f, _wrap_deg(u))


def nodal_factor_array(constituent, state: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized :func:`nodal_factor` over a state produced by
    :func:`_astro_state_array` (arrays of ``f`` and ``u`` in degrees)."""
    f = _F_DISPATCH[constituent.nodal](state) ** constituent.f_power
    u = _U_DISPATCH[constituent.nodal](state) * constituent.u_power
    return f, _wrap_deg(u)
