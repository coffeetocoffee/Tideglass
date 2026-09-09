"""Constituent catalog for Marea Core.

A declarative table. Each entry stores integer Doodson multipliers of the
``(tau, s, h, p, Np, p1)`` arguments; the angular speed (°/h) is *derived* as::

    speed = Σ_k doodson[k] * RATES[k]

so the catalog is trivially extensible and valid at any epoch.

Doodson numbers follow the standard equilibrium-argument convention
(Schureman SP-98; same numbering as NOAA/``pytides``). ``phase0`` carries the
constant ±90° phase offset some species need; ``nodal`` selects the
Schureman node-factor family; ``f_power``/``u_power`` compose the shallow-water
multiples (M4 = 2×M2 → power 2, M6 = 3×M2 → power 3, …).
"""

from __future__ import annotations

from dataclasses import dataclass

from tideglass.marea.astronomy import ARG_ORDER, RATES


@dataclass(frozen=True)
class Constituent:
    name: str
    doodson: tuple[int, int, int, int, int, int]
    phase0: float = 0.0
    species: str = "unknown"
    nodal: str = "unity"
    f_power: float = 1.0
    u_power: float = 1.0


def speed(c: Constituent) -> float:
    """Derived angular speed of a constituent in degrees per hour."""
    return sum(d * RATES[k] for d, k in zip(c.doodson, ARG_ORDER))


CATALOG: list[Constituent] = [
    # --- Principal 8 -------------------------------------------------------
    Constituent("M2", (2, 0, 0, 0, 0, 0), 0.0, "semidiurnal", "M2"),
    Constituent("S2", (2, 2, -2, 0, 0, 0), 0.0, "semidiurnal", "unity"),
    Constituent("N2", (2, -1, 0, 1, 0, 0), 0.0, "semidiurnal", "M2"),
    Constituent("K2", (2, 2, 0, 0, 0, 0), 0.0, "semidiurnal", "K2"),
    Constituent("K1", (1, 1, 0, 0, 0, 0), -90.0, "diurnal", "K1"),
    Constituent("O1", (1, -1, 0, 0, 0, 0), 90.0, "diurnal", "O1"),
    Constituent("P1", (1, 1, -2, 0, 0, 0), 90.0, "diurnal", "unity"),
    Constituent("Q1", (1, -2, 0, 1, 0, 0), 90.0, "diurnal", "O1"),
    # --- Shallow-water / overtides -----------------------------------------
    Constituent("M4", (4, 0, 0, 0, 0, 0), 0.0, "shallow", "M2", 2.0, 2.0),
    Constituent("MS4", (4, 2, -2, 0, 0, 0), 0.0, "shallow", "M2", 1.0, 1.0),
    Constituent("MN4", (4, -1, 0, 1, 0, 0), 0.0, "shallow", "M2", 2.0, 2.0),
    Constituent("M6", (6, 0, 0, 0, 0, 0), 0.0, "shallow", "M2", 3.0, 3.0),
    Constituent("S4", (4, 4, -4, 0, 0, 0), 0.0, "shallow", "unity"),
    # --- Long-period ---------------------------------------------------------
    Constituent("Mf", (0, 2, 0, 0, 0, 0), 0.0, "long", "Mf"),
    Constituent("Mm", (0, 1, 0, -1, 0, 0), 0.0, "long", "Mm"),
    Constituent("Ssa", (0, 0, 2, 0, 0, 0), 0.0, "long", "unity"),
    Constituent("Sa", (0, 0, 1, 0, 0, 0), 0.0, "long", "unity"),
]

BY_NAME: dict[str, Constituent] = {c.name: c for c in CATALOG}

PRINCIPAL_8 = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"]


def get(name: str) -> Constituent:
    """Look up a constituent by name (raises ``KeyError`` if unknown).

    Resolves built-in catalog entries first, then any registered plugin pack
    (see :mod:`tideglass.marea.plugins`).
    """
    if name in BY_NAME:
        return BY_NAME[name]
    from tideglass.marea import plugins

    return plugins.find(name)  # raises KeyError if unknown to plugins too


def principal() -> list[Constituent]:
    """The 8 principal constituents."""
    return [BY_NAME[n] for n in PRINCIPAL_8]
