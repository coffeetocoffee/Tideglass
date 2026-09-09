"""Plugin constituent packs for Marea Core — v0.9 ecosystem lock-in.

The built-in :data:`tideglass.marea.constituents.CATALOG` is fixed, but real
coasts need more: freshwater estuaries carry high-order shallow-water
compounds, the Great Lakes need seiche-scale compounds, and solid-earth / load
tide needs its own species. Rather than fork the catalog, v0.9 makes the
catalog *pluggable*: third parties register extra constituents (or drop a JSON
pack into a directory) and they become first-class — resolvable by
:func:`tideglass.marea.constituents.get`, selectable as fit candidates, and
consumed by every downstream module (EOF, transfer, kriging, calibration).

Three illustrated seed packs ship with the engine:

* ``rivers`` — high-order shallow-water compounds of the principal 8 (estuaries
  where the harmonic signal is dominated by overtides).
* ``great_lakes`` — seiche-scale and compound species (closed-basin coasts).
* ``solid_earth`` — load / solid-earth-tide species (permanent + compound).

The pack constituents are defined as exact integer combinations of the verified
principal Doodson numbers, so their engine-derived speeds are physically
correct; ``nodal="unity"`` is the safe default for seed packs (a user pack can
carry the proper node-factor family). Like the harmonics database, the packs are
*illustrative* — drop authoritative local definitions in via
:func:`load_pack_file` to make them operational.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass

from tideglass.marea import constituents as CON

# Each pack constituent is an integer combination of the verified principal
# Doodson numbers, so CON.speed() reproduces the correct physical rate. Names
# are composition-based (and clash-free with the built-in catalog).
_RIVERS = [
    ("2M2", (4, 0, 0, 0, 0, 0)),     # 2×M2
    ("2S2", (4, 4, -4, 0, 0, 0)),    # 2×S2
    ("2N2", (4, -2, 0, 2, 0, 0)),    # 2×N2
    ("2O1", (2, -2, 0, 0, 0, 0)),    # 2×O1
    ("MSK6", (6, 4, -2, 0, 0, 0)),   # M2 + S2 + K2
]
_GREAT_LAKES = [
    ("2SM6", (8, 4, -4, 0, 0, 0)),   # 2×(M2 + S2)
    ("2MK6", (6, 2, 0, 0, 0, 0)),    # 2×M2 + K2
    ("MK3", (3, 1, 0, 0, 0, 0)),     # M2 + K1
    ("MO3", (3, -1, 0, 0, 0, 0)),    # M2 + O1
    ("2MN6", (6, -1, 0, 1, 0, 0)),   # 2×M2 + N2
]
_SOLID_EARTH = [
    ("SK3", (3, 3, -2, 0, 0, 0)),    # S2 + K1
    ("2MN2", (6, -1, 0, 1, 0, 0)),   # 2×M2 + N2
    ("Q1N2", (3, -3, 0, 2, 0, 0)),   # Q1 + N2
    ("2MS6", (6, 2, -2, 0, 0, 0)),   # 2×M2 + S2
]

_SPECIES = {
    "rivers": "shallow",
    "great_lakes": "compound",
    "solid_earth": "load",
}


def _make(name: str, doodson, species: str) -> CON.Constituent:
    return CON.Constituent(
        name, tuple(int(d) for d in doodson), 0.0, species, "unity", 1.0, 1.0)


@dataclass(frozen=True)
class ConstituentPack:
    """A named bundle of plugin constituents (built-in seed or user-loaded)."""

    name: str
    constituents: list[CON.Constituent]
    source: str = "builtin"

    def names(self) -> list[str]:
        return [c.name for c in self.constituents]


# name -> ConstituentPack
PLUGINS: dict[str, ConstituentPack] = {}


def _register_builtins() -> None:
    if PLUGINS:  # idempotent
        return
    for pack, entries in (
        ("rivers", _RIVERS),
        ("great_lakes", _GREAT_LAKES),
        ("solid_earth", _SOLID_EARTH),
    ):
        PLUGINS[pack] = ConstituentPack(
            pack,
            [_make(name, d, _SPECIES[pack]) for name, d in entries],
            source="builtin",
        )


_register_builtins()


# --- public API ----------------------------------------------------------------


def register_pack(name: str, constituents: Sequence[CON.Constituent],
                  source: str = "user") -> ConstituentPack:
    """Register (or replace) a constituent pack by name."""
    pack = ConstituentPack(name, list(constituents), source=source)
    PLUGINS[name] = pack
    return pack


def register_constituent(constituent: CON.Constituent, pack: str = "user") -> None:
    """Add a single constituent to a (created-on-demand) pack."""
    existing = PLUGINS.get(pack)
    if existing is None:
        PLUGINS[pack] = ConstituentPack(pack, [constituent], source="user")
    else:
        if any(c.name == constituent.name for c in existing.constituents):
            # Replace in place.
            kept = [c for c in existing.constituents if c.name != constituent.name]
            PLUGINS[pack] = ConstituentPack(
                pack, kept + [constituent], source=existing.source)
        else:
            PLUGINS[pack] = ConstituentPack(
                pack, list(existing.constituents) + [constituent],
                source=existing.source)


def list_packs() -> dict[str, list[str]]:
    """Map each registered pack name to its constituent names."""
    return {name: p.names() for name, p in PLUGINS.items()}


def find(name: str) -> CON.Constituent:
    """Resolve a constituent by name across built-in + plugin packs.

    Built-in catalog entries take precedence on a name clash. Raises KeyError if
    unknown to *both*.
    """
    if name in CON.BY_NAME:
        return CON.BY_NAME[name]
    for pack in PLUGINS.values():
        for c in pack.constituents:
            if c.name == name:
                return c
    raise KeyError(f"unknown constituent {name!r} (not in catalog or any pack)")


def all_constituents() -> list[CON.Constituent]:
    """Every available constituent: built-in catalog first, then plugins."""
    seen: set[str] = set()
    out: list[CON.Constituent] = []
    for c in CON.CATALOG:
        out.append(c)
        seen.add(c.name)
    for pack in PLUGINS.values():
        for c in pack.constituents:
            if c.name not in seen:
                out.append(c)
                seen.add(c.name)
    return out


def load_pack_file(path: str, name: str | None = None,
                   pack: str | None = None) -> ConstituentPack:
    """Load a constituent pack from JSON and register it.

    Accepts either a bare list of constituent dicts, or an object
    ``{"name": ..., "constituents": [...]}``. Each constituent dict:
    ``{"name", "doodson": [int×6], "phase0"?, "species"?, "nodal"?, "f_power"?,
    "u_power"?}``.
    """
    with open(path) as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        consts = data.get("constituents", [])
        pack_name = name or data.get("name") or os.path.splitext(
            os.path.basename(path))[0]
    else:
        consts = data
        pack_name = name or os.path.splitext(os.path.basename(path))[0]
    objs = [
        CON.Constituent(
            c["name"], tuple(int(d) for d in c["doodson"]),
            float(c.get("phase0", 0.0)),
            c.get("species", "plugin"),
            c.get("nodal", "unity"),
            float(c.get("f_power", 1.0)),
            float(c.get("u_power", 1.0)),
        )
        for c in consts
    ]
    return register_pack(pack_name if pack is None else pack, objs, source=path)


def load_pack_dir(directory: str) -> dict[str, ConstituentPack]:
    """Load every ``*.json`` constituent pack in a directory."""
    out: dict[str, ConstituentPack] = {}
    for fn in sorted(os.listdir(directory)):
        if fn.endswith(".json"):
            p = os.path.join(directory, fn)
            if os.path.isfile(p):
                try:
                    out[os.path.splitext(fn)[0]] = load_pack_file(p)
                except (OSError, ValueError, KeyError):
                    continue
    return out
