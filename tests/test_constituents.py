"""Tests for the constituent catalog: derived speeds vs published values."""

import pytest

from tideglass.marea import constituents as C

# Published mean speeds (°/h), Schureman SP-98 / NOAA tables.
PUBLISHED = {
    # Principal 8
    "M2": 28.984104,
    "S2": 30.000000,
    "N2": 28.439730,
    "K2": 30.082137,
    "K1": 15.041069,
    "O1": 13.943035,
    "P1": 14.958931,
    "Q1": 13.398741,
    # Shallow-water / overtides
    "M4": 57.968208,
    "MS4": 58.984104,
    "MN4": 57.423834,
    "M6": 86.952313,
    "S4": 60.000000,
    # Long-period
    "Mf": 1.098033,
    "Mm": 0.544374,
    "Ssa": 0.082137,
    "Sa": 0.041069,
}


@pytest.mark.parametrize("name,expected", sorted(PUBLISHED.items()))
def test_derived_speed_matches_published(name, expected):
    assert abs(C.speed(C.get(name)) - expected) < 1e-4


def test_principal_8_present():
    assert [c.name for c in C.principal()] == ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"]


def test_phase_offsets():
    assert C.get("O1").phase0 == 90.0
    assert C.get("K1").phase0 == -90.0
    assert C.get("M2").phase0 == 0.0


def test_unknown_raises():
    with pytest.raises(KeyError):
        C.get("NOPE")
