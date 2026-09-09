"""Tests for the Marea Core astronomical kernel."""

from datetime import datetime, timedelta, timezone

from tideglass.marea import astronomy as A
from tideglass.marea import constituents as C

J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_julian_day_j2000():
    assert A.julian_day(J2000) == 2451545.0


def test_mean_longitudes_keys_and_j2000_values():
    ml = A.mean_longitudes(J2000)
    assert set(ml) == {"s", "h", "p", "N", "p1"}
    # Meeus polynomial constants at J2000 (mod 360).
    assert abs(ml["s"] - 218.3164591) < 0.01
    assert abs(ml["h"] - 280.46645) < 0.01
    assert abs(ml["p"] - 83.3532430) < 0.01
    assert abs(ml["N"] - 125.0445550) < 0.01
    assert abs(ml["p1"] - (-77.06265 % 360.0)) < 0.01


def test_rates_match_expected_basis():
    # Rates are derived from full-precision polynomial coefficients, so they
    # match the rounded published table values to ~1e-8.
    for key, expected in {
        "s": 0.54901653,
        "h": 0.04106864,
        "p": 0.00464184,
        "Np": -0.00220641,
        "p1": 0.00000196,
        "tau": 14.49205211,
    }.items():
        assert abs(A.RATES[key] - expected) < 1e-7, key


def test_doodson_args_shape_and_tau_rate():
    args = A.doodson_args(J2000)
    assert len(args) == 6
    later = A.doodson_args(J2000 + timedelta(hours=1))
    dtau = (later[0] - args[0]) % 360.0
    assert abs(dtau - 14.49205211) < 1e-4


def test_naive_datetime_treated_as_utc():
    naive = datetime(2000, 1, 1, 12, 0, 0)
    assert A.doodson_args(naive) == A.doodson_args(J2000)


def test_nodal_factor_unity():
    f, u = A.nodal_factor(C.get("S2"), J2000)
    assert f == 1.0
    assert u == 0.0


def test_nodal_factor_m2_sane_and_time_varying():
    f1, u1 = A.nodal_factor(C.get("M2"), J2000)
    f2, u2 = A.nodal_factor(C.get("M2"), datetime(2005, 1, 1, tzinfo=timezone.utc))
    assert 0.7 < f1 < 1.3
    assert -180.0 <= u1 < 180.0
    # Nodal modulation over the 18.6-yr cycle must actually move.
    assert abs(f1 - f2) > 1e-4 or abs(u1 - u2) > 1e-4


def test_nodal_factor_shallow_water_composes():
    f_m2, u_m2 = A.nodal_factor(C.get("M2"), J2000)
    f_m4, u_m4 = A.nodal_factor(C.get("M4"), J2000)
    assert abs(f_m4 - f_m2**2) < 1e-12
    # u is wrapped to [-180, 180); compare modulo 360.
    assert abs(((u_m4 - 2 * u_m2 + 180.0) % 360.0) - 180.0) < 1e-9
