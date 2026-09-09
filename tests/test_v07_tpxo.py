"""Tests for genuine TPXO/FES ingestion (v0.7.1).

NetCDF3 fixtures are written in-test by a minimal writer below — no licensed
TPXO/FES bytes anywhere in the repository (OSU/AVISO forbid redistribution).
"""

import math
import struct
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tideglass.cli import main as cli_main
from tideglass.marea import constituents as CON
from tideglass.marea import harmonics_db as HDB
from tideglass.marea.solver import rad_per_hour
from tideglass.marea.tpxo import (
    format_self_check,
    native_to_model,
    read_tpxo,
    self_check_tpxo,
    tpxo_model_at,
)

UTC = timezone.utc

# TPXO-style native constants for the fixtures: {CODE: (amp_m, greenwich_deg)}.
FIX_CONSTS = {"M2": (0.50, 120.0), "S2": (0.14, 170.0), "K1": (0.16, 150.0)}
FIX_LONS = [-122.75, -122.50, -122.25]
FIX_LATS = [37.60, 37.80, 38.00]


# --- minimal NetCDF3-classic writer (test fixtures only) -----------------------

_TYPE_SIZE = {1: 1, 3: 2, 5: 4, 6: 8}
_TYPE_FMT = {1: ">b", 3: ">h", 5: ">f", 6: ">d"}


def _pname(b: bytes) -> bytes:
    return struct.pack(">i", len(b)) + b + b"\x00" * ((4 - len(b) % 4) % 4)


def _write_tpxo_nc(path, lons, lats, consts, mask=None, int16_phase=(),
                    weird_names=False, gatt=True):
    """Write a TPXO-shaped NetCDF3 file: dims nx/ny, lon_z/lat_z, h_am/h_pu."""
    nx, ny = len(lons), len(lats)
    specs = [("lon_z", [0], 6, {}, np.asarray(lons, float)),
             ("lat_z", [1], 6, {}, np.asarray(lats, float))]
    for code, (amp, ph) in consts.items():
        c = code.lower()
        an = f"amp{c}x" if weird_names else f"h_am_{c}"
        pn = f"pha{c}x" if weird_names else f"h_pu_{c}"
        a = np.full((ny, nx), amp)
        p = np.full((ny, nx), ph)
        specs.append((an, [1, 0], 6, {}, a))
        if code in int16_phase:
            specs.append((pn, [1, 0], 3, {"scale_factor": (5, [0.01])},
                          np.round(p / 0.01).astype(np.int16)))
        else:
            specs.append((pn, [1, 0], 6, {}, p))
    if mask is not None:
        specs.append(("mask_z", [1, 0], 1, {}, np.asarray(mask, np.int8)))

    hdr = bytearray(b"CDF\x01") + struct.pack(">i", 0)  # numrecs
    hdr += struct.pack(">i", 0x0A) + struct.pack(">i", 2)
    hdr += _pname(b"nx") + struct.pack(">i", nx)
    hdr += _pname(b"ny") + struct.pack(">i", ny)
    hdr += struct.pack(">i", 0x0C) + struct.pack(">i", 1 if gatt else 0)
    if gatt:
        hdr += _pname(b"Conventions") + struct.pack(">i", 2)
        val = b"CF-1.6"
        hdr += struct.pack(">i", len(val)) + val + b"\x00" * ((4 - len(val) % 4) % 4)
    hdr += struct.pack(">i", 0x0B) + struct.pack(">i", len(specs))

    regions = []
    var_entries = []
    for vname, dimids, vtype, attrs, arr in specs:
        e = _pname(vname.encode())
        e += struct.pack(">i", len(dimids)) + b"".join(struct.pack(">i", d) for d in dimids)
        e += struct.pack(">i", 0x0C) + struct.pack(">i", len(attrs))
        for an, (at, vals) in attrs.items():
            e += _pname(an.encode()) + struct.pack(">i", at) + struct.pack(">i", len(vals))
            for v in vals:
                e += struct.pack(_TYPE_FMT[at], v)
            nbytes = _TYPE_SIZE[at] * len(vals)
            e += b"\x00" * ((4 - nbytes % 4) % 4)
        e += struct.pack(">i", vtype)
        nels = int(np.prod([nx if d == 0 else ny for d in dimids])) if dimids else 1
        e += struct.pack(">i", nels * _TYPE_SIZE[vtype])
        e += struct.pack(">i", 0)  # begin placeholder (patched in pass 2)
        var_entries.append(e)
        regions.append((vtype, arr.ravel()))
    # Header length determines data begins.
    hlen = len(hdr) + sum(len(e) for e in var_entries)
    hlen += (4 - hlen % 4) % 4
    # Data begins follow in var order right after the padded header.
    begins = []
    # NOTE: header entries sit before `hlen`; data starts at hlen.
    data = bytearray()
    for (vtype, flat) in regions:
        begins.append(hlen + len(data))
        for v in flat:
            data += struct.pack(_TYPE_FMT[vtype], v)
        while len(data) % 4:
            data += b"\x00"
    # Rewrite header with correct begins (entries are fixed-size records here
    # except names/attrs, so rebuild fully instead of patching).
    hdr2 = bytearray(b"CDF\x01") + struct.pack(">i", 0)
    hdr2 += struct.pack(">i", 0x0A) + struct.pack(">i", 2)
    hdr2 += _pname(b"nx") + struct.pack(">i", nx)
    hdr2 += _pname(b"ny") + struct.pack(">i", ny)
    hdr2 += struct.pack(">i", 0x0C) + struct.pack(">i", 1 if gatt else 0)
    if gatt:
        hdr2 += _pname(b"Conventions") + struct.pack(">i", 2)
        val = b"CF-1.6"
        hdr2 += struct.pack(">i", len(val)) + val + b"\x00" * ((4 - len(val) % 4) % 4)
    hdr2 += struct.pack(">i", 0x0B) + struct.pack(">i", len(specs))
    for (vname, dimids, vtype, attrs, _arr), begin in zip(specs, begins):
        hdr2 += _pname(vname.encode())
        hdr2 += struct.pack(">i", len(dimids)) + b"".join(struct.pack(">i", d) for d in dimids)
        hdr2 += struct.pack(">i", 0x0C) + struct.pack(">i", len(attrs))
        for an, (at, vals) in attrs.items():
            hdr2 += _pname(an.encode()) + struct.pack(">i", at) + struct.pack(">i", len(vals))
            for v in vals:
                hdr2 += struct.pack(_TYPE_FMT[at], v)
            nbytes = _TYPE_SIZE[at] * len(vals)
            hdr2 += b"\x00" * ((4 - nbytes % 4) % 4)
        hdr2 += struct.pack(">i", vtype)
        nels = int(np.prod([nx if d == 0 else ny for d in dimids])) if dimids else 1
        hdr2 += struct.pack(">i", nels * _TYPE_SIZE[vtype])
        hdr2 += struct.pack(">i", begin)
    while len(hdr2) % 4:
        hdr2 += b"\x00"
    assert len(hdr2) == hlen, (len(hdr2), hlen)
    with open(path, "wb") as fh:
        fh.write(bytes(hdr2) + bytes(data))


def _fixture(path, **kw):
    amps = {c: v[0] for c, v in FIX_CONSTS.items()}
    _write_tpxo_nc(path, FIX_LONS, FIX_LATS,
                    {c: (np.full((3, 3), a), np.full((3, 3), p))
                     for c, (a, p) in FIX_CONSTS.items()}, **kw)
    return str(path)


# --- reader --------------------------------------------------------------------

def test_reader_parses_layout_and_bbox(tmp_path):
    p = _fixture(tmp_path / "t.nc")
    full = read_tpxo(p, constituents=["M2", "S2", "K1"])
    assert len(full["points"]) == 9
    assert full["constituents"] == ["K1", "M2", "S2"]
    assert full["missing"] == []
    assert all(full["water"].values())
    sub = read_tpxo(p, bbox=(-122.6, -122.4, 37.7, 37.9))
    assert len(sub["points"]) == 1
    key = next(iter(sub["points"]))
    assert key == (-122.5, 37.8)
    assert sub["points"][key]["M2"] == (0.50, 120.0)


def test_reader_wraps_lon_to_pm180(tmp_path):
    p = tmp_path / "w.nc"
    _write_tpxo_nc(str(p), [237.5, 238.0], [37.8],
                    {"M2": (np.full((1, 2), 0.5), np.full((1, 2), 120.0))})
    grid = read_tpxo(str(p))
    assert set(grid["points"]) == {(-122.5, 37.8), (-122.0, 37.8)}


def test_reader_reports_missing_constituents(tmp_path):
    p = _fixture(tmp_path / "t.nc")
    grid = read_tpxo(p, constituents=["M2", "XYZ"])
    assert grid["missing"] == ["XYZ"]
    assert "M2" in grid["constituents"]


def test_reader_var_map_override(tmp_path):
    p = _fixture(tmp_path / "t.nc", weird_names=True)
    # Default discovery finds nothing TPXO-shaped; the explicit map does.
    grid = read_tpxo(p, constituents=["M2"],
                     var_map={"M2": ("ampm2x", "pham2x")})
    assert grid["constituents"] == ["M2"]
    assert grid["missing"] == []


def test_reader_rejects_non_netcdf(tmp_path):
    p = tmp_path / "x.nc"
    p.write_text("not a netcdf file")
    with pytest.raises(ValueError):
        read_tpxo(str(p))


def test_reader_applies_scale_factor(tmp_path):
    p = _fixture(tmp_path / "t.nc", int16_phase={"M2"})
    grid = read_tpxo(p, constituents=["M2"])
    pt = next(iter(grid["points"].values()))
    assert pt["M2"][1] == pytest.approx(120.0, abs=0.02)


# --- native -> kappa conversion --------------------------------------------------

def _native_series(native, n, code_order=None):
    hvec = np.arange(n, dtype=float)
    y = np.zeros(n)
    for code, (amp, ph) in native.items():
        om = float(rad_per_hour(CON.speed(CON.get(code))))
        y = y + amp * np.cos(om * hvec - math.radians(ph))
    return y


def test_native_roundtrip_is_exact(tmp_path):
    from tideglass.marea.astronomy import doodson_args, nodal_factor

    p = _fixture(tmp_path / "t.nc")
    grid = read_tpxo(p)
    native = next(iter(grid["points"].values()))
    model = native_to_model(native, station="T", fit_days=60)
    assert model.meta["tpxo_skipped"] == []
    assert model.meta["source"] == "tpxo"  # provenance (object tag stays "fit")
    n = 60 * 24
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    times = [t0 + timedelta(hours=h) for h in range(n)]
    pred = model.predict(times).mean
    # Nodal modulation is absorbed by the fit: ~1e-3 on metre-scale signals.
    assert np.max(np.abs(pred - _native_series(native, n))) < 5e-3
    # But the phase-convention alignment itself is strict: fitted kappa must
    # equal the equilibrium argument at the fit epoch plus native g.
    args = doodson_args(t0)
    fits = {f.name: f for f in model.constituents()}
    for code, (_amp, g) in native.items():
        c = CON.get(code)
        V = (sum(n_ * a for n_, a in zip(c.doodson, args)) + c.phase0) % 360.0
        _f, u = nodal_factor(c, t0)
        expect = (V + u + g) % 360.0
        d = (fits[code].phase_deg - expect) % 360.0
        assert min(d, 360.0 - d) < 2.0, (code, fits[code].phase_deg, expect)


def test_native_skips_unknown_catalog_entries():
    native = {"M2": (0.5, 120.0), "XX": (0.1, 10.0)}
    model = native_to_model(native, fit_days=30)
    assert model.meta["tpxo_skipped"] == ["XX"]
    assert {f.name for f in model.constituents()} == {"M2"}


def test_native_empty_raises():
    with pytest.raises(ValueError):
        native_to_model({})


def test_tpxo_model_at_picks_nearest_water(tmp_path):
    mask = np.ones((3, 3), np.int8)
    mask[1, 1] = 0  # land at (-122.50, 37.80)
    p = _fixture(tmp_path / "t.nc", mask=mask)
    # Gauge sits exactly on the land point: must fall back to nearest water.
    model = tpxo_model_at(str(p), -122.50, 37.80, fit_days=30)
    assert model.meta["tpxo_point"] != (-122.5, 37.8)
    assert model.meta["tpxo_file"] == "t.nc"
    assert model.meta["source"].startswith("tpxo:")
    assert {f.name for f in model.constituents()} == {"M2", "S2", "K1"}


# --- self-check ------------------------------------------------------------------

def _publish_from_fixture(monkeypatch, tmp_path, factor=1.0):
    p = _fixture(tmp_path / "t.nc")
    grid = read_tpxo(str(p))
    native = next(iter(grid["points"].values()))
    model = native_to_model(native, fit_days=30)
    entries = [(f.name, f.amplitude * factor, f.phase_deg)
               for f in model.constituents()]
    monkeypatch.setitem(HDB.PUBLIC_HARMONICS, "Test Region", [{
        "name": "Testport", "lat": 37.8, "lon": -122.5, "mean": 0.0,
        "constituents": entries}])
    return str(p)


def test_self_check_passes_on_matching_station(tmp_path, monkeypatch, capsys):
    p = _publish_from_fixture(monkeypatch, tmp_path)
    rep = self_check_tpxo(p, -122.5, 37.8, "Test Region", "Testport")
    assert rep["n_shared"] == 3
    assert rep["pass"] is True
    out = format_self_check(rep)
    assert "PASS" in out and "Testport" in out and "M2" in out


def test_self_check_fails_on_divergent_station(tmp_path, monkeypatch):
    p = _publish_from_fixture(monkeypatch, tmp_path, factor=3.0)
    rep = self_check_tpxo(p, -122.5, 37.8, "Test Region", "Testport")
    assert rep["pass"] is False
    assert all(r["damp"] is not None and r["damp"] > 0.05 for r in rep["rows"])
    assert "FAIL" in format_self_check(rep)


def test_self_check_unknown_station_raises(tmp_path):
    p = _fixture(tmp_path / "t.nc")
    with pytest.raises(ValueError):
        self_check_tpxo(p, -122.5, 37.8, "No Such Region", "Nope")


# --- CLI .nc dispatch ---------------------------------------------------------------

def _gauge_csv(path, days=40, seed=21):
    from tideglass.marea import constituents as C

    TRUE = {"M2": (1.0, 0.5), "S2": (0.30, -1.2), "N2": (0.20, 2.0),
            "K1": (0.15, 0.0), "O1": (0.10, 2.8)}
    T0 = datetime(2024, 1, 1, tzinfo=UTC)
    times = [T0 + timedelta(hours=h) for h in range(days * 24)]
    t = np.arange(len(times), dtype=float)
    y = np.full_like(t, 0.7)
    for n, (a, ph) in TRUE.items():
        w = float(rad_per_hour(C.speed(C.get(n))))
        y = y + a * np.cos(w * t - ph)
    y = y + np.random.default_rng(seed).normal(0.0, 0.02, size=t.size)
    with open(path, "w") as fh:
        fh.write("time,height\n")
        fh.writelines("%s,%.4f\n" % (ti.isoformat(), hi) for ti, hi in zip(times, y))
    return str(path)


def test_cli_bench_nc_end_to_end(tmp_path, capsys):
    csv = _gauge_csv(tmp_path / "gauge.csv")
    nc = _fixture(tmp_path / "g.nc")
    rc = cli_main(["bench", csv, "--station", "SYN", "--against", "tpxo",
                   "--global-model", nc, "--lon", "-122.5", "--lat", "37.8",
                   "--consts", "M2,S2,K1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "global" in out and "marea/global rmse ratio" in out
    assert "winner (rmse)" in out


def test_cli_bench_nc_missing_degrades(tmp_path, capsys):
    csv = _gauge_csv(tmp_path / "gauge.csv", days=10)
    rc = cli_main(["bench", csv, "--against", "tpxo", "--global-model",
                   str(tmp_path / "missing.nc"), "--lon", "-122.5",
                   "--lat", "37.8"])
    assert rc == 0
    assert "skipped" in capsys.readouterr().out
