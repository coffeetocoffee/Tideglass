"""Genuine TPXO/FES ingestion — closes the v0.7 benchmark caveat.

`marea/bench_global.py` compares Marea Core against global hydrodynamic models,
but only from a hand-written CSV schema. The genuine models — TPXO (Oregon
State University) and FES (LEGOS/CNES, via AVISO) — publish harmonic constants
(amplitude ``H`` + Greenwich phase ``g``) on regular lat/lon grids inside
NetCDF elevation files (``h_tpxo9.v1.nc`` and friends). This module reads those
files with a **dependency-free NetCDF3-classic parser** (big-endian XDR, NumPy
only — the same zero-dependency stance as :mod:`tideglass.marea.export`) and
converts the native ``H·cos(ωt − g)`` constants into Tideglass convention.

**Licensing (load-bearing):** TPXO and FES files are free for research use but
*forbid redistribution* — they can never ship in this repository. Operators
register at https://www.tpxo.net (TPXO) or https://www.aviso.altimetry.fr
(FES), download the elevation file themselves, and point
``tideglass bench --against tpxo --global-model h_tpxo9.v1.nc`` at it.
``data/tpxo_sample.csv`` remains the license-clean stand-in for CI and demos.

**Phase-convention conversion without formula risk:** instead of converting
Greenwich phases to NOAA kappa (κ) by sign/epoch algebra, we predict the model
*in its native convention* at N sample times and re-fit that series with the
exact SVD solver (:meth:`~tideglass.marea.model.TideModel.fit`, fixed
constituent set). The phase alignment is exact by construction; the round-trip
reproduces the native series up to the slow nodal modulation the Tideglass
basis carries (~1e-3 on metre-scale signals) — the correct mean-epoch behavior
for a global product. Ordinary ``predict`` then works at any epoch through the
dynamic Doodson kernel.

Only harmonic math is touched; no network calls.
"""

from __future__ import annotations

import math
import os
import struct
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

import numpy as np

from tideglass.marea import constituents as CON
from tideglass.marea import harmonics_db as HDB
from tideglass.marea.model import TideModel
from tideglass.marea.solver import rad_per_hour

# --- NetCDF3-classic reader (general; export.py only handles its own files) ---

_MAGIC = b"CDF\x01"
_NC_DIMENSION = 0x0A
_NC_VARIABLE = 0x0B
_NC_ATTRIBUTE = 0x0C
# ntype -> (struct fmt, itemsize). NC_CHAR (2) is handled separately.
_NC_TYPES = {1: (">b", 1), 3: (">h", 2), 4: (">i", 4), 5: (">f", 4), 6: (">d", 8)}


class _Cursor:
    def __init__(self, data: bytes):
        self.d = data
        self.off = 0

    def i4(self) -> int:
        v = struct.unpack(">i", self.d[self.off:self.off + 4])[0]
        self.off += 4
        return v

    def name(self) -> str:
        n = self.i4()
        s = self.d[self.off:self.off + n].decode("utf-8", "replace")
        self.off += n + (4 - n % 4) % 4  # counted strings pad to 4 bytes
        return s

    def attr_vals(self, ntype: int, nvals: int):
        if ntype == 2:  # NC_CHAR
            raw = self.d[self.off:self.off + nvals]
            self.off += nvals + (4 - nvals % 4) % 4
            return raw.split(b"\x00")[0].decode("utf-8", "replace")
        fmt, size = _NC_TYPES[ntype]
        vals = [
            struct.unpack(fmt, self.d[self.off + i * size:self.off + (i + 1) * size])[0]
            for i in range(nvals)
        ]
        self.off += nvals * size + (4 - (nvals * size) % 4) % 4
        return vals


def _read_nc_header(path: str) -> tuple[list, dict, list]:
    """Parse a NetCDF3-classic header.

    Returns ``(dims, gattrs, variables)`` where ``dims`` is ``[(name, length)]``
    (unlimited dim has length -1), ``gattrs`` maps names to values, and each
    variable is ``{"name", "dimids", "attrs" (name -> value), "type", "begin"}``.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:4] != _MAGIC:
        raise ValueError(f"{path!r} is not a NetCDF3 classic file "
                         f"(magic {data[:4]!r}; NetCDF4/HDF5 needs h5py)")
    cur = _Cursor(data)
    cur.off = 4
    _numrecs = cur.i4()
    if cur.i4() != _NC_DIMENSION:
        raise ValueError(f"{path!r}: bad dimension section")
    dims = [(cur.name(), cur.i4()) for _ in range(cur.i4())]
    if cur.i4() != _NC_ATTRIBUTE:
        raise ValueError(f"{path!r}: bad global-attribute section")

    def _attrs(n: int) -> dict:
        out = {}
        for _ in range(n):
            nm, tp, nv = cur.name(), cur.i4(), cur.i4()
            out[nm] = cur.attr_vals(tp, nv)
        return out

    gattrs = _attrs(cur.i4())
    if cur.i4() != _NC_VARIABLE:
        raise ValueError(f"{path!r}: bad variable section")
    variables = []
    for _ in range(cur.i4()):
        nm = cur.name()
        dimids = [cur.i4() for _ in range(cur.i4())]
        if cur.i4() != _NC_ATTRIBUTE:
            raise ValueError(f"{path!r}: bad variable-attribute section")
        attrs = _attrs(cur.i4())
        vtype, _vsize, begin = cur.i4(), cur.i4(), cur.i4()
        variables.append({"name": nm, "dimids": dimids, "attrs": attrs,
                          "type": vtype, "begin": begin})
    unlim = next((i for i, (_, ln) in enumerate(dims) if ln == -1), None)
    for v in variables:
        if unlim is not None and unlim in v["dimids"]:
            raise ValueError(f"{path!r}: record (unlimited-dimension) variable "
                             f"{v['name']!r} is unsupported — TPXO/FES elevation "
                             f"files are fixed-grid")
    return dims, gattrs, variables, data


def _read_nc_var(data: bytes, var: dict, dims: list) -> np.ndarray:
    """Read one fixed-grid variable (applies scale_factor/add_offset/_FillValue)."""
    shape = tuple(dims[d][1] for d in var["dimids"])
    count = int(np.prod(shape, dtype=np.int64)) if shape else 1
    ntype = var["type"]
    begin = var["begin"]
    if ntype == 2:
        raw = data[begin:begin + count]
        return np.frombuffer(raw, dtype="S1")
    fmt, size = _NC_TYPES[ntype]
    buf = data[begin:begin + count * size]
    if len(buf) < count * size:
        raise ValueError(f"variable {var['name']!r} overruns the file")
    raw = np.frombuffer(buf, dtype=np.dtype(fmt)).astype(float).reshape(shape or (1,))
    fill = None
    for key in ("_FillValue", "missing_value"):
        if key in var["attrs"]:
            try:
                fill = float(np.asarray(var["attrs"][key]).ravel()[0])
            except (TypeError, ValueError):
                pass
    arr = raw.copy()
    if fill is not None and not math.isnan(fill):
        arr[raw == fill] = np.nan
    if "scale_factor" in var["attrs"]:
        arr = arr * float(np.asarray(var["attrs"]["scale_factor"]).ravel()[0])
    if "add_offset" in var["attrs"]:
        arr = arr + float(np.asarray(var["attrs"]["add_offset"]).ravel()[0])
    return arr


# --- TPXO/FES layout discovery -------------------------------------------------

_LON_NAMES = ("lon_z", "lon", "longitude", "nav_lon", "glon", "lons", "x")
_LAT_NAMES = ("lat_z", "lat", "latitude", "nav_lat", "glat", "lats", "y")
_MASK_NAMES = ("mask_z", "mask_t", "mask", "wet_mask", "land_mask", "masks")


def _canon_const(name: str) -> str:
    """Normalize a constituent code to catalog spelling (``m2`` -> ``M2``)."""
    s = str(name).strip().lower()
    return {"mf": "Mf", "mm": "Mm"}.get(s, s.upper())


def _amp_candidates(code: str) -> list[str]:
    c = code.lower()
    return [f"h_am_{c}", f"ham_{c}", f"amp_{c}", f"amplitude_{c}",
            f"h_{c}", f"{c}_am", f"{c}_amp"]


def _phase_candidates(code: str) -> list[str]:
    c = code.lower()
    return [f"h_pu_{c}", f"hph_{c}", f"pha_{c}", f"phase_{c}", f"g_{c}",
            f"ph_{c}", f"{c}_ph", f"{c}_pu", f"greenwich_{c}"]


def _pick(names: set[str], candidates: Sequence[str]) -> str | None:
    for cand in candidates:
        if cand in names:
            return cand
    return None


def _wrap_lon(lon: np.ndarray) -> np.ndarray:
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0


def _haversine_km(lon1, lat1, lon2, lat2) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (lon1, lat1, lon2, lat2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2.0 * math.asin(min(1.0, math.sqrt(a)))


# --- Public API -----------------------------------------------------------------

DEFAULT_CONSTS = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"]


def read_tpxo(
    path: str,
    bbox: tuple[float, float, float, float] | None = None,
    constituents: Sequence[str] | None = None,
    var_map: dict[str, tuple[str, str]] | None = None,
    mask_water_nonzero: bool = True,
) -> dict[str, Any]:
    """Read a TPXO/FES-style NetCDF3 elevation file into a harmonic grid.

    :param path: ``h_tpxo9.v1.nc`` (or FES equivalent) — NetCDF3 classic only.
    :param bbox: ``(lon0, lon1, lat0, lat1)`` in -180..180° (antimeridian-safe);
        ``None`` reads the whole file.
    :param constituents: codes to extract (default: the principal 8).
        Missing variables are reported in ``"missing"``, not fatal.
    :param var_map: explicit ``{CODE: (amp_var, phase_var)}`` override when the
        file's naming differs from the TPXO9 ``h_am_<c>`` / ``h_pu_<c>`` default.
    :returns: ``{"points": {(lon, lat): {CODE: (amp_m, greenwich_phase_deg)}},
        "water": {(lon, lat): bool}, "constituents": [...], "missing": [...],
        "source": path}``. Longitudes are wrapped to -180..180°.
    """
    dims, _gattrs, variables, data = _read_nc_header(path)
    by_name = {v["name"].lower(): v for v in variables}
    names = set(by_name)

    def _get(varname: str) -> np.ndarray:
        return _read_nc_var(data, by_name[varname.lower()], dims)

    lon_var = _pick(names, _LON_NAMES)
    lat_var = _pick(names, _LAT_NAMES)
    if lon_var is None or lat_var is None:
        raise ValueError(f"{path!r}: no lon/lat variables found "
                         f"(have: {sorted(names)})")
    lon1 = np.asarray(_get(lon_var), dtype=float).ravel()
    lat1 = np.asarray(_get(lat_var), dtype=float).ravel()
    mask_var = _pick(names, _MASK_NAMES)

    codes = [_canon_const(c) for c in (constituents or DEFAULT_CONSTS)]
    var_map = {str(k).strip().upper(): v for k, v in (var_map or {}).items()}
    pairs: dict[str, tuple[str, str]] = {}
    missing: list[str] = []
    for code in codes:
        if code.upper() in var_map:
            pairs[code] = var_map[code.upper()]
            continue
        a = _pick(names, _amp_candidates(code))
        p = _pick(names, _phase_candidates(code))
        if a is None or p is None:
            missing.append(code)
        else:
            pairs[code] = (a, p)

    # Assemble the 2-D fields. Regular grids expose 1-D axes; curvilinear grids
    # expose 2-D lon/lat (used as-is); flat fallbacks are reshaped to (ny, nx).
    ny = nx = None
    fields: dict[str, np.ndarray] = {}
    for code, (av, pv) in pairs.items():
        for vn in (av, pv):
            if vn not in fields:
                arr = np.asarray(_get(vn), dtype=float)
                fields[vn] = arr
                if arr.ndim == 2:
                    ny, nx = arr.shape
    if ny is None:  # degenerate single-point file
        ny = nx = 1
        fields = {k: np.asarray(v, dtype=float).reshape(1, 1)
                  for k, v in fields.items()}
    else:
        for k, v in fields.items():
            if v.ndim == 1 and v.size == ny * nx:
                fields[k] = v.reshape(ny, nx)
            elif v.ndim != 2:
                raise ValueError(f"{path!r}: variable {k!r} has unexpected "
                                 f"shape {v.shape}")

    if lon1.size == nx and lat1.size == ny:
        lon2, lat2 = np.meshgrid(lon1, lat1)
    elif lon1.size == ny * nx and lat1.size == ny * nx:
        lon2, lat2 = lon1.reshape(ny, nx), lat1.reshape(ny, nx)
    else:
        raise ValueError(f"{path!r}: lon/lat axes {lon1.size}/{lat1.size} do not "
                         f"match field shape ({ny}, {nx})")
    lon2 = _wrap_lon(lon2)

    if mask_var is not None:
        m = np.asarray(_get(mask_var), dtype=float)
        m = m.reshape(ny, nx) if m.ndim == 1 and m.size == ny * nx else m
        water = (m != 0) if mask_water_nonzero else (m == 0)
        water = np.asarray(water, dtype=bool).reshape(ny, nx)
    else:
        water = np.ones((ny, nx), dtype=bool)

    if bbox is not None:
        lon0, lon1b, lat0, lat1b = bbox
        if lon0 <= lon1b:
            keep = (lon2 >= lon0) & (lon2 <= lon1b)
        else:  # antimeridian-crossing window
            keep = (lon2 >= lon0) | (lon2 <= lon1b)
        keep &= (lat2 >= lat0) & (lat2 <= lat1b)
    else:
        keep = np.ones((ny, nx), dtype=bool)

    points: dict[tuple[float, float], dict[str, tuple[float, float]]] = {}
    wet: dict[tuple[float, float], bool] = {}
    for j in range(ny):
        for i in range(nx):
            if not keep[j, i]:
                continue
            key = (round(float(lon2[j, i]), 6), round(float(lat2[j, i]), 6))
            cell: dict[str, tuple[float, float]] = {}
            for code, (av, pv) in pairs.items():
                a, p = float(fields[av][j, i]), float(fields[pv][j, i])
                if math.isnan(a) or math.isnan(p):
                    continue
                cell[code] = (a, p)
            if cell:
                points[key] = cell
                wet[key] = bool(water[j, i])
    if not points:
        raise ValueError(f"{path!r}: no grid points in bbox {bbox}")
    return {"points": points, "water": wet, "constituents": sorted(pairs),
            "missing": missing, "source": path, "grid_shape": (ny, nx)}


def native_to_model(
    native: dict[str, tuple[float, float]],
    station: str | None = None,
    source: str | None = None,
    fit_days: int = 370,
    fit_epoch: datetime | None = None,
) -> TideModel:
    """Convert native ``{CODE: (amp_m, greenwich_phase_deg)}`` to a TideModel.

    Predicts the model *in its native convention* — ``Σ A·cos(ωt − g)`` — over
    a ``fit_days``-long hourly window and re-fits that series with the exact
    solver on the fixed constituent set. The phase alignment is exact by
    construction (no sign/epoch algebra to get wrong); the round-trip
    reproduces the native series up to absorbed nodal modulation (~1e-3).

    Constituents outside the Marea catalog are skipped and recorded in
    ``meta["tpxo_skipped"]``.
    """
    fit_epoch = fit_epoch or datetime(2024, 1, 1, tzinfo=timezone.utc)
    candidates: list = []
    skipped: list[str] = []
    amps: dict[str, float] = {}
    phases: dict[str, float] = {}
    for raw, (amp, ph) in native.items():
        code = _canon_const(raw)
        try:
            candidates.append(CON.get(code))
        except KeyError:
            skipped.append(code)
            continue
        amps[code] = float(amp)
        phases[code] = float(ph)
    if not candidates:
        raise ValueError(f"no catalogued constituents in {sorted(native)} "
                         f"(skipped: {skipped})")
    n = fit_days * 24
    times = [fit_epoch + _td(hours=h) for h in range(n)]
    hvec = np.arange(n, dtype=float)
    y = np.zeros(n)
    for c in candidates:
        om = float(rad_per_hour(CON.speed(c)))
        g = math.radians(phases[c.name])
        y = y + amps[c.name] * np.cos(om * hvec - g)
    model = TideModel.fit(times, y, auto_select=False, candidates=candidates,
                          station=station, source=source or "tpxo")
    model.meta["tpxo_skipped"] = skipped
    return model


def _td(**kw):
    from datetime import timedelta

    return timedelta(**kw)


def tpxo_model_at(
    path: str,
    lon: float,
    lat: float,
    constituents: Sequence[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    fit_days: int = 370,
    station: str | None = None,
    var_map: dict[str, tuple[str, str]] | None = None,
) -> TideModel:
    """Build a convention-aligned :class:`TideModel` for a gauge from a TPXO file.

    Takes the nearest *water* grid point (land-masked points are skipped) and
    converts it via :func:`native_to_model`. The chosen point and file are
    pinned in ``meta`` (``tpxo_point`` / ``tpxo_file``).
    """
    if bbox is None:
        bbox = (lon - 2.5, lon + 2.5, max(lat - 2.5, -90.0), min(lat + 2.5, 90.0))
    grid = read_tpxo(path, bbox=bbox, constituents=constituents, var_map=var_map)
    wet = [pt for pt, w in grid["water"].items() if w]
    if not wet:
        raise ValueError(f"{path!r}: no water points near ({lon}, {lat}) — "
                         f"widen the window or check the land mask")
    best = min(wet, key=lambda p: _haversine_km(lon, lat, p[0], p[1]))
    model = native_to_model(
        grid["points"][best],
        station=station or f"tpxo@{lon:.2f},{lat:.2f}",
        source=f"tpxo:{os.path.basename(path)}",
        fit_days=fit_days,
    )
    model.meta["tpxo_point"] = best
    model.meta["tpxo_file"] = os.path.basename(path)
    return model


def self_check_tpxo(
    path: str,
    lon: float,
    lat: float,
    region: str,
    station: str,
    constituents: Sequence[str] | None = None,
    tol_amp_m: float = 0.05,
    tol_phase_deg: float = 5.0,
    **kw,
) -> dict[str, Any]:
    """Validate the TPXO pipeline against NOAA published constants.

    Converts the TPXO file at the gauge and compares per-constituent amplitude
    and (circular) phase against ``harmonics_db.load_station(region, station)``.
    TPXO assimilates tide gauges, so at a well-observed port the two should
    agree within a few cm / a few degrees — if they don't, either the file's
    naming/layout differs (see ``var_map``) or the download is corrupt, *not*
    the engine. Returns ``{"pass", "rows", "n_shared", ...}``; render with
    :func:`format_self_check`.
    """
    try:
        pub = HDB.load_station(region, station)
    except KeyError as exc:
        raise ValueError(f"unknown published station {station!r} in region "
                         f"{region!r} (regions: {HDB.list_regions()})") from exc
    conv = tpxo_model_at(path, lon, lat, constituents=constituents,
                         station=f"tpxo-check:{station}", **kw)
    pub_f = {f.name: f for f in pub.constituents()}
    conv_f = {f.name: f for f in conv.constituents()}
    rows = []
    for name in pub_f:
        if name not in conv_f:
            rows.append({"constituent": name, "tpxo_amp": None,
                         "pub_amp": pub_f[name].amplitude, "damp": None,
                         "tpxo_phase": None, "pub_phase": pub_f[name].phase_deg,
                         "dphase": None, "ok": False, "note": "missing in TPXO"})
            continue
        damp = abs(conv_f[name].amplitude - pub_f[name].amplitude)
        dphase = (conv_f[name].phase_deg - pub_f[name].phase_deg + 180.0) % 360.0 - 180.0
        ok = damp <= tol_amp_m and abs(dphase) <= tol_phase_deg
        rows.append({"constituent": name,
                     "tpxo_amp": conv_f[name].amplitude,
                     "pub_amp": pub_f[name].amplitude, "damp": damp,
                     "tpxo_phase": conv_f[name].phase_deg,
                     "pub_phase": pub_f[name].phase_deg, "dphase": dphase,
                     "ok": ok, "note": ""})
    shared = [r for r in rows if r["tpxo_amp"] is not None]
    return {"station": station, "region": region,
            "point": conv.meta.get("tpxo_point"), "file": conv.meta.get("tpxo_file"),
            "n_shared": len(shared), "rows": rows,
            "tol_amp_m": tol_amp_m, "tol_phase_deg": tol_phase_deg,
            "pass": bool(shared) and all(r["ok"] for r in rows)}


def format_self_check(rep: dict[str, Any]) -> str:
    """Render a :func:`self_check_tpxo` report as a text table."""
    lines = [f"self-check: {rep['station']} ({rep['region']}) "
             f"via {rep['file']} at {rep['point']}"]
    lines.append(f"{'constituent':<12}{'tpxo_amp':>10}{'pub_amp':>10}"
                 f"{'damp':>8}{'dphase':>8}  ok")
    for r in rep["rows"]:
        ta = f"{r['tpxo_amp']:.4f}" if r["tpxo_amp"] is not None else "n/a"
        tp = f"{r['tpxo_phase']:.2f}" if r["tpxo_phase"] is not None else "n/a"
        da = f"{r['damp']:.4f}" if r["damp"] is not None else "n/a"
        dp = f"{r['dphase']:+.2f}" if r["dphase"] is not None else "n/a"
        note = f"  ({r['note']})" if r["note"] else ""
        lines.append(f"{r['constituent']:<12}{ta:>10}{r['pub_amp']:>10.4f}"
                     f"{da:>8}{dp:>8}  {'OK' if r['ok'] else '--'}{note}")
    lines.append(f"verdict: {'PASS' if rep['pass'] else 'FAIL'} "
                 f"({rep['n_shared']} shared constituents, "
                 f"tol ±{rep['tol_amp_m']} m / ±{rep['tol_phase_deg']}°)")
    return "\n".join(lines)
