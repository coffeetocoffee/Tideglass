"""Export adapters for Marea Core — the v0.4 "feeds" deliverable.

Turns a fitted :class:`~tideglass.marea.model.TideModel` (or its prediction) into
the interchange formats operators actually want:

* :func:`to_json` / :func:`write_json` — the round-trippable artifact (also the
  native ``TideModel.to_artifact`` form).
* :func:`write_csv` — a flat ``time,height,lower,upper,se`` feed.
* :func:`to_xtide` / :func:`from_xtide` — a human-readable harmonic-constants
  file (one ``name amplitude phase`` line per constituent) in the spirit of
  XTide's harmonic-constants interchange.
* :func:`write_netcdf` / :func:`read_netcdf` — a self-contained **NetCDF3
  classic** writer/reader (big-endian XDR, no external dependency) so the
  prediction curve can be dropped into oceanographic pipelines.

Only the standard library and numpy are required.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Sequence
from datetime import datetime, timezone

import numpy as np

from tideglass.marea.model import TideModel

# --- JSON --------------------------------------------------------------------

def to_json(model: TideModel) -> dict:
    """The round-trippable model artifact (see ``TideModel.to_artifact``)."""
    return model.to_artifact()


def write_json(model: TideModel, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(to_json(model), fh, indent=2)


# --- CSV feed ----------------------------------------------------------------

def write_csv(model: TideModel, times: Sequence[datetime], path: str) -> None:
    """Write ``time,height,lower,upper,se`` rows (metres) for ``times``."""
    pred = model.predict(times)
    with open(path, "w", newline="") as fh:
        fh.write("time,height_m,lower_m,upper_m,se_m\n")
        fh.writelines(f"{t.isoformat()},{m:.6f},{lo:.6f},{hi:.6f},{se:.6f}\n" for t, m, lo, hi, se in zip(times, pred.mean, pred.lower, pred.upper, pred.se))


# --- XTide-style harmonic constants (plain text) -----------------------------

def to_xtide(model: TideModel, station: str | None = None, datum: str = "MLLW") -> str:
    """A plain-text harmonic-constants file (name amplitude phase per line).

    The format mirrors the human-readable harmonic-constants interchange used
    alongside XTide: a station header followed by one line per constituent with
    its equilibrium amplitude (m) and phase (°). Round-trips via :func:`from_xtide`.
    """
    station = station or model.station or "unknown"
    lines = [
        "# Tideglass harmonic constants",
        f"# station: {station}",
        f"# datum: {datum}",
        f"# mean_level_m: {model._coef[0]:.6f}",
        "# format: name amplitude_m phase_deg",
    ]
    for f in model.constituents():
        lines.append(f"{f.name} {f.amplitude:.6f} {f.phase_deg:.4f}")
    return "\n".join(lines) + "\n"


def from_xtide(text: str) -> dict:
    """Parse an :func:`to_xtide` file back into the artifact dict."""
    station = None
    datum = "MLLW"
    mean = 0.0
    consts = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            if line.startswith("# station:"):
                station = line.split(":", 1)[1].strip()
            elif line.startswith("# datum:"):
                datum = line.split(":", 1)[1].strip()
            elif line.startswith("# mean_level_m:"):
                mean = float(line.split(":", 1)[1].strip())
            continue
        parts = line.split()
        if len(parts) == 3:
            consts.append({
                "name": parts[0],
                "amplitude": float(parts[1]),
                "phase": float(parts[2]),
            })
    return {"station": station, "datum": datum, "mean": mean, "constituents": consts}


def write_xtide(model: TideModel, path: str, station: str | None = None, datum: str = "MLLW") -> None:
    with open(path, "w") as fh:
        fh.write(to_xtide(model, station, datum))


# --- NetCDF3 classic (dependency-free) ---------------------------------------

_NC_DIMENSION = 0x0A
_NC_VARIABLE = 0x0B
_NC_ATTRIBUTE = 0x0C
_NC_DOUBLE = 6
_MAGIC = b"CDF\x01"


def write_netcdf(
    model: TideModel,
    times: Sequence[datetime],
    path: str,
    time_units: str = "hours since 2000-01-01T00:00:00Z",
) -> None:
    """Write ``time, height, lower, upper`` as a NetCDF3 classic file.

    A single unlimited (record) dimension ``time`` carries four double
    variables. Big-endian XDR, so the file is readable by standard NetCDF tools.
    """
    pred = model.predict(times)
    t = np.asarray(times, dtype=object)
    if isinstance(t[0], datetime) and t[0].tzinfo is None:
        t = np.array([x.replace(tzinfo=timezone.utc) for x in t])
    epoch = datetime(2000, 1, 1, tzinfo=timezone.utc)
    tnum = np.array([
        (x - epoch).total_seconds() / 3600.0 for x in t
    ], dtype=">f8")  # big-endian double
    height = np.asarray(pred.mean, dtype=">f8")
    lower = np.asarray(pred.lower, dtype=">f8")
    upper = np.asarray(pred.upper, dtype=">f8")
    n = int(tnum.size)

    # --- header -----------------------------------------------------------
    hdr = bytearray(_MAGIC)
    hdr += struct.pack(">i", n)  # numrecs

    # dim_list: one "time" dimension, unlimited (length 0xFFFFFFFF).
    dim_block = bytearray()
    dim_block += struct.pack(">i", _NC_DIMENSION)
    dim_block += struct.pack(">i", 1)  # one dimension
    name = b"time"
    dim_block += struct.pack(">i", len(name)) + name
    dim_block += struct.pack(">i", -1)  # unlimited dimension (0xFFFFFFFF as signed)
    hdr += dim_block

    # gatt_list: no global attributes.
    hdr += struct.pack(">i", _NC_ATTRIBUTE) + struct.pack(">i", 0)

    # var_list: time, height, lower, upper (all record vars, double).
    var_names = ["time", "height", "lower", "upper"]
    var_arrays = [tnum, height, lower, upper]
    var_block = bytearray()
    var_block += struct.pack(">i", _NC_VARIABLE)
    var_block += struct.pack(">i", len(var_names))
    for vname in var_names:
        nm = vname.encode()
        var_block += struct.pack(">i", len(nm)) + nm
        var_block += struct.pack(">i", 1)  # one dimension
        var_block += struct.pack(">i", 0)  # dim id 0 (time)
        var_block += struct.pack(">i", _NC_ATTRIBUTE) + struct.pack(">i", 0)  # no attrs
        var_block += struct.pack(">i", _NC_DOUBLE)
        var_block += struct.pack(">i", 8)  # vsize: 1 double = 8 bytes (4-aligned)
        var_block += struct.pack(">i", 0)  # begin (unused for record vars)
    hdr += var_block

    # Pad header to a 4-byte boundary.
    while len(hdr) % 4:
        hdr += b"\x00"

    # --- data: record-interleaved, explicit big-endian XDR ---------------
    # numpy keeps the buffer in native byte order even for a ">f8" dtype, so we
    # pack each value explicitly with struct to guarantee XDR (big-endian).
    rec = bytearray()
    for i in range(n):
        for arr in var_arrays:
            rec += struct.pack(">d", float(arr[i]))
    # Each record is 4*8 = 32 bytes, already 4-aligned; pad defensively.
    while len(rec) % 4:
        rec += b"\x00"

    with open(path, "wb") as fh:
        fh.write(bytes(hdr))
        fh.write(bytes(rec))


def read_netcdf(path: str) -> dict:
    """Read a NetCDF3 classic file produced by :func:`write_netcdf`.

    Returns ``{"time": np.ndarray, "height": ..., "lower": ..., "upper": ...}``
    with ``time`` in hours since the 2000-01-01 epoch (decode with the same
    epoch if needed).
    """
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:4] != _MAGIC:
        raise ValueError("not a NetCDF3 classic file")
    off = 4
    (numrecs,) = struct.unpack(">i", data[off:off + 4])
    off += 4

    def _read_name():
        nonlocal off
        (nlen,) = struct.unpack(">i", data[off:off + 4])
        off += 4
        name = data[off:off + nlen].decode()
        off += nlen
        return name

    # dim_list
    (tag,) = struct.unpack(">i", data[off:off + 4])
    off += 4
    assert tag == _NC_DIMENSION
    (ndim,) = struct.unpack(">i", data[off:off + 4])
    off += 4
    for _ in range(ndim):
        _read_name()
        (dlen,) = struct.unpack(">i", data[off:off + 4])
        off += 4
        if dlen < 0:  # unlimited dimension (stored as -1 / 0xFFFFFFFF)
            pass  # record count already captured in numrecs

    # gatt_list
    (tag,) = struct.unpack(">i", data[off:off + 4])
    off += 4
    assert tag == _NC_ATTRIBUTE
    (ngatt,) = struct.unpack(">i", data[off:off + 4])
    off += 4
    for _ in range(ngatt):
        _read_name()
        (ntype,) = struct.unpack(">i", data[off:off + 4])
        off += 4
        (nvals,) = struct.unpack(">i", data[off:off + 4])
        off += 4
        nbytes = ntype_size(ntype) * nvals
        off += nbytes
        off += (4 - nbytes % 4) % 4  # attribute values padded to 4 bytes

    # var_list
    (tag,) = struct.unpack(">i", data[off:off + 4])
    off += 4
    assert tag == _NC_VARIABLE
    (nvars,) = struct.unpack(">i", data[off:off + 4])
    off += 4
    vars_ = []
    for _ in range(nvars):
        vname = _read_name()
        (nd,) = struct.unpack(">i", data[off:off + 4])
        off += 4
        dim_ids = []
        for _ in range(nd):
            (d,) = struct.unpack(">i", data[off:off + 4])
            off += 4
            dim_ids.append(d)
        # vatt_list
        (vtag,) = struct.unpack(">i", data[off:off + 4])
        off += 4
        assert vtag == _NC_ATTRIBUTE
        (nvatt,) = struct.unpack(">i", data[off:off + 4])
        off += 4
        for _ in range(nvatt):
            _read_name()
            (ntype,) = struct.unpack(">i", data[off:off + 4])
            off += 4
            (nvals,) = struct.unpack(">i", data[off:off + 4])
            off += 4
            off += ntype_size(ntype) * nvals
        (vtype,) = struct.unpack(">i", data[off:off + 4])
        off += 4
        (vsize,) = struct.unpack(">i", data[off:off + 4])
        off += 4
        (begin,) = struct.unpack(">i", data[off:off + 4])
        off += 4
        vars_.append((vname, dim_ids, vtype, vsize, begin))

    while off % 4:
        off += 1

    # Record variables are stored interleaved (all vars' values for record r,
    # then record r+1, ...). Read per-record so each variable lands correctly.
    rec_vars = [(vname, dim_ids, vtype, vsize, begin)
                for (vname, dim_ids, vtype, vsize, begin) in vars_
                if any(d == 0 for d in dim_ids) and len(dim_ids) == 1]
    recsize = sum(v[3] for v in rec_vars)  # bytes per record (all record vars)
    out: dict[str, np.ndarray] = {v[0]: np.empty(numrecs, dtype=">f8") for v in rec_vars}
    for r in range(numrecs):
        base = off + r * recsize
        cursor = 0
        for vname, _d, _t, vsize, _b in rec_vars:
            out[vname][r] = struct.unpack(">d", data[base + cursor:base + cursor + 8])[0]
            cursor += vsize
    # Any non-record variables would be read from their `begin`; we only write
    # record variables, so nothing else to do here.
    return out


def ntype_size(ntype: int) -> int:
    return {1: 1, 2: 2, 3: 4, 4: 4, 5: 4, 6: 8}.get(ntype, 8)
