"""NOAA CO-OPS gauge ingestion for Tideglass.

Pulls water levels from the NOAA Tides & Currents API and emits the
``time,height`` CSV that ``tideglass fit`` / ``bench`` already consume — so
the engine can be run on any public station/date range without hand-fetched
files.

API reference: https://api.tidesandcurrents.noaa.gov/api/prod/
"""

from __future__ import annotations

import csv as _csv
from collections.abc import Callable
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import urlopen

_NOAA_BASE = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
# (url, timeout) -> binary file-like, so tests can inject a fake getter.
_Getter = Callable[[str, int], object]


def _fmt_date(d: str) -> str:
    """Accept ``YYYY-MM-DD`` (or ``YYYYMMDD``) → API ``YYYYMMDD``."""
    d = d.strip().replace("-", "")
    if len(d) != 8 or not d.isdigit():
        raise ValueError(f"bad date {d!r} (want YYYY-MM-DD)")
    return d


def fetch_noaa(
    station: str,
    begin: str,
    end: str,
    datum: str = "MLLW",
    interval: str = "h",
    timeout: int = 30,
    _getter: _Getter | None = None,
) -> list[tuple[datetime, float]]:
    """Download ``(datetime, height_m)`` rows for a NOAA station.

    :param station: NOAA CO-OPS station id, e.g. ``9414290`` (San Francisco).
    :param begin, end: inclusive date range as ``YYYY-MM-DD`` (UTC).
    :param datum: NOAA vertical datum (default ``MLLW``).
    :param interval: ``h`` hourly, ``1`` 6-minute, ``hilo`` high/low only.
    """
    params = {
        "product": "water_level",
        "station": str(station),
        "begin_date": _fmt_date(begin),
        "end_date": _fmt_date(end),
        "datum": datum,
        "units": "metric",
        "time_zone": "GMT",
        "interval": interval,
        "format": "csv",
    }
    url = f"{_NOAA_BASE}?{urlencode(params)}"
    getter = _getter or urlopen
    with getter(url, timeout=timeout) as resp:  # type: ignore[operator]
        text = resp.read().decode("utf-8")  # type: ignore[attr-defined]
    return parse_noaa_csv(text)


def parse_noaa_csv(text: str) -> list[tuple[datetime, float]]:
    """Parse a NOAA ``water_level`` CSV body into ``(datetime, height)`` rows.

    Robust to the header column order and to missing ``Water Level`` samples
    (NOAA emits a blank for gaps), which are skipped.
    """
    rows: list[tuple[datetime, float]] = []
    reader = _csv.reader(text.splitlines())
    header = next(reader, None)
    if not header:
        return rows
    norm = [h.strip() for h in header]
    try:
        i_t = norm.index("Date Time")
        i_h = norm.index("Water Level")
    except ValueError:
        raise ValueError("NOAA CSV missing 'Date Time'/'Water Level' columns")

    for r in reader:
        if len(r) <= max(i_t, i_h):
            continue
        ts = r[i_t].strip().replace(" ", "T")
        try:
            t = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        raw = r[i_h].strip()
        if not raw:
            continue
        try:
            h = float(raw)
        except ValueError:
            continue
        rows.append((t, h))
    return rows


def write_csv(rows: list[tuple[datetime, float]], path: str) -> None:
    """Write ``time,height`` rows in the format ``tideglass fit`` expects."""
    with open(path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["time", "height"])
        for t, h in rows:
            w.writerow([t.isoformat(), f"{h:.4f}"])
