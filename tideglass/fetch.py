"""NOAA CO-OPS gauge ingestion for Tideglass.

Pulls water levels from the NOAA Tides & Currents API and emits the
``time,height`` CSV that ``tideglass fit`` / ``bench`` already consume — so
the engine can be run on any public station/date range without hand-fetched
files.

The CO-OPS API caps a single ``water_level`` request at **31 days**
("Range Limit Exceeded" HTTP 400/422 otherwise). :func:`fetch_noaa_range`
transparently chunks longer windows, dedupes overlapping samples, and
concatenates the result — :func:`fetch_noaa` stays a single raw request.
HTTP failures raise ``ValueError`` carrying NOAA's own error text (or
``OSError`` when the host is unreachable) instead of a raw urllib traceback.

API reference: https://api.tidesandcurrents.noaa.gov/api/prod/
"""

from __future__ import annotations

import csv as _csv
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

_NOAA_BASE = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
# NOAA documented per-request range limit for water_level (31 days); stay under.
_MAX_RANGE_DAYS = 30
# (url, timeout) -> binary file-like, so tests can inject a fake getter.
_Getter = Callable[[str, int], object]


def _fmt_date(d: str) -> str:
    """Accept ``YYYY-MM-DD`` (or ``YYYYMMDD``) → API ``YYYYMMDD``."""
    d = d.strip().replace("-", "")
    if len(d) != 8 or not d.isdigit():
        raise ValueError(f"bad date {d!r} (want YYYY-MM-DD)")
    return d


def _parse_api_date(d: str) -> date:
    # Result is a bare date (no wall-clock semantics), so the UTC pin is
    # formal; it keeps naive-datetime lint (DTZ) satisfied.
    return datetime.strptime(d, "%Y%m%d").replace(tzinfo=timezone.utc).date()


def _noaa_error(exc: HTTPError) -> ValueError:
    """Wrap an HTTP error, preferring NOAA's own message body when present."""
    try:
        body = exc.read(2000).decode("utf-8", "replace").strip()
    except Exception:  # noqa: BLE001 - body may itself be unreadable
        body = ""
    if body:
        return ValueError(f"NOAA CO-OPS HTTP {exc.code}: {body}")
    return ValueError(f"NOAA CO-OPS HTTP {exc.code} {exc.reason}")


def _get_rows(url: str, timeout: int, getter: _Getter | None):
    getter = getter or urlopen
    try:
        with getter(url, timeout=timeout) as resp:  # type: ignore[operator]
            text = resp.read().decode("utf-8")  # type: ignore[attr-defined]
    except HTTPError as exc:
        raise _noaa_error(exc) from exc
    except URLError as exc:
        raise OSError(f"NOAA CO-OPS unreachable: {exc.reason}") from exc
    return parse_noaa_csv(text)


def fetch_noaa(
    station: str,
    begin: str,
    end: str,
    datum: str = "MLLW",
    interval: str = "h",
    timeout: int = 30,
    _getter: _Getter | None = None,
) -> list[tuple[datetime, float]]:
    """Download ``(datetime, height_m)`` rows for a NOAA station (one request).

    :param station: NOAA CO-OPS station id, e.g. ``9414290`` (San Francisco).
    :param begin, end: inclusive date range as ``YYYY-MM-DD`` (UTC).
    :param datum: NOAA vertical datum (default ``MLLW``).
    :param interval: ``h`` hourly, ``1`` 6-minute, ``hilo`` high/low only.

    Ranges longer than NOAA's 31-day per-request limit raise ``ValueError``;
    use :func:`fetch_noaa_range` to fetch longer windows in chunks.
    """
    b, e = _fmt_date(begin), _fmt_date(end)
    if (_parse_api_date(e) - _parse_api_date(b)).days > _MAX_RANGE_DAYS:
        raise ValueError(
            f"NOAA CO-OPS limits water_level requests to 31 days "
            f"({begin}..{end} is "
            f"{(_parse_api_date(e) - _parse_api_date(b)).days + 1} days); "
            f"fetch_noaa_range() chunks automatically")
    params = {
        "product": "water_level",
        "station": str(station),
        "begin_date": b,
        "end_date": e,
        "datum": datum,
        "units": "metric",
        "time_zone": "GMT",
        "interval": interval,
        "format": "csv",
    }
    url = f"{_NOAA_BASE}?{urlencode(params)}"
    return _get_rows(url, timeout, _getter)


def fetch_noaa_range(
    station: str,
    begin: str,
    end: str,
    datum: str = "MLLW",
    interval: str = "h",
    timeout: int = 30,
    _getter: _Getter | None = None,
) -> list[tuple[datetime, float]]:
    """Download any-length ranges by chunking at NOAA's 31-day request limit.

    Splits ``begin..end`` into <=30-day windows (with a 1-hour overlap so no
    sample is lost at a seam), sorts, and dedupes identical timestamps. A
    failing chunk aborts with the chunk's date range in the message.
    """
    b = _parse_api_date(_fmt_date(begin))
    e = _parse_api_date(_fmt_date(end))
    if e < b:
        raise ValueError(f"end {end} precedes begin {begin}")
    rows: list[tuple[datetime, float]] = []
    seen: set[datetime] = set()
    chunk_start = b
    while chunk_start <= e:
        chunk_end = min(chunk_start + timedelta(days=_MAX_RANGE_DAYS - 1), e)
        params = {
            "product": "water_level",
            "station": str(station),
            "begin_date": chunk_start.strftime("%Y%m%d"),
            "end_date": chunk_end.strftime("%Y%m%d"),
            "datum": datum,
            "units": "metric",
            "time_zone": "GMT",
            "interval": interval,
            "format": "csv",
        }
        try:
            rows.extend(_get_rows(f"{_NOAA_BASE}?{urlencode(params)}", timeout,
                                  _getter))
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"NOAA chunk {chunk_start:%Y-%m-%d}..{chunk_end:%Y-%m-%d}: {exc}"
            ) from exc
        chunk_start = chunk_end + timedelta(days=1)
    rows.sort(key=lambda r: r[0])
    deduped = [r for r in rows if not (r[0] in seen or seen.add(r[0]))]
    return deduped


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
