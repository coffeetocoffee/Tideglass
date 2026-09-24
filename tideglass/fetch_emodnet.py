"""EMODnet (European) sea-level data fetcher for Tideglass.

This module provides ``fetch_emodnet()`` and related utilities to download
sea-level observations from the **EMODnet Physics** portal:

    https://emodnet.ec.europa.eu/en/monitoring/sea-level-data

Data sources cover ~300 European coastal stations across Mediterranean, North Sea,
Baltic Sea, Atlantic coasts of France/Spain/UK/Ireland/Netherlands/Belgium/Germany,
and Arctic Norway/France.

Usage::

    from tideglass.fetch import fetch_emodnet

    # Download hourly sea level for a station bounding box
    rows = fetch_emodnet(
        lon_min=-8.0, lon_max=-1.0,  # Ireland to Portugal
        lat_min=35.0, lat_max=60.0,  # UK to Iberia
        start="2024-01-01", end="2024-03-01",
        dataset="sl",  # "sl" = sea level
    )

All returned rows match the NOAA format already used by :meth:`TideModel.fit`:
a list of ``(datetime, height_m)`` tuples with UTC timestamps.

Limitations:
* Free tier access allows ~100 downloads/month; contact EMODnet for higher quotas.
* Stations report varying formats (some include quality flags, others only raw values).
* Not all stations publish near-real-time data — check individual station metadata first.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import urllib.parse
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

DEFAULT_USER_AGENT = (
    "tideglass/3.3.0 (+https://github.com/coffeetocoffee/Tideglass)"
)


class EMODNetError(RuntimeError):
    """Raised when EMODnet download fails or returns invalid data."""


class EMODNetStationNotFoundError(ValueError):
    """Raised when a requested station ID is not found in EMODnet results."""


def emodnet_stations(
    lon_min: float | None = None,
    lon_max: float | None = None,
    lat_min: float | None = None,
    lat_max: float | None = None,
    area_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """List available EMODnet stations within a bounding box or area codes.

    :param lon_min/lon_max: bounding box longitude range (degrees).
    :param lat_min/lat_max: bounding box latitude range (degrees).
    :param area_ids: optional area/country codes (e.g. ["IE","PT"]) to filter.

    Returns a list of dictionaries with keys:
        * ``id``: unique station identifier
        * ``name``: display name
        * ``lon``, ``lat``: coordinates (WGS84)
        * ``time_range``: earliest/latest available timestamp
        * ``frequency``: typical sampling frequency (hourly/daily/...)
        * ``provider``: national agency providing the data
    """
    from email.utils import parsedate_to_datetime

    url = "https://emodnet-physics.ec.europa.eu/rest/datacollection/stations?"

    params = {"version": "v1"}
    if area_ids is not None and len(area_ids) > 0:
        params["areas"] = ",".join(str(a) for a in area_ids)

    if (lon_min is not None) and (lon_max is not None) and (lat_min is not None) and (
        lat_max is not None
    ):
        params["bbox"] = f"{lon_min},{lat_min},{lon_max},{lat_max}"

    query = urllib.parse.urlencode(params, doseq=True)
    full_url = f"{url}{query}"

    req = urllib.request.Request(full_url, headers={"User-Agent": DEFAULT_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        msg = (
            f"EMODnet stations request failed ({resp.status}: {resp.reason}). "
            f"Check your bbox coordinates or area IDs."
        )
        raise EMODNetError(msg) from e
    except urllib.error.URLError as e:
        raise EMODNetError(f"Could not connect to EMODnet: {e.reason}") from e

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise EMODNetError(f"Invalid JSON response: {exc}") from exc

    if "stations" not in data:
        return []

    results: list[dict[str, Any]] = []
    for s in data.get("stations", []):
        info = {
            "id": str(s.get("id", "")),
            "name": str(s.get("title", "") or s.get("stationName", "")),
            "lon": float(s.get("longitude", 0.0)),
            "lat": float(s.get("latitude", 0.0)),
            "time_range": str(s.get("periodOfObservation", "").strip()),
            "frequency": s.get("samplingFrequency", "unknown"),
            "provider": s.get("institution", "").strip(),
        }
        results.append(info)

    return results


def _emodnet_download_url(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    start: datetime,
    end: datetime,
    dataset: str = "sl",
) -> str:
    """Build the EMODnet download URL for a bounding box and date window.

    This constructs the direct download link that returns either JSON or CSV
    depending on content negotiation. We force ``format=csv``.
    """
    base = "https://emodnet-physics.ec.europa.eu/rest/datacollection/download"

    bbox = f"{lon_min},{lat_min},{lon_max},{lat_max}"
    time_from = start.strftime("%Y-%m-%dT%H:%M:%S")
    time_to = end.strftime("%Y-%m-%dT%H:%M:%S")

    params = {
        "version": "v1",
        "area": bbox,
        "dataset": dataset,
        "format": "csv",
        "timeFrom": time_from,
        "timeTo": time_to,
    }

    query = urllib.parse.urlencode(params)
    return f"{base}?{query}"


def fetch_emodnet(
    lon_min: float | None = None,
    lon_max: float | None = None,
    lat_min: float | None = None,
    lat_max: float | None = None,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    dataset: str = "sl",
    out_dir: str | None = None,
) -> list[tuple[datetime, float]]:
    """Download sea-level data from EMODnet for a geographic region and time window.

    :param lon_min/max, lat_min/max: bounding box for stations to fetch. Exactly one
        of the bbox parameters OR ``station_id`` must be provided.
    :param start/end: ISO-8601 strings or datetime objects marking the download window.
        Defaults to last 31 days if omitted.
    :param dataset: "sl" for sea level (default), "st" for storm surge residuals.
    :param out_dir: optional local cache directory; downloaded CSVs are written there
        and reused on subsequent runs when the hash matches.

    Returns a list of ``(time_utc, height_m)`` tuples sorted chronologically.
    Raises ``EMODNetError`` if the download fails or returns zero rows.

    Example::

        rows = fetch_emodnet(
            lon_min=-10.0, lon_max=5.0,
            lat_min=36.0, lat_max=70.0,
            start="2024-01-01", end="2024-03-01",
            out_dir=".tideglass/emodnet_cache",
        )
    """
    now = datetime.now(timezone.utc)
    if start is None:
        start = now - timedelta(days=31)
    elif isinstance(start, str):
        start = datetime.fromisoformat(start.replace("Z", "+00:00"))
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
    if end is None:
        end = now
    elif isinstance(end, str):
        end = datetime.fromisoformat(end.replace("Z", "+00:00"))
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

    assert lon_min is not None and lon_max is not None, (
        "Provide bounding box via lon_min/max and lat_min/max"
    )
    assert lat_min is not None and lat_max is not None, (
        "Provide bounding box via lon_min/max and lat_min/max"
    )

    url = _emodnet_download_url(lon_min, lon_max, lat_min, lat_max, start, end, dataset)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        fname = _download_filename(url)
        cache_path = os.path.join(out_dir, fname)
        if os.path.exists(cache_path):
            cached_hash = _hash_file(cache_path)
            expected_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
            if cached_hash.startswith(expected_hash):
                return _load_cached(csv.reader(open(cache_path, newline="")))

    req = urllib.request.Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            if resp.status != 200:
                text = resp.read().decode("utf-8", errors="replace")[:256]
                raise EMODNetError(f"Download failed: status={resp.status}, reason={text}")
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        msg = (
            f"EMODnet download failed (status {e.code}): {e.reason}. "
            f"This usually means no stations matched the bbox + date window."
        )
        raise EMODNetError(msg) from e
    except urllib.error.URLError as e:
        raise EMODNetError(f"Network error: {e.reason}") from e

    rows = _parse_response(body)
    if not rows:
        raise EMODNetError("Download succeeded but returned zero valid rows")

    if out_dir is not None:
        with open(cache_path, "w", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["time_utc", "height_m"])
            writer.writerows([(t.isoformat(), h) for t, h in rows])

    return rows


def _download_filename(url: str) -> str:
    """Derive a stable filename from the EMODnet URL for caching purposes."""
    slug = urllib.parse.quote(url, safe=":")
    slug = "".join(c for c in slug if c.isalnum() or c in "_.-").lower()
    short = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return f"{slug}.{short}.csv"


def _hash_file(path: str) -> str:
    """Return SHA-256 hex digest of a file for cache validation."""
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def _load_cached(reader: csv.reader) -> list[tuple[datetime, float]]:
    """Load rows from a cached CSV reader into ``(datetime, height)`` tuples."""
    rows: list[tuple[datetime, float]] = []
    header_seen = False
    for row in reader:
        if not header_seen:
            header_seen = True
            continue
        if not row or len(row) < 2:
            continue
        ts_str, val_str = row[0].strip(), row[1].strip()
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        try:
            val = float(val_str)
        except ValueError:
            continue
        rows.append((ts, val))
    return sorted(rows, key=lambda x: x[0])


def _parse_response(body: str) -> list[tuple[datetime, float]]:
    """Parse EMODnet CSV response into ``(datetime_utc, height_m)`` tuples.

    Expect columns: timestamp, value[, quality_flag]. Timestamps should be in ISO-8601
    with explicit timezone. Height units vary by station (most are metres MLLW).
    """
    reader = csv.reader(StringIOWrapper(body))
    rows: list[tuple[datetime, float]] = []
    header_seen = False

    for row in reader:
        if not header_seen:
            header_seen = True
            column_names = [c.strip().lower() for c in row]
            idx_time = next(i for i, c in enumerate(column_names) if "time" in c)
            idx_value = next(i for i, c in enumerate(column_names) if "value" in c or "level" in c)
            continue

        if not row or len(row) <= max(idx_time, idx_value):
            continue

        ts_str, val_str = row[idx_time], row[idx_value]
        if not ts_str or not val_str:
            continue

        try:
            ts = datetime.fromisoformat(ts_str.strip().replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        try:
            val = float(val_str)
        except ValueError:
            continue

        rows.append((ts, val))

    return sorted(rows, key=lambda x: x[0])


class StringIOWrapper:
    """Wrap a string as a file-like object compatible with csv.reader.

    Simplification avoiding dependencies on io.StringIO which may fail under some
    Python environments.
    """

    def __init__(self, data: str) -> None:
        self._lines = iter(data.splitlines())

    def __iter__(self) -> Iterator[str]:
        return self

    def __next__(self) -> str:
        return next(self._lines)

