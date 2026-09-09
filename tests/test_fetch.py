"""Tests for NOAA CO-OPS ingestion (parse path, no network)."""

import io
import re
from datetime import datetime, timedelta
from urllib.error import HTTPError

import pytest

from tideglass.fetch import (
    _MAX_RANGE_DAYS,
    _fmt_date,
    fetch_noaa,
    fetch_noaa_range,
    parse_noaa_csv,
    write_csv,
)

_SAMPLE = """Date Time,Water Level,Sigma,OO,O,F,R,L,Q
2024-01-01 00:00,1.234,0.001,0,1,0,0,0,1
2024-01-01 01:00, ,0.001,0,1,0,0,0,1
2024-01-01 02:00,1.456,0.002,0,1,0,0,0,1
"""


def test_fmt_date_accepts_dashed_and_plain():
    assert _fmt_date("2024-01-01") == "20240101"
    assert _fmt_date("20240101") == "20240101"


def test_parse_skips_missing_samples():
    rows = parse_noaa_csv(_SAMPLE)
    assert len(rows) == 2  # the blank Water Level row is skipped
    assert rows[0][0].isoformat().startswith("2024-01-01T00:00:00")
    assert abs(rows[0][1] - 1.234) < 1e-9
    assert abs(rows[1][1] - 1.456) < 1e-9


def test_write_then_readable_by_cli(tmp_path):
    from tideglass.cli import read_csv

    rows = parse_noaa_csv(_SAMPLE)
    out = str(tmp_path / "gauge.csv")
    write_csv(rows, out)
    times, heights = read_csv(out)
    assert len(times) == 2
    assert abs(heights[0] - 1.234) < 1e-9


def test_fetch_noaa_uses_getter(monkeypatch):
    from tideglass.fetch import fetch_noaa

    class _Fake:
        def __init__(self, text):
            self._t = text.encode("utf-8")

        def read(self):
            return self._t

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    captured = {}

    def _getter(url, timeout=30):
        captured["url"] = url
        return _Fake(_SAMPLE)

    rows = fetch_noaa("9414290", "2024-01-01", "2024-01-01", _getter=_getter)
    assert len(rows) == 2
    assert "station=9414290" in captured["url"]
    assert "begin_date=20240101" in captured["url"]
    assert "end_date=20240101" in captured["url"]


class _Fake:
    """Minimal context-manager response for injected getters."""

    def __init__(self, text):
        self._t = text.encode("utf-8")

    def read(self):
        return self._t

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_fetch_noaa_rejects_over_31_days():
    with pytest.raises(ValueError, match="31 days"):
        fetch_noaa("9414290", "2024-01-01", "2024-03-01",
                   _getter=lambda *a, **k: (_ for _ in ()).throw(
                       AssertionError("must not hit the API")))


def test_fetch_noaa_http_error_surfaces_noaa_message():
    def _getter(url, timeout=30):
        raise HTTPError(url, 400, "Bad Request",
                        {"content-type": "text/plain"},
                        io.BytesIO(b" Wrong Date: Range Limit Exceeded: "
                                  b"31 days "))

    with pytest.raises(ValueError, match="Range Limit Exceeded"):
        fetch_noaa("9414290", "2024-01-01", "2024-01-02", _getter=_getter)


def test_fetch_range_chunks_long_windows():
    urls: list[str] = []

    def _getter(url, timeout=30):
        urls.append(url)
        m = re.search(r"begin_date=(\d{8})&end_date=(\d{8})", url)
        b = datetime.strptime(m.group(1), "%Y%m%d")
        e = datetime.strptime(m.group(2), "%Y%m%d")
        # Emit one hourly row per requested day, tagged with the chunk index.
        rows = ["Date Time,Water Level"]
        day = b
        while day <= e:
            rows.append(f"{day:%Y-%m-%d} 00:00,{day.timetuple().tm_yday % 10}.000")
            day += timedelta(days=1)
        return _Fake("\n".join(rows) + "\n")

    rows = fetch_noaa_range("9414290", "2024-01-01", "2024-03-01", _getter=_getter)
    # 61 days -> 3 chunks at the 30-day cap.
    assert len(urls) == 3
    assert "begin_date=20240101" in urls[0]
    assert "end_date=20240130" in urls[0]
    assert "begin_date=20240131" in urls[1]
    assert "end_date=20240229" in urls[1]
    assert "begin_date=20240301" in urls[2]
    # 61 unique daily rows, sorted, none lost at chunk seams.
    assert len(rows) == 61
    assert all(rows[i][0] < rows[i + 1][0] for i in range(len(rows) - 1))


def test_fetch_range_dedupes_seam_overlap():
    def _getter(url, timeout=30):
        return _Fake(_SAMPLE)  # same rows for every chunk

    rows = fetch_noaa_range("9414290", "2024-01-01", "2024-02-15", _getter=_getter)
    stamps = [r[0] for r in rows]
    # The sample's blank Water Level row is skipped; the 2 real timestamps
    # appear in every chunk and must be deduped to one occurrence each.
    assert len(stamps) == len(set(stamps)) == 2
    assert _MAX_RANGE_DAYS == 30


def test_fetch_range_empty_range_raises():
    with pytest.raises(ValueError, match="precedes"):
        fetch_noaa_range("9414290", "2024-02-01", "2024-01-01",
                         _getter=lambda *a, **k: (_ for _ in ()).throw(
                             AssertionError("must not hit the API")))
