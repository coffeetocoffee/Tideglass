"""Tests for NOAA CO-OPS ingestion (parse path, no network)."""

from tideglass.fetch import _fmt_date, parse_noaa_csv, write_csv

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
