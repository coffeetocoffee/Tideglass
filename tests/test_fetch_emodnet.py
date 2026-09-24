"""Tests for EMODnet fetcher and European data integration."""

import pytest

from tideglass.fetch_emodnet import (
    EMODNetError,
    fetch_emodnet,
)


def test_fetch_emodnet_raises_on_invalid_bbox():
    """Invalid bbox should raise clearly."""
    with pytest.raises(EMODNetError):
        # Bounding box reversed: invalid
        fetch_emodnet(lon_min=5.0, lon_max=-10.0, lat_min=35.0, lat_max=60.0)


def test_fetch_emodnet_requires_bbox():
    """Must provide bounding box parameters."""
    with pytest.raises(AssertionError):
        # Missing required bbox params
        fetch_emodnet()  # type: ignore[call-arg]


def test_fetch_emodnet_returns_sorted_rows(tmp_path):
    """Downloaded rows should be sorted by timestamp and have valid dt/height pairs."""
    # This test assumes EMODnet is reachable; skip if network unavailable
    try:
        rows = fetch_emodnet(
            lon_min=-10.0, lon_max=5.0,  # Ireland to Iberia
            lat_min=35.0, lat_max=60.0,
            start="2024-02-01", end="2024-02-10",
        )
    except (EMODNetError, OSError):
        pytest.skip("EMODnet unreachable or no stations matched")

    assert len(rows) > 0
    ts_prev = None
    for t, h in rows:
        assert isinstance(t, type(ts_prev)) if ts_prev else None
        assert not math.isnan(h)
        if ts_prev is not None:
            assert t >= ts_prev
        ts_prev = t
