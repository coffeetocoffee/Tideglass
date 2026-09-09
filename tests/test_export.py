"""Tests for v0.4 export adapters (JSON / CSV / XTide / NetCDF3)."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tideglass.marea import constituents as C
from tideglass.marea import export as EX
from tideglass.marea.model import TideModel
from tideglass.marea.solver import rad_per_hour

UTC = timezone.utc
TRUE = {"M2": (1.0, 0.5), "S2": (0.30, -1.2), "N1": (0.0, 0.0)}
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _hours(times):
    return np.array([(t - T0).total_seconds() / 3600.0 for t in times])


def _synthetic(times, noise=0.01, seed=0):
    t = _hours(times)
    y = np.full_like(t, 0.7)
    for name, (amp, phi) in TRUE.items():
        if name == "N1":
            continue
        w = float(rad_per_hour(C.speed(C.get(name))))
        y = y + amp * np.cos(w * t - phi)
    if noise:
        y = y + np.random.default_rng(seed).normal(0, noise, size=t.size)
    return y


def _hourly(start, n):
    return [start + timedelta(hours=h) for h in range(n)]


@pytest.fixture
def model():
    train = _hourly(T0, 30 * 24)
    return TideModel.fit(train, _synthetic(train, seed=5), alpha=1e-4, station="X")


def test_xtide_roundtrip(model):
    text = EX.to_xtide(model, station="X")
    assert "# station: X" in text
    art = EX.from_xtide(text)
    assert art["station"] == "X"
    names = {c["name"] for c in art["constituents"]}
    assert {"M2", "S2"} <= names
    # Re-load via the artifact form and confirm it predicts again.
    reloaded = TideModel.load_harmonic(art)
    pred = reloaded.predict(_hourly(T0, 24))
    assert pred.mean.shape == (24,)


def test_write_csv(tmp_path, model):
    times = _hourly(T0, 24)
    out = tmp_path / "feed.csv"
    EX.write_csv(model, times, str(out))
    lines = out.read_text().splitlines()
    assert lines[0] == "time,height_m,lower_m,upper_m,se_m"
    assert len(lines) == 25


def test_netcdf_roundtrip(tmp_path, model):
    times = _hourly(T0, 50)
    out = tmp_path / "curve.nc"
    EX.write_netcdf(model, times, str(out))
    back = EX.read_netcdf(str(out))
    assert set(back) == {"time", "height", "lower", "upper"}
    pred = model.predict(times)
    assert np.allclose(back["height"], pred.mean, atol=1e-6)
    assert np.allclose(back["lower"], pred.lower, atol=1e-6)
    assert back["time"].size == 50


def test_netcdf_crosscheck_with_scipy(tmp_path, model):
    sp = pytest.importorskip("scipy.io.netcdf_file")
    times = _hourly(T0, 40)
    out = tmp_path / "curve.nc"
    EX.write_netcdf(model, times, str(out))
    with sp(str(out), "r") as nc:
        height = np.array(nc.variables["height"][:])
    assert np.allclose(height, model.predict(times).mean, atol=1e-6)
