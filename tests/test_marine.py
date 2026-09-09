"""Tests for the marine layer (species / harvesting / rip / advisor)."""

import numpy as np
import pytest
from datetime import datetime, timedelta, timezone

from tideglass import TideAdvisor, TideModel
from tideglass.cli import main as cli_main
from tideglass.marea import constituents as C
from tideglass.marea.solver import rad_per_hour
from tideglass.marine import harvesting as HV
from tideglass.marine import rip as RIP
from tideglass.marine import species as SP

UTC = timezone.utc
T0 = datetime(2024, 6, 1, tzinfo=UTC)


def _semidiurnal(days=4, amp=1.0):
    times = [T0 + timedelta(hours=h) for h in range(days * 24)]
    t = np.arange(len(times), dtype=float)
    w = float(rad_per_hour(C.speed(C.get("M2"))))
    return times, amp * np.cos(w * t)


def test_exposure_windows_and_fraction():
    times, h = _semidiurnal()
    wins = SP.exposure_windows(times, h, threshold_m=0.0, min_hours=1.0)
    assert len(wins) == 8  # one per low water over 4 days
    for s, e in wins:
        assert e > s
    frac = SP.exposure_fraction(times, h, 0.0)
    assert frac == pytest.approx(0.5, abs=0.02)
    with pytest.raises(ValueError):
        SP.exposure_windows(times, h[:-1], 0.0)


def test_safe_windows_and_surge_guard():
    times, h = _semidiurnal()
    wins = HV.safe_windows(times, h, low_threshold_m=0.0, min_hours=1.0)
    assert len(wins) == 8
    flags = [False] * len(times)
    flags[6] = True  # surge just before the first low-water window
    guarded = HV.safe_windows(times, h, flags, low_threshold_m=0.0, min_hours=1.0)
    assert len(guarded) < len(wins)  # 24 h guard drops the nearby windows
    assert guarded[0].start > wins[0].start
    with pytest.raises(ValueError):
        HV.safe_windows(times, h, [False])


def test_rip_risk_flat_vs_steep():
    times, _ = _semidiurnal()
    flat = RIP.risk(times, np.zeros(len(times)))
    assert (flat.score == 0.0).all() and set(flat.category) == {"Low"}
    _, steep = _semidiurnal(amp=2.0)
    r = RIP.risk(times, steep)
    assert "High" in r.category and r.score.max() <= 1.0
    assert set(r.category) <= {"Low", "Moderate", "High"}
    with pytest.raises(ValueError):
        RIP.risk([times[0]], [1.0])


def test_advisor_end_to_end(tmp_path):
    times, h = _semidiurnal(days=10)
    y = h + 0.7 + np.random.default_rng(12).normal(0, 0.01, len(times))
    model = TideModel.fit(times, y, auto_select=False,
                          candidates=[C.get("M2")], station="Cove")
    adv = TideAdvisor(model).advise(times[:48])
    assert adv.station == "Cove"
    assert len(adv.harvest) >= 1
    assert set(adv.exposure) == {"Pacific oyster", "Blue mussel", "Manila clam"}
    assert "safe harvest window" in adv.summary


def test_cli_advise(tmp_path, capsys):
    times, h = _semidiurnal(days=10)
    csv = tmp_path / "cove.csv"
    with open(csv, "w") as fh:
        fh.write("time,height\n")
        for t, v in zip(times, h + 0.7):
            fh.write(f"{t.isoformat()},{v:.4f}\n")
    store = str(tmp_path / "store")
    assert cli_main(["fit", str(csv), "--station", "Cove", "--store", store,
                     "--no-select"]) == 0
    capsys.readouterr()
    day = T0.strftime("%Y-%m-%d")
    assert cli_main(["advise", "Cove", day, "--store", store]) == 0
    out = capsys.readouterr().out
    assert "safe harvest window" in out and "rip risk" in out
