"""Tests for the marine layer (species / harvesting / rip / advisor)."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

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


def test_rip_exposes_decomposition_and_missing_inputs():
    times, _ = _semidiurnal()
    _, steep = _semidiurnal(amp=2.0)
    r = RIP.risk(times, steep)
    assert r.rate_term.shape == r.score.shape
    assert r.range_term.shape == r.score.shape
    assert r.wave_term.shape == r.score.shape
    total = r.rate_term + r.range_term + r.wave_term
    assert np.allclose(r.score, np.clip(total, 0, 1))
    # A tide-only score must name the dominant inputs it is missing.
    assert "wave_height_m" in r.missing_inputs
    assert "bathymetry" in r.missing_inputs
    assert r.confidence == "screening"
    assert r.usable_for_safety is False
    assert "not for public-safety decisions" in r.basis


def test_rip_wave_informed_confidence():
    times, _ = _semidiurnal()
    _, steep = _semidiurnal(amp=2.0)
    r = RIP.risk(times, steep, wave_weight=0.25, wave_height_m=2.0,
                 wave_period_s=10.0)
    assert r.confidence == "wave-informed"
    assert "wave_height_m" not in r.missing_inputs
    assert (r.wave_term > 0).all()
    assert r.score.max() >= RIP.risk(times, steep).score.max()


def test_rip_uncertainty_widens_score():
    times, _ = _semidiurnal()
    flat = np.zeros(len(times))
    assert RIP.risk(times, flat).score.max() == 0.0
    widened = RIP.risk(times, flat, sigma_m=0.10)
    assert widened.score.max() > 0.0
    # More height uncertainty can only widen, never narrow, the score.
    more = RIP.risk(times, flat, sigma_m=0.30)
    assert more.score.max() >= widened.score.max()


def test_advisor_rip_sigma_skips_models_without_bands():
    from tideglass.marine.advisor import _mean_prediction_sigma

    published = TideModel.load_harmonic({
        "station": "P", "mean": 0.0,
        "constituents": [{"name": "M2", "amplitude": 1.0, "phase": 0.0}],
    })
    with pytest.warns(UserWarning):
        pred = published.predict([T0 + timedelta(hours=h) for h in range(4)])
    assert _mean_prediction_sigma(pred) is None

    train = [T0 + timedelta(hours=h) for h in range(10 * 24)]
    y = _semidiurnal(days=10)[1] + 0.7 + np.random.default_rng(3).normal(
        0, 0.02, len(train)
    )
    fitted = TideModel.fit(train, y, auto_select=False,
                           candidates=[C.get("M2")], station="F")
    sigma = _mean_prediction_sigma(fitted.predict(train[:8]))
    assert sigma is not None and sigma > 0.0


def test_advice_surfaces_screening_caveat():
    train = [T0 + timedelta(hours=h) for h in range(10 * 24)]
    y = _semidiurnal(days=10)[1] + 0.7 + np.random.default_rng(4).normal(
        0, 0.02, len(train)
    )
    model = TideModel.fit(train, y, auto_select=False,
                          candidates=[C.get("M2")], station="Cave")
    adv = TideAdvisor(model).advise(train[:48])
    assert "rip risk confidence: screening" in adv.summary
    assert "not for public-safety decisions" in adv.summary
    assert all(w.caution for w in adv.harvest)


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
        fh.writelines(f"{t.isoformat()},{v:.4f}\n" for t, v in zip(times, h + 0.7))
    store = str(tmp_path / "store")
    assert cli_main(["fit", str(csv), "--station", "Cove", "--store", store,
                     "--no-select"]) == 0
    capsys.readouterr()
    day = T0.strftime("%Y-%m-%d")
    assert cli_main(["advise", "Cove", day, "--store", store]) == 0
    out = capsys.readouterr().out
    assert "safe harvest window" in out and "rip risk" in out
