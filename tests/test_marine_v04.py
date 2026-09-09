"""Tests for v0.4 marine depth: region rulesets, wave-coupled rip, surge alerts."""

from datetime import datetime, timedelta

import numpy as np

from tideglass.marea import constituents as C
from tideglass.marea.model import TideModel
from tideglass.marea.solver import rad_per_hour
from tideglass.marine import alerting, knowledge
from tideglass.marine.advisor import TideAdvisor
from tideglass.marine.rip import risk

T0 = datetime(2024, 1, 1)


def _times(n):
    return [T0 + timedelta(hours=h) for h in range(n)]


def _fit_model(n=200):
    t = np.arange(n)
    y = np.full(n, 0.7)
    for nm, amp in (("M2", 0.8), ("S2", 0.3)):
        w = float(rad_per_hour(C.speed(C.get(nm))))
        y = y + amp * np.cos(w * t - 0.5)
    return TideModel.fit(_times(n), y, auto_select=False)


def test_region_ruleset_merges_defaults():
    base = knowledge.get_region(None)
    assert "harvest" in base and "rip" in base
    pnw = knowledge.get_region("PacificNW")
    assert pnw["harvest"]["min_hours"] == 3.0
    assert pnw["harvest"]["surge_guard_hours"] == 36.0
    assert pnw["rip"]["wave_weight"] == 0.25
    assert pnw["rip"]["ref_rate_m_per_h"] == knowledge.RIP_DEFAULTS["ref_rate_m_per_h"]


def test_advisor_uses_region_and_waves():
    m = _fit_model()
    times = _times(48)
    adv = TideAdvisor(m, region="PacificNW")
    a = adv.advise(times, wave_height_m=2.5, wave_period_s=12.0)
    assert a.region == "PacificNW"
    assert "wave" in a.rip.basis.lower()
    a2 = TideAdvisor(m).advise(times)
    assert "waves not supplied" in a2.rip.basis


def test_wave_coupling_increases_risk():
    n = 100
    t = np.arange(n)
    h = 1.0 + np.sin(2 * np.pi * t / 12.0)
    times = _times(n)
    no_wave = risk(times, h, wave_weight=0.3, ref_wave_height=2.0)
    with_wave = risk(times, h, wave_weight=0.3, ref_wave_height=2.0,
                     wave_height_m=2.0, wave_period_s=14.0)
    assert with_wave.score.max() >= no_wave.score.max()


def test_surge_alerts_group_events():
    n = 200
    t = np.arange(n)
    # Same M2/S2 tide the model is fit on, so the residual is ~0 except where
    # we inject surge events.
    y = np.full(n, 0.7)
    for nm, amp in (("M2", 0.8), ("S2", 0.3)):
        w = float(rad_per_hour(C.speed(C.get(nm))))
        y = y + amp * np.cos(w * t - 0.5)
    times = _times(n)
    m = TideModel.fit(times, y, auto_select=False)
    observed = y.copy()
    observed[50:56] += 0.8
    observed[120:123] += 1.2
    events = alerting.surge_events(observed, m.predict(times).mean, times, threshold_m=0.3)
    assert len(events) == 2
    peaks = sorted(e.peak_residual_m for e in events)
    assert peaks[-1] > 1.0
