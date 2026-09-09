"""Tests for DCDM constituent selection."""

import numpy as np
import pytest

from tideglass.marea import constituents as C
from tideglass.marea import selection as SEL
from tideglass.marea import solver as S

TRUE = {"M2": (1.0, 0.5), "S2": (0.30, -1.2), "N2": (0.20, 2.0),
        "K1": (0.15, 0.0), "O1": (0.10, 2.8)}
SUPERSET = (["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"]
            + ["M4", "MS4", "MN4", "M6", "S4"]
            + ["Mf", "Mm", "Ssa", "Sa"])


def _candidates(names=SUPERSET):
    omegas = S.rad_per_hour([C.speed(C.get(n)) for n in names])
    return [SEL.Candidate(n, float(w)) for n, w in zip(names, omegas)]


def _signal(t, noise=0.0, seed=0):
    y = np.full_like(t, 0.7)
    for name, (amp, phi) in TRUE.items():
        w = float(S.rad_per_hour(C.speed(C.get(name))))
        y = y + amp * np.cos(w * t - phi)
    if noise:
        y = y + np.random.default_rng(seed).normal(0.0, noise, size=t.size)
    return y


def test_f_sf_known_critical_values():
    # 95th percentiles of standard F tables -> p ≈ 0.05.
    assert abs(SEL.f_sf(4.1028, 2, 10) - 0.05) < 1e-3
    assert abs(SEL.f_sf(161.4476, 1, 1) - 0.05) < 1e-3
    assert abs(SEL.f_sf(2.9957, 2, 20000) - 0.05) < 1e-3
    assert SEL.f_sf(0.0, 2, 10) == 1.0
    assert SEL.f_sf(1e6, 2, 10) < 1e-9
    assert SEL.f_sf(10.0, 2, 10) > SEL.f_sf(20.0, 2, 10)


def test_exact_subset_recovery():
    t = np.arange(0.0, 90 * 24, 1.0)  # 90 days hourly
    y = _signal(t, noise=0.02, seed=7)
    res = SEL.select(t, y, _candidates(), alpha=1e-4)
    assert res.selected[0] == "M2"
    assert set(res.selected) == set(TRUE)
    assert res.n_params == 1 + 2 * len(TRUE)
    # p-values strictly tighten the evidence bar: all decisive.
    assert all(p < 1e-4 for p in res.p_values)


def test_no_signal_selects_nothing():
    t = np.arange(0.0, 30 * 24, 1.0)
    y = np.full_like(t, 1.5)
    res = SEL.select(t, y, _candidates(["M2", "S2", "O1"]), start=())
    assert res.selected == []


def test_rayleigh_gate_blocks_unresolvable():
    # 10 days cannot resolve the M2/S2 beat (needs ~15 days).
    t = np.arange(0.0, 10 * 24, 1.0)
    y = _signal(t)
    cands = _candidates(["M2", "S2"])
    gated = SEL.select(t, y, cands, alpha=1e-4, min_separation_cycles=1.0)
    assert gated.selected == ["M2"]
    assert gated.excluded == ["S2"]
    ungated = SEL.select(t, y, cands, alpha=1e-4, min_separation_cycles=0)
    assert set(ungated.selected) == {"M2", "S2"}


def test_unresolvable_pair_resolves_to_dominant():
    # K1/P1 need ~182 days to separate: on 30 days P1 is excluded up front
    # and K1 is solved for — deterministically, regardless of noise.
    t = np.arange(0.0, 30 * 24, 1.0)
    for seed in (1, 7, 21, 42):
        y = _signal(t, noise=0.02, seed=seed)
        res = SEL.select(t, y, _candidates(), alpha=1e-4)
        assert "P1" in res.excluded and "K2" in res.excluded
        assert set(res.selected) == set(TRUE)


def test_long_period_excluded_from_short_record():
    # Ssa/Sa need months to separate from the mean level: on 40 days they
    # are excluded instead of aliasing anomalies into fake cycles.
    t = np.arange(0.0, 40 * 24, 1.0)
    y = _signal(t, noise=0.02, seed=11)
    res = SEL.select(t, y, _candidates(), alpha=1e-4)
    assert "Ssa" in res.excluded and "Sa" in res.excluded
    assert set(res.selected) == set(TRUE)


def test_duplicate_names_raise():
    t = np.arange(0.0, 100.0)
    y = np.sin(t)
    with pytest.raises(ValueError):
        SEL.select(t, y, [SEL.Candidate("M2", 0.5), SEL.Candidate("M2", 0.5)])
    with pytest.raises(ValueError):
        SEL.select(t, y[:-1], _candidates(["M2"]))
