"""Tests for residual/surge decomposition."""

import numpy as np

from tideglass.marea import surge as SU


def test_decompose_flags_injected_surge():
    n = 720
    astro = np.sin(np.arange(n) * 0.5)
    obs = astro + np.random.default_rng(4).normal(0.0, 0.02, n)
    obs[100:106] += 0.5  # 6-hour surge
    dec = SU.decompose(obs, astro)
    assert dec.flags[100:106].all()
    assert dec.flags.sum() <= 10
    assert dec.threshold == 3.0 * dec.sigma > 0


def test_decompose_mismatch_raises():
    try:
        SU.decompose([1.0, 2.0], [1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_fit_ar1_recovers_phi():
    rng = np.random.default_rng(9)
    x = [0.0]
    for _ in range(2000):
        x.append(0.7 * x[-1] + rng.normal())
    ar = SU.fit_ar1(np.array(x))
    assert abs(ar.phi - 0.7) < 0.05
    assert ar.sigma_eps > 0 and ar.sigma > 0


def test_forecast_decays_and_grows():
    resid = np.linspace(1.0, 0.0, 50)
    ar = SU.fit_ar1(np.random.default_rng(2).normal(0, 1, 500))
    mean, std = SU.forecast_ar1(resid, ar, 12)
    assert mean.shape == std.shape == (12,)
    assert (np.diff(std) >= -1e-12).all()  # uncertainty is non-decreasing
    m0, s0 = SU.forecast_ar1(resid, SU.AR1(phi=0.0, sigma_eps=0.5, sigma=0.5), 4)
    assert (m0 == 0.0).all() and (s0 == 0.5).all()
