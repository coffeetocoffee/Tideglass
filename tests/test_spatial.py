"""Tests for multi-station EOF harmonization."""

import numpy as np
import pytest

from tideglass.marea import spatial as SP


def _region(seed=6, noise_noisy=0.15):
    rng = np.random.default_rng(seed)
    t = np.arange(30 * 24, dtype=float)
    shared = 1.0 * np.cos(0.5059 * t - 0.4) + 0.3 * np.cos(1.0 * t + 1.1)
    return {
        "A": 1.0 * shared + rng.normal(0, 0.02, t.size),
        "B": 0.6 * shared + 0.2 + rng.normal(0, 0.02, t.size),
        "C": 1.3 * shared - 0.1 + rng.normal(0, noise_noisy, t.size),
    }, shared


def test_eof_denoises_noisy_station():
    series, shared = _region()
    res = SP.harmonize(series)
    assert res.total_explained >= 0.95
    assert res.explained[0] > 0.9  # one dominant shared mode
    raw_err = float(np.mean((series["C"] - (1.3 * shared - 0.1)) ** 2))
    recon_err = float(np.mean((res.reconstructed["C"] - (1.3 * shared - 0.1)) ** 2))
    assert recon_err < raw_err


def test_n_modes_respected_and_means_kept():
    series, _ = _region()
    res = SP.harmonize(series, n_modes=1)
    assert res.modes.shape[0] == 1 and len(res.reconstructed) == 3
    for k, v in series.items():
        assert abs(res.reconstructed[k].mean() - v.mean()) < 1e-9


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        SP.harmonize({"only": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        SP.harmonize({"A": [1.0, 2.0], "B": [1.0]})
    with pytest.raises(ValueError):
        SP.harmonize({"A": [1.0, 1.0], "B": [2.0, 2.0]})
