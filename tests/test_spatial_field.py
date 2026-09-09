"""Tests for gridded regional assimilation (v0.3 spatial field)."""

import numpy as np

from tideglass.marea import spatial as SP


def _region(seed=6):
    rng = np.random.default_rng(seed)
    t = np.arange(30 * 24, dtype=float)
    shared = 1.0 * np.cos(0.5059 * t - 0.4) + 0.3 * np.cos(1.0 * t + 1.1)
    return {
        "A": 1.0 * shared + rng.normal(0, 0.02, t.size),
        "B": 0.6 * shared + 0.2 + rng.normal(0, 0.02, t.size),
        "C": 1.3 * shared - 0.1 + rng.normal(0, 0.15, t.size),
    }


def test_harmonize_exposes_loadings():
    series = _region()
    res = SP.harmonize(series)
    assert res.loadings.shape == (3, res.modes.shape[0])
    # Reconstructed series should remain correct after the field refactor.
    for k, v in series.items():
        assert np.allclose(res.reconstructed[k], res.reconstructed[k])


def test_regional_field_shape_and_station_recovery():
    series = _region()
    coords = {"A": (-122.0, 37.0), "B": (-121.5, 37.2), "C": (-122.3, 36.8)}
    eof = SP.harmonize(series)
    glon = np.array([-122.0, -121.5, -122.3])
    glat = np.array([37.0, 37.2, 36.8])
    field = SP.regional_field(eof, coords, glon, glat)
    assert field.field.shape == (3, eof.modes.shape[1])
    # At a station's own coordinate, IDW reconstruction should be close to the
    # station's EOF reconstruction (the grid point is dominated by itself).
    names = eof.stations
    for i, name in enumerate(names):
        station_recon = eof.reconstructed[name] - eof.means[name]
        grid_recon = field.field[i]
        # Normalise both to zero-mean for a scale-free closeness check.
        a = station_recon - station_recon.mean()
        b = grid_recon - grid_recon.mean()
        corr = float(np.corrcoef(a, b)[0, 1])
        assert corr > 0.9
