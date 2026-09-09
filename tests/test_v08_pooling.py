"""Tests for v0.8 — hierarchical partial pooling across the station network."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tideglass import HierarchicalPool, TideModel  # top-level exports (v0.8)
from tideglass.cli import main as cli_main
from tideglass.marea import constituents as C
from tideglass.marea.solver import rad_per_hour

UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)
NAMES = ["M2", "S2", "N2", "K1", "O1"]


def _hourly(start, n):
    return [start + timedelta(hours=h) for h in range(n)]


def _tide(times, amps, phases):
    t = np.array([(x - T0).total_seconds() / 3600.0 for x in times])
    y = np.zeros_like(t)
    for name, amp, phi in zip(NAMES, amps, phases):
        w = float(rad_per_hour(C.speed(C.get(name))))
        y = y + amp * np.cos(w * t - phi)
    return y


# Three well-observed neighbours: same ocean, slightly different constants.
NET = {
    "north": ([1.00, 0.30, 0.20, 0.15, 0.10], [0.5, -1.2, 2.0, 0.0, 2.8]),
    "mid": ([1.04, 0.31, 0.19, 0.16, 0.10], [0.5, -1.2, 2.0, 0.0, 2.8]),
    "south": ([0.96, 0.29, 0.21, 0.14, 0.11], [0.5, -1.2, 2.0, 0.0, 2.8]),
}
SHORT_TRUE = ([1.02, 0.30, 0.20, 0.15, 0.10], [0.5, -1.2, 2.0, 0.0, 2.8])


def _network(noise=0.02, seed=0, days=60):
    rng = np.random.default_rng(seed)
    models = {}
    for station, (amps, phases) in NET.items():
        times = _hourly(T0, days * 24)
        y = _tide(times, amps, phases) + 0.7 + rng.normal(0, noise, days * 24)
        models[station] = TideModel.fit(times, y, alpha=1e-4, station=station)
    return models


def _amp(model, name):
    return next(f.amplitude for f in model.constituents() if f.name == name)


# --- core behaviour ------------------------------------------------------------


def test_pool_improves_short_noisy_record():
    models = _network()
    pool = HierarchicalPool(models)
    short_t = _hourly(T0, 3 * 24)  # 3 days: M2/S2 inseparable, very noisy
    rng = np.random.default_rng(42)
    short_y = (_tide(short_t, *SHORT_TRUE) + 0.7
               + rng.normal(0, 0.20, len(short_t)))
    own = TideModel.fit(short_t, short_y, auto_select=False,
                        candidates=pool.union, station="short")
    pooled = pool.seed_short(short_t, short_y, "short")

    held = _hourly(short_t[-1] + timedelta(hours=1), 10 * 24)
    truth = _tide(held, *SHORT_TRUE) + 0.7
    from tideglass.marea.metrics import rmse
    assert rmse(pooled.predict(held).mean, truth) < rmse(
        own.predict(held).mean, truth)

    # The short record leans on the network (high shrinkage), M2 near truth.
    assert pooled.meta["mean_shrinkage"] > 0.25
    assert abs(_amp(pooled, "M2") - 1.02) < 0.08
    assert {f.name for f in pooled.constituents()} == {
        f.name for f in own.constituents()}


def test_pool_leaves_long_clean_records_alone():
    models = _network(noise=0.01, days=90)
    pool = HierarchicalPool(models)
    for station in NET:
        pooled = pool.pool(station)
        # Clean records earn their weight: pooled ≈ own fit ...
        assert abs(_amp(pooled, "M2") - _amp(models[station], "M2")) < 0.005
        assert pooled.meta["mean_shrinkage"] < 0.1
        # ... and predict identically on held-out data.
        held = _hourly(T0, 10 * 24)
        from tideglass.marea.metrics import rmse
        assert rmse(pooled.predict(held).mean,
                    models[station].predict(held).mean) < 0.005


def test_pool_imputes_constituent_target_never_selected():
    models = _network()
    # A member fit on M2+K1 only: S2 must be borrowed from the network.
    times = _hourly(T0, 60 * 24)
    amps, phases = NET["mid"]
    y = _tide(times, amps, phases) + 0.7
    thin = TideModel.fit(
        times, y, auto_select=False,
        candidates=[C.get("M2"), C.get("K1")], station="thin")
    pool2 = HierarchicalPool({**models, "thin": thin})
    pooled = pool2.pool("thin")
    names = [f.name for f in pooled.constituents()]
    assert "S2" in names  # imputed from the regional prior
    assert abs(_amp(pooled, "S2") - 0.30) < 0.05


def test_report_shrinkage_in_unit_interval():
    pool = HierarchicalPool(_network())
    for r in pool.report("mid"):
        assert 0.0 <= r.shrinkage <= 1.0
        assert r.tau >= 0.0
        assert r.amplitude >= 0.0


def test_pool_needs_two_stations_raises():
    models = _network()
    with pytest.raises(ValueError):
        HierarchicalPool({"only": models["mid"]})


def test_pool_unknown_station_raises():
    pool = HierarchicalPool(_network())
    with pytest.raises(ValueError):
        pool.pool("ghost")


# --- spatial prior -------------------------------------------------------------


def test_spatial_prior_pulls_toward_nearby_neighbour():
    rng = np.random.default_rng(3)
    # Amplitudes grow eastward: 1.00 / 1.10 / 1.20 at lon 0 / 1 / 2.
    models = {}
    for station, lon, amp in [("w", 0.0, 1.00), ("c", 1.0, 1.10), ("e", 2.0, 1.20)]:
        times = _hourly(T0, 60 * 24)
        y = _tide(times, [amp, 0.3, 0.2, 0.15, 0.1],
                  SHORT_TRUE[1]) + 0.7 + rng.normal(0, 0.02, len(times))
        models[station] = TideModel.fit(times, y, alpha=1e-4, station=station)
    coords = {"w": (0.0, 0.0), "c": (1.0, 0.0), "e": (2.0, 0.0)}
    pool = HierarchicalPool(models, coords=coords)
    short_t = _hourly(T0, 2 * 24)
    short_y = (_tide(short_t, [1.19, 0.3, 0.2, 0.15, 0.1], SHORT_TRUE[1])
               + 0.7 + rng.normal(0, 0.25, len(short_t)))
    pooled = pool.seed_short(short_t, short_y, "new", target_coords=(1.99, 0.0))
    # The eastern neighbour (1.20) dominates the IDW prior: pooled M2 lands
    # above the plain network mean (~1.10).
    assert _amp(pooled, "M2") > 1.12


# --- CLI -----------------------------------------------------------------------


def _write_csv(path, times, heights):
    with open(path, "w") as fh:
        fh.write("time,height\n")
        fh.writelines(f"{t.isoformat()},{h:.4f}\n" for t, h in zip(times, heights))


def test_cli_pool_end_to_end(tmp_path, capsys):
    rng = np.random.default_rng(13)
    store = str(tmp_path / "store")
    for station, (amps, phases) in NET.items():
        times = _hourly(T0, 60 * 24)
        y = _tide(times, amps, phases) + 0.7 + rng.normal(0, 0.02, len(times))
        _write_csv(tmp_path / f"{station}.csv", times, y)
        rc = cli_main(["fit", str(tmp_path / f"{station}.csv"),
                       "--station", station, "--store", store])
        assert rc == 0
        capsys.readouterr()
    short_t = _hourly(T0, 3 * 24)
    short_y = (_tide(short_t, *SHORT_TRUE) + 0.7
               + rng.normal(0, 0.20, len(short_t)))
    _write_csv(tmp_path / "short.csv", short_t, short_y)
    rc = cli_main(["pool", str(tmp_path / "short.csv"), "--station", "short",
                   "--store", store])
    assert rc == 0
    out = capsys.readouterr().out
    assert "shrinkage" in out
    assert (tmp_path / "store" / "short.json").exists()
