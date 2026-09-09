"""Tests for TideModel and the CLI (Step 4 acceptance)."""

import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np

from tideglass import TideModel  # top-level export (MVP definition of done)
from tideglass.cli import main as cli_main
from tideglass.marea import constituents as C
from tideglass.marea.astronomy import doodson_args, nodal_factor
from tideglass.marea.solver import rad_per_hour

UTC = timezone.utc
TRUE = {"M2": (1.0, 0.5), "S2": (0.30, -1.2), "N2": (0.20, 2.0),
        "K1": (0.15, 0.0), "O1": (0.10, 2.8)}
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _hours(times):
    t0 = T0
    return np.array([(t - t0).total_seconds() / 3600.0 for t in times])


def _synthetic(times, noise=0.0, seed=0):
    t = _hours(times)
    y = np.full_like(t, 0.7)
    for name, (amp, phi) in TRUE.items():
        w = float(rad_per_hour(C.speed(C.get(name))))
        y = y + amp * np.cos(w * t - phi)
    if noise:
        y = y + np.random.default_rng(seed).normal(0.0, noise, size=t.size)
    return y


def _hourly(start, n):
    return [start + timedelta(hours=h) for h in range(n)]


def test_fit_selects_exact_subset_and_covers():
    train = _hourly(T0, 60 * 24)
    y = _synthetic(train, noise=0.02, seed=3)
    model = TideModel.fit(train, y, alpha=1e-4, station="test")
    assert {f.name for f in model.constituents()} == set(TRUE)

    held = _hourly(train[-1] + timedelta(hours=1), 30 * 24)
    truth = _synthetic(held)
    noisy = _synthetic(held, noise=0.02, seed=99)
    pred = model.predict(held)
    cover = np.mean((pred.lower <= noisy) & (noisy <= pred.upper))
    assert cover >= 0.90  # 95% prediction interval
    assert math.sqrt(np.mean((pred.mean - truth) ** 2)) < 0.02


def test_constituent_amplitudes_close():
    # The fit solves for *equilibrium* amplitudes (normalized to f=1) on the
    # nodal-modulated basis, while the plain synthetic carries the locally
    # modulated amplitude — so compare A_fit * fbar against truth.
    from tideglass.marea.astronomy import nodal_factor as _nf

    train = _hourly(T0, 60 * 24)
    model = TideModel.fit(train, _synthetic(train, noise=0.02, seed=3), alpha=1e-4)
    sample = train[::24]
    for f in model.constituents():
        c = C.get(f.name)
        fbar = sum(_nf(c, t)[0] for t in sample) / len(sample)
        assert abs(f.amplitude * fbar - TRUE[f.name][0]) / TRUE[f.name][0] < 0.03
        assert 0.0 <= f.phase_deg < 360.0


def test_noaa_load_matches_equilibrium_argument():
    t0 = datetime(2026, 9, 10, 12, tzinfo=UTC)
    model = TideModel.load_harmonic({
        "station": "X", "mean": 1.0,
        "constituents": [{"name": "M2", "amplitude": 2.0, "phase": 30.0}],
    })
    tau, s, h, p, N, p1 = doodson_args(t0)
    V = math.radians(2 * tau)  # M2 = 2τ, phase0 = 0
    f, u = nodal_factor(C.get("M2"), t0)
    expected = 1.0 + f * 2.0 * math.cos(V + math.radians(u - 30.0))
    pred = model.predict([t0])
    assert abs(pred.mean[0] - expected) < 1e-9
    # No covariance on the loaded path: band collapses to the mean.
    assert pred.lower[0] == pred.mean[0] == pred.upper[0]


def test_artifact_roundtrip(tmp_path):
    train = _hourly(T0, 30 * 24)
    model = TideModel.fit(train, _synthetic(train, noise=0.01, seed=5), alpha=1e-4)
    blob = json.dumps(model.to_artifact())
    via_str = TideModel.load_harmonic(blob)
    path = tmp_path / "m.json"
    path.write_text(blob)
    via_file = TideModel.load_harmonic(str(path))
    held = _hourly(train[-1] + timedelta(hours=1), 48)
    assert np.max(np.abs(via_str.predict(held).mean - model.predict(held).mean)) < 1e-12
    assert np.max(np.abs(via_file.predict(held).mean - model.predict(held).mean)) < 1e-12


def test_cli_fit_predict_roundtrip(tmp_path, capsys):
    train = _hourly(T0, 60 * 24)
    y = _synthetic(train, noise=0.02, seed=11)
    csv = tmp_path / "obs.csv"
    with open(csv, "w") as fh:
        fh.write("time,height\n")
        fh.writelines(f"{t.isoformat()},{h:.4f}\n" for t, h in zip(train, y))
    store = str(tmp_path / "store")
    assert cli_main(["fit", str(csv), "--station", "T", "--store", store,
                     "--alpha", "1e-4"]) == 0
    assert (tmp_path / "store" / "T.json").exists()

    day = (train[-1] + timedelta(hours=1)).strftime("%Y-%m-%d")
    capsys.readouterr()
    assert cli_main(["predict", "T", day, "--store", store]) == 0
    rows = [ln for ln in capsys.readouterr().out.splitlines()
            if ln and not ln.startswith("#")]
    assert len(rows) == 24
    for ln in rows:
        _, m, lo, hi = ln.split()
        assert abs(float(m)) < 3.0 and float(lo) <= float(m) <= float(hi)


def test_cli_errors(tmp_path):
    assert cli_main(["predict", "NOPE", "2026-09-10",
                     "--store", str(tmp_path)]) == 2
    assert cli_main(["predict", "NOPE", "not-a-date",
                     "--store", str(tmp_path)]) == 2
    assert cli_main(["fit", str(tmp_path / "missing.csv")]) == 2


def test_naive_datetimes_are_utc():
    aware = _hourly(T0, 200)
    naive = [t.replace(tzinfo=None) for t in aware]
    y = _synthetic(aware, seed=1)
    m1 = TideModel.fit(aware, y, auto_select=False)
    m2 = TideModel.fit(naive, y, auto_select=False)
    p1, p2 = m1.predict(aware), m2.predict(naive)
    assert np.max(np.abs(p1.mean - p2.mean)) == 0.0


def test_basis_matrix_matches_per_element():
    from tideglass.marea import constituents as C
    from tideglass.marea.model import _basis_matrix, _design_row

    times = _hourly(T0, 73)
    consts = C.principal()
    A = _basis_matrix(consts, times)
    n, m = len(times), len(consts)
    assert A.shape == (n, 1 + 2 * m)
    ref = np.empty_like(A)
    ref[:, 0] = 1.0
    for j, c in enumerate(consts):
        for i, t in enumerate(times):
            co, si = _design_row(c, t)
            ref[i, 1 + 2 * j] = co
            ref[i, 2 + 2 * j] = si
    assert np.max(np.abs(A - ref)) < 1e-12
