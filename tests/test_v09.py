"""v0.9 — optional accelerated kernel, API freeze, plugin packs, report."""

import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tideglass import TideModel
from tideglass.marea import constituents as C
from tideglass.marea.contract import PUBLIC_API, api_version, verify_public_api
from tideglass.marea.kernel import (
    _HAS_NUMBA,
    available_backends,
    basis_matrix,
)
from tideglass.marea.plugins import (
    all_constituents,
    find,
    list_packs,
    load_pack_file,
    register_constituent,
    register_pack,
)
from tideglass.marea.report import build_report, gauge_files, validate_gauge

UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _hourly(hours):
    return [T0 + timedelta(hours=int(h)) for h in hours]


# --- kernel -------------------------------------------------------------------


def test_backends_always_include_numpy():
    assert "numpy" in available_backends()
    if _HAS_NUMBA:
        assert available_backends() == ["numpy", "numba"]


def test_kernel_rejects_unknown_backend():
    with pytest.raises(ValueError):
        basis_matrix(C.principal(), _hourly(range(3)), backend="rust")


def test_numba_kernel_matches_numpy_exactly():
    if not _HAS_NUMBA:
        pytest.skip("numba not installed")
    times = _hourly(range(0, 24 * 90, 3))
    for consts in (C.principal(), C.CATALOG):
        A_np = basis_matrix(consts, times, "numpy")
        A_nb = basis_matrix(consts, times, "numba")
        assert A_np.shape == A_nb.shape
        assert np.allclose(A_np, A_nb, atol=1e-9)


def test_auto_backend_is_safe_without_numba():
    times = _hourly(range(0, 48, 3))
    A_auto = basis_matrix(C.principal(), times, "auto")
    A_np = basis_matrix(C.principal(), times, "numpy")
    assert np.allclose(A_auto, A_np)


def test_fit_and_predict_with_numba_kernel_match_numpy():
    if not _HAS_NUMBA:
        pytest.skip("numba not installed")
    from tideglass.marea.solver import rad_per_hour

    def synth(times):
        t = np.array([(x - T0).total_seconds() / 3600.0 for x in times])
        y = np.full_like(t, 0.7)
        for name, (a, p) in {"M2": (1.0, 0.5), "S2": (0.3, -1.2),
                             "K1": (0.15, 0.0), "O1": (0.1, 2.8)}.items():
            w = rad_per_hour(C.speed(C.get(name)))
            y = y + a * np.cos(w * t - p)
        return y

    train = _hourly(range(60 * 24))
    m_np = TideModel.fit(train, synth(train), alpha=1e-4, kernel="numpy")
    m_nb = TideModel.fit(train, synth(train), alpha=1e-4, kernel="numba")
    test = _hourly(range(60 * 24, 60 * 24 + 48))
    assert np.allclose(m_np.predict(test).mean, m_nb.predict(test).mean,
                       atol=1e-9)
    assert np.allclose(m_np._coef, m_nb._coef, atol=1e-9)


# --- public API freeze --------------------------------------------------------


def test_api_contract_holds():
    verify_public_api()


def test_api_contract_is_frozen_v1():
    assert api_version == "1.0"
    assert isinstance(PUBLIC_API, frozenset)
    # A few names that MUST be in the frozen contract.
    for name in ("TideModel", "verify_public_api", "register_pack",
                 "basis_matrix", "available_backends", "api_version"):
        assert name in PUBLIC_API


# --- plugin constituent packs -------------------------------------------------


def test_seed_packs_registered():
    packs = list_packs()
    for name in ("rivers", "great_lakes", "solid_earth"):
        assert name in packs and len(packs[name]) >= 4


def test_find_resolves_builtin_then_plugin():
    assert find("M2") is C.get("M2")          # built-in wins
    assert find("2M2").doodson == (4, 0, 0, 0, 0, 0)
    with pytest.raises(KeyError):
        find("NO_SUCH_CONSTITUENT_XYZ")


def test_plugin_speeds_are_composition_exact():
    # 2M2 must be exactly twice M2's derived speed (linearity of the Doodson
    # argument), and 2SM6 exactly 2x(M2+S2).
    s = C.speed
    assert s(find("2M2")) == pytest.approx(2 * s(find("M2")))
    assert s(find("2SM6")) == pytest.approx(2 * (s(find("M2")) + s(find("S2"))))
    assert s(find("MK3")) == pytest.approx(s(find("M2")) + s(find("K1")))


def test_all_constituents_includes_plugins_without_duplicates():
    allc = all_constituents()
    names = [c.name for c in allc]
    assert len(names) == len(set(names))
    assert "M2" in names and "2M2" in names
    # built-ins come first
    assert names.index("M2") < names.index("2M2")


def test_constituents_get_resolves_plugin_names():
    assert C.get("2M2").name == "2M2"


def test_register_pack_and_constituent(tmp_path):
    register_pack("test_pack", [C.Constituent("TEST9", (9, 0, 0, 0, 0, 0))])
    assert "TEST9" in list_packs()["test_pack"]
    register_constituent(C.Constituent("TEST10", (10, 0, 0, 0, 0, 0)),
                         pack="test_pack")
    assert "TEST10" in list_packs()["test_pack"]
    # replace-on-same-name is idempotent
    register_constituent(C.Constituent("TEST9", (9, 0, 0, 0, 0, 0)),
                         pack="test_pack")
    assert list_packs()["test_pack"].count("TEST9") == 1


def test_load_pack_file_json(tmp_path):
    payload = {
        "name": "my_coast",
        "constituents": [
            {"name": "LOCAL1", "doodson": [4, 0, 0, 0, 0, 0],
             "species": "local"},
            {"name": "LOCAL2", "doodson": [1, 1, 0, 0, 0, 0], "phase0": 90.0},
        ],
    }
    path = tmp_path / "my_coast.json"
    path.write_text(json.dumps(payload))
    pack = load_pack_file(str(path))
    assert pack.name == "my_coast"
    assert pack.source == str(path)
    assert find("LOCAL1").species == "local"
    assert find("LOCAL2").phase0 == 90.0


def test_load_pack_dir_skips_garbage(tmp_path):
    (tmp_path / "good.json").write_text(
        json.dumps([{"name": "DIRC1", "doodson": [2, 0, 0, 0, 0, 0]}]))
    (tmp_path / "bad.json").write_text("{not json")
    (tmp_path / "notes.txt").write_text("ignore me")
    from tideglass.marea.plugins import load_pack_dir

    loaded = load_pack_dir(str(tmp_path))
    assert "good" in loaded and "bad" not in loaded
    assert find("DIRC1").name == "DIRC1"


def test_plugin_constituent_fits_and_recovers_amplitude():
    from tideglass.marea.solver import rad_per_hour

    c = find("2M2")
    times = _hourly(range(30 * 24))
    t = np.arange(len(times), dtype=float)
    y = 0.5 + 0.8 * np.cos(rad_per_hour(C.speed(c)) * t - 0.3)
    model = TideModel.fit(times, y, auto_select=False, candidates=[c])
    fit = model.constituents()[0]
    assert fit.name == "2M2"
    assert fit.amplitude == pytest.approx(0.8, abs=0.01)


def test_builtin_catalog_untouched_by_plugins():
    assert len(C.CATALOG) == 17
    assert [c.name for c in C.principal()] == \
        ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"]


# --- global validation report --------------------------------------------------

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def test_gauge_files_finds_bundled_records():
    files = gauge_files(DATA)
    assert files and all(f.endswith(".csv") for f in files)


def test_validate_gauge_bench_and_eva():
    files = gauge_files(DATA)
    longest = max(files, key=lambda p: os.path.getsize(p))
    res = validate_gauge(longest)
    assert "error" not in res
    assert 0 < res["rmse"] < 0.5
    assert 0.0 <= res["coverage"] <= 1.0
    assert "eva" in res


def test_validate_gauge_reports_error_for_bad_input(tmp_path):
    bad = tmp_path / "empty.csv"
    bad.write_text("time,height\n")
    res = validate_gauge(str(bad))
    assert "error" in res


def test_build_report_is_one_page_markdown():
    md = build_report(DATA)
    assert md.startswith("# Tideglass — Global Validation Report")
    for section in ("## 1. Coverage", "## 2. Per-region validation",
                    "## 3. Gauge records", "## 4. Engine inventory"):
        assert section in md
    # every harmonics region appears
    for region in ("US West Coast", "Europe (Atlantic)", "Pacific (Asia)"):
        assert region in md
    # engine inventory mentions the plugin packs and the backends
    assert "rivers" in md and "solid_earth" in md
    assert "numpy" in md
