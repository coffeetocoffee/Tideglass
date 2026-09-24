"""Benchmark tests: metrics are unit-tested in test_metrics.py; here the CLI
head-to-head (Marea vs real pytides) on synthetic gauge data."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import sys

from tideglass.cli import _load_pytides, _pytides_env, build_parser
from tideglass.cli import main as cli_main
from tideglass.marea import constituents as C
from tideglass.marea.solver import rad_per_hour

UTC = timezone.utc
TRUE = {"M2": (1.0, 0.5), "S2": (0.30, -1.2), "N2": (0.20, 2.0),
        "K1": (0.15, 0.0), "O1": (0.10, 2.8)}
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _gauge_csv(path, days=40, seed=21):
    times = [T0 + timedelta(hours=h) for h in range(days * 24)]
    t = np.arange(len(times), dtype=float)
    y = np.full_like(t, 0.7)
    for n, (a, p) in TRUE.items():
        w = float(rad_per_hour(C.speed(C.get(n))))
        y = y + a * np.cos(w * t - p)
    y = y + np.random.default_rng(seed).normal(0.0, 0.02, size=t.size)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("time,height\n")
        fh.writelines(f"{ti.isoformat()},{hi:.4f}\n" for ti, hi in zip(times, y))
    return str(path)


def _table(out):
    rows = {}
    for ln in out.splitlines():
        parts = ln.split()
        if parts and parts[0] in ("marea", "pytides"):
            rows[parts[0]] = parts[1:]
    return rows


def test_bench_marea_beats_pytides(tmp_path, capsys):
    try:
        _load_pytides()
    except Exception as exc:
        pytest.skip(f"pytides unavailable: {exc}")
    csv = _gauge_csv(tmp_path / "gauge.csv")
    assert cli_main(["bench", csv, "--station", "SYN",
                     "--test-fraction", "0.25", "--alpha", "1e-4"]) == 0
    out = capsys.readouterr().out
    assert "winner (rmse): marea" in out
    rows = _table(out)
    assert float(rows["marea"][0]) <= float(rows["pytides"][0])
    assert float(rows["marea"][2]) >= 0.90  # CI coverage


def test_bench_without_pytides(tmp_path, capsys):
    csv = _gauge_csv(tmp_path / "gauge.csv")
    assert cli_main(["bench", csv, "--against", "none"]) == 0
    out = capsys.readouterr().out
    assert "marea" in out and "uncontested" in out


def test_pytides_shims_do_not_leak_into_host_process():
    """The Py2-era compat shims must be scoped to the benchmark.

    pytides resolves these names lazily, so they have to be installed while it
    runs -- but leaking a patched numpy.float or a reduce injected into
    builtins would silently change every other library in the process.
    """
    import builtins
    import collections

    import numpy as np

    before = {
        "np_float": hasattr(np, "float"),
        "reduce": hasattr(builtins, "reduce"),
        "collections_abc": hasattr(collections, "Iterable"),
        "sys_path_len": len(sys.path),
    }
    try:
        with _pytides_env() as tide_mod:
            assert hasattr(tide_mod, "Tide")
            # inside the scope the shims are present
            assert hasattr(np, "float")
            assert hasattr(builtins, "reduce")
            assert hasattr(collections, "Iterable")
    except Exception as exc:
        pytest.skip(f"pytides unavailable: {exc}")
    assert hasattr(np, "float") == before["np_float"]
    assert hasattr(builtins, "reduce") == before["reduce"]
    assert hasattr(collections, "Iterable") == before["collections_abc"]
    assert len(sys.path) == before["sys_path_len"]


def test_bench_reports_scope_and_basis_caveat(tmp_path, capsys):
    csv = _gauge_csv(tmp_path / "gauge.csv")
    assert cli_main(["bench", csv, "--station", "SYN", "--against", "none",
                     "--test-fraction", "0.25", "--alpha", "1e-4"]) == 0
    out = capsys.readouterr().out
    assert "scope:" in out
    assert "not a multi-station result" in out


def test_help_formats():
    # Regression: unescaped % in help strings crashes argparse.
    parser = build_parser()
    assert "advise" in parser.format_help()
    for action in parser._subparsers._group_actions:
        for name, sub in action.choices.items():
            assert name in sub.format_help()
