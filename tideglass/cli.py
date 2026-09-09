"""Command-line interface for Tideglass.

    tideglass fit <csv> [--station NAME] [--store DIR] [--alpha A] [--no-select]
    tideglass predict <station> <date> [--store DIR] [--days N]
    tideglass bench <csv> [--test-fraction F] [--against pytides|none]
    tideglass advise <station> <date> [--store DIR] [--days N]
    tideglass fetch <station> <begin> <end> [--out CSV] [--datum D] [--interval I]

``fit`` reads ``time,height`` rows (ISO-8601 datetimes, metres) and saves a
model artifact; ``predict`` prints an hourly height curve with 95% bands;
``fetch`` downloads a gauge CSV directly from NOAA CO-OPS.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timedelta, timezone

from tideglass.marea.metrics import evaluate, peak_tide_error, rmse
from tideglass.marea.model import TideModel

DEFAULT_STORE = ".tideglass"


def parse_time(s: str) -> datetime:
    s = s.strip()
    if s.endswith(("Z", "z")):
        s = s[:-1] + "+00:00"
    t = datetime.fromisoformat(s)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t


def read_csv(path: str):
    times, heights = [], []
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if not row or not row[0].strip():
                continue
            try:
                t = parse_time(row[0])
            except ValueError:
                continue  # header line
            times.append(t)
            heights.append(float(row[1]))
    if not times:
        raise ValueError(f"no readable time,height rows in {path!r}")
    return times, heights


def cmd_fit(args) -> int:
    try:
        times, heights = read_csv(args.csv)
    except (OSError, ValueError) as exc:
        print(f"tideglass fit: {exc}", file=sys.stderr)
        return 2
    station = args.station or os.path.splitext(os.path.basename(args.csv))[0]
    try:
        model = TideModel.fit(
            times, heights,
            auto_select=not args.no_select,
            alpha=args.alpha,
            station=station,
        )
    except ValueError as exc:
        print(f"tideglass fit: {exc}", file=sys.stderr)
        return 2
    os.makedirs(args.store, exist_ok=True)
    path = os.path.join(args.store, f"{station}.json")
    with open(path, "w") as fh:
        json.dump(model.to_artifact(), fh, indent=2)
    print(f"station: {station}")
    print(f"observations: {len(times)}  rmse: {model.meta['rmse']:.4f} m")
    print("constituents:")
    for f in model.constituents():
        print(f"  {f.name:<5} A={f.amplitude:.4f} m  kappa={f.phase_deg:7.2f} deg")
    print(f"saved: {path}")
    return 0


def cmd_predict(args) -> int:
    try:
        day = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        print(f"tideglass predict: bad date {args.date!r} (want YYYY-MM-DD)",
              file=sys.stderr)
        return 2
    path = os.path.join(args.store, f"{args.station}.json")
    if not os.path.exists(path):
        print(f"tideglass predict: no model for station {args.station!r} "
              f"(looked in {args.store!r})", file=sys.stderr)
        return 2
    model = TideModel.load_harmonic(path)
    n = args.days * 24
    times = [day + timedelta(hours=h) for h in range(n)]
    pred = model.predict(times)
    print(f"# station={model.station} date={args.date} n={n}")
    print("# time height_m lower_m upper_m")
    for t, m, lo, hi in zip(times, pred.mean, pred.lower, pred.upper):
        print(f"{t.isoformat()} {m:.4f} {lo:.4f} {hi:.4f}")
    return 0


def _load_pytides():
    """Import pytides with minimal Py2-era compat shims.

    The canonical package (0.0.4) is unmaintained: it uses ``collections.Iterable``,
    ``functools.reduce``-as-builtin, ``numpy.float`` and top-level intra-package
    imports. We shim those at runtime purely to run the head-to-head benchmark.
    Raises ImportError/RuntimeError with a clear message when unavailable.
    """
    import builtins
    import collections
    import collections.abc
    import functools
    import importlib.util

    for _name in ("Iterable", "Mapping", "Sequence"):
        if not hasattr(collections, _name):
            setattr(collections, _name, getattr(collections.abc, _name))
    if not hasattr(builtins, "reduce"):
        builtins.reduce = functools.reduce
    import numpy as _np

    if not hasattr(_np, "float"):
        _np.float = _np.float64  # noqa: NP001, runtime shim for pytides only

    spec = importlib.util.find_spec("pytides")
    if spec is None or not spec.origin:
        raise ImportError("pytides is not installed (pip install pytides)")
    pkg_dir = os.path.dirname(spec.origin)
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)
    try:
        import tide as pytides_tide
    except Exception as exc:
        raise RuntimeError(f"could not import pytides: {exc}") from exc
    return pytides_tide


def _fmt(x: float) -> str:
    if x != x:  # noqa: PLR0124 - the standard NaN check idiom
        return "n/a"
    return f"{x:.4f}"


def cmd_bench(args) -> int:
    import numpy as np

    try:
        times, heights = read_csv(args.csv)
    except (OSError, ValueError) as exc:
        print(f"tideglass bench: {exc}", file=sys.stderr)
        return 2
    order = sorted(range(len(times)), key=lambda i: times[i])
    times = [times[i] for i in order]
    y = np.array([heights[i] for i in order], dtype=float)
    n_test = max(24, round(len(times) * args.test_fraction))
    train_t, train_y = times[:-n_test], y[:-n_test]
    test_t, test_y = times[-n_test:], y[-n_test:]
    station = args.station or os.path.splitext(os.path.basename(args.csv))[0]

    model = TideModel.fit(train_t, train_y, alpha=args.alpha, station=station)
    marea = evaluate(model.predict(test_t), test_y)

    pytides_row = None
    if args.against == "pytides":
        try:
            tide_mod = _load_pytides()
            naive = [t.replace(tzinfo=None) for t in train_t]
            pt = tide_mod.Tide.decompose(np.asarray(train_y, dtype=float), t=naive)
            py_mean = np.asarray(
                pt.at([t.replace(tzinfo=None) for t in test_t]), dtype=float
            ).ravel()
            pytides_row = {
                "rmse": rmse(py_mean, test_y),
                "peak_error": peak_tide_error(py_mean, test_y),
            }
        except Exception as exc:  # noqa: BLE001 - degrade to Marea-only metrics
            print(f"# pytides comparison skipped: {exc}")

    print(f"station: {station}  train: {len(train_t)}  test: {len(test_t)}")
    print(f"{'model':<10}{'rmse(m)':>10}{'peak_err':>10}{'coverage':>10}")
    print(f"{'marea':<10}{marea['rmse']:>10.4f}"
          f"{_fmt(marea['peak_error']):>10}{marea['coverage']:>10.4f}")
    if pytides_row is not None:
        print(f"{'pytides':<10}{pytides_row['rmse']:>10.4f}"
              f"{_fmt(pytides_row['peak_error']):>10}{'n/a':>10}")
        winner = "marea" if marea["rmse"] <= pytides_row["rmse"] else "pytides"
        print(f"winner (rmse): {winner}")
    else:
        print("winner (rmse): marea (uncontested — pytides unavailable)")
    return 0


def _load_station(store: str, station: str):
    from tideglass.marea.model import TideModel

    path = os.path.join(store, f"{station}.json")
    if not os.path.exists(path):
        print(f"tideglass: no model for station {station!r} "
              f"(looked in {store!r})", file=sys.stderr)
        return None
    return TideModel.load_harmonic(path)


def cmd_advise(args) -> int:
    from tideglass.marine.advisor import TideAdvisor

    try:
        day = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        print(f"tideglass advise: bad date {args.date!r} (want YYYY-MM-DD)",
              file=sys.stderr)
        return 2
    model = _load_station(args.store, args.station)
    if model is None:
        return 2
    times = [day + timedelta(hours=h) for h in range(args.days * 24)]
    print(TideAdvisor(model).advise(times).summary)
    return 0


def cmd_fetch(args) -> int:
    from tideglass.fetch import fetch_noaa, write_csv

    out = args.out or f"{args.station}.csv"
    try:
        rows = fetch_noaa(args.station, args.begin, args.end,
                          datum=args.datum, interval=args.interval)
    except (OSError, ValueError) as exc:
        print(f"tideglass fetch: {exc}", file=sys.stderr)
        return 2
    if not rows:
        print(f"tideglass fetch: no data for station {args.station!r}", file=sys.stderr)
        return 2
    write_csv(rows, out)
    print(f"station: {args.station}  rows: {len(rows)}  datum: {args.datum}")
    print(f"saved: {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tideglass", description="Tide intelligence engine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_fit = sub.add_parser("fit", help="fit a model from observations")
    p_fit.add_argument("csv", help="CSV file with time,height rows")
    p_fit.add_argument("--station", default=None, help="station name (default: CSV stem)")
    p_fit.add_argument("--store", default=DEFAULT_STORE, help="artifact directory")
    p_fit.add_argument("--alpha", type=float, default=0.05, help="selection significance")
    p_fit.add_argument("--no-select", action="store_true", help="fit principal 8 directly")
    p_fit.set_defaults(func=cmd_fit)

    p_pred = sub.add_parser("predict", help="height curve with 95%% bands")
    p_pred.add_argument("station", help="station name")
    p_pred.add_argument("date", help="YYYY-MM-DD (UTC)")
    p_pred.add_argument("--store", default=DEFAULT_STORE, help="artifact directory")
    p_pred.add_argument("--days", type=int, default=1, help="days from midnight")
    p_pred.set_defaults(func=cmd_predict)

    p_bench = sub.add_parser("bench", help="benchmark Marea vs pytides on a gauge CSV")
    p_bench.add_argument("csv", help="CSV file with time,height rows")
    p_bench.add_argument("--station", default=None, help="station name (default: CSV stem)")
    p_bench.add_argument("--test-fraction", type=float, default=0.25,
                         help="held-out fraction (chronological tail)")
    p_bench.add_argument("--alpha", type=float, default=0.05, help="selection significance")
    p_bench.add_argument("--against", choices=["pytides", "none"], default="pytides",
                         help="baseline to beat")
    p_bench.set_defaults(func=cmd_bench)

    p_adv = sub.add_parser("advise", help="harvesting + rip + species advice")
    p_adv.add_argument("station", help="station name")
    p_adv.add_argument("date", help="YYYY-MM-DD (UTC)")
    p_adv.add_argument("--store", default=DEFAULT_STORE, help="artifact directory")
    p_adv.add_argument("--days", type=int, default=1, help="days from midnight")
    p_adv.set_defaults(func=cmd_advise)

    p_fetch = sub.add_parser("fetch", help="download NOAA CO-OPS gauge CSV")
    p_fetch.add_argument("station", help="NOAA station id (e.g. 9414290)")
    p_fetch.add_argument("begin", help="start date YYYY-MM-DD (UTC)")
    p_fetch.add_argument("end", help="end date YYYY-MM-DD (UTC)")
    p_fetch.add_argument("--out", default=None, help="output CSV (default: <station>.csv)")
    p_fetch.add_argument("--datum", default="MLLW", help="NOAA vertical datum")
    p_fetch.add_argument("--interval", default="h", choices=["h", "1", "hilo"],
                         help="passed to NOAA (observed water levels are 6-min)")
    p_fetch.set_defaults(func=cmd_fetch)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
