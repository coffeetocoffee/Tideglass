"""Command-line interface for Tideglass.

    tideglass fit <csv> [--station NAME] [--store DIR] [--alpha A] [--no-select]
                 [--source S]
    tideglass predict <station> <date> [--store DIR] [--days N]
    tideglass bench <csv> [--test-fraction F] [--against pytides|tpxo|none]
                 [--global-model PATH --lon X --lat Y [--consts M2,S2,...]]
                 (PATH may be a CSV grid or a genuine TPXO/FES NetCDF3 .nc file)
    tideglass advise <station> <date> [--store DIR] [--days N]
                  [--flood M] [--surge-sigma S]
    tideglass fetch <station> <begin> <end> [--out CSV] [--datum D] [--interval I] [--met]
    tideglass surge <obs.csv> <met.csv> [--station S] [--store DIR] [--max-lag H]
                  [--forecast MET.csv] [--hours H] [--flood M] [--cost C --loss L]
    tideglass nowcast <station> <feed.csv> [--store DIR] [--alpha A]
                 [--auto-refit | --no-auto-refit] [--hours H]
    tideglass poll <station> [--store DIR] [--lookback-h H] [--repeat N]
                 [--sleep-s S] [--datum D] [--alpha A] [--no-auto-refit]
    tideglass contribute <csv> <lon> <lat> --station NAME [--store DIR] [--source S]
    tideglass network [--store DIR] [--threshold F]
    tideglass validate [--region R]

``fit`` reads ``time,height`` rows (ISO-8601 datetimes, metres) and saves a
model artifact; ``predict`` prints an hourly height curve with 95% bands;
``fetch`` downloads a gauge CSV directly from NOAA CO-OPS (``--met``: wind +
pressure instead); ``surge`` learns the local wind/pressure → surge response
and forecasts surge from met forcing; ``nowcast``
assimilates a fresh feed into the deployed model (with drift check and
optional auto-refit); ``poll`` repeats that live against NOAA on a sleep loop;
``pool`` fits a short record by borrowing strength from the network;
``extremes`` prints skew-surge stats, GPD return levels, and joint
tide/surge exceedance; ``plugins`` lists constituent packs; ``report``
renders the one-page global validation report.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timedelta, timezone

from tideglass.marea import export as EXPORT
from tideglass.marea.calibration import (
    constituent_attribution,
    evaluate_calibration,
)
from tideglass.marea.kalman import JointModel
from tideglass.marea.metrics import evaluate, peak_tide_error, rmse
from tideglass.marea.model import TideModel
from tideglass.marine.alerting import surge_events
from tideglass.tui import build_dashboard
from tideglass.web import run_server

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
    source = args.source or f"csv:{args.csv}"
    try:
        model = TideModel.fit(
            times, heights,
            auto_select=not args.no_select,
            alpha=args.alpha,
            station=station,
            source=source,
            kernel=args.kernel,
        )
    except ValueError as exc:
        print(f"tideglass fit: {exc}", file=sys.stderr)
        return 2
    os.makedirs(args.store, exist_ok=True)
    path = os.path.join(args.store, f"{station}.json")
    with open(path, "w") as fh:
        json.dump(model.to_artifact(), fh, indent=2)
    prov = model.meta
    print(f"station: {station}")
    print(f"observations: {len(times)}  rmse: {model.meta['rmse']:.4f} m")
    print(f"source: {prov.get('source')}  window: {prov.get('obs_start')} .. "
          f"{prov.get('obs_end')}")
    print(f"data_sha256: {prov.get('data_sha256')}  "
          f"tideglass: {prov.get('tideglass_version')}")
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


def cmd_smooth(args) -> int:
    try:
        times, heights = read_csv(args.csv)
    except (OSError, ValueError) as exc:
        print(f"tideglass smooth: {exc}", file=sys.stderr)
        return 2
    try:
        model = JointModel.fit(
            times, heights,
            auto_select=not args.no_select,
            alpha=args.alpha,
            station=args.station,
        )
    except ValueError as exc:
        print(f"tideglass smooth: {exc}", file=sys.stderr)
        return 2
    fit = model.fit_result
    print(f"# station={fit.meta.get('station', args.station)} n={fit.observed.size}")
    print(f"# secular_trend_mm_yr={fit.trend_mm_yr:+.3f} "
          f"±{fit.trend_mm_yr_se:.3f}  surge_phi={fit.phi:.3f}  rmse={fit.meta['rmse']:.4f}")
    print("# time observed model tide surge trend")
    for t, o, mo, ti, su, tr in zip(
        times, fit.observed, fit.model, fit.tide, fit.surge, fit.trend
    ):
        print(f"{t.isoformat()} {o:.4f} {mo:.4f} {ti:.4f} {su:.4f} {tr:.4f}")
    return 0


def cmd_calibrate(args) -> int:
    import numpy as np

    from tideglass.marea.calibration import (
        conformalize,
        pit_histogram,
        reliability_curve,
    )

    try:
        times, heights = read_csv(args.csv)
    except (OSError, ValueError) as exc:
        print(f"tideglass calibrate: {exc}", file=sys.stderr)
        return 2
    order = sorted(range(len(times)), key=lambda i: times[i])
    times = [times[i] for i in order]
    y = np.array([heights[i] for i in order], dtype=float)
    # Chronological three-way split: fit on the head, conformal scores on the
    # middle, distribution diagnostics on the tail.
    n_test = max(24, round(len(times) * args.test_fraction))
    n_calib = max(24, round(len(times) * args.calib_fraction))
    if len(times) < n_test + n_calib + 72:
        print(f"tideglass calibrate: need ≥ {n_test + n_calib + 72} rows "
              f"for a train/calib/test split, got {len(times)}",
              file=sys.stderr)
        return 2
    train_t, train_y = times[:-(n_test + n_calib)], y[:-(n_test + n_calib)]
    calib_t, calib_y = times[-(n_test + n_calib):-n_test], y[-(n_test + n_calib):-n_test]
    test_t, test_y = times[-n_test:], y[-n_test:]
    station = args.station or os.path.splitext(os.path.basename(args.csv))[0]

    model = TideModel.fit(train_t, train_y, alpha=args.alpha, station=station)
    pred = model.predict(test_t)
    cal = evaluate_calibration(pred, test_y)
    sig_test = (np.asarray(pred.upper) - np.asarray(pred.lower)) / (2.0 * 1.96)
    pred_calib = model.predict(calib_t)
    sig_calib = ((np.asarray(pred_calib.upper) - np.asarray(pred_calib.lower))
                 / (2.0 * 1.96))

    print(f"station: {station}  train: {len(train_t)}  "
          f"calib: {len(calib_t)}  test: {len(test_t)}")
    print(f"{'metric':<22}{'value':>12}")
    print(f"{'crps(m)':<22}{cal['crps']:>12.4f}")
    print(f"{'rmse(m)':<22}{cal['rmse']:>12.4f}")
    print(f"{'coverage (empirical)':<22}{cal['coverage_empirical']:>12.4f}")
    print(f"{'coverage (nominal)':<22}{cal['coverage_nominal']:>12.4f}")

    pit = pit_histogram(test_y, pred.mean, sig_test)
    print(f"{'pit_mean (want ~0.5)':<22}{pit['pit_mean']:>12.4f}")

    rel = reliability_curve(pred, test_y)
    print("reliability (nominal -> empirical):")
    for nom, emp in zip(rel["levels"], rel["empirical"]):
        print(f"  {nom:>6.2f} -> {emp:6.4f}")

    lo, hi, q = conformalize(pred.mean, sig_test, pred_calib.mean,
                             sig_calib, calib_y, alpha=args.alpha_level)
    conf_cov = float(np.mean((lo <= test_y) & (test_y <= hi)))
    print(f"conformal q (1-a={1.0 - args.alpha_level:.2f}): {q:.4f}  "
          f"coverage: {conf_cov:.4f}")

    attr = constituent_attribution(model, test_t)
    print("per-constituent variance share:")
    for name, share in sorted(
        zip(attr["names"], attr["share"]), key=lambda kv: -kv[1]
    ):
        print(f"  {name:<5} {share:7.4f}")
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

    if args.against == "tpxo" and (not args.global_model
                                   or args.lon is None or args.lat is None):
        print("tideglass bench: --against tpxo requires --global-model PATH "
              "and --lon/--lat (gauge coordinates)", file=sys.stderr)
        return 2
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

    rival_name = "pytides" if args.against == "pytides" else "global"
    rival_row = None
    if args.against == "pytides":
        try:
            tide_mod = _load_pytides()
            naive = [t.replace(tzinfo=None) for t in train_t]
            pt = tide_mod.Tide.decompose(np.asarray(train_y, dtype=float), t=naive)
            py_mean = np.asarray(
                pt.at([t.replace(tzinfo=None) for t in test_t]), dtype=float
            ).ravel()
            rival_row = {
                "rmse": rmse(py_mean, test_y),
                "peak_error": peak_tide_error(py_mean, test_y),
            }
        except Exception as exc:  # noqa: BLE001 - degrade to Marea-only metrics
            print(f"# pytides comparison skipped: {exc}")
    elif args.against == "tpxo":
        try:
            from tideglass.marea.bench_global import compare as compare_global

            if not args.global_model:
                raise ValueError("--against tpxo requires --global-model PATH")
            if args.lon is None or args.lat is None:
                raise ValueError("--against tpxo requires --lon/--lat (gauge coords)")
            consts = [c.strip() for c in (args.consts or "").split(",") if c.strip()]
            if args.global_model.lower().endswith(".nc"):
                # Genuine TPXO/FES elevation file (NetCDF3 classic, no extra deps).
                from tideglass.marea.tpxo import tpxo_model_at

                global_model = tpxo_model_at(
                    args.global_model, args.lon, args.lat,
                    constituents=consts or None, station=station)
            else:
                from tideglass.marea.bench_global import (
                    global_model_at,
                    read_harmonic_grid,
                )

                grid = read_harmonic_grid(args.global_model)
                global_model = global_model_at(grid, args.lon, args.lat)
            cmp = compare_global(global_model, model, test_t, test_y)
            rival_row = {
                "rmse": cmp["global"]["rmse"],
                "peak_error": cmp["global"]["peak_error"],
                "ratio": cmp["rmse_ratio_marea_over_global"],
            }
        except (OSError, ValueError) as exc:
            print(f"# global-model comparison skipped: {exc}")

    print(f"station: {station}  train: {len(train_t)}  test: {len(test_t)}")
    print(f"{'model':<10}{'rmse(m)':>10}{'peak_err':>10}{'coverage':>10}")
    print(f"{'marea':<10}{marea['rmse']:>10.4f}"
          f"{_fmt(marea['peak_error']):>10}{marea['coverage']:>10.4f}")
    if rival_row is not None:
        print(f"{rival_name:<10}{rival_row['rmse']:>10.4f}"
              f"{_fmt(rival_row['peak_error']):>10}{'n/a':>10}")
        winner = "marea" if marea["rmse"] <= rival_row["rmse"] else rival_name
        print(f"winner (rmse): {winner}")
        if "ratio" in rival_row:
            print(f"marea/global rmse ratio: {rival_row['ratio']:.3f}")
    else:
        print("winner (rmse): marea (uncontested - baseline unavailable)")
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
    if (args.cost is not None or args.loss is not None) and args.flood is None:
        print("tideglass advise: --cost/--loss require --flood (the event "
              "threshold the decision prices)", file=sys.stderr)
        return 2
    times = [day + timedelta(hours=h) for h in range(args.days * 24)]
    print(TideAdvisor(model).advise(
        times, flood_threshold_m=args.flood,
        surge_sigma_m=args.surge_sigma,
        decision_threshold_m=args.flood,
        cost=args.cost, loss=args.loss).summary)
    return 0


def cmd_fetch(args) -> int:
    if args.met:
        from tideglass.fetch import fetch_met_range, write_met_csv

        out = args.out or f"{args.station}.met.csv"
        try:
            rows = fetch_met_range(args.station, args.begin, args.end)
        except (OSError, ValueError) as exc:
            print(f"tideglass fetch: {exc}", file=sys.stderr)
            return 2
        if not rows:
            print(f"tideglass fetch: no data for station {args.station!r}",
                  file=sys.stderr)
            return 2
        write_met_csv(rows, out)
        print(f"station: {args.station}  rows: {len(rows)} (wind + pressure)")
        print(f"saved: {out}")
        return 0

    from tideglass.fetch import fetch_noaa_range, write_csv

    out = args.out or f"{args.station}.csv"
    try:
        rows = fetch_noaa_range(args.station, args.begin, args.end,
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


def cmd_nowcast(args) -> int:
    from tideglass.marea import ops as OPS

    os.makedirs(args.store, exist_ok=True)
    try:
        rep = OPS.rerun(
            args.station, args.feed, store=args.store, alpha=args.alpha,
            auto_refit=args.auto_refit,
        )
    except (OSError, ValueError) as exc:
        print(f"tideglass nowcast: {exc}", file=sys.stderr)
        return 2
    print(f"station: {rep.station}  new: {rep.n_new}  skipped: {rep.n_skipped}")
    if rep.log is not None:
        print(f"assimilated: {rep.feed_start} .. {rep.feed_end}")
        print(f"innovation: mean={rep.log.mean_innovation:+.4f} m  "
              f"rms={rep.log.rms_innovation:.4f} m  "
              f"surge={rep.log.last_surge:+.4f} m")
    print(str(rep.health))
    if rep.refit_done:
        print("auto-refit: performed (artifact replaced, lineage in refit_history)")
    elif rep.health.needs_refit:
        print("recommendation: REFIT (re-run with --auto-refit to apply)")
    if args.hours > 0:
        from tideglass.marea.nowcast import load_state as _load_state

        model = TideModel.load_harmonic(
            os.path.join(args.store, f"{args.station}.json"))
        eng = _load_state(
            model, os.path.join(args.store, f"{args.station}.nowcast.json"))
        last = eng.last_time
        if last is not None:
            times = [last + timedelta(hours=h + 1) for h in range(args.hours)]
            pred = eng.predict(times)
            print(f"# nowcast {args.hours}h from {last.isoformat()}")
            print("# time height_m lower_m upper_m")
            for t, m, lo, hi in zip(times, pred.mean, pred.lower, pred.upper):
                print(f"{t.isoformat()} {m:.4f} {lo:.4f} {hi:.4f}")
    return 0


def cmd_poll(args) -> int:
    import time

    from tideglass.marea import ops as OPS

    os.makedirs(args.store, exist_ok=True)
    passes = 0
    while True:
        try:
            rep = OPS.poll(
                args.station, store=args.store,
                lookback_hours=args.lookback_h, alpha=args.alpha,
                datum=args.datum, auto_refit=args.auto_refit,
            )
        except (OSError, ValueError) as exc:
            print(f"tideglass poll: {exc}", file=sys.stderr)
            return 2
        except KeyboardInterrupt:
            print("tideglass poll: interrupted", file=sys.stderr)
            return 130
        passes += 1
        if rep is None:
            print(f"[pass {passes}] station={args.station}: no new rows")
        else:
            flag = "REFIT" if rep.health.needs_refit and not rep.refit_done else (
                "refit-applied" if rep.refit_done else "ok")
            print(f"[pass {passes}] station={args.station} new={rep.n_new} "
                  f"coverage={rep.health.coverage:.3f} "
                  f"rmse={rep.health.rmse:.4f} status={flag}")
        if args.repeat and passes >= args.repeat:
            break
        try:
            time.sleep(args.sleep_s)
        except KeyboardInterrupt:
            print("tideglass poll: interrupted", file=sys.stderr)
            return 130
    return 0


def _build_day_times(date: str, days: int):
    day = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return [day + timedelta(hours=h) for h in range(max(1, int(days)) * 24)]


def cmd_export(args) -> int:
    times = _build_day_times(args.date, args.days)
    model = _load_station(args.store, args.station)
    if model is None:
        return 2
    out = args.out or f"{args.station}.{args.format}"
    try:
        if args.format == "json":
            EXPORT.write_json(model, out)
        elif args.format == "csv":
            EXPORT.write_csv(model, times, out)
        elif args.format == "xtide":
            EXPORT.write_xtide(model, out, station=args.station)
        elif args.format == "netcdf":
            EXPORT.write_netcdf(model, times, out)
        else:
            print(f"tideglass export: unknown format {args.format!r}", file=sys.stderr)
            return 2
    except ValueError as exc:
        print(f"tideglass export: {exc}", file=sys.stderr)
        return 2
    print(f"exported: {out} ({args.format})")
    return 0


def cmd_serve(args) -> int:
    print(f"tideglass serve: http://{args.host}:{args.port} "
          f"(store={args.store}; Ctrl-C to stop)")
    run_server(args.store, args.host, args.port)
    return 0


def cmd_tui(args) -> int:
    try:
        times = _build_day_times(args.date, args.days)
    except ValueError:
        print(f"tideglass tui: bad date {args.date!r} (want YYYY-MM-DD)", file=sys.stderr)
        return 2
    model = _load_station(args.store, args.station)
    if model is None:
        return 2
    print(build_dashboard(model, times))
    return 0


def cmd_contribute(args) -> int:
    from tideglass.marea.crowdsource import GaugeStore

    gs = GaugeStore(args.store)
    try:
        n = gs.add_csv(args.csv, args.station, args.lon, args.lat,
                       source=args.source)
    except (OSError, ValueError) as exc:
        print(f"tideglass contribute: {exc}", file=sys.stderr)
        return 2
    print(f"station: {args.station}  observations: {n}  source: {args.source}")
    print(f"network size: {gs.count} gauges (store={args.store})")
    return 0


def cmd_network(args) -> int:
    from tideglass.marea.crowdsource import GaugeStore

    gs = GaugeStore(args.store)
    if gs.count < 2:
        print(f"tideglass network: only {gs.count} gauge(s) in {args.store!r}; "
              f"add at least 2 with `tideglass contribute` to form a network")
        return 2
    eff = gs.network_effect(variance_threshold=args.threshold)
    print(f"network: {eff['n_total']} gauges, variance threshold "
          f"{eff['variance_threshold']:.2f}")
    print(f"{'n':>3}{'total_explained':>16}{'modes_to_thr':>14}")
    for s in eff["steps"]:
        print(f"{s['n_stations']:>3}{s['total_explained']:>16.4f}"
              f"{s['modes_to_threshold']:>14}")
    print(f"network-effect gain in explained variance: "
          f"{eff['gain_total_explained']:+.4f}")
    return 0


def cmd_qc(args) -> int:
    from tideglass.marea.qc import clean_series, qc_check

    try:
        times, heights = read_csv(args.csv)
    except (OSError, ValueError) as exc:
        print(f"tideglass qc: {exc}", file=sys.stderr)
        return 2
    rep = qc_check(times, heights)
    print(str(rep))
    if rep.spike_idx:
        print(f"flagged indices: {rep.spike_idx}")
    if rep.passed:
        _, cleaned = clean_series(times, heights, rep)
        print(f"cleaned observations: {len(cleaned)}")
    return 0 if rep.passed else 2


def cmd_federate(args) -> int:
    from tideglass.marea.federation import FederatedRefit

    try:
        times, heights = read_csv(args.csv)
    except (OSError, ValueError) as exc:
        print(f"tideglass federate: {exc}", file=sys.stderr)
        return 2
    fed = FederatedRefit(args.store, variance_threshold=0.95)
    try:
        rep = fed.refit(
            args.station, args.lon, args.lat, times, heights,
            source=args.source, alpha=args.alpha,
        )
    except (ValueError, OSError) as exc:
        print(f"tideglass federate: {exc}", file=sys.stderr)
        return 2
    print(str(rep))
    if rep.network_before and rep.network_after:
        gain = (rep.network_after["gain_total_explained"]
                - rep.network_before["gain_total_explained"])
        print(f"network-effect gain: {gain:+.4f} explained variance "
              f"({rep.network_before['n_total']} -> {rep.network_after['n_total']} gauges)")
    if rep.trust is not None:
        print(f"trust: {rep.trust:.3f} (QC reputation; low trust down-weights "
              "the sensor in the pool)")
    if rep.improved:
        print(f"improved model saved: {os.path.join(fed.store.root, args.station + '.json')}")
    return 0 if rep.qc.passed else 2


def cmd_correct(args) -> int:

    from tideglass.marea.residual import learn_residual

    try:
        times, heights = read_csv(args.csv)
    except (OSError, ValueError) as exc:
        print(f"tideglass correct: {exc}", file=sys.stderr)
        return 2
    model = _load_station(args.store, args.station)
    if model is None:
        return 2
    try:
        rm, diag = learn_residual(
            model, times, heights, bins=args.bins,
            holdout=args.holdout, alpha=args.alpha)
    except ValueError as exc:
        print(f"tideglass correct: {exc}", file=sys.stderr)
        return 2
    model.attach_residual(rm)
    path = os.path.join(args.store, f"{args.station}.json")
    with open(path, "w") as fh:
        json.dump(model.to_artifact(), fh, indent=2)
    print(f"station: {args.station}  n: {diag['n']}  bins: {diag['bins']} "
          f"(day-of-year climatology)")
    print(f"max |bias|: {diag['max_abs_bias']:.4f} m")
    print(f"rmse: {diag['rmse_before']:.4f} -> {diag['rmse_after']:.4f} m "
          f"(held-out tail, n={diag['n_test']})")
    if diag.get("coverage_after") is not None:
        print(f"coverage: {diag['coverage_before']:.4f} -> "
              f"{diag['coverage_after']:.4f} "
              f"(conformal q={diag['conformal_q']:.3f})")
    print(f"saved: {path} (bias table applied by every predict)")
    return 0


def cmd_surge(args) -> int:
    import numpy as np
    from tideglass.marea.decision import decision_curve
    from tideglass.marea.extremes import flood_probability
    from tideglass.marea.met import learn_met_response, read_met_csv
    from tideglass.marea.surge import fit_ar1

    try:
        times, heights = read_csv(args.csv)
        mt, ws, wd, pr = read_met_csv(args.met_csv)
    except (OSError, ValueError) as exc:
        print(f"tideglass surge: {exc}", file=sys.stderr)
        return 2
    order = sorted(range(len(times)), key=lambda i: times[i])
    times = [times[i] for i in order]
    heights = np.array([heights[i] for i in order], dtype=float)

    station = args.station or os.path.splitext(os.path.basename(args.csv))[0]
    model_path = os.path.join(args.store, f"{station}.json")
    if os.path.exists(model_path):
        model = TideModel.load_harmonic(model_path)
    else:
        # self-contained: fit (and save) a harmonic model from the same record
        try:
            model = TideModel.fit(times, heights, station=station,
                                  source=f"csv:{args.csv}")
        except ValueError as exc:
            print(f"tideglass surge: {exc}", file=sys.stderr)
            return 2
        os.makedirs(args.store, exist_ok=True)
        with open(model_path, "w") as fh:
            json.dump(model.to_artifact(), fh, indent=2)

    pred = model.predict(times)
    resid = heights - np.asarray(pred.mean, dtype=float)
    try:
        resp, diag = learn_met_response(
            times, resid, mt, ws, wd, pr,
            max_lag_hours=args.max_lag, p_ref=args.p_ref)
    except ValueError as exc:
        print(f"tideglass surge: {exc}", file=sys.stderr)
        return 2

    os.makedirs(args.store, exist_ok=True)
    met_path = os.path.join(args.store, f"{station}.met.json")
    with open(met_path, "w") as fh:
        json.dump(resp.to_dict(), fh, indent=2)

    print(f"station: {station}  n: {diag['n']}  lag: {diag['lag']} h")
    print("response: surge = c0 + a*tau_u + b*tau_v + beta*dp")
    print(f"  intercept: {resp.intercept:+.4f} m")
    print(f"  stress u:  {resp.stress_u:+.5f} m/(m2/s2)")
    print(f"  stress v:  {resp.stress_v:+.5f} m/(m2/s2)")
    print(f"  barometer: {resp.barometer:+.5f} m/hPa  "
          f"(theory {diag['barometer_theory']:+.5f}, "
          f"ratio {diag['barometer_ratio']:.2f})")
    print(f"  fit: r2 {diag['r2']:.3f}  sigma {diag['sigma']:.4f} m  "
          f"(rmse {diag['rmse_before']:.4f} -> {diag['rmse_after']:.4f} m)")
    print(f"saved: {met_path}")

    if args.forecast:
        try:
            ft, fws, fwd, fpr = read_met_csv(args.forecast)
        except (OSError, ValueError) as exc:
            print(f"tideglass surge: {exc}", file=sys.stderr)
            return 2
        if ft:
            f0 = ft[0]
            keep = [i for i, t in enumerate(ft)
                    if (t - f0).total_seconds() / 3600.0 < args.hours]
            ft = [ft[i] for i in keep]
            fws = np.asarray(fws)[keep]
            fwd = np.asarray(fwd)[keep]
            fpr = np.asarray(fpr)[keep]
        try:
            ar = fit_ar1(resid)
        except ValueError:
            ar = None
        # Prepend the met history tail so the fitted lag can interpolate real
        # forcing (not clamps) for the first lag hours of the forecast.
        pre = max(resp.lag_hours + 1, 1)
        gt = list(mt[-pre:]) + list(ft)
        gws = np.concatenate([np.asarray(ws[-pre:]), np.asarray(fws)])
        gwd = np.concatenate([np.asarray(wd[-pre:]), np.asarray(fwd)])
        gpr = np.concatenate([np.asarray(pr[-pre:]), np.asarray(fpr)])
        fc_all = resp.forecast(gt, gws, gwd, gpr, ar=ar,
                               last_residual=float(resid[-1]), origin=times[-1])
        fc_mean = np.asarray(fc_all.mean, dtype=float)[pre:]
        fc_sigma = np.asarray(fc_all.sigma, dtype=float)[pre:]
        tide_pred = model.predict(ft)
        p = None
        if args.flood is not None and len(ft):
            p = flood_probability(tide_pred, args.flood,
                                  surge_mean=fc_mean, surge_sigma=fc_sigma)
        print(f"\nsurge forecast ({args.hours} h from {ft[0].isoformat()}):")
        for i, t in enumerate(ft):
            line = (f"  {t.isoformat()}  {fc_mean[i]:+.3f} m "
                    f"+/- {1.96 * fc_sigma[i]:.3f}")
            if p is not None:
                line += f"   P(flood>{args.flood:.2f})={p[i]:.4f}"
            print(line)
        if (args.cost is not None or args.loss is not None) \
                and args.flood is None:
            print("tideglass surge: --cost/--loss require --flood "
                  "(the event threshold)", file=sys.stderr)
            return 2
        if (args.cost is not None) != (args.loss is not None):
            print("tideglass surge: --cost and --loss must be given together",
                  file=sys.stderr)
            return 2
        if args.cost is not None:
            try:
                curve = decision_curve(
                    ft, tide_pred, args.flood, args.cost, args.loss,
                    surge_mean=fc_mean, surge_sigma=fc_sigma)
            except ValueError as exc:
                print(f"tideglass surge: {exc}", file=sys.stderr)
                return 2
            print()
            print(str(curve))
    return 0


def cmd_validate(args) -> int:
    from tideglass.marea import harmonics_db as HDB

    rep = HDB.coverage_report()
    print(f"global coverage: {rep['n_stations']} stations across "
          f"{rep['n_regions']} regions (published harmonics, no fit needed)")
    for region, n in rep["regions"].items():
        print(f"  {region:<20} {n} stations")
    print()
    regions = [args.region] if args.region else HDB.list_regions()
    for region in regions:
        print(f"region: {region}")
        print(f"{'station':<16}{'range_m':>10}{'dominant':>10}"
              f"{'n_const':>10}")
        for row in HDB.benchmark_region(region):
            print(f"{row['station']:<16}{row['range_m']:>10.3f}"
                  f"{row['dominant']:>10}{row['n_constituents']:>10}")
        print()
    return 0


def cmd_report(args) -> int:
    from tideglass.marea.report import build_report

    md = build_report(args.data_dir)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(md)
        print(f"report written: {args.out}")
        return 0
    print(md, end="")
    return 0


def cmd_plugins(args) -> int:
    from tideglass.marea.plugins import all_constituents, list_packs

    packs = list_packs()
    total = sum(len(v) for v in packs.values())
    print(f"constituent packs: {len(packs)} ({total} plugin constituents)")
    for name in sorted(packs):
        print(f"  {name}: {', '.join(packs[name])}")
    if args.speeds:
        from tideglass.marea.constituents import speed

        print()
        print(f"{'constituent':<14}{'speed(d/h)':>12}{'species':>12}")
        for c in all_constituents():
            print(f"{c.name:<14}{speed(c):>12.5f}{c.species:>12}")
    return 0


def cmd_alert(args) -> int:
    try:
        times, heights = read_csv(args.csv)
    except (OSError, ValueError) as exc:
        print(f"tideglass alert: {exc}", file=sys.stderr)
        return 2
    model = _load_station(args.store, args.station)
    if model is None:
        return 2
    events = surge_events(
        heights, model.predict(times).mean, times,
        station=args.station, threshold_m=args.threshold,
    )
    print(f"station: {args.station}  surge events: {len(events)}")
    for e in events:
        print(f"  {e.start.isoformat()} -> {e.end.isoformat()}  "
              f"peak |residual|={e.peak_residual_m:.3f} m")
    return 0


def cmd_pool(args) -> int:
    from tideglass.marea.pooling import HierarchicalPool

    try:
        times, heights = read_csv(args.csv)
    except (OSError, ValueError) as exc:
        print(f"tideglass pool: {exc}", file=sys.stderr)
        return 2
    if not os.path.isdir(args.store):
        print(f"tideglass pool: no artifact store at {args.store!r} "
              "(fit the network stations first)", file=sys.stderr)
        return 2
    models = {}
    for fn in sorted(os.listdir(args.store)):
        if not fn.endswith(".json") or fn[:-5] == args.station:
            continue
        path = os.path.join(args.store, fn)
        try:
            models[fn[:-5]] = TideModel.load_harmonic(path)
        except (OSError, ValueError, KeyError):
            print(f"# skipping {fn}: not a harmonic artifact")
    if len(models) < 2:
        print(f"tideglass pool: need ≥ 2 network models in {args.store!r}, "
              f"found {len(models)}", file=sys.stderr)
        return 2
    try:
        pool = HierarchicalPool(models)
        pooled = pool.seed_short(times, heights, args.station)
    except ValueError as exc:
        print(f"tideglass pool: {exc}", file=sys.stderr)
        return 2
    os.makedirs(args.store, exist_ok=True)
    path = os.path.join(args.store, f"{args.station}.json")
    with open(path, "w") as fh:
        json.dump(pooled.to_artifact(), fh, indent=2)
    print(f"station: {args.station}  network: {len(models)} stations  "
          f"obs: {len(times)}")
    print(f"mean_shrinkage: {pooled.meta['mean_shrinkage']:.4f} "
          "(0 = own data, 1 = pure prior)")
    print(f"{'constituent':<12}{'own_amp':>10}{'pooled_amp':>12}{'shrinkage':>11}")
    own_names = {f.name: f for f in
                 TideModel.fit(times, heights, auto_select=False,
                               candidates=pool.union).constituents()}
    for f in pooled.constituents():
        own_amp = own_names[f.name].amplitude if f.name in own_names else 0.0
        print(f"{f.name:<12}{own_amp:>10.4f}{f.amplitude:>12.4f}"
              f"{pooled.meta['shrinkage'][f.name]:>11.4f}")
    print(f"saved: {path}")
    return 0


def cmd_extremes(args) -> int:
    import numpy as np

    from tideglass.marea.extremes import (
        annual_rate,
        decluster,
        fit_gpd,
        joint_exceedance_probability,
        skew_surge,
    )

    try:
        times, heights = read_csv(args.csv)
    except (OSError, ValueError) as exc:
        print(f"tideglass extremes: {exc}", file=sys.stderr)
        return 2
    order = sorted(range(len(times)), key=lambda i: times[i])
    times = [times[i] for i in order]
    y = np.array([heights[i] for i in order], dtype=float)
    station = args.station or os.path.splitext(os.path.basename(args.csv))[0]
    try:
        model = TideModel.fit(
            times, y, auto_select=not args.no_select, alpha=args.alpha,
            station=station)
        sk = skew_surge(times, y, model.predict(times).mean)
        thresh = float(np.quantile(sk.skew, args.threshold_q))
        _, kv = decluster(sk.times, sk.skew, gap_hours=args.gap_h,
                          threshold=thresh)
        g = fit_gpd(kv, threshold=thresh)
    except ValueError as exc:
        print(f"tideglass extremes: {exc}", file=sys.stderr)
        return 2
    rate = annual_rate(times, g.n_peaks)
    print(f"station: {station}  obs: {len(times)}  "
          f"high waters: {len(sk.times)}")
    print(f"skew surge: max={float(np.max(sk.skew)):.4f} m  "
          f"mean={float(np.mean(sk.skew)):.4f} m")
    print(f"declustered peaks: {len(kv)}  threshold: {thresh:.4f} m  "
          f"exceedances: {g.n_peaks}  rate: {rate:.2f}/yr")
    print(f"GPD: loc={g.loc:.4f} m  scale={g.scale:.4f} m  shape={g.shape:+.4f}")
    try:
        periods = [float(p) for p in args.return_periods.split(",") if p.strip()]
    except ValueError:
        print("tideglass extremes: bad --return-periods "
              "(want e.g. '1,10,100')", file=sys.stderr)
        return 2
    print("return level (m):")
    for t in periods:
        print(f"  {t:>8g}-yr  {g.return_level(t, rate):.4f}")
    resid = y - model.predict(times).mean
    total = model.predict(times).mean + resid
    levels = sorted({float(np.quantile(total, 0.99)),
                     float(np.quantile(total, 0.999))}
                    | ({float(args.flood)} if args.flood is not None else set()))
    print("joint exceedance P(tide+surge > level):")
    for z in levels:
        p = joint_exceedance_probability(
            model.predict(times).mean, resid, z)
        print(f"  {z:7.4f} m  p_hour={p:.6f}  ~{p * 8760.0:.1f} h/yr")
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
    p_fit.add_argument("--source", default=None,
                       help="provenance tag pinned in the artifact (default: csv:<path>)")
    p_fit.add_argument("--kernel", choices=["numpy", "numba", "auto"], default="numpy",
                       help="numeric backend for the design matrix (default: numpy)")
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
    p_bench.add_argument("--against", choices=["pytides", "tpxo", "none"],
                         default="pytides",
                         help="baseline to beat (tpxo = global model grid)")
    p_bench.add_argument("--global-model", default=None,
                         help="TPXO/FES harmonic grid: CSV (lon,lat,constituent,"
                              "amplitude,phase) or genuine NetCDF3 .nc elevation "
                              "file, for --against tpxo")
    p_bench.add_argument("--consts", default="M2,S2,N2,K2,K1,O1,P1,Q1",
                         help="comma-separated constituents to extract from a "
                              ".nc global file (default: principal 8)")
    p_bench.add_argument("--lon", type=float, default=None,
                         help="gauge longitude for --against tpxo")
    p_bench.add_argument("--lat", type=float, default=None,
                         help="gauge latitude for --against tpxo")
    p_bench.set_defaults(func=cmd_bench)

    p_smooth = sub.add_parser(
        "smooth", help="joint Kalman smooth: tide + surge + secular trend"
    )
    p_smooth.add_argument("csv", help="CSV file with time,height rows")
    p_smooth.add_argument("--station", default=None, help="station name (default: CSV stem)")
    p_smooth.add_argument("--alpha", type=float, default=0.05, help="selection significance")
    p_smooth.add_argument("--no-select", action="store_true", help="fit principal 8 directly")
    p_smooth.set_defaults(func=cmd_smooth)

    p_cal = sub.add_parser("calibrate", help="CRPS + PIT + reliability + "
                                            "conformal calibration on held-out data")
    p_cal.add_argument("csv", help="CSV file with time,height rows")
    p_cal.add_argument("--station", default=None, help="station name (default: CSV stem)")
    p_cal.add_argument("--test-fraction", type=float, default=0.25,
                       help="held-out fraction for diagnostics (chronological tail)")
    p_cal.add_argument("--calib-fraction", type=float, default=0.2,
                       help="held-out fraction for conformal scores "
                            "(chronological middle)")
    p_cal.add_argument("--alpha", type=float, default=0.05, help="selection significance")
    p_cal.add_argument("--alpha-level", type=float, default=0.05,
                       help="conformal miscoverage level (bands cover 1-alpha)")
    p_cal.set_defaults(func=cmd_calibrate)

    p_adv = sub.add_parser("advise", help="harvesting + rip + species advice")
    p_adv.add_argument("station", help="station name")
    p_adv.add_argument("date", help="YYYY-MM-DD (UTC)")
    p_adv.add_argument("--store", default=DEFAULT_STORE, help="artifact directory")
    p_adv.add_argument("--days", type=int, default=1, help="days from midnight")
    p_adv.add_argument("--flood", type=float, default=None,
                       help="alarm level (m): price P(level > flood) per time "
                            "(e.g. a GPD return level from `tideglass extremes`)")
    p_adv.add_argument("--surge-sigma", type=float, default=0.0,
                       help="surge std (m) folded into the flood probability")
    p_adv.add_argument("--cost", type=float, default=None,
                       help="cost of the protective action per hour; with "
                            "--loss, prices the act/wait decision policy "
                            "(requires --flood as the event threshold)")
    p_adv.add_argument("--loss", type=float, default=None,
                       help="loss if the event hits an unprepared hour; with "
                            "--cost, prices the act/wait decision policy "
                            "(requires --flood as the event threshold)")
    p_adv.set_defaults(func=cmd_advise)

    p_fetch = sub.add_parser("fetch", help="download NOAA CO-OPS gauge CSV")
    p_fetch.add_argument("station", help="NOAA station id (e.g. 9414290)")
    p_fetch.add_argument("begin", help="start date YYYY-MM-DD (UTC)")
    p_fetch.add_argument("end", help="end date YYYY-MM-DD (UTC)")
    p_fetch.add_argument("--out", default=None, help="output CSV (default: <station>.csv)")
    p_fetch.add_argument("--datum", default="MLLW", help="NOAA vertical datum")
    p_fetch.add_argument("--interval", default="h", choices=["h", "1", "hilo"],
                          help="passed to NOAA (observed water levels are 6-min)")
    p_fetch.add_argument("--met", action="store_true",
                         help="fetch meteorological wind + pressure instead of "
                              "water levels (writes <station>.met.csv)")
    p_fetch.set_defaults(func=cmd_fetch)

    p_now = sub.add_parser(
        "nowcast",
        help="assimilate a fresh feed into the deployed model (drift check + refit)",
    )
    p_now.add_argument("station", help="station name (model artifact in store)")
    p_now.add_argument("feed", help="CSV file with time,height rows (recent feed)")
    p_now.add_argument("--store", default=DEFAULT_STORE, help="artifact directory")
    p_now.add_argument("--alpha", type=float, default=0.05,
                       help="selection significance for auto-refit")
    p_now.add_argument("--auto-refit", dest="auto_refit", action="store_true",
                       default=True, help="refit automatically when stale (default)")
    p_now.add_argument("--no-auto-refit", dest="auto_refit", action="store_false",
                       help="only report staleness, never refit")
    p_now.add_argument("--hours", type=int, default=24,
                       help="nowcast horizon in hours after the feed (0 to skip)")
    p_now.set_defaults(func=cmd_nowcast)

    p_poll = sub.add_parser(
        "poll", help="live loop: fetch NOAA, assimilate, health-check, refit"
    )
    p_poll.add_argument("station", help="NOAA station id / station name")
    p_poll.add_argument("--store", default=DEFAULT_STORE, help="artifact directory")
    p_poll.add_argument("--lookback-h", type=int, default=72,
                        help="hours to fetch back on the first pass")
    p_poll.add_argument("--repeat", type=int, default=0,
                        help="passes to run (0 = run forever)")
    p_poll.add_argument("--sleep-s", type=float, default=3600.0,
                        help="seconds between passes")
    p_poll.add_argument("--datum", default="MLLW", help="NOAA vertical datum")
    p_poll.add_argument("--alpha", type=float, default=0.05,
                        help="selection significance for auto-refit")
    p_poll.add_argument("--auto-refit", dest="auto_refit", action="store_true",
                        default=True, help="refit automatically when stale (default)")
    p_poll.add_argument("--no-auto-refit", dest="auto_refit", action="store_false",
                        help="only report staleness, never refit")
    p_poll.set_defaults(func=cmd_poll)

    p_export = sub.add_parser("export", help="write model prediction in a feed format")
    p_export.add_argument("station", help="station name")
    p_export.add_argument("date", help="YYYY-MM-DD (UTC)")
    p_export.add_argument("--store", default=DEFAULT_STORE, help="artifact directory")
    p_export.add_argument("--days", type=int, default=1, help="days from midnight")
    p_export.add_argument("--format", choices=["json", "csv", "xtide", "netcdf"],
                          default="csv", help="output format")
    p_export.add_argument("--out", default=None, help="output path (default: <station>.<fmt>)")
    p_export.set_defaults(func=cmd_export)

    p_serve = sub.add_parser("serve", help="HTTP API: /predict and /advise")
    p_serve.add_argument("--store", default=DEFAULT_STORE, help="artifact directory")
    p_serve.add_argument("--host", default="127.0.0.1", help="bind host")
    p_serve.add_argument("--port", type=int, default=8000, help="bind port")
    p_serve.set_defaults(func=cmd_serve)

    p_tui = sub.add_parser("tui", help="terminal dashboard (predict + advise)")
    p_tui.add_argument("station", help="station name")
    p_tui.add_argument("date", help="YYYY-MM-DD (UTC)")
    p_tui.add_argument("--store", default=DEFAULT_STORE, help="artifact directory")
    p_tui.add_argument("--days", type=int, default=1, help="days from midnight")
    p_tui.set_defaults(func=cmd_tui)

    p_alert = sub.add_parser("alert", help="surge-flag alerts for a watched station")
    p_alert.add_argument("csv", help="CSV file with time,height rows (observations)")
    p_alert.add_argument("--station", default=None, help="station name (default: CSV stem)")
    p_alert.add_argument("--store", default=DEFAULT_STORE, help="artifact directory")
    p_alert.add_argument("--threshold", type=float, default=0.3,
                         help="|residual| (m) that triggers an alert")
    p_alert.set_defaults(func=cmd_alert)

    p_contrib = sub.add_parser("contribute", help="upload a crowd-sourced gauge CSV")
    p_contrib.add_argument("csv", help="CSV with time,height rows (cheap-sensor upload)")
    p_contrib.add_argument("lon", type=float, help="station longitude (deg)")
    p_contrib.add_argument("lat", type=float, help="station latitude (deg)")
    p_contrib.add_argument("--station", required=True, help="unique station id")
    p_contrib.add_argument("--store", default=DEFAULT_STORE, help="gauge store dir")
    p_contrib.add_argument("--source", default="crowd", help="uploader/source tag")
    p_contrib.set_defaults(func=cmd_contribute)

    p_net = sub.add_parser("network", help="show the crowd-sourced network effect")
    p_net.add_argument("--store", default=DEFAULT_STORE, help="gauge store dir")
    p_net.add_argument("--threshold", type=float, default=0.95,
                       help="EOF variance threshold (0..1)")
    p_net.set_defaults(func=cmd_network)

    p_val = sub.add_parser("validate", help="global coverage + per-region benchmark")
    p_val.add_argument("--region", default=None, help="limit to one region")
    p_val.set_defaults(func=cmd_validate)

    p_qc = sub.add_parser("qc", help="quality-check a crowd-sourced sensor upload")
    p_qc.add_argument("csv", help="CSV with time,height rows (cheap-sensor upload)")
    p_qc.add_argument("lon", type=float, help="station longitude (deg)")
    p_qc.add_argument("lat", type=float, help="station latitude (deg)")
    p_qc.add_argument("--station", required=True, help="unique station id")
    p_qc.add_argument("--store", default=DEFAULT_STORE, help="gauge store dir")
    p_qc.add_argument("--source", default="crowd", help="uploader/source tag")
    p_qc.set_defaults(func=cmd_qc)

    p_fed = sub.add_parser(
        "federate", help="QC + local fit + federated pooled refit of a sensor")
    p_fed.add_argument("csv", help="CSV with time,height rows (cheap-sensor upload)")
    p_fed.add_argument("lon", type=float, help="station longitude (deg)")
    p_fed.add_argument("lat", type=float, help="station latitude (deg)")
    p_fed.add_argument("--station", required=True, help="unique station id")
    p_fed.add_argument("--store", default=DEFAULT_STORE, help="gauge store dir")
    p_fed.add_argument("--source", default="crowd", help="uploader/source tag")
    p_fed.add_argument("--alpha", type=float, default=0.05,
                       help="selection significance for the local fit")
    p_fed.set_defaults(func=cmd_federate)

    p_pool = sub.add_parser(
        "pool", help="partially-pooled model for a short record "
                     "(borrows strength from the network in --store)")
    p_pool.add_argument("csv", help="CSV file with time,height rows (short record)")
    p_pool.add_argument("--station", required=True, help="new station name")
    p_pool.add_argument("--store", default=DEFAULT_STORE, help="network artifact directory")
    p_pool.set_defaults(func=cmd_pool)

    p_ext = sub.add_parser(
        "extremes", help="skew-surge + GPD return levels + joint exceedance")
    p_ext.add_argument("csv", help="CSV file with time,height rows (long record)")
    p_ext.add_argument("--station", default=None, help="station name (default: CSV stem)")
    p_ext.add_argument("--alpha", type=float, default=0.05, help="selection significance")
    p_ext.add_argument("--no-select", action="store_true", help="fit principal 8 directly")
    p_ext.add_argument("--threshold-q", type=float, default=0.9,
                       help="POT threshold as a quantile of the skew-surge series")
    p_ext.add_argument("--gap-h", type=float, default=72.0,
                       help="declustering gap in hours (one storm = one peak)")
    p_ext.add_argument("--return-periods", default="1,2,5,10,25,50,100",
                       help="comma-separated return periods (years)")
    p_ext.add_argument("--flood", type=float, default=None,
                       help="alarm level (m) for the joint-exceedance table")
    p_ext.set_defaults(func=cmd_extremes)

    p_rep = sub.add_parser(
        "report", help="one-page global validation report (markdown)")
    p_rep.add_argument("--data-dir", default="data",
                       help="directory holding gauge CSV records")
    p_rep.add_argument("--out", default=None,
                       help="write markdown to a file (default: stdout)")
    p_rep.set_defaults(func=cmd_report)

    p_corr = sub.add_parser(
        "correct",
        help="learn a station's residual bias table and attach it to the model")
    p_corr.add_argument("csv", help="CSV with time,height rows (recent record)")
    p_corr.add_argument("--station", default=None,
                        help="station name (default: CSV stem)")
    p_corr.add_argument("--store", default=DEFAULT_STORE, help="artifact directory")
    p_corr.add_argument("--bins", type=int, default=12,
                        help="day-of-year climatology bins (12 = monthly)")
    p_corr.add_argument("--holdout", type=float, default=0.25,
                        help="chronological tail fraction for scoring")
    p_corr.add_argument("--alpha", type=float, default=0.05,
                        help="conformal miscoverage level for the corrected band")
    p_corr.set_defaults(func=cmd_correct)

    p_plug = sub.add_parser(
        "plugins", help="list constituent packs (builtin + user JSON packs)")
    p_plug.add_argument("--speeds", action="store_true",
                        help="also print every constituent's derived speed")
    p_plug.set_defaults(func=cmd_plugins)

    p_surge = sub.add_parser(
        "surge", help="learn the wind/pressure surge response; forecast surge "
                      "from a met forecast (v1.2)")
    p_surge.add_argument("csv", help="CSV file with time,height observations")
    p_surge.add_argument("met_csv", help="concurrent met CSV: time,wind_speed,"
                                         "wind_dir,pressure")
    p_surge.add_argument("--station", default=None,
                         help="station name (default: CSV stem)")
    p_surge.add_argument("--store", default=DEFAULT_STORE, help="artifact directory")
    p_surge.add_argument("--max-lag", type=int, default=6,
                         help="maximum forcing-to-surge lag scanned (hours)")
    p_surge.add_argument("--p-ref", type=float, default=1013.25,
                         help="reference pressure for the inverse-barometer "
                              "anomaly (hPa)")
    p_surge.add_argument("--forecast", default=None,
                         help="met forecast CSV (same format) to run through "
                              "the learned response")
    p_surge.add_argument("--hours", type=int, default=48,
                         help="forecast horizon kept from the met CSV (hours)")
    p_surge.add_argument("--flood", type=float, default=None,
                         help="alarm level (m): prints P(level > flood) per "
                              "forecast hour")
    p_surge.add_argument("--cost", type=float, default=None,
                         help="cost of the protective action per hour; with "
                              "--loss, prices the act/wait decision (needs "
                              "--flood)")
    p_surge.add_argument("--loss", type=float, default=None,
                         help="loss if the event hits an unprepared hour; "
                              "with --cost, prices the act/wait decision "
                              "(needs --flood)")
    p_surge.set_defaults(func=cmd_surge)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
