"""Global validation report — v0.9 "publish the artifact".

One page that every region, every source has to answer to: the published
harmonics database (per-region self-consistency) *and* every local gauge record
under ``data/`` (hold-out accuracy + extreme-value analysis), rendered as a
single deterministic markdown document.

This is the artifact competitors have to match — produced entirely offline by
:func:`build_report` (or ``tideglass report``), reproducible from the repository
with no network access.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone

import numpy as np

from tideglass.marea import extremes as EVA
from tideglass.marea import harmonics_db
from tideglass.marea.metrics import evaluate
from tideglass.marea.model import TideModel

_DEFAULT_RETURN_PERIODS = (2.0, 10.0, 100.0)


def _parse_ts(text: str) -> datetime:
    return datetime.fromisoformat(text.strip().replace("Z", "+00:00"))


def read_gauge_csv(path: str) -> tuple[list[datetime], np.ndarray]:
    """Read a ``time,height`` CSV (the NOAA CO-OPS gauge format)."""
    times: list[datetime] = []
    heights: list[float] = []
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if not row or not row[0].strip():
                continue
            try:
                t = _parse_ts(row[0])
            except ValueError:
                continue  # header line
            times.append(t)
            heights.append(float(row[1]))
    if not times:
        raise ValueError(f"no readable time,height rows in {path!r}")
    order = sorted(range(len(times)), key=lambda i: times[i])
    return [times[i] for i in order], np.array([heights[i] for i in order])


def validate_gauge(path: str, station: str | None = None,
                   test_fraction: float = 0.25,
                   threshold_q: float = 0.75,
                   return_periods=_DEFAULT_RETURN_PERIODS) -> dict:
    """Hold-out benchmark + skew-surge EVA for one gauge record.

    Returns a dict with bench metrics (``rmse``, ``peak_error``, ``coverage``)
    and an ``eva`` sub-dict (storm count, GPD parameters, annual rate and
    return levels), or ``"error"`` describing why validation was not possible.

    ``threshold_q`` is the POT threshold as a quantile of the skew-surge series.
    Unlike the CLI default (0.9, suited to multi-year records) the report
    defaults to 0.75 so short (1-3 month) bundled records still yield enough
    independent storms for a defensible tail fit.
    """
    station = station or os.path.splitext(os.path.basename(path))[0]
    try:
        times, y = read_gauge_csv(path)
    except (OSError, ValueError) as exc:
        return {"station": station, "error": str(exc)}
    n_test = max(24, round(len(times) * test_fraction))
    train_t, train_y = times[:-n_test], y[:-n_test]
    test_t, test_y = times[-n_test:], y[-n_test:]

    try:
        model = TideModel.fit(train_t, train_y, alpha=1e-4, station=station)
    except ValueError as exc:
        return {"station": station, "error": str(exc)}
    pred = model.predict(test_t)
    bench = evaluate(pred, test_y)

    eva: dict = {}
    try:
        full_pred = model.predict(times)
        surge = EVA.skew_surge(times, y, full_pred.mean)
        thresh = float(np.quantile(surge.skew, threshold_q))
        peaks_t, peaks_v = EVA.decluster(surge.times, surge.skew,
                                         threshold=thresh)
        # Bundled records are 1-3 months: the standard "median of peaks" POT
        # threshold would halve an already small storm sample. The report fits
        # over the 5th-largest peak (leaving the top 4 excesses), and reports
        # "-" when even that leaves too few storms for a defensible fit.
        if len(peaks_v) < 6:
            raise ValueError(
                f"only {len(peaks_v)} independent storms in the record; "
                "too few for a tail fit")
        gpd = EVA.fit_gpd(
            peaks_v,
            threshold=float(np.sort(np.asarray(peaks_v))[-5]),
            min_peaks=4)
        rate = EVA.annual_rate(peaks_t, gpd.n_peaks)
        eva = {
            "n_storms": gpd.n_peaks,
            "threshold_m": round(gpd.loc, 3),
            "scale_m": round(gpd.scale, 3),
            "shape": round(gpd.shape, 3),
            "rate_per_year": round(rate, 2),
            "return_levels_m": {
                f"{p:g}yr": round(gpd.return_level(p, rate), 3)
                for p in return_periods
            },
        }
    except ValueError as exc:
        eva = {"error": str(exc)}

    return {
        "station": station,
        "n_obs": len(times),
        "span_days": round((times[-1] - times[0]).total_seconds() / 86400.0, 1),
        "n_train": len(train_t),
        "n_test": n_test,
        "rmse": bench["rmse"],
        "peak_error": bench["peak_error"],
        "coverage": bench["coverage"],
        "eva": eva,
    }


def gauge_files(data_dir: str) -> list[str]:
    """Every ``*.csv`` gauge record under ``data_dir`` (sorted)."""
    if not os.path.isdir(data_dir):
        return []
    return [
        os.path.join(data_dir, fn)
        for fn in sorted(os.listdir(data_dir))
        if fn.endswith(".csv") and os.path.isfile(os.path.join(data_dir, fn))
    ]


def build_report(data_dir: str = "data") -> str:
    """Render the one-page global validation report as markdown."""
    lines: list[str] = []
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    cov = harmonics_db.coverage_report()

    # Deferred: engine inventory details (backends, packs, api_version) so the
    # report stays import-cheap and the imports keep their single home.
    import tideglass
    from tideglass.marea.contract import api_version
    from tideglass.marea.kernel import available_backends
    from tideglass.marea.plugins import list_packs

    lines += [
        "# Tideglass — Global Validation Report",
        "",
        (f"*Every region, every source, one page. Generated {now} by "
         f"tideglass {tideglass.__version__} (API contract {api_version}). "
         "Fully reproducible offline: `tideglass report`.*"),
        "",
        "## 1. Coverage (published harmonics)",
        "",
        (f"{cov['n_stations']} stations across {cov['n_regions']} regions "
         "served from published harmonic constants alone — no fitting required:"),
        "",
        "| Region | Stations |",
        "|---|---|",
    ]
    for region, n in sorted(cov["regions"].items()):
        lines.append(f"| {region} | {n} |")

    lines += ["", "## 2. Per-region validation (self-consistency)", ""]
    for region in harmonics_db.list_regions():
        rows = harmonics_db.benchmark_region(region)
        lines += [
            f"### {region}",
            "",
            "| Station | Range (m) | Dominant | Constituents |",
            "|---|---|---|---|",
        ]
        for r in rows:
            lines.append(
                f"| {r['station']} | {r['range_m']:.2f} | {r['dominant']} "
                f"| {r['n_constituents']} |")
        lines.append("")

    lines += ["## 3. Gauge records (hold-out + extreme value analysis)", ""]
    paths = gauge_files(data_dir)
    if not paths:
        lines += [
            (f"No gauge records under `{data_dir}/`. Drop NOAA CO-OPS "
             "`time,height` CSVs there and re-run to extend this table."), ""]
    else:
        lines += [
            ("| Station | Obs | Span (d) | RMSE (m) | Peak err (m) | "
             "CI coverage | Storms | 2-yr (m) | 10-yr (m) | 100-yr (m) |"),
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for path in paths:
            res = validate_gauge(path)
            if "error" in res:
                lines.append(
                    f"| {res['station']} | - | - | - | - | - | - | - | - | - |")
                lines.append("")
                lines.append(f"*{res['station']}: {res['error']}*")
                lines.append("")
                continue
            rl = res["eva"].get("return_levels_m", {})
            eva_ok = "error" not in res["eva"]

            def fmt(k: str, rl: dict = rl) -> str:
                return f"{rl[k]:.2f}" if k in rl else "-"
            lines.append(
                f"| {res['station']} | {res['n_obs']} | {res['span_days']} "
                 f"| {res['rmse']:.3f} | {res['peak_error']:.3f} "
                 f"| {res['coverage']:.2f} "
                 f"| {res['eva'].get('n_storms', '-') if eva_ok else '-'} "
                 f"| {fmt('2yr')} | {fmt('10yr')} | {fmt('100yr')} |")
        lines.append("")

    packs = list_packs()
    lines += [
        "## 4. Engine inventory",
        "",
        (f"- Numeric backends: {', '.join(available_backends())} "
         "(numpy always; numba when installed)"),
        (f"- Constituent packs: {', '.join(sorted(packs)) or 'none'} "
         f"({sum(len(v) for v in packs.values())} plugin constituents)"),
        "- Public API contract: frozen (see `tideglass.marea.contract`)",
        "",
        "---",
        "",
        ("*Numbers produced by the engine, not the marketing department. "
         "Regenerate with `tideglass report`.*"),
    ]
    return "\n".join(lines) + "\n"
