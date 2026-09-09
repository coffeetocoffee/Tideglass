# Tideglass

A tide + marine-life intelligence engine. The **math core (Marea Core)** is the
real product; the marine layer is a thin, well-fed consumer of
`predict(height, time) ± σ`.

Marea Core beats `pytides` where it matters: exact linear solve instead of
nonlinear `leastsq`, automatic DCDM constituent selection with F-test
stopping, covariance-based prediction intervals, a dynamic astronomical
kernel valid at any epoch, surge decomposition, and multi-station EOF
harmonization. v0.3 adds the engine depth: a **joint Kalman smoother** over
harmonics + secular trend + surge (replacing fit-then-decompose), **proper
scoring (CRPS)** calibration with per-constituent error attribution, and
**gridded regional assimilation** of the EOF field. v0.4 adds the **product
surface**: an HTTP API, a terminal dashboard, feed exports (CSV/JSON/XTide/
  NetCDF3), and marine depth (region rulesets, wave-coupled rip, surge alerts).

v0.5 is the ambition tier: a **crowd-sourced gauge network** (`GaugeStore`)
where every cheap-sensor upload improves the shared EOF regional field — a real
network effect — plus **global coastal coverage** from public harmonic databases
(`harmonics_db`, benchmarked per region) and a published **validation write-up**
(`VALIDATION.md`) with bench tables vs `pytides` and `UTide`.

## Install

```bash
pip install -e .
# optional, for the bench baseline:
pip install pytides
```

Requires Python ≥ 3.10 and numpy. `pytides` 0.0.4 is unmaintained and broken
on Python 3.12; `bench` applies minimal runtime shims purely to run the
head-to-head comparison.

## Quickstart

```bash
# Fit a model from gauge observations (time,height CSV, ISO datetimes, metres)
tideglass fit data/noaa_9414290_20240101_20240301.csv --station SF

# Height curve with 95% bands
tideglass predict SF 2024-02-01

# Benchmark against pytides on held-out data
tideglass bench data/noaa_9414290_20240101_20240301.csv --station SF

# Marine advice: harvest windows, rip risk, species exposure
tideglass advise SF 2024-02-01

# Joint tide + surge + secular-trend smoothing (v0.3)
tideglass smooth data/noaa_9414290_20240101_20240301.csv --station SF

# Uncertainty calibration: CRPS + coverage + per-constituent variance shares
tideglass calibrate data/noaa_9414290_20240101_20240301.csv --station SF
```  # (v0.8: + PIT + reliability curve + split-conformal bands)

```bash
# v0.8 — uncertainty as the product: pooling, extremes, calibration
tideglass pool short.csv --station pier07 --store .tideglass  # borrow strength
tideglass extremes gauge3yr.csv --station SF --flood 2.5      # GPD return levels
```

# v0.4 — product surface
tideglass export SF 2024-02-01 --format netcdf --out sf.nc   # CSV/JSON/XTide/NetCDF
tideglass tui SF 2024-02-01                                 # terminal dashboard
tideglass alert data/noaa_9414290_20240101_20240301.csv --station SF --threshold 0.5
tideglass serve --host 127.0.0.1 --port 8000               # GET /predict, /advise
```

```bash
# v0.5 — crowd-sourced gauges + global coverage
tideglass contribute upload.csv -122.34 47.60 --station pier07   # cheap-sensor upload
tideglass network                                             # the network effect
tideglass validate                                            # coverage + per-region bench
```

```bash
# v0.6 — operational core: nowcast, drift watch, auto-refit
tideglass fit train.csv --station SF --source noaa-coops:9414290  # provenance pinned
tideglass nowcast SF feed.csv --hours 24        # assimilate + health + refit if stale
tideglass poll 9414290 --repeat 24 --sleep-s 3600  # live loop: fetch, nowcast, refit
```

```bash
# v0.7 — spatial moat: kriging fields, transfer, global-model bench
# regional_field() now kriges EOF loadings -> field + field_var (GP uncertainty)
# response-function transfer seeds short-record stations from a reference port
tideglass bench data/noaa_9414290_20240101_20240301.csv --station SF \
    --against tpxo --global-model data/grids/tpxo_sample.csv --lon -122.47 --lat 37.81
# v0.7.1 — genuine TPXO/FES files (register at tpxo.net / aviso.altimetry.fr;
# free for research, no redistribution — download the elevation file yourself):
tideglass bench data/noaa_9414290_20240101_20240301.csv --station SF \
    --against tpxo --global-model h_tpxo9.v1.nc --lon -122.47 --lat 37.81
```

```python
from tideglass import TideModel, TideAdvisor, JointModel, build_dashboard
from tideglass import NowcastEngine, HealthMonitor, rerun  # v0.6: operations
from tideglass import ResponseTransfer, krige_regional, global_model_at  # v0.7
from tideglass import read_tpxo, tpxo_model_at  # v0.7.1: genuine TPXO/FES
from tideglass import self_check_tpxo, format_self_check
from tideglass import HierarchicalPool, fit_gpd  # v0.8: uncertainty as product
from tideglass import reliability_curve, conformalize, flood_probability
from tideglass.marea import export as EX

model = TideModel.fit(times, heights)          # auto-selects constituents
pred = model.predict(future_times)             # mean / lower / upper

# v0.6: live assimilation + drift watch (no batch refit)
eng = NowcastEngine(model)
eng.update(new_times, new_heights)              # recursive Kalman update
rep = rerun("SF", list(zip(new_times, new_heights)), store=".tideglass")
print(rep.health)                               # ok, or REFIT + auto-refit applied

# v0.7: gridded field with uncertainty + short-record transfer
field = regional_field(eof, coords, grid_lons, grid_lats)  # kriging default
#   field.field (grid x time), field.field_var (GP kriging variance)
tgt = ResponseTransfer(ref_model, ref_coords, neighbors, tgt_coords
                       ).refine(short_times, short_heights)

# v0.7.1: genuine global model at a gauge + pipeline self-check
glob = tpxo_model_at("h_tpxo9.v1.nc", -122.47, 37.81)  # nearest water point
print(format_self_check(self_check_tpxo(                      # vs NOAA pub.
    "h_tpxo9.v1.nc", -122.47, 37.81, "US West Coast", "San Francisco")))

# v0.8: uncertainty as the product — pooling, extremes, calibration
short = HierarchicalPool(network_models).seed_short(         # borrow strength
    short_times, short_heights, "pier07")      # shrinkage in .meta
gpd = fit_gpd(declustered_skew_surges, threshold=0.13)        # POT tail
print(gpd.return_level(100.0, rate_per_year=17.0))            # 100-yr level
print(reliability_curve(pred, held_out)["empirical"])        # ≈ nominal
lo, hi, q = conformalize(pred.mean, sig, cal.mean, cal_sig, cal_y)
print(TideAdvisor(model).advise(
    future_times, flood_threshold_m=2.5).summary)  # probabilistic threshold

# v0.3: estimate tide + surge + secular trend jointly
joint = JointModel.fit(times, heights)
print(joint.fit_result.trend_mm_yr)            # e.g. +3.12 mm/yr ± 0.40
print(TideAdvisor(model).advise(future_times).summary)

# v0.4: feeds + dashboard
EX.write_netcdf(model, future_times, "sf.nc")  # dependency-free NetCDF3
print(build_dashboard(model, future_times))
```

## Measured proof (NOAA San Francisco 9414290, held-out tail)

Winter 2024, storm season (41 d train / 14 d test):

| model   | rmse(m) | peak_err | coverage |
|---------|---------|----------|----------|
| marea   | 0.1267  | 0.1004   | 0.9668   |
| pytides | 0.7770  | 0.6693   | n/a      |

Summer 2024, calmer spell (28 d train / 10 d test):

| model   | rmse(m) | peak_err | coverage |
|---------|---------|----------|----------|
| marea   | 0.0869  | 0.0618   | 0.7588   |
| pytides | 0.2606  | 0.1592   | n/a      |

Coverage dips below nominal when genuine non-tidal variance (storm setup,
upwelling anomalies) exceeds the training residual — that signal belongs to
`surge.decompose`, not the harmonic engine.

## Layout

```
tideglass/
├── marea/            # math engine (no marine imports)
│   ├── astronomy.py  # Doodson kernel + nodal factors (Schureman SP-98)
│   ├── constituents.py
│   ├── solver.py     # exact SVD least-squares + covariance
│   ├── selection.py  # DCDM + F-test + Rayleigh gate
│   ├── model.py      # TideModel: fit / predict / load_harmonic
│   ├── metrics.py    # rmse / peak error / coverage
│   ├── surge.py      # residual decomposition + AR(1) tracker
│   ├── kalman.py     # v0.3: joint tide+surge+trend state-space smoother
│   ├── calibration.py # v0.3: CRPS + coverage + per-constituent attribution
│   ├── export.py      # v0.4: CSV/JSON/XTide/NetCDF3 feed adapters
│   ├── spatial.py    # multi-station EOF harmonization + gridded field
│   ├── crowdsource.py # v0.5: crowd-sourced gauge network (network effect)
│   ├── harmonics_db.py # v0.5: global coverage from public harmonic databases
│   ├── nowcast.py    # v0.6: recursive Kalman assimilation + AR(1) surge nowcast
│   ├── drift.py      # v0.6: rolling coverage/RMSE/bias monitor + refit trigger
│   ├── provenance.py # v0.6: obs window + source + sha256 pinned in artifacts
│   ├── ops.py        # v0.6: cold_start / rerun / poll operational flywheel
│   ├── krige.py      # v0.7: ordinary-kriging / GP spatial harmonics + variance
│   ├── transfer.py   # v0.7: response-function transfer from reference ports
│   ├── bench_global.py # v0.7: benchmark vs TPXO/FES-style global grids
│   └── tpxo.py       # v0.7.1: genuine TPXO/FES NetCDF3 ingestion + self-check
├── marine/           # domain layer (consumes predict() only)
│   ├── knowledge.py / species.py / harvesting.py / rip.py / advisor.py
│   └── alerting.py   # v0.4: surge-event alerts for watched stations
├── web.py            # v0.4: stdlib HTTP API (GET /predict, /advise)
├── tui.py            # v0.4: terminal dashboard
├── cli.py            # fit / predict / bench / advise / smooth / calibrate /
│                    #   export / serve / tui / alert / contribute / network /
│                    #   validate / nowcast / poll (v0.6)
data/                 # sample NOAA gauge CSVs (SF 9414290); grids/ has the
│                     #   TPXO-style harmonic-grid stand-in (tpxo_sample.csv)
```

See `architecture.md` for module boundaries and math, `MVP.md` for the build
record with acceptance criteria.

## Testing

```bash
pip install pytest
python -m pytest tests -q
```

## License & credits

- Marea Core: MIT. Marine ruleset: CC0-style defaults (tune locally).
- Astronomical basis: Schureman, *Special Publication 98*; Meeus,
  *Astronomical Algorithms*; IERS conventions.
- Gauge data: NOAA CO-OPS (San Francisco station 9414290).
- Inspired by (and benchmarked against) `sam-cox/pytides`.
