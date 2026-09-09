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

# v0.4 — product surface
tideglass export SF 2024-02-01 --format netcdf --out sf.nc   # CSV/JSON/XTide/NetCDF
tideglass tui SF 2024-02-01                                 # terminal dashboard
tideglass alert data/noaa_9414290_20240101_20240301.csv --station SF --threshold 0.5
tideglass serve --host 127.0.0.1 --port 8000               # GET /predict, /advise
```

```python
from tideglass import TideModel, TideAdvisor, JointModel, build_dashboard
from tideglass.marea import export as EX

model = TideModel.fit(times, heights)          # auto-selects constituents
pred = model.predict(future_times)             # mean / lower / upper

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
│   └── spatial.py    # multi-station EOF harmonization + gridded field
├── marine/           # domain layer (consumes predict() only)
│   ├── knowledge.py / species.py / harvesting.py / rip.py / advisor.py
│   └── alerting.py   # v0.4: surge-event alerts for watched stations
├── web.py            # v0.4: stdlib HTTP API (GET /predict, /advise)
├── tui.py            # v0.4: terminal dashboard
├── cli.py            # fit / predict / bench / advise / smooth / calibrate /
│                    #   export / serve / tui / alert
data/                 # sample NOAA gauge CSVs (San Francisco 9414290)
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
