# Tideglass

A tide + marine-life intelligence engine. The **math core (Marea Core)** is the
product; the marine layer (advice, harvesting, rip risk) is a thin consumer of
`predict(time) → height ± σ`.

## Why not just pytides?

Marea Core beats `pytides` on the same public gauge data — exact linear solve
instead of nonlinear `leastsq`, automatic constituent selection, honest
uncertainty, and an engine that *learns*:

- **Exact solver** — SVD least-squares with full covariance; prediction
  intervals from parameter + residual variance
- **Automatic constituent selection** — DCDM greedy orthogonal selection with
  F-test stopping; no hand-picked harmonic sets
- **Dynamic astronomy** — Doodson arguments + nodal factors (Schureman SP-98),
  valid at any epoch
- **It learns** — a day-of-year residual bias layer conformalized on held-out
  data, plus trust-weighted pooling over a crowd-sourced sensor network
- **Real uncertainty** — CRPS/PIT calibration, conformal bands, GPD extreme
  return levels, cost-loss-priced decisions
- **Beyond single stations** — joint Kalman tide+surge+trend smoothing, kriged
  regional fields, response-function transfer, genuine TPXO/FES ingestion

## Install

```bash
pip install -e .
# optional, for the bench baseline:
pip install pytides
```

Requires Python ≥ 3.10 and numpy (the only hard dependency).

## Quickstart

```bash
tideglass fit data/noaa_9414290_20240101_20240301.csv --station SF
tideglass predict SF 2024-02-01          # hourly curve with 95% bands
tideglass bench data/noaa_9414290_20240101_20240301.csv --station SF
tideglass advise SF 2024-02-01           # harvest windows, rip risk, species
```

```python
from tideglass import TideModel, TideAdvisor

model = TideModel.fit(times, heights)   # auto-selects constituents
pred = model.predict(future_times)      # mean / lower / upper

print(TideAdvisor(model).advise(future_times).summary)
```

`tideglass --help` lists all 23 subcommands — `smooth`, `calibrate`, `extremes`,
`nowcast`, `poll`, `pool`, `federate`, `correct`, `export`, `serve`, `tui`, …

## Measured proof

NOAA San Francisco 9414290, held-out tail (marea vs pytides):

| record                | marea rmse (m) | pytides rmse (m) | coverage |
|-----------------------|---------------:|-----------------:|---------:|
| winter 2024 (storms)  | **0.1267**     | 0.7770           | 0.9668   |
| summer 2024 (calm)    | **0.0869**     | 0.2606           | 0.7588   |

Coverage dips below nominal when genuine non-tidal variance (storm setup)
exceeds the training residual — that signal belongs to the surge model, not
the harmonic engine. Full details: `VALIDATION.md`, `GLOBAL_VALIDATION.md`.

## What's inside

```
tideglass/
├── marea/                  # math engine (numpy-only, no marine imports)
│   ├── astronomy / constituents / solver / selection / model
│   ├── kalman / surge / nowcast / ops / drift / provenance
│   ├── spatial / krige / transfer / crowdsource / pooling
│   ├── calibration / extremes / metrics / residual / federation / decision
│   ├── qc / kernel / contract / plugins / report
│   └── export / tpxo / bench_global / harmonics_db
├── marine/                 # thin consumer: advisor, harvesting, rip, species
├── cli.py · web.py · tui.py · fetch.py
data/                       # sample NOAA gauge CSVs (SF 9414290)
GLOBAL_VALIDATION.md        # regenerate: tideglass report
```

`architecture.md` covers module boundaries and math; `MVP.md` is the build
record with acceptance criteria.

## Testing

```bash
pip install pytest
python -m pytest tests -q
```

## License & credits

- Marea Core: MIT. Marine rulesets: CC0-style defaults (tune locally).
- Astronomical basis: Schureman, *Special Publication 98*; IERS conventions.
- Gauge data: NOAA CO-OPS (San Francisco 9414290).
- Inspired by (and benchmarked against) `sam-cox/pytides`.
