# Tideglass

A tide + marine-life intelligence engine. The **math core (Marea Core)** is the
real product; the marine layer is a thin, well-fed consumer of
`predict(height, time) ± σ`.

Marea Core beats `pytides` where it matters: exact linear solve instead of
nonlinear `leastsq`, automatic DCDM constituent selection with F-test
stopping, covariance-based prediction intervals, a dynamic astronomical
kernel valid at any epoch, surge decomposition, and multi-station EOF
harmonization.

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
```

```python
from tideglass import TideModel, TideAdvisor

model = TideModel.fit(times, heights)          # auto-selects constituents
pred = model.predict(future_times)             # mean / lower / upper
print(TideAdvisor(model).advise(future_times).summary)
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
│   └── spatial.py    # multi-station EOF harmonization
├── marine/           # domain layer (consumes predict() only)
│   ├── knowledge.py / species.py / harvesting.py / rip.py / advisor.py
├── cli.py            # fit / predict / bench / advise
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
