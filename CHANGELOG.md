# Changelog

All notable changes to Tideglass are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/); this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.5.0]
Milestone "Ambition tier" — released.

### Added
- `marea/crowdsource.py` — `GaugeStore`, a dependency-free local store for
  crowd-sourced cheap-sensor uploads (`time,height` CSV + lon/lat). Multiple
  gauges are merged into the EOF network so every upload makes the shared
  regional field (and each station's denoised reconstruction) strictly better.
- `GaugeStore.network_effect()` quantifies the network effect: explained
  regional variance rises as each station joins.
- `marea/harmonics_db.py` — global coastal coverage from public harmonic
  databases: a curated public-domain seed table (NOAA CO-OPS + IHO/TPXO-style
  values) keyed by region, plus `load_station` / `benchmark_region` /
  `coverage_report` / `add_harmonic_file` to extend coverage from authoritative
  files.
- CLI: `tideglass contribute <csv> <lon> <lat> --station NAME`, `tideglass
  network` (the network effect), `tideglass validate` (global coverage +
  per-region benchmark).
- `VALIDATION.md` — published validation write-up with bench tables vs
  `pytides` (and `UTide` discussion) plus the v0.5 regional coverage and
  network-effect results.

### Changed
- Public API now exports `GaugeStore`, `add_harmonic_file`, `benchmark_region`,
  `coverage_report`, `load_station`.

## [0.4.0]
Milestone "Product surface" — released.

### Added
- `marea/export.py` — feed adapters: `write_csv` / `to_json` / `write_xtide`
  (XTide-style plain-text harmonic constants, round-trippable via `from_xtide`),
  and a dependency-free **NetCDF3 classic** writer/reader (`write_netcdf` /
  `read_netcdf`) so predictions drop into oceanographic pipelines without scipy.
- `web.py` — stdlib `http.server` API serving `GET /predict` and `GET /advise`
  (`tideglass serve [--host --port --store]`); swappable for Flask/FastAPI later.
- `tui.py` and `build_dashboard` — a terminal dashboard (ASCII sparkline,
  high/low, rip risk, harvest windows, species exposure) for `tideglass tui`.
- `marine/alerting.py` — `surge_events` / `StationWatch` group flagged residuals
  into discrete storm-surge events for watched stations (`tideglass alert`).
- `marine/knowledge.REGION_RULESETS` (PacificNW / GulfCoast / NortheastUS) plus
  `get_region` to override harvest + rip defaults per region.
- `marine/rip.risk` now accepts `wave_height_m` / `wave_period_s` for a
  wave-coupled rip-risk term (blended with the tidal rate/range heuristic).
- CLI: `tideglass export`, `serve`, `tui`, `alert`. Top-level `from tideglass
  import build_dashboard, run_server, surge_events, write_netcdf, …` works.

### Changed
- `marine/advisor.TideAdvisor` gains `region=` and wave-coupling parameters; its
  harvest windows and rip risk now honour region rulesets.

## [0.3.0]
Milestone "Deepen the engine" — released.

### Added
- `marea/kalman.py` — `JointModel`, a single linear-Gaussian state-space model
  over harmonics + secular trend + AR(1) surge, smoothed with the RTS smoother.
  This replaces fit-then-decompose: tide, trend (mm/yr, with uncertainty), and
  surge are estimated jointly; the existing `surge` AR(1) tracker becomes the
  process model.
- `marea/calibration.py` — proper-scoring uncertainty calibration: `crps_gaussian`
  / `crps_interval` (CRPS), `evaluate_calibration` (CRPS + coverage reliability),
  and `constituent_attribution` for per-constituent error attribution.
- `marea/spatial.regional_field` — gridded regional assimilation: interpolates
  EOF station loadings onto a continuous lon/lat grid (IDW over haversine
  distances) for a true continuous field, not a per-station series.
- CLI: `tideglass smooth` (joint tide/surge/trend) and `tideglass calibrate`
  (CRPS + coverage + per-constituent variance shares).

### Changed
- `marea/spatial.harmonize` now also returns `loadings` (station × mode spatial
  weights) to support `regional_field`.

## [0.2.0]
Milestone "Harden the core" — released.

### Added
- `tideglass fetch <station> <begin> <end>` — download NOAA CO-OPS gauge CSVs
  directly, so `fit`/`bench` run on any public station without hand-fetched
  files.
- Vectorized nodal kernel: `_basis_matrix` now evaluates the astronomical state
  on arrays instead of a per-(constituent, time) Python loop (`marea/astronomy`
  gains `_astro_state_array` / `nodal_factor_array`). Same math, far cheaper on
  large datasets.
- CI: scheduled weekly real-data benchmark against live NOAA gauges
  (`.github/workflows/ci.yml`, `realdata-bench` job).
- Release engineering: PyPI trusted-publishing workflow and this changelog.

### Changed
- CI matrix now runs `pytest` on Python 3.10 / 3.11 / 3.12 and adds a `ruff`
  lint job.

## [0.1.0] - initial MVP

### Added
- Marea Core math engine: exact SVD linear solver, DCDM constituent selection
  with F-test stopping, covariance-based prediction intervals, dynamic Doodson
  astronomical kernel with nodal corrections, and a declarative constituent
  catalog.
- `tideglass fit` / `predict` / `bench` / `advise` CLI and the `TideModel` /
  `TideAdvisor` public API.
- Post-MVP differentiators: `surge` (residual decomposition), `spatial` (EOF
  harmonization), and the `marine` advice layer.
- Benchmark proof: Marea Core beats `pytides` on NOAA San Francisco 9414290.
