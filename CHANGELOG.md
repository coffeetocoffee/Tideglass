# Changelog

All notable changes to Tideglass are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/); this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- `fetch` / `poll` no longer fail on NOAA CO-OPS' **31-day per-request range
  limit** (the CI `realdata-bench` job's 45-day window got HTTP 400/422
  "Range Limit Exceeded"). New `fetch_noaa_range` transparently chunks long
  windows at 30 days with a 1-hour seam overlap, sorts, and dedupes;
  `tideglass fetch` and `ops.poll` use it. Raw `urllib` HTTPError tracebacks
  are replaced by `ValueError`/`OSError` messages carrying NOAA's own error
  text (e.g. "Range Limit Exceeded"), so failures are diagnosable at a glance.

## [0.7.1]
Closes the v0.7 TPXO caveat: genuine global-model files are now a supported
bench path (no more "drop a real extraction in the same schema" hope).

### Added
- `marea/tpxo.py` — genuine TPXO/FES ingestion: a dependency-free
  NetCDF3-classic parser (big-endian XDR, NumPy only; fixed-grid variables,
  `scale_factor`/`add_offset`/`_FillValue` honored, record variables refused
  with a clear error) plus fuzzy TPXO9 layout discovery
  (`h_am_<c>`/`h_pu_<c>`, lon/lat/mask) with an explicit `var_map` override,
  lon wrapping to -180..180°, bbox cropping (antimeridian-safe), and
  nearest-*water* point selection (land-masked points skipped).
- `native_to_model` — phase-convention conversion *without formula risk*:
  predicts the global model in its native `Σ A·cos(ωt − g)` convention and
  re-fits with the exact solver on the fixed constituent set. Phase alignment
  is exact by construction; the round-trip reproduces the native series up to
  absorbed nodal modulation (~1e-3). Off-catalog constituents are skipped and
  recorded in `meta["tpxo_skipped"]`; chosen point/file pinned in meta.
- `self_check_tpxo` / `format_self_check` — validates the pipeline against
  NOAA published constants (`harmonics_db`): per-constituent amplitude and
  circular-phase diffs with a PASS/FAIL verdict. A mismatch indicts the
  file/layout, not the engine.
- CLI: `tideglass bench --against tpxo --global-model h_tpxo9.v1.nc
  --lon X --lat Y [--consts M2,S2,...]` (`.nc` dispatches to the TPXO reader;
  CSV grids keep working). Missing-file behavior unchanged (degrades to
  marea-only); missing grid/coords remains a hard usage error.
- Public API now exports `read_tpxo`, `native_to_model`, `tpxo_model_at`,
  `self_check_tpxo`, `format_self_check`.

### Notes
- TPXO (OSU) and FES (AVISO/LEGOS) files are free for research but forbid
  redistribution: they can never ship in this repo. Operators register at
  https://www.tpxo.net / https://www.aviso.altimetry.fr, download the
  elevation file, and bench it directly. `data/grids/tpxo_sample.csv` stays as the
  license-clean CI stand-in; all NetCDF3 test fixtures are generated in-test
  (no licensed bytes anywhere).

## [0.7.0]
Milestone "From interpolation to physics (the spatial moat)" — released.

### Added
- `marea/krige.py` — ordinary kriging / Gaussian-process spatial harmonics:
  per-EOF-mode loadings are kriged over great-circle distances with an
  exponential covariance (auto range = 0.6× network extent, small nugget);
  the regional field is rebuilt as `Σ_k s_k·ŵ_k(grid) ⊗ mode_k` and now carries
  a **variance field** (`field_var`), so the gridded regional field is itself a
  random field. IDW fallback preserves the <3-station cases.
- `regional_field(method="krige")` — kriging is now the default interpolation
  (`method="idw"` keeps the legacy behavior; `RegionalField.field_var` added).
- `marea/transfer.py` — `ResponseTransfer`: classic admiralty response-function
  transfer from a well-observed reference port. Per-constituent gain + phase
  lag are estimated from neighbour ports (IDW over distance), seeding a target
  station whose short record is then used only to **refine** (fixed constituent
  set, no DCDM) — the missing piece for sparse gauges.
- `marea/bench_global.py` — benchmark against global hydrodynamic models
  (TPXO/FES): reads a `lon,lat,constituent,amplitude,phase` harmonic grid,
  builds the global model at the gauge (nearest or IDW blend), and scores
  marea vs global with the same held-out metrics as the pytides bench.
  `data/grids/tpxo_sample.csv` bundles a clearly-labelled representative grid
  (derived from NOAA SF constants, coarse mesh, 5 constituents); drop a real
  TPXO/FES extraction in the same schema to bench the genuine model.
- CLI: `tideglass bench --against tpxo --global-model PATH --lon X --lat Y`
  (prints marea vs global RMSE + ratio); missing grid/coords is a hard usage
  error, an unreadable grid degrades to a marea-only bench.
- Public API now exports `KrigeField`, `krige_field`, `krige_regional`,
  `ResponseTransfer`, `TransferCoefficients`, `read_harmonic_grid`,
  `global_model_at`, `compare`.

### Changed
- `RegionalField` gained `field_var` and `method`; `regional_field` defaults to
  kriging and adds the blended mean level to the gridded field (IDW previously
  reconstructed the anomaly only).

## [0.6.0]
Milestone "Operational core" — released.

### Added
- `marea/nowcast.py` — `NowcastEngine`: recursive Kalman/RLS assimilation of
  streaming gauge observations into a fitted `TideModel` (static harmonic state
  with random-walk process noise), plus an AR(1) surge tracker so forecasts
  ahead of the last observation carry a decayed surge nudge with growing bands.
  Observation noise defaults to the base model's own residual variance, keeping
  nowcast bands calibrated. State round-trips to `<store>/<station>.nowcast.json`.
- `marea/drift.py` — `HealthMonitor`: rolling-window staleness detection
  (CI-coverage collapse, RMSE blow-up vs the fit baseline, bias drift), with
  `needs_refit` reasons and a persistable rolling buffer.
- `marea/provenance.py` — data versioning: every `TideModel.fit` pins
  `source`, `obs_start`/`obs_end`, `n_obs`, a canonical `data_sha256` of the
  training rows, `tideglass_version`, and `fitted_at`; artifacts round-trip the
  block through `to_artifact`/`load_harmonic`.
- `marea/ops.py` — the operational flywheel: `cold_start` (first fit from a
  fetched window), `rerun` (assimilate a feed, score health, auto-refit from
  accumulated history with `refit_history` lineage, replay-safe skips), and
  `poll` (one live pass: fetch since the persisted cursor, then `rerun`).
- CLI: `tideglass fit --source S` (provenance tag, printed with the fit),
  `tideglass nowcast <station> <feed.csv> [--auto-refit|--no-auto-refit]
  [--hours H]` (assimilate + health + optional forecast), and `tideglass poll
  <station> [--lookback-h H] [--repeat N] [--sleep-s S]` (live NOAA loop).
- Public API now exports `NowcastEngine`, `UpdateLog`, `HealthMonitor`,
  `HealthReport`, `OpsReport`, `cold_start`, `rerun`, `poll`,
  `canonical_digest`, `provenance`, `load_state`, `save_state`.

### Fixed
- `TideModel.to_artifact` no longer clobbers the pinned provenance `source`
  with the coarse construction tag; `load_harmonic` restores the full `meta`.

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
