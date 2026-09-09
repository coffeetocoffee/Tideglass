# Changelog

All notable changes to Tideglass are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/); this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `tideglass fetch <station> <begin> <end>` — download NOAA CO-OPS gauge CSVs
  directly, so `fit`/`bench` run on any public station without hand-fetched
  files (v0.2 data-ingestion milestone).
- Vectorized nodal kernel: `_basis_matrix` now evaluates the astronomical state
  on arrays instead of a per-(constituent, time) Python loop (`marea/astronomy`
  gains `_astro_state_array` / `nodal_factor_array`). Same math, far cheaper on
  large datasets (v0.2 performance milestone).
- CI: scheduled weekly real-data benchmark against live NOAA gauges
  (`.github/workflows/ci.yml`, `realdata-bench` job).
- Release engineering: PyPI trusted-publishing workflow and this changelog
  (v0.2 milestone).

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
