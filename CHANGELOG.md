# Changelog

All notable changes to Tideglass are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/); this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added — v2.1 "The protocol grows teeth"
- **Gossip + deltas** (`marea/federation.py`) — peers exchange bundle
  manifests and pull only changed stations: `content_digest` (an
  alias-independent hash of one station's exported constants),
  `PeerBundle.manifest()` (per-station digests + one `bundle_sha256`),
  `PeerBundle.delta()` → `PeerDelta` (added/updated/removed/unchanged).
  `GlobalFederation.ingest` returns the applied delta and skips rewriting the
  stored bundle when it is empty. Bundles carry **lineage**
  (`StationContribution.lineage`: `data_sha256`, the full refit `chain`,
  `fitted_at` from the v0.6 provenance block), which round-trips through
  save/load and is restored into federated models' `meta["lineage"]`.
  CLI: `tideglass sync` prints the manifest digest; `tideglass merge` prints
  each bundle's delta.
- **Peer-level trust** — `GlobalFederation.peer_trust(local_obs)` scores each
  peer by leave-one-peer-out validation: the regional prior induced by all
  peers, and again with one peer removed, predicts your stations' held-out
  tails. Poisoned peers (removal improves your error) decay Gaussian in the
  relative worsening (floor 0.05); duplicated peers are penalized by their
  redundant share, so two identical copies carry one copy's weight; stale
  peers degrade like poisoned ones. Returns `PeerTrustScore` per peer;
  `trust_weights()` expands scores into `HierarchicalPool` weights.
  CLI: `tideglass peers --store DIR [--obs local.csv ...]`.
- **Optional DP noise** — `dp_noisify(bundle, epsilon, delta, clip_m, seed)`
  applies the calibrated Gaussian mechanism to every shared coefficient entry
  (clip, then N(0, σ²) with σ = clip·√(2·ln(1.25/δ))/ε), recorded in bundle
  `meta["dp"]`. CLI: `tideglass sync ... --epsilon E`.
- Top-level exports (additive, contract stays `1.0`): `PeerDelta`,
  `PeerTrustScore`, `dp_noisify`.

### Added — v1.2 "Surge becomes a real forecast"
- `marea/met.py` — **met-forced surge response**, the last physics gap: surge
  was AR(1) — it decayed, it never predicted. `learn_met_response` regresses
  gauge residuals (obs − tide) onto **wind-stress components**
  (τ ∝ −|W|²·(sin θ, cos θ), quadratic drag, meteorological from-direction)
  and the **inverse-barometer** pressure anomaly by OLS, scanning the
  forcing→surge lag (0..max-lag h, best fit wins). The learned barometer is
  sanity-checked against the theoretical −1/(ρ·g) ≈ −0.0099 m/hPa
  (`BARO_THEORY_M_PER_HPA`); `MetResponse.surge` applies the lagged response
  to any forcing, `MetResponse.forecast` produces the 48-h surge forecast —
  the met-driven mean does **not** decay with lead time (the nowcast→forecast
  jump), with the v0.6 AR(1) layered on the unexplained residual (band grows
  toward the residual marginal). Persists as `<station>.met.json`.
- **Met ingestion** — `fetch.fetch_met` / `fetch_met_range` pull NOAA CO-OPS
  `wind` + `air_pressure` products (merged on timestamp, chunked at the 31-day
  cap, deduped at seams); `write_met_csv` / `met.read_met_csv` round-trip the
  `time,wind_speed,wind_dir,pressure` format any forecast source (NWS export,
  hand file) can emit. CLI `tideglass fetch --met`.
- **Downstream wiring** — `extremes.flood_probability` and
  `decision.decision_curve` accept **per-time** `surge_mean`/`surge_sigma`
  arrays (backward-compatible scalars), so a met-forced surge forecast prices
  the act/wait decision directly.
- CLI: `tideglass surge <obs.csv> <met.csv> [--station --store --max-lag
  --p-ref] [--forecast met.csv --hours 48] [--flood M] [--cost C --loss L]` —
  learns the response (auto-fitting and saving the harmonic model when absent),
  prints diagnostics, hourly surge ± band, P(flood), and the priced
  `DecisionCurve`.
- Top-level exports (additive, contract stays `1.0`): `MetResponse`,
  `SurgeForecast`, `learn_met_response`.

### Added — v1.3 "Accountability + regime-aware uncertainty"
- **Decision ledger** (`marea/ledger.py`) — every priced act/wait decision
  (from a `DecisionCurve`) is logged with its probability, action, and
  economics; realized water levels are recorded later and reconciled.
  `LedgerReport` reports the **realized cost vs the always-act and never-act
  baselines** — "the forecast earned X" in the operator's currency — plus event
  recall / action precision, turning the advisor into an auditable, B2B-grade
  system. Persists as `<station>.ledger.json`; CLI `tideglass ledger
  <station> [--outcome o.csv]` audits, and `advise ... --ledger PATH` writes
  each advisory's priced decision into the ledger.
- **Regime-conditional calibration** (`marea/regimes.py`) — the single
  conformal quantile is stratified by regime so each band uses the quantile of
  *its own* error distribution and widens exactly when it should. **spring/neap**
  is read from the local predicted tidal range (the M2/S2 fortnightly envelope);
  **storm/non-storm** from the residual (or, at forecast time, the v1.2
  met-forced surge). `learn_regime_calibration` fits per-regime conformal
  quantiles; `RegimeCalibration.apply` rescales a prediction's band per time.
  Persists as `<station>.regimes.json`; `tideglass calibrate --regimes` learns
  and saves it.
- Top-level exports (additive, contract stays `1.0`): `DecisionLedger`,
  `RegimeCalibration`, `learn_regime_calibration`.

### Added — v1.1 "The engine that learns"
- `marea/residual.py` — **residual memory**: a learned day-of-year bias table
  (`learn_residual` → `ResidualModel`) fitted to post-harmonic residuals with
  robust circular bins, conformalized on a held-out tail. Attached via
  `TideModel.attach_residual`, it is applied by every `predict()` and survives
  the artifact round-trip (`meta["residual_bias"]`). The harmonics stay the
  physics; the bias table is the asset pytides/TPXO cannot have. CLI
  `tideglass correct <csv> --station S` prints before/after RMSE + coverage.
- **Trust-weighted federation** — `federation.trust_score` /
  `TrustLedger` turn v1.0's QC history into a per-sensor reputation in (0, 1];
  `HierarchicalPool(..., trust=)` inflates low-trust members' own uncertainty
  (`s²/w`) so unreliable sensors shrink toward the regional prior instead of
  contaminating it. `FederatedRefit` records every QC pass and pools with
  ledger weights; `FederatedReport.trust` / CLI `federate` report it.
- Top-level exports (additive, contract stays `1.0`): `ResidualModel`,
  `learn_residual`, `TrustLedger`.

## [1.0.0]
Milestone "The moat itself" — the roadmap's final tier. The crowd network
becomes self-improving, and the marine layer *prices* uncertainty instead of
just displaying it.

### Added
- `marea/qc.py` — **QC pipeline for cheap-sensor uploads** (spike / datum /
  drift): robust MAD spike detection on the local-tide residual, datum-shift
  detection (vs a reference model, else step change between record halves),
  robust drift slope on a long-window smoother, plus hard sanity rejections
  (too few points, non-finite, non-monotonic time, flatlining).
  `clean_series` interpolates flagged spikes for fitting. `QcConfig`,
  `QcReport`, `qc_check`, `clean_series` exported at the top level.
- `marea/federation.py` — **federated refits**: `FederatedRefit.refit`
  QC-checks an upload, adds it to the `GaugeStore`, then folds it into the
  network via `HierarchicalPool` (`pool` for dense gauges, `seed_short` for
  short records) and persists the improved model; per-round `network_effect`
  before/after makes "every sensor helps everyone" measurable.
  `FederatedRefit`, `FederatedReport`, `federate` exported.
- `marea/decision.py` — **decision-theoretic pricing**: the classical
  cost-loss model over the prediction CI. `decision_curve` prices the act/wait
  policy per hour (act where `P(event) > cost/loss`, the break-even
  probability; optimal expected cost `min(cost, p·loss)`) and reports value vs
  the forecast-blind baselines (always-act / never-act). `CostLoss`,
  `DecisionCurve`, `decision_curve` exported; `PUBLIC_API` grows additively
  (contract stays `1.0`).
- `TideAdvisor.advise(..., cost=, loss=, decision_threshold_m=)` attaches a
  priced `DecisionCurve` to the `Advice` (threshold defaults to
  `flood_threshold_m`) — the marine layer stays a thin consumer.
- CLI: `tideglass qc <csv> <lon> <lat> --station S`,
  `tideglass federate <csv> <lon> <lat> --station S`, and
  `tideglass advise ... --flood Z --cost C --loss L`.
- Project docs `MVP.md` and `architecture.md` are now part of the repository.

### Fixed
- `fetch` / `poll` no longer fail on NOAA CO-OPS' **31-day per-request range
  limit** (the CI `realdata-bench` job's 45-day window got HTTP 400/422
  "Range Limit Exceeded"). New `fetch_noaa_range` transparently chunks long
  windows at 30 days with a 1-hour seam overlap, sorts, and dedupes;
  `tideglass fetch` and `ops.poll` use it. Raw `urllib` HTTPError tracebacks
  are replaced by `ValueError`/`OSError` messages carrying NOAA's own error
  text (e.g. "Range Limit Exceeded"), so failures are diagnosable at a glance.

## [0.9.0]
Milestone "Ecosystem lock-in" — the artifact competitors have to answer.

### Added
- `marea/kernel.py` — **optional accelerated numeric kernel** for the design
  matrix (the engine's hot loop): a fused Numba JIT path (Julian date → mean
  longitudes → Schureman node factors → design rows in one typed loop) that is
  numerically identical to the numpy reference (max |Δ| ≈ 1e-15 over the full
  catalog × 4,000 hours). Numba is never a hard dependency — the package still
  installs numpy-only, `kernel="numpy"` remains the default everywhere, and
  `"auto"`/`"numba"` fall back cleanly with a clear error when Numba is absent.
  `TideModel.fit`/`predict` accept `kernel=`; `tideglass fit --kernel numba`.
- **Public API freeze** (`marea/contract.py`) — `PUBLIC_API` pins the exact
  `tideglass.__all__` surface; `verify_public_api()` fails loudly on any
  accidental add/remove/rename (enforced by the test suite), and `api_version`
  ("1.0") is the semver of the *contract* itself. `__version__` now exposed.
- `marea/plugins.py` — **plugin constituent packs**: register extra
  constituents beyond the built-in catalog (seed packs `rivers`,
  `great_lakes`, `solid_earth`; 14 composition-exact compound constituents —
  e.g. 2M2 = 2×M2's Doodson, so derived speeds are physically correct).
  `register_pack` / `register_constituent`, JSON pack files (`load_pack_file`,
  `load_pack_dir`), `find` / `all_constituents`; `constituents.get` resolves
  plugin names, so packs are first-class fit candidates. `tideglass plugins`.
- `marea/report.py` + `tideglass report` — **the one-page global validation
  report**: coverage and per-region self-consistency from the published
  harmonics DB, hold-out bench + skew-surge GPD return levels for every gauge
  record under `data/`, and the engine inventory (backends, packs, API
  contract) in a single deterministic markdown page — `GLOBAL_VALIDATION.md`
  is committed and reproducible offline.

### Changed
- `pyproject.toml` version → 0.9.0; `tideglass.__version__` added.
- Top-level exports: kernel (`available_backends`, `basis_matrix`), plugin
  (`register_pack`, `register_constituent`, `load_pack_file`, `load_pack_dir`,
  `list_packs`, `find`, `all_constituents`), contract (`api_version`,
  `verify_public_api`), constituents (`CATALOG`, `Constituent`, `get`,
  `speed`, `principal`), harmonics (`list_regions`, `stations_in_region`).

## [0.8.0]
Milestone "Uncertainty as the product" — released.

### Added
- `marea/pooling.py` — hierarchical partial pooling across the station
  network: per-constituent empirical-Bayes shrinkage of `(a, b)` toward the
  regional prior (DerSimonian–Laird between-station spread; inverse-variance
  mean, or inverse-distance neighbour mean when coords are known). Long
  records keep weight ≈ 1; short/cheap sensors shrink hard and impute
  unselected constituents from the prior. H0 stays local (datums differ);
  output is a plain `TideModel`. `HierarchicalPool.pool` / `seed_short` /
  `report`; `tideglass pool <short.csv> --station NAME --store DIR`.
- `marea/extremes.py` — extreme value analysis: `skew_surge` (obs−pred at
  predicted high waters), runs-declustering (`decluster`), peaks-over-
  threshold GPD fit by maximum likelihood (`fit_gpd`, dependency-free
  Nelder–Mead), `GPD.survival` / `return_level`, `annual_rate`,
  `joint_exceedance_probability`, and `flood_probability` — the probabilistic
  threshold replacing heuristic cutoffs. CLI `tideglass extremes <csv>
  [--threshold-q Q] [--gap-h H] [--return-periods ...] [--flood M]`.
- Calibration graduation (`marea/calibration.py`): `pit_values` /
  `pit_histogram` (exact-erf PIT uniformity), `reliability_curve` (empirical
  vs nominal coverage via Acklam's inverse normal CDF), `conformal_quantile`
  / `conformalize` (normalized split-conformal bands with the finite-sample
  guarantee). `tideglass calibrate` now runs a chronological train/calib/test
  split (`--calib-fraction`, `--alpha-level`).
- Marine hook: `TideAdvisor.advise(..., flood_threshold_m=,
  surge_sigma_m=)` prices `P(level > threshold)` per time from the prediction
  band; `Advice` gains `flood` / `flood_threshold_m` (default `None`, thin
  consumer unchanged otherwise).
- Public API now exports `HierarchicalPool`, `PooledConstituent`, `GPD`,
  `SkewSurge`, `fit_gpd`, `skew_surge`, `decluster`, `annual_rate`,
  `joint_exceedance_probability`, `flood_probability`, `pit_values`,
  `pit_histogram`, `reliability_curve`, `conformal_quantile`, `conformalize`.

### Notes
- Real-data checks (NOAA San Francisco 9414290, 2022–2024 hourly): 51
  declustered storms, GPD(0.13 m, σ=0.082 m, ξ≈+0.01), 10-yr skew-surge level
  0.56 m / 100-yr 0.76 m; reliability 0.50→0.49 … 0.99→0.99 on a 3-yr
  train/calib/test split, while the PIT mean (0.30) flags a −0.6σ seasonal
  low-water bias the symmetric band hides. EVA refuses short records with a
  clear error (GPD needs ≥ 10 exceedances) instead of fitting junk.

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
