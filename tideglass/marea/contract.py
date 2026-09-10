"""Public API freeze for Tideglass — v0.9 semver lock-in.

From v0.9 onward the **public** surface of Tideglass is a contract. Adding,
removing, or renaming a symbol in :data:`tideglass.__all__` is a deliberate,
reviewed change — it bumps the major version. :func:`verify_public_api` pins the
current surface so an accidental export change fails CI loudly (the test suite
calls it). Internal modules (``tideglass.marea.*``) remain free to change.

``api_version`` is the semantic version of the *contract*, independent of the
package release version: the contract is ``1.0`` and is guaranteed stable across
all ``1.x`` releases (additive, backward-compatible additions only).
"""

from __future__ import annotations

# Semantic version of the *contract* (independent of the package release).
api_version = "1.0"

# The frozen public surface. Every name here is part of the supported API; order
# is irrelevant but the *set* is what is locked. It mirrors ``tideglass.__all__``
# exactly — ``verify_public_api`` enforces that they never drift apart.
PUBLIC_API: frozenset[str] = frozenset({
    "GPD", "Advice", "CATALOG", "Constituent", "GaugeStore", "GlobalFederation",
    "HealthMonitor",
    "HealthReport", "HierarchicalPool", "JointModel", "KrigeField",
    "MetResponse", "NowcastEngine", "OpsReport", "PooledConstituent",
    "Prediction", "PeerBundle", "ResponseTransfer", "SkewSurge", "StationWatch",
    "SurgeForecast", "TideAdvisor", "TideModel",
    "TransferCoefficients", "UpdateLog", "add_harmonic_file", "all_constituents",
    "annual_rate", "api_version", "available_backends", "basis_matrix",
    "benchmark_region", "build_dashboard", "cold_start", "compare",
    "conformal_quantile", "conformalize", "constituent_attribution",
    "coverage_report", "crps_gaussian",     "CostLoss", "decision_curve",
    "DecisionCurve", "DecisionLedger", "decluster", "evaluate_calibration",
    "FederatedRefit", "FederatedReport", "federate", "find", "fit_gpd",
    "flood_probability", "format_self_check", "get",
    "global_model_at", "joint_exceedance_probability", "krige_field",
    "krige_regional", "learn_met_response", "learn_regime_calibration",
    "learn_residual", "list_packs",
    "list_regions", "load_pack_dir",
    "load_pack_file", "load_station", "marea", "marine", "native_to_model",
    "pit_histogram", "pit_values", "poll", "principal", "qc_check",
    "QcConfig", "QcReport", "read_harmonic_grid",
    "read_netcdf", "read_tpxo", "register_constituent", "register_pack",
    "reliability_curve", "RegimeCalibration", "ResidualModel",
    "learn_residual", "rerun",
    "run_server", "self_check_tpxo", "skew_surge",
    "speed", "stations_in_region", "surge_events", "to_json", "to_xtide",
    "tpxo_model_at", "TrustLedger",
    "verify_public_api", "write_csv", "write_json",
    "write_netcdf", "write_xtide", "clean_series",
})


def verify_public_api() -> None:
    """Ensure the live ``tideglass.__all__`` exactly matches :data:`PUBLIC_API`.

    Raises ``AssertionError`` if a public symbol was added, removed, or renamed
    without an intentional contract update (and the matching major bump).
    """
    import tideglass

    live = frozenset(tideglass.__all__)
    assert live == PUBLIC_API, (
        "Public API drift detected. "
        f"added={sorted(live - PUBLIC_API)} "
        f"removed={sorted(PUBLIC_API - live)}. "
        "Update PUBLIC_API deliberately (with a major-version bump) if this is "
        "intentional."
    )
