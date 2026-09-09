"""Tideglass — tide + marine-life intelligence engine.

Marea Core (``tideglass.marea``) is the math engine; the marine layer
(``tideglass.marine``) is a thin consumer of its predictions. The v0.4 product
surface adds export feeds, a stdlib HTTP API, and a terminal dashboard.
"""

from tideglass import marea, marine

__version__ = "0.9.0"
from tideglass.marea.bench_global import compare, global_model_at, read_harmonic_grid
from tideglass.marea.calibration import (
    conformal_quantile,
    conformalize,
    constituent_attribution,
    crps_gaussian,
    evaluate_calibration,
    pit_histogram,
    pit_values,
    reliability_curve,
)
from tideglass.marea.constituents import (
    CATALOG,
    Constituent,
    get,
    principal,
    speed,
)
from tideglass.marea.contract import api_version, verify_public_api
from tideglass.marea.crowdsource import GaugeStore
from tideglass.marea.drift import HealthMonitor, HealthReport
from tideglass.marea.export import (
    read_netcdf,
    to_json,
    to_xtide,
    write_csv,
    write_json,
    write_netcdf,
    write_xtide,
)
from tideglass.marea.extremes import (
    GPD,
    SkewSurge,
    annual_rate,
    decluster,
    fit_gpd,
    flood_probability,
    joint_exceedance_probability,
    skew_surge,
)
from tideglass.marea.harmonics_db import (
    add_harmonic_file,
    benchmark_region,
    coverage_report,
    list_regions,
    load_station,
    stations_in_region,
)
from tideglass.marea.kalman import JointModel
from tideglass.marea.kernel import available_backends, basis_matrix
from tideglass.marea.krige import KrigeField, krige_field, krige_regional
from tideglass.marea.model import Prediction, TideModel
from tideglass.marea.nowcast import NowcastEngine, UpdateLog
from tideglass.marea.ops import OpsReport, cold_start, poll, rerun
from tideglass.marea.plugins import (
    all_constituents,
    find,
    list_packs,
    load_pack_dir,
    load_pack_file,
    register_constituent,
    register_pack,
)
from tideglass.marea.pooling import HierarchicalPool, PooledConstituent
from tideglass.marea.tpxo import (
    format_self_check,
    native_to_model,
    read_tpxo,
    self_check_tpxo,
    tpxo_model_at,
)
from tideglass.marea.transfer import ResponseTransfer, TransferCoefficients
from tideglass.marine.advisor import Advice, TideAdvisor
from tideglass.marine.alerting import StationWatch, surge_events
from tideglass.tui import build_dashboard
from tideglass.web import run_server

__all__ = [
    "CATALOG",
    "GPD",
    "Advice",
    "Constituent",
    "GaugeStore",
    "HealthMonitor",
    "HealthReport",
    "HierarchicalPool",
    "JointModel",
    "KrigeField",
    "NowcastEngine",
    "OpsReport",
    "PooledConstituent",
    "Prediction",
    "ResponseTransfer",
    "SkewSurge",
    "StationWatch",
    "TideAdvisor",
    "TideModel",
    "TransferCoefficients",
    "UpdateLog",
    "add_harmonic_file",
    "all_constituents",
    "annual_rate",
    "api_version",
    "available_backends",
    "basis_matrix",
    "benchmark_region",
    "build_dashboard",
    "cold_start",
    "compare",
    "conformal_quantile",
    "conformalize",
    "constituent_attribution",
    "coverage_report",
    "crps_gaussian",
    "decluster",
    "evaluate_calibration",
    "find",
    "fit_gpd",
    "flood_probability",
    "format_self_check",
    "get",
    "global_model_at",
    "joint_exceedance_probability",
    "krige_field",
    "krige_regional",
    "list_packs",
    "list_regions",
    "load_pack_dir",
    "load_pack_file",
    "load_station",
    "marea",
    "marine",
    "native_to_model",
    "pit_histogram",
    "pit_values",
    "poll",
    "principal",
    "read_harmonic_grid",
    "read_netcdf",
    "read_tpxo",
    "register_constituent",
    "register_pack",
    "reliability_curve",
    "rerun",
    "run_server",
    "self_check_tpxo",
    "skew_surge",
    "speed",
    "stations_in_region",
    "surge_events",
    "to_json",
    "to_xtide",
    "tpxo_model_at",
    "verify_public_api",
    "write_csv",
    "write_json",
    "write_netcdf",
    "write_xtide",
]
