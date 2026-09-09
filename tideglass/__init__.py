"""Tideglass — tide + marine-life intelligence engine.

Marea Core (``tideglass.marea``) is the math engine; the marine layer
(``tideglass.marine``) is a thin consumer of its predictions. The v0.4 product
surface adds export feeds, a stdlib HTTP API, and a terminal dashboard.
"""

from tideglass import marea, marine
from tideglass.marea.bench_global import compare, global_model_at, read_harmonic_grid
from tideglass.marea.calibration import (
    constituent_attribution,
    crps_gaussian,
    evaluate_calibration,
)
from tideglass.marea.crowdsource import GaugeStore
from tideglass.marea.export import (
    read_netcdf,
    to_json,
    to_xtide,
    write_csv,
    write_json,
    write_netcdf,
    write_xtide,
)
from tideglass.marea.harmonics_db import (
    add_harmonic_file,
    benchmark_region,
    coverage_report,
    load_station,
)
from tideglass.marea.drift import HealthMonitor, HealthReport
from tideglass.marea.kalman import JointModel
from tideglass.marea.krige import KrigeField, krige_field, krige_regional
from tideglass.marea.model import Prediction, TideModel
from tideglass.marea.nowcast import NowcastEngine, UpdateLog
from tideglass.marea.ops import OpsReport, cold_start, poll, rerun
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
    "Advice",
    "GaugeStore",
    "HealthMonitor",
    "HealthReport",
    "JointModel",
    "KrigeField",
    "NowcastEngine",
    "OpsReport",
    "Prediction",
    "ResponseTransfer",
    "StationWatch",
    "TideAdvisor",
    "TideModel",
    "TransferCoefficients",
    "UpdateLog",
    "add_harmonic_file",
    "benchmark_region",
    "build_dashboard",
    "cold_start",
    "compare",
    "constituent_attribution",
    "coverage_report",
    "crps_gaussian",
    "evaluate_calibration",
    "format_self_check",
    "global_model_at",
    "krige_field",
    "krige_regional",
    "load_station",
    "marea",
    "marine",
    "native_to_model",
    "poll",
    "read_harmonic_grid",
    "read_netcdf",
    "read_tpxo",
    "rerun",
    "run_server",
    "self_check_tpxo",
    "surge_events",
    "to_json",
    "to_xtide",
    "tpxo_model_at",
    "write_csv",
    "write_json",
    "write_netcdf",
    "write_xtide",
]
