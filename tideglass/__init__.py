"""Tideglass — tide + marine-life intelligence engine.

Marea Core (``tideglass.marea``) is the math engine; the marine layer
(``tideglass.marine``) is a thin consumer of its predictions. The v0.4 product
surface adds export feeds, a stdlib HTTP API, and a terminal dashboard.
"""

from tideglass import marea, marine
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
from tideglass.marea.kalman import JointModel
from tideglass.marea.model import Prediction, TideModel
from tideglass.marine.advisor import Advice, TideAdvisor
from tideglass.marine.alerting import StationWatch, surge_events
from tideglass.tui import build_dashboard
from tideglass.web import run_server

__all__ = [
    "Advice",
    "GaugeStore",
    "JointModel",
    "Prediction",
    "StationWatch",
    "TideAdvisor",
    "TideModel",
    "add_harmonic_file",
    "benchmark_region",
    "build_dashboard",
    "constituent_attribution",
    "coverage_report",
    "crps_gaussian",
    "evaluate_calibration",
    "load_station",
    "marea",
    "marine",
    "read_netcdf",
    "run_server",
    "surge_events",
    "to_json",
    "to_xtide",
    "write_csv",
    "write_json",
    "write_netcdf",
    "write_xtide",
]
