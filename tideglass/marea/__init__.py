"""Marea Core — the math engine."""

from tideglass.marea import (
    astronomy,
    calibration,
    constituents,
    export,
    kalman,
    metrics,
    model,
    selection,
    solver,
    spatial,
    surge,
)
from tideglass.marea.calibration import (
    constituent_attribution,
    crps_gaussian,
    crps_interval,
    evaluate_calibration,
)
from tideglass.marea.kalman import JointFit, JointModel
from tideglass.marea.model import Fit, Prediction, TideModel
from tideglass.marea.spatial import EOFResult, harmonize, regional_field

__all__ = [
    "EOFResult",
    "Fit",
    "JointFit",
    "JointModel",
    "Prediction",
    "TideModel",
    "astronomy",
    "calibration",
    "constituent_attribution",
    "constituents",
    "crps_gaussian",
    "crps_interval",
    "evaluate_calibration",
    "export",
    "harmonize",
    "kalman",
    "metrics",
    "model",
    "regional_field",
    "selection",
    "solver",
    "spatial",
    "surge",
]
