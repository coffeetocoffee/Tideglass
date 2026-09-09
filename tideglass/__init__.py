"""Tideglass — tide + marine-life intelligence engine.

Marea Core (``tideglass.marea``) is the math engine; the marine layer
(not yet built) will consume its predictions.
"""

from tideglass import marea, marine
from tideglass.marea.model import Prediction, TideModel
from tideglass.marine.advisor import Advice, TideAdvisor

__all__ = ["Advice", "Prediction", "TideAdvisor", "TideModel", "marea", "marine"]
