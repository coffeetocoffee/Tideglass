"""Curated marine ruleset (CC0-style defaults with cited sources).

These are *illustrative starting points* grounded in standard references —
tune thresholds to local regulations and conditions. The marine layer never
touches signal math; it consumes ``MareaCore.predict()`` output plus these
rules.
"""

SOURCES = [
    "FAO Fisheries Technical Paper — bivalve sanitation and growing-area management",
    "NOAA National Ocean Service — tide predictions and rip-current science",
    "USLA / NOAA — rip currents correlate with wave energy and tidal stage",
    "General intertidal ecology — zonation by emersion tolerance",
]

# Harvesting defaults: shellfish hand-collection during low-water exposure.
HARVEST_DEFAULTS = {
    "low_threshold_m": 0.3,  # exposed enough to walk/work the beds
    "min_hours": 2.0,  # minimum workable window
    "surge_guard_hours": 24.0,  # avoid beds after a flagged surge event
}

# Rip-current heuristic weights (see rip.py).
RIP_DEFAULTS = {
    "ref_rate_m_per_h": 0.5,  # rate of change that alone means high risk
    "ref_range_m": 2.0,  # spring-range scale
    "rate_weight": 0.7,
    "range_weight": 0.3,
}
