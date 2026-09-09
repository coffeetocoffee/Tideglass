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
    "wave_weight": 0.0,  # wave-coupled term (see rip.py); 0 keeps legacy behaviour
    "ref_wave_height_m": 2.0,  # breaker height that alone means high risk
}

# --- Region-specific regulatory rulesets (v0.4 marine depth) ------------------
# Each entry overrides the harvest/rip defaults for a named region. Merge with
# get_region() so a region only needs to specify what differs from the baseline.
REGION_RULESETS: dict[str, dict] = {
    "PacificNW": {
        "harvest": {
            "low_threshold_m": 0.2,  # exposed sooner on the open coast
            "min_hours": 3.0,  # longer exposure needed for safe digging
            "surge_guard_hours": 36.0,
        },
        "rip": {
            "wave_weight": 0.25,  # heavy winter swell drives rips here
            "ref_wave_height_m": 3.0,
        },
    },
    "GulfCoast": {
        "harvest": {
            "surge_guard_hours": 48.0,  # tropical surge/resuspension lingers
        },
        "rip": {
            "wave_weight": 0.15,
            "ref_wave_height_m": 1.5,
        },
    },
    "NortheastUS": {
        "harvest": {
            "low_threshold_m": 0.35,
            "min_hours": 2.5,
        },
        "rip": {
            "wave_weight": 0.2,
            "ref_wave_height_m": 2.5,
        },
    },
}


def get_region(region: str | None) -> dict:
    """Merge a region's overrides onto the global defaults.

    Returns ``{"harvest": {...}, "rip": {...}}`` with every key filled in from
    ``HARVEST_DEFAULTS`` / ``RIP_DEFAULTS`` then overlaid by the region.
    """
    harvest = dict(HARVEST_DEFAULTS)
    rip = dict(RIP_DEFAULTS)
    if region and region in REGION_RULESETS:
        rule = REGION_RULESETS[region]
        harvest.update(rule.get("harvest", {}))
        rip.update(rule.get("rip", {}))
    return {"harvest": harvest, "rip": rip}

