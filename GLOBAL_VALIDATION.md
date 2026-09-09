# Tideglass — Global Validation Report

*Every region, every source, one page. Generated 2026-09-09 by tideglass 0.9.0 (API contract 1.0). Fully reproducible offline: `tideglass report`.*

## 1. Coverage (published harmonics)

12 stations across 5 regions served from published harmonic constants alone — no fitting required:

| Region | Stations |
|---|---|
| Europe (Atlantic) | 2 |
| Gulf of Mexico | 2 |
| Pacific (Asia) | 2 |
| US East Coast | 3 |
| US West Coast | 3 |

## 2. Per-region validation (self-consistency)

### Europe (Atlantic)

| Station | Range (m) | Dominant | Constituents |
|---|---|---|---|
| Liverpool | 8.76 | M2 | 8 |
| Brest | 6.44 | M2 | 8 |

### Gulf of Mexico

| Station | Range (m) | Dominant | Constituents |
|---|---|---|---|
| Galveston | 1.17 | M2 | 8 |
| Key West | 1.01 | M2 | 8 |

### Pacific (Asia)

| Station | Range (m) | Dominant | Constituents |
|---|---|---|---|
| Tokyo | 2.71 | M2 | 8 |
| Hong Kong | 2.72 | M2 | 8 |

### US East Coast

| Station | Range (m) | Dominant | Constituents |
|---|---|---|---|
| Boston | 2.56 | M2 | 8 |
| New York | 2.63 | M2 | 8 |
| Portland ME | 2.95 | M2 | 8 |

### US West Coast

| Station | Range (m) | Dominant | Constituents |
|---|---|---|---|
| San Francisco | 1.78 | M2 | 8 |
| Los Angeles | 1.55 | M2 | 8 |
| Astoria | 2.79 | M2 | 8 |

## 3. Gauge records (hold-out + extreme value analysis)

| Station | Obs | Span (d) | RMSE (m) | Peak err (m) | CI coverage | Storms | 2-yr (m) | 10-yr (m) | 100-yr (m) |
|---|---|---|---|---|---|---|---|---|---|
| noaa_9414290_20240101_20240301 | 1323 | 61.0 | 0.127 | 0.101 | 0.98 | 4 | 0.62 | 0.81 | 1.08 |
| noaa_9414290_20240701_20240831 | 912 | 38.0 | 0.088 | 0.059 | 0.76 | - | - | - | - |

## 4. Engine inventory

- Numeric backends: numpy, numba (numpy always; numba when installed)
- Constituent packs: great_lakes, rivers, solid_earth (14 plugin constituents)
- Public API contract: frozen (see `tideglass.marea.contract`)

---

*Numbers produced by the engine, not the marketing department. Regenerate with `tideglass report`.*
