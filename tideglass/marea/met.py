"""Meteorological surge response — the v1.2 physics upgrade.

Until now surge was AR(1): it decayed, it never *predicted*. This module
learns a station's local response to the weather and turns it into a real
forecast: given forecast wind + pressure (NOAA/NWS feeds), the learned
response produces a **48-hour surge forecast** — the one place a data network
beats even a perfect harmonic model, because a harmonic model has no idea
what the atmosphere will do next.

The response is the standard reduced storm-surge model::

    surge(t) = c0 + a·τ_u(t−L) + b·τ_v(t−L) + β·Δp(t−L)

* **wind-stress vector regression** — the wind pushes water downwind with a
  force ∝ |W|·W (quadratic drag). ``τ_u, τ_v`` are the east/north stress
  components (up to the constant ρ·C_D factor, absorbed into ``a``/``b``);
  per-station coefficients encode the local shoreline's exposure.
* **inverse-barometer term** — sea level rises ≈ 1 cm per hPa of pressure
  *drop*; ``β`` is learned and sanity-checked against the theoretical
  −1/(ρ·g) ≈ −0.0099 m/hPa.
* **lag L** — the response to forcing arrives hours later (shelf dynamics);
  the lag is scanned and chosen by best fit.

:func:`learn_met_response` fits the coefficients by ordinary least squares on
concurrent gauge residuals (obs − tide) and meteorological history.
:meth:`MetResponse.forecast` applies them to forecast forcing, optionally
layering the v0.6 AR(1) tracker on the unexplained residual for the first
hours of lead. The mean level does **not** decay with lead time — that is the
difference between a nowcast and a forecast.

Met CSV format (any source, incl. NOAA/NWS exports): ``time,wind_speed,
wind_dir,pressure`` — ISO datetimes, wind speed m/s, wind direction degrees
(meteorological: the direction the wind blows *from*), pressure hPa.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from tideglass.marea.surge import AR1

# Theoretical inverse-barometer slope: −1/(ρ_w·g) per Pa, 1 hPa = 100 Pa.
BARO_THEORY_M_PER_HPA = -100.0 / (1025.0 * 9.81)  # ≈ −0.00994 m/hPa
DEFAULT_P_REF = 1013.25  # standard atmosphere (hPa)
_MIN_OBS = 48


def _parse_iso(s: str) -> datetime:
    s = s.strip()
    if s.endswith(("Z", "z")):
        s = s[:-1] + "+00:00"
    t = datetime.fromisoformat(s)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t


def _hours_since(t0: datetime, times: Sequence[datetime]) -> np.ndarray:
    return np.array([(t - t0).total_seconds() / 3600.0 for t in times],
                    dtype=float)


def stress_features(wind_speed, wind_dir_deg):
    """East/north wind-stress proxy components (m²/s², ρ·C_D factored out).

    ``wind_dir_deg`` uses the meteorological convention — the direction the
    wind blows *from* (0° = from north). The drag force on the water points
    *downwind*, i.e. toward ``(u, v) = (−|W|·sin θ, −|W|·cos θ)``, and stress
    is quadratic: ``τ = (−|W|²·sin θ, −|W|²·cos θ)``.
    """
    w = np.asarray(wind_speed, dtype=float).ravel()
    d = np.asarray(wind_dir_deg, dtype=float).ravel()
    if w.shape != d.shape:
        raise ValueError(f"{w.size} wind speeds but {d.size} directions")
    if np.any(w < 0):
        raise ValueError("wind speeds must be ≥ 0")
    rad = np.deg2rad(d)
    return -(w * w) * np.sin(rad), -(w * w) * np.cos(rad)


@dataclass
class MetResponse:
    """A station's learned local response of surge to wind and pressure."""

    intercept: float  # c0 (m)
    stress_u: float  # a: m of surge per eastward stress unit (m per m²/s²)
    stress_v: float  # b: same, northward
    barometer: float  # β: m per hPa (≈ −0.0099 for a healthy fit)
    lag_hours: int  # fitted forcing→response lag (hours)
    sigma: float  # std of the fit residuals (m)
    r_squared: float  # variance explained by the response
    n: int  # samples fitted
    p_ref: float = DEFAULT_P_REF  # pressure reference (hPa)

    # -- applying the response -------------------------------------------------

    def _features(self, times, wind_speed, wind_dir_deg, pressure):
        """Stress/barometer features lagged onto ``times`` (hourly interp)."""
        times = list(times)
        if len(times) != np.asarray(wind_speed).size:
            raise ValueError(
                f"{len(times)} times but {np.asarray(wind_speed).size} "
                "met samples")
        su, sv = stress_features(wind_speed, wind_dir_deg)
        dp = np.asarray(pressure, dtype=float).ravel() - self.p_ref
        if su.size != dp.size:
            raise ValueError(
                f"{su.size} wind samples but {dp.size} pressure samples")
        if self.lag_hours:
            x = _hours_since(times[0], times)
            su = np.interp(x - self.lag_hours, x, su)
            sv = np.interp(x - self.lag_hours, x, sv)
            dp = np.interp(x - self.lag_hours, x, dp)
        return su, sv, dp

    def surge(self, times, wind_speed, wind_dir_deg, pressure) -> np.ndarray:
        """Mean surge (m) under the given (concurrent) met forcing."""
        su, sv, dp = self._features(times, wind_speed, wind_dir_deg, pressure)
        return (self.intercept + self.stress_u * su + self.stress_v * sv
                + self.barometer * dp)

    def forecast(self, times, wind_speed, wind_dir_deg, pressure,
                 ar: AR1 | None = None, last_residual: float = 0.0,
                 origin: datetime | None = None) -> SurgeForecast:
        """Surge forecast (m) under *forecast* met forcing.

        The met-driven mean does not decay with lead time. With ``ar`` (an
        AR(1) fitted to the response's unexplained residuals) the last
        residual ``last_residual`` (at ``origin``, or one step before the
        first forecast time) is layered on with its usual decay, and the band
        grows with the AR(1) h-step forecast variance toward the residual
        marginal. Without ``ar`` the band is the constant fit ``sigma``.
        """
        times = list(times)
        mean = self.surge(times, wind_speed, wind_dir_deg, pressure)
        n = len(times)
        if ar is None:
            sigma = np.full(n, self.sigma)
        else:
            if origin is not None:
                dt = np.maximum(_hours_since(origin, times), 0.0)
            else:
                dt = np.arange(1, n + 1, dtype=float)
            mean = mean + float(last_residual) * ar.phi ** dt
            if abs(ar.phi) < 1.0:
                var = (ar.sigma_eps ** 2
                       * (1.0 - ar.phi ** (2.0 * dt)) / (1.0 - ar.phi ** 2))
            else:  # non-stationary fallback: random-walk growth
                var = ar.sigma_eps ** 2 * dt
            sigma = np.sqrt(np.maximum(var, 0.0))
        return SurgeForecast(times=times, mean=mean, sigma=sigma)

    # -- persistence -----------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "intercept": float(self.intercept),
            "stress_u": float(self.stress_u),
            "stress_v": float(self.stress_v),
            "barometer": float(self.barometer),
            "lag_hours": int(self.lag_hours),
            "sigma": float(self.sigma),
            "r_squared": float(self.r_squared),
            "n": int(self.n),
            "p_ref": float(self.p_ref),
        }

    @classmethod
    def from_dict(cls, d: dict) -> MetResponse:
        return cls(
            intercept=float(d["intercept"]),
            stress_u=float(d["stress_u"]),
            stress_v=float(d["stress_v"]),
            barometer=float(d["barometer"]),
            lag_hours=int(d["lag_hours"]),
            sigma=float(d["sigma"]),
            r_squared=float(d["r_squared"]),
            n=int(d["n"]),
            p_ref=float(d.get("p_ref", DEFAULT_P_REF)),
        )


@dataclass
class SurgeForecast:
    """Per-time surge forecast: mean (m) and std (m)."""

    times: list
    mean: np.ndarray
    sigma: np.ndarray


def learn_met_response(
    times: Sequence[datetime],
    surge,
    met_times: Sequence[datetime],
    wind_speed,
    wind_dir_deg,
    pressure,
    max_lag_hours: int = 6,
    p_ref: float = DEFAULT_P_REF,
    min_coverage: float = 0.9,
) -> tuple[MetResponse, dict]:
    """Fit the local wind/pressure → surge response by least squares.

    :param times: observation timestamps (chronological).
    :param surge: concurrent surge series (m) — typically ``obs −
        model.predict()`` from a fitted harmonic model.
    :param met_times: met timestamps (own grid; need not match ``times``).
    :param wind_speed: m/s; ``wind_dir_deg`` meteorological direction (from);
        ``pressure`` hPa.
    :param max_lag_hours: lag scanned 0..max (integer hours); the best fit
        wins.
    :param p_ref: pressure reference for the inverse-barometer anomaly.
    :param min_coverage: fraction of ``t − lag`` that must fall inside the met
        window for a lag to be a candidate.
    :returns: ``(MetResponse, diagnostics)``.
    """
    times = list(times)
    mt = list(met_times)
    y = np.asarray(surge, dtype=float).ravel()
    ws = np.asarray(wind_speed, dtype=float).ravel()
    wd = np.asarray(wind_dir_deg, dtype=float).ravel()
    pr = np.asarray(pressure, dtype=float).ravel()
    if y.size != len(times):
        raise ValueError(f"{len(times)} times but {y.size} surge samples")
    if not (ws.size == wd.size == pr.size == len(mt)):
        raise ValueError(
            f"met arrays mismatch: {len(mt)} times, {ws.size} speeds, "
            f"{wd.size} directions, {pr.size} pressures")
    if y.size < _MIN_OBS:
        raise ValueError(
            f"record too short to learn a response (need ≥ {_MIN_OBS} obs, "
            f"got {y.size})")
    if max_lag_hours < 0:
        raise ValueError("max_lag_hours must be ≥ 0")
    if len(mt) < max_lag_hours + 2:
        raise ValueError("met record shorter than the lag scan")
    med_p = float(np.median(pr))
    if not 500.0 <= med_p <= 1200.0:
        raise ValueError(
            f"median pressure {med_p:.1f} does not look like hPa "
            "(want ~1013; check the met feed units)")
    order = np.argsort(_hours_since(times[0], mt), kind="stable")
    mt = [mt[i] for i in order]
    ws, wd, pr = ws[order], wd[order], pr[order]

    su, sv = stress_features(ws, wd)
    dp = pr - float(p_ref)
    h_obs = _hours_since(times[0], times)
    h_met = _hours_since(times[0], mt)

    y_var = float(np.var(y))
    if y_var < 1e-12:
        raise ValueError("surge series is constant; nothing to learn")

    best = None
    r2_lag0 = None
    for lag in range(int(max_lag_hours) + 1):
        tq = h_obs - lag
        covered = float(np.mean((tq >= h_met[0]) & (tq <= h_met[-1])))
        if covered < min_coverage:
            continue
        X = np.column_stack([
            np.ones(y.size),
            np.interp(tq, h_met, su),
            np.interp(tq, h_met, sv),
            np.interp(tq, h_met, dp),
        ])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        ss_res = float(resid @ resid)
        r2 = 1.0 - ss_res / (y_var * y.size)
        if lag == 0:
            r2_lag0 = r2
        if best is None or r2 > best[0]:
            best = (r2, lag, coef, resid)
    if best is None:
        raise ValueError(
            "met record does not cover the observation window "
            f"({mt[0].isoformat()} .. {mt[-1].isoformat()})")
    r2, lag, coef, resid = best

    sigma = float(np.std(resid, ddof=X.shape[1])) if y.size > X.shape[1] \
        else float(np.std(resid))
    # Lag-1 correlation of the unexplained residual — a diagnostic the CLI uses
    # to seed the v0.6 AR(1) tracker on the response's leftover memory.
    lag1 = None
    try:
        if resid.size > 2:
            lag1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    except Exception:  # noqa: BLE001 - diagnostic only
        lag1 = None
    resp = MetResponse(
        intercept=float(coef[0]), stress_u=float(coef[1]),
        stress_v=float(coef[2]), barometer=float(coef[3]),
        lag_hours=lag, sigma=sigma, r_squared=float(max(r2, 0.0)),
        n=int(y.size), p_ref=float(p_ref),
    )
    diag = {
        "n": int(y.size),
        "lag": int(lag),
        "r2": float(max(r2, 0.0)),
        "r2_lag0": (None if r2_lag0 is None else float(max(r2_lag0, 0.0))),
        "sigma": sigma,
        "rmse_before": float(np.sqrt(np.mean(y ** 2))),
        "rmse_after": float(np.sqrt(np.mean(resid ** 2))),
        "barometer_theory": BARO_THEORY_M_PER_HPA,
        "barometer_ratio": (float(coef[3] / BARO_THEORY_M_PER_HPA)
                            if abs(BARO_THEORY_M_PER_HPA) > 0 else None),
        "lag1_corr": lag1,
    }
    return resp, diag


# --- met CSV I/O ----------------------------------------------------------------


def read_met_csv(path: str):
    """Read a met CSV (``time,wind_speed,wind_dir,pressure``).

    Columns are matched by header name when present (extra columns ignored);
    otherwise the canonical order above is assumed. Blank rows and rows with
    unparsable cells are skipped. Returns ``(times, wind_speed, wind_dir,
    pressure)``.
    """
    wanted = {"time": None, "wind_speed": None, "wind_dir": None,
              "pressure": None}
    times: list[datetime] = []
    speed: list[float] = []
    direc: list[float] = []
    press: list[float] = []
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    cols = None
    for row in rows:
        if not row or all(not c.strip() for c in row):
            continue
        if cols is None:
            head = [c.strip().lower() for c in row]
            if "time" in head:
                for key in wanted:
                    wanted[key] = head.index(key) if key in head else None
                if wanted["time"] is None:
                    raise ValueError(f"{path!r}: met CSV has no 'time' column")
                if any(wanted[k] is None for k in
                       ("wind_speed", "wind_dir", "pressure")):
                    raise ValueError(
                        f"{path!r}: met CSV needs wind_speed, wind_dir and "
                        "pressure columns")
                cols = wanted
                continue
            cols = {"time": 0, "wind_speed": 1, "wind_dir": 2, "pressure": 3}
        try:
            t = _parse_iso(row[cols["time"]])
            s = float(row[cols["wind_speed"]])
            d = float(row[cols["wind_dir"]])
            p = float(row[cols["pressure"]])
        except (ValueError, IndexError):
            continue
        times.append(t)
        speed.append(s)
        direc.append(d)
        press.append(p)
    if not times:
        raise ValueError(f"no readable met rows in {path!r}")
    return times, np.asarray(speed), np.asarray(direc), np.asarray(press)


def write_met_csv(rows, path: str) -> None:
    """Write ``(time, wind_speed, wind_dir, pressure)`` rows as a met CSV."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "wind_speed", "wind_dir", "pressure"])
        for t, s, d, p in rows:
            w.writerow([t.isoformat(), f"{s:.2f}", f"{d:.1f}", f"{p:.2f}"])


# ---------------------------------------------------------------------------
# v2.3 — estuarine river-discharge coupling
# ---------------------------------------------------------------------------
# In estuaries, a fraction of the residual water level is driven by antecedent
# river discharge (Q), not local weather. If ignored, that component inflates the
# met-response's ``sigma`` and biases its coefficients. This models it
# additively on the *met-unexplained* residual and learns it by least squares.
#
# Discharge CSV (any source, e.g. USGS): ``time,discharge`` -- ISO datetimes,
# discharge in m^3/s.


def _rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
    """Trailing uniform rolling mean with window ``win`` (samples)."""
    if win <= 1:
        return x.copy()
    if win >= x.size:
        return np.full_like(x, float(np.mean(x)))
    cs = np.cumsum(np.concatenate([[0.0], x]))
    out = np.empty_like(x)
    w = int(win)
    out[: w - 1] = cs[1:w] / np.arange(1, w, dtype=float)
    out[w - 1 :] = (cs[w:] - cs[:-w]) / float(w)
    return out


def _golden_section_1d(f, lo: float, hi: float, tol: float = 1e-7,
                       maxiter: int = 500) -> float:
    """Dependency-free golden-section minimiser over ``[lo, hi]``."""
    gr = 0.5 * (math.sqrt(5.0) - 1.0)
    a, b = float(lo), float(hi)
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(maxiter):
        if abs(b - a) < tol * (abs(c) + abs(d) + 1e-9):
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


@dataclass
class DischargeCoupling:
    """Learned estuarine surge coupling to river discharge (v2.3).

    Models the discharge-driven component additively::

        surge_estuary(t) += alpha * Q_antecedent(t)^beta

    where ``Q_antecedent`` is a trailing mean of recent discharge (window
    ``tau_hours``) capturing the catchment's memory, and ``beta < 1`` reflects
    sublinear channel storage. ``alpha > 0`` (discharge only pushes water up).
    """

    alpha: float  # m of surge per (m^3/s)^beta
    beta: float  # exponent (sublinear)
    tau_hours: int  # antecedent averaging window (h)
    sigma: float  # residual std after coupling (m)
    r_squared: float  # variance explained by the coupling term
    n: int  # samples fitted

    def apply(self, times, discharge_times, discharge) -> np.ndarray:
        """Mean discharge-driven surge (m) at ``times``."""
        times = list(times)
        q = _rolling_mean(np.asarray(discharge, dtype=float).ravel(),
                          max(1, self.tau_hours))
        h_times = _hours_since(times[0], times)
        h_disc = _hours_since(next(iter(discharge_times)), list(discharge_times))
        q_at_t = np.interp(h_times, h_disc, q)
        return self.alpha * np.maximum(q_at_t, 0.0) ** self.beta

    def to_dict(self) -> dict:
        return {
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "tau_hours": int(self.tau_hours),
            "sigma": float(self.sigma),
            "r_squared": float(max(self.r_squared, 0.0)),
            "n": int(self.n),
        }

    @classmethod
    def from_dict(cls, d: dict) -> DischargeCoupling:
        return cls(
            alpha=float(d["alpha"]),
            beta=float(d["beta"]),
            tau_hours=int(d["tau_hours"]),
            sigma=float(d["sigma"]),
            r_squared=float(d.get("r_squared", 0.0)),
            n=int(d["n"]),
        )


def learn_discharge_coupling(
    times: Sequence[datetime],
    met_residual,
    discharge_times: Sequence[datetime],
    discharge,
    tau_hours: int = 48,
) -> tuple[DischargeCoupling, dict]:
    """Fit the discharge-driven component of an estuarine residual (v2.3).

    ``met_residual`` is the *unexplained* residual left by the v1.2 met
    response (``surge - MetResponse.surge(...)``), so the discharge term is
    isolated cleanly. Returns ``(DischargeCoupling, diagnostics)``.
    """
    times = list(times)
    y = np.asarray(met_residual, dtype=float).ravel()
    if y.size != len(times):
        raise ValueError(f"{len(times)} times but {y.size} residual samples")
    if y.size < _MIN_OBS:
        raise ValueError(
            f"record too short to learn discharge coupling (need >= {_MIN_OBS}, "
            f"got {y.size})")
    if tau_hours <= 0:
        raise ValueError("tau_hours must be positive")
    q_raw = np.asarray(discharge, dtype=float).ravel()
    h_disc = _hours_since(next(iter(discharge_times)), list(discharge_times))
    q = _rolling_mean(q_raw, max(1, tau_hours))
    h_times = _hours_since(times[0], times)
    q_at_t = np.maximum(np.interp(h_times, h_disc, q), 0.0)
    y_var = float(np.var(y))
    if y_var < 1e-12:
        raise ValueError("met residual is constant; nothing to couple")

    pos = q_at_t > 0.0
    qpos, ypos = q_at_t[pos], y[pos]
    if qpos.size < max(_MIN_OBS // 2, 8):
        raise ValueError(
            f"discharge series does not cover the observation window "
            f"({int(pos.sum())} positive samples, need >= {max(_MIN_OBS // 2, 8)})")
    qv = float(np.var(qpos))
    if qv < 1e-12:
        raise ValueError("discharge variance is zero over the window")

    def _ss(beta: float) -> float:
        qb = qpos ** float(beta)
        den = float(qb @ qb)
        if den == 0.0:
            return math.inf
        a = float((qb @ ypos) / den)
        if a < 0.0:
            return math.inf  # physical: discharge pushes water up
        resid = y - a * q_at_t ** float(beta)
        return float(resid @ resid)

    beta = _golden_section_1d(_ss, 0.05, 2.0)
    qb = qpos ** beta
    alpha = float((qb @ ypos) / float(qb @ qb))
    alpha = max(alpha, 0.0)
    resid_all = y - alpha * q_at_t ** beta
    ss_res = float(resid_all @ resid_all)
    r2 = max(0.0, 1.0 - ss_res / (y_var * y.size))
    sigma = float(np.std(resid_all, ddof=2)) if y.size > 2 \
        else float(np.std(resid_all))
    dc = DischargeCoupling(alpha=alpha, beta=beta,
                           tau_hours=int(tau_hours),
                           sigma=sigma, r_squared=r2, n=int(y.size))
    diag = {
        "n": int(y.size),
        "tau_hours": int(tau_hours),
        "beta": beta,
        "alpha": alpha,
        "r2": r2,
        "sigma": sigma,
        "rmse_before": float(np.sqrt(np.mean(y ** 2))),
        "rmse_after": float(np.sqrt(np.mean(resid_all ** 2))),
    }
    return dc, diag


def read_discharge_csv(path: str):
    """Read a ``time,discharge`` CSV (ISO datetimes, m^3/s).

    Header-aware: columns are matched by name when a ``time`` header is
    present, otherwise the canonical ``time,discharge`` order is assumed.
    Blank/unparsable rows are skipped. Returns ``(times, discharge)``.
    """
    wanted = {"time": None, "discharge": None}
    times: list[datetime] = []
    vals: list[float] = []
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    cols = None
    for row in rows:
        if not row or all(not c.strip() for c in row):
            continue
        if cols is None:
            head = [c.strip().lower() for c in row]
            if "time" in head:
                for key in wanted:
                    wanted[key] = head.index(key) if key in head else None
                if wanted["time"] is None or wanted["discharge"] is None:
                    raise ValueError(
                        f"{path!r}: discharge CSV needs 'time' and "
                        "'discharge' columns")
                cols = wanted
                continue
            cols = {"time": 0, "discharge": 1}
        try:
            t = _parse_iso(row[cols["time"]])
            v = float(row[cols["discharge"]])
        except (ValueError, IndexError):
            continue
        times.append(t)
        vals.append(v)
    if not times:
        raise ValueError(f"no readable discharge rows in {path!r}")
    return times, np.asarray(vals, dtype=float)
