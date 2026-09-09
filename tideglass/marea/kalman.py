"""Joint tide + surge + trend estimation via Kalman smoother.

This replaces the old fit-then-decompose pipeline. Instead of fitting harmonics,
subtracting them to get a residual, then fitting an AR(1) to that residual, we
estimate **everything at once** in a single linear-Gaussian state-space model
and smooth it with the Rauch–Tung–Striebel (RTS) smoother.

State at step ``k`` (observation index ``k``, time ``t_k``)::

    x_k = [L_k, β, ξ_k, a_1, b_1, ..., a_m, b_m]

where

    L_k = mean sea level *including* the linear secular trend
          L_k = L_{k-1} + Δt_k · β
    β   = secular trend slope (m per hour; reported as mm/yr)
    ξ_k = non-tidal surge, an AR(1) process:  ξ_k = φ·ξ_{k-1} + w_k
    a_i, b_i = amplitude/phase coefficients of constituent i
               (constant in time, estimated online by the filter)

Observation::

    y_k = L_k + Σ_i [a_i·cos(ω_i t_k) + b_i·sin(ω_i t_k)] + ξ_k + ε_k

The model is exactly linear in the state, so the Kalman filter and RTS
smoother are exact (no linearization / EKF needed). The surge AR(1) tracker
(:mod:`tideglass.marea.surge`) supplies ``φ`` and the innovation variance, which
become the process model — i.e. the AR(1) tracker *is* the surge dynamics here,
not a separate post-step.

Output (a :class:`JointFit`) carries the smoothed tide curve, surge series, the
secular trend with uncertainty, and the per-constituent coefficients with their
smoothed standard errors (the basis for per-constituent error attribution).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from tideglass.marea import constituents as CON
from tideglass.marea.astronomy import julian_day
from tideglass.marea.solver import rad_per_hour
from tideglass.marea.surge import fit_ar1

_HOURS_PER_YEAR = 24.0 * 365.25
_MM_PER_M = 1000.0
# Prior std on the secular trend slope (~tens of mm/yr). This keeps the trend
# from absorbing the low-frequency wander of the (long-memory) surge AR(1): the
# slope is estimated but regularized toward 0, so genuine multi-decadal drift
# shows up as a trend rather than a ramp fit to storm noise.
_TREND_PRIOR_MM_YR = 30.0


def _hours_since_j2000(times: Sequence[datetime]) -> np.ndarray:
    return np.array([(julian_day(t) - 2451545.0) * 24.0 for t in times], dtype=float)


# --- Generic Kalman filter / RTS smoother ------------------------------------


def _kalman_filter(
    y: np.ndarray,
    F: list[np.ndarray],
    H: list[np.ndarray],
    Q: list[np.ndarray],
    R: float,
    x0: np.ndarray,
    P0: np.ndarray,
    mask: np.ndarray | None = None,
):
    """Forward Kalman filter for a time-varying linear-Gaussian model.

    Returns per-step predicted and filtered state/covariance. ``mask`` is a
    bool array marking *missing* observations (skipped in the update step).
    """
    n = len(y)
    p = x0.size
    x_pred = np.empty((n, p))
    P_pred = np.empty((n, p, p))
    x_post = np.empty((n, p))
    P_post = np.empty((n, p, p))
    x = x0.copy()
    P = P0.copy()
    I = np.eye(p)
    Rmat = np.array([[R]])
    for k in range(n):
        x = F[k] @ x
        P = F[k] @ P @ F[k].T + Q[k]
        x_pred[k] = x
        P_pred[k] = P
        if mask is None or not mask[k]:
            S = H[k] @ P @ H[k].T + Rmat
            K = P @ H[k].T @ np.linalg.inv(S)
            innov = y[k] - H[k] @ x
            x = x + K @ innov
            P = (I - K @ H[k]) @ P
        x_post[k] = x
        P_post[k] = P
    return x_pred, P_pred, x_post, P_post


def _rts_smoother(x_pred, P_pred, x_post, P_post, F):
    """Backward Rauch–Tung–Striebel smoother."""
    n = x_post.shape[0]
    x_sm = np.empty_like(x_post)
    P_sm = np.empty_like(P_post)
    x_sm[-1] = x_post[-1]
    P_sm[-1] = P_post[-1]
    for k in range(n - 2, -1, -1):
        C = P_post[k] @ F[k + 1].T @ np.linalg.inv(P_pred[k + 1])
        x_sm[k] = x_post[k] + C @ (x_sm[k + 1] - x_pred[k + 1])
        P_sm[k] = P_post[k] + C @ (P_sm[k + 1] - P_pred[k + 1]) @ C.T
    return x_sm, P_sm


# --- Output containers --------------------------------------------------------


@dataclass
class JointFit:
    """Smoothed joint estimate of tide + surge + trend."""

    times: np.ndarray  # hours since t0
    observed: np.ndarray
    trend: np.ndarray  # smoothed mean sea level incl. linear trend (m)
    tide: np.ndarray  # smoothed harmonic tide (m)
    surge: np.ndarray  # smoothed non-tidal surge (m)
    model: np.ndarray  # trend + tide + surge (m)
    trend_mm_yr: float  # secular trend (mm/yr)
    trend_mm_yr_se: float  # its standard error (mm/yr)
    coefficients: dict[str, dict]  # name -> {a, b, se_a, se_b, amp, phase}
    phi: float  # surge AR(1) coefficient used in the process model
    constituents: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"joint fit: n={self.observed.size} constituents={len(self.coefficients)}",
            f"secular trend: {self.trend_mm_yr:+.2f} ± {self.trend_mm_yr_se:.2f} mm/yr",
            f"surge AR(1) φ={self.phi:.3f}",
            "constituents:",
        ]
        for name, c in self.coefficients.items():
            lines.append(
                f"  {name:<5} A={c['amp']:.4f} m  φ={math.degrees(c['phase']):7.2f}°"
            )
        return "\n".join(lines)


# --- JointModel ---------------------------------------------------------------


class JointModel:
    """Joint state-space estimate of tide harmonics + surge + secular trend."""

    def __init__(
        self,
        constituents: Sequence[CON.Constituent],
        speeds_rad_per_h: np.ndarray,
        phi: float,
        q_xi: float,
        fit: JointFit,
    ):
        self._constituents = list(constituents)
        self._omega = np.asarray(speeds_rad_per_h, dtype=float).ravel()
        self._phi = float(phi)
        self._q_xi = float(q_xi)
        self._fit = fit

    # -- construction --------------------------------------------------------

    @classmethod
    def fit(
        cls,
        times: Sequence[datetime],
        heights,
        auto_select: bool = True,
        candidates: Sequence[CON.Constituent] | None = None,
        alpha: float = 0.05,
        phi: float | None = None,
        q_xi: float | None = None,
        q_level: float = 0.0,
        r: float = 1e-6,
        station: str | None = None,
    ) -> JointModel:
        """Estimate tide + surge + trend jointly from gauge observations.

        Constituent selection reuses DCDM (which *chooses* constituents); the
        coefficients, the secular trend, and the surge are then estimated
        together in the smoother rather than by decompose-then-fit.

        The level follows a *local linear trend*: ``L_k = L_{k-1} + β·Δt + η_k``
        with a small random-walk increment ``η_k ~ N(0, q_level)``. This lets the
        level absorb slow, non-linear non-tidal drift (ENSO, seasonal setup) so
        the slope ``β`` stays a clean *secular* trend rather than fitting the
        low-frequency wander of the (long-memory) surge AR(1).

        :param phi: surge AR(1) coefficient; if ``None``, fit from the residual
            of a quick harmonic warm-start (the AR(1) tracker becomes the process
            model).
        :param q_xi: surge innovation variance; defaults to the AR(1) fit's
            ``sigma_eps**2`` (or a small floor when no surge is present).
        :param q_level: random-walk variance of the mean level (m²/hr); larger
            values let the level follow more slow non-tidal drift.
        :param r: observation noise variance (metres²).
        """
        times = list(times)
        y = np.asarray(heights, dtype=float).ravel()
        if len(times) != y.size:
            raise ValueError(f"{len(times)} times but {y.size} heights")

        if candidates is None:
            candidates = list(CON.CATALOG) if auto_select else CON.principal()
        consts: list[CON.Constituent]
        if auto_select:
            from tideglass.marea.model import TideModel
            from tideglass.marea.selection import Candidate, select

            speeds = rad_per_hour([CON.speed(c) for c in candidates])
            res = select(
                _hours_since_j2000(times) - _hours_since_j2000(times)[0],
                y,
                [Candidate(c.name, float(w)) for c, w in zip(candidates, speeds)],
                alpha=alpha,
            )
            by_name = {c.name: c for c in candidates}
            consts = [by_name[n] for n in res.selected]
        else:
            consts = list(candidates)
        if not consts:
            raise ValueError("no constituents selected/fitted")

        omega = rad_per_hour([CON.speed(c) for c in consts])
        m = len(consts)
        p = 3 + 2 * m
        t_hours = _hours_since_j2000(times)
        t0 = t_hours[0]
        dt = np.diff(t_hours, prepend=t_hours[0])
        dt[0] = 0.0

        # Warm start: quick harmonic solve for initial coefficients + residual,
        # then AR(1) on the residual -> surge process model.
        from tideglass.marea.model import TideModel, _basis_matrix  # noqa: F401

        A = _basis_matrix(consts, times)
        coef0, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef0
        # Detrend the residual before fitting the AR(1): otherwise the secular
        # trend would leak into the surge process model (inflating q_xi) and the
        # smoother would prefer to put the ramp in surge rather than in β.
        tt = t_hours - t_hours.mean()
        slope_res, intercept = np.polyfit(tt, resid, 1)
        resid_detr = resid - (slope_res * tt + intercept)
        if phi is None:
            try:
                ar = fit_ar1(resid_detr)
                phi_eff = float(ar.phi)
                q_xi_eff = float(ar.sigma_eps) ** 2
            except ValueError:
                phi_eff, q_xi_eff = 0.95, float(np.var(resid_detr))
        else:
            phi_eff = float(phi)
            q_xi_eff = float(q_xi) if q_xi is not None else float(np.var(resid_detr))
        if q_xi is not None:
            q_xi_eff = float(q_xi)
        if q_xi_eff <= 0.0:
            q_xi_eff = 1e-6

        # Initial state: [L0, β, ξ0, a1, b1, ...]
        x0 = np.zeros(p)
        x0[0] = float(y.mean())
        x0[3::2] = coef0[1::2]  # a_i (cos)
        x0[4::2] = coef0[2::2]  # b_i (sin)

        P0 = np.eye(p)
        prior_beta = (_TREND_PRIOR_MM_YR / _MM_PER_M) / _HOURS_PER_YEAR  # m/hr
        P0[0, 0] = max(float(np.var(y)), 1e-2)
        P0[1, 1] = prior_beta**2  # tight prior on the trend slope
        P0[2, 2] = max(float(np.var(y)), 1e-2)
        for j in range(m):
            P0[3 + 2 * j, 3 + 2 * j] = max(x0[3 + 2 * j] ** 2, 0.1)
            P0[4 + 2 * j, 4 + 2 * j] = max(x0[4 + 2 * j] ** 2, 0.1)

        # Build per-step F, H, Q.
        Fs, Hs, Qs = [], [], []
        for k in range(y.size):
            F = np.eye(p)
            F[0, 1] = dt[k]
            F[2, 2] = phi_eff
            Fs.append(F)

            H = np.zeros((1, p))
            H[0, 0] = 1.0  # mean level / trend
            H[0, 2] = 1.0  # surge
            tk = t_hours[k] - t0
            for j in range(m):
                H[0, 3 + 2 * j] = math.cos(omega[j] * tk)
                H[0, 4 + 2 * j] = math.sin(omega[j] * tk)
            Hs.append(H)

            Q = np.zeros((p, p))
            Q[0, 0] = q_level
            Q[2, 2] = q_xi_eff
            Qs.append(Q)

        x_pred, P_pred, x_post, P_post = _kalman_filter(
            y, Fs, Hs, Qs, r, x0, P0
        )
        x_sm, P_sm = _rts_smoother(x_pred, P_pred, x_post, P_post, Fs)

        # Extract smoothed components.
        trend = x_sm[:, 0]
        surge = x_sm[:, 2]
        a = x_sm[:, 3::2]  # (n, m)
        b = x_sm[:, 4::2]  # (n, m)
        tide = np.zeros(y.size)
        for j in range(m):
            tk = t_hours - t0
            tide += a[-1, j] * np.cos(omega[j] * tk) + b[-1, j] * np.sin(omega[j] * tk)

        beta = float(x_sm[-1, 1])
        beta_se = float(math.sqrt(max(P_sm[-1, 1, 1], 0.0)))
        trend_mm_yr = beta * _HOURS_PER_YEAR * _MM_PER_M
        trend_mm_yr_se = beta_se * _HOURS_PER_YEAR * _MM_PER_M

        coefficients: dict[str, dict] = {}
        for j, c in enumerate(consts):
            av, bv = float(a[-1, j]), float(b[-1, j])
            se_a = float(math.sqrt(max(P_sm[-1, 3 + 2 * j, 3 + 2 * j], 0.0)))
            se_b = float(math.sqrt(max(P_sm[-1, 4 + 2 * j, 4 + 2 * j], 0.0)))
            coefficients[c.name] = {
                "a": av, "b": bv, "se_a": se_a, "se_b": se_b,
                "amp": math.hypot(av, bv),
                "phase": math.atan2(bv, av),
            }

        fit = JointFit(
            times=np.array(t_hours),
            observed=y,
            trend=trend,
            tide=tide,
            surge=surge,
            model=trend + tide + surge,
            trend_mm_yr=trend_mm_yr,
            trend_mm_yr_se=trend_mm_yr_se,
            coefficients=coefficients,
            phi=phi_eff,
            constituents=list(consts),
            meta={"n_obs": y.size, "rmse": math.sqrt(float(np.mean((y - (trend + tide + surge)) ** 2)))},
        )
        return cls(consts, omega, phi_eff, q_xi_eff, fit)

    # -- use ---------------------------------------------------------------

    @property
    def fit_result(self) -> JointFit:
        return self._fit

    def predict(self, times: Sequence[datetime], z: float = 1.96):
        """Forecast the tide curve (surge mean-reverts) with growing bands.

        Returns a :class:`~tideglass.marea.model.Prediction`-like object with
        ``mean`` (= trend + tide; surge expectation decays toward 0), ``lower``,
        ``upper`` (95% interval from the propagated state covariance), and ``se``.
        """
        from tideglass.marea.model import Prediction

        times = list(times)
        t_hours = _hours_since_j2000(times)
        t0 = self._fit.times[0]
        prev = self._fit.times[-1] - t0
        # Reconstruct the final smoothed state vector from the stored fit.
        p = 3 + 2 * len(self._constituents)
        x_vec = np.zeros(p)
        x_vec[0] = float(self._fit.trend[-1])
        x_vec[1] = self._fit.trend_mm_yr / (_HOURS_PER_YEAR * _MM_PER_M)
        x_vec[2] = float(self._fit.surge[-1])
        for j, c in enumerate(self._constituents):
            x_vec[3 + 2 * j] = self._fit.coefficients[c.name]["a"]
            x_vec[4 + 2 * j] = self._fit.coefficients[c.name]["b"]
        P = np.eye(p) * 1e-8

        mean = np.empty(len(times))
        se = np.empty(len(times))
        m = len(self._constituents)
        for k, th in enumerate(t_hours):
            dt = th - t0 - prev
            F = np.eye(p)
            F[0, 1] = dt
            F[2, 2] = self._phi
            Q = np.zeros((p, p))
            Q[1, 1] = 1e-12
            Q[2, 2] = self._q_xi
            x_vec = F @ x_vec
            P = F @ P @ F.T + Q
            H = np.zeros((1, p))
            H[0, 0] = 1.0
            H[0, 2] = 1.0
            tk = th - t0
            for j in range(m):
                H[0, 3 + 2 * j] = math.cos(self._omega[j] * tk)
                H[0, 4 + 2 * j] = math.sin(self._omega[j] * tk)
            mu = float((H @ x_vec)[0])
            var = float((H @ P @ H.T + 1e-6)[0, 0])
            mean[k] = mu
            se[k] = math.sqrt(max(var, 0.0))
            prev = th - t0
        half = z * np.sqrt(se**2 + 1e-6)
        return Prediction(mean=mean, lower=mean - half, upper=mean + half, se=se)
