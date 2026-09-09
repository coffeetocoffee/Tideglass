"""Live assimilation — the v0.6 operational nowcast.

Rather than re-fitting the whole record whenever new gauge observations arrive,
a :class:`NowcastEngine` keeps the harmonic state vector

    x = [mean, a_1, b_1, ..., a_m, b_m]

and applies a **recursive Kalman / recursive-least-squares** update as each new
observation streams in. The state is (nearly) static, so the transition is the
identity with a small *random-walk* process noise ``q`` that lets slow
amplitude drift (seasonal nodal modulation, equipment bias) be absorbed between
refits without discarding history.

The per-observation innovation (residual) drives a lightweight **AR(1) surge
tracker** carried over from the v0.3 ``JointModel`` skeleton: predictions ahead
of the last observation are nudged by the *decayed* last surge level
``ξ·φ^{Δt}``, and the surge uncertainty grows toward its marginal value with
lead time. This is what turns the engine from a batch refit into a *nowcast*.

Everything here is linear-Gaussian and exact (no EKF). The engine is
cheap enough to update on every 6-minute NOAA sample.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from tideglass.marea.model import Prediction, TideModel, _basis_matrix


_HOURS_PER_YEAR = 24.0 * 365.25
_MAX_INNOV = 2000  # rolling history kept for surge fitting / diagnostics


def _as_hours(times: Sequence[datetime]) -> np.ndarray:
    from tideglass.marea.astronomy import julian_day

    return np.array([(julian_day(t) - 2451545.0) * 24.0 for t in times], dtype=float)


@dataclass
class UpdateLog:
    """What a call to :meth:`NowcastEngine.update` produced."""

    n_obs: int
    start: datetime
    end: datetime
    mean_innovation: float  # mean residual (m) over the batch
    rms_innovation: float  # RMS residual (m)
    last_surge: float  # smoothed surge level after the last observation
    n_refit: int  # number of constituent re-selections performed (0 normally)


@dataclass
class NowcastState:
    """Serializable engine state (round-trips to ``<store>/<station>.nowcast.json``)."""

    coef: np.ndarray
    covariance: np.ndarray
    last_time: str  # ISO of the most recent assimilated observation
    surge_level: float
    phi: float
    q_mean: float
    q_coef: float
    r: float
    base_sha256: str  # digest of the model artifact this engine was initialised from
    n_updates: int
    innovations: list[float] = field(default_factory=list)
    history: list[list] = field(default_factory=list)  # capped [[iso, height], ...]
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "coef": self.coef.tolist(),
            "covariance": self.covariance.tolist(),
            "last_time": self.last_time,
            "surge_level": self.surge_level,
            "phi": self.phi,
            "q_mean": self.q_mean,
            "q_coef": self.q_coef,
            "r": self.r,
            "base_sha256": self.base_sha256,
            "n_updates": self.n_updates,
            "innovations": self.innovations,
            "history": self.history,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> NowcastState:
        return cls(
            coef=np.asarray(d["coef"], dtype=float),
            covariance=np.asarray(d["covariance"], dtype=float),
            last_time=d["last_time"],
            surge_level=float(d["surge_level"]),
            phi=float(d["phi"]),
            q_mean=float(d["q_mean"]),
            q_coef=float(d["q_coef"]),
            r=float(d["r"]),
            base_sha256=d["base_sha256"],
            n_updates=int(d["n_updates"]),
            innovations=list(d.get("innovations", [])),
            history=list(d.get("history", [])),
            meta=dict(d.get("meta", {})),
        )


class NowcastEngine:
    """Recursive Kalman update of a :class:`TideModel` as data streams in."""

    def __init__(
        self,
        model: TideModel,
        q_mean: float = 1e-8,
    q_coef: float = 1e-10,
    r: float | None = None,
    phi: float = 0.9,
    base_sha256: str | None = None,
    max_history: int = 5000,
):
        """Initialise from a fitted (or loaded) :class:`TideModel`.

        :param q_mean: random-walk variance of the mean level per *hour* (m²/hr).
            Tiny by default — the level is essentially static between refits but
            can absorb slow bias drift.
        :param q_coef: random-walk variance of each harmonic coefficient per
            *hour* (m²/hr). Smaller than ``q_mean``; permits very slow amplitude
            adaptation without forgetting the historical fit.
        :param r: observation noise variance (m²) for the Kalman update.
            Defaults to the base model's own residual variance σ² so the
            nowcast bands stay calibrated to what the model actually explains
            (1e-4 floor for loaded harmonics, which carry no covariance).
        :param phi: AR(1) coefficient for the surge tracker (0 disables surge).
        :param base_sha256: digest of the originating model artifact, used to
            detect when the base model changed (in which case the engine should
            be re-seated). Defaults to the model's ``data_sha256`` if present.
        """
        self._model = model
        self.q_mean = float(q_mean)
        self.q_coef = float(q_coef)
        if r is None:
            sig2 = float(getattr(model, "_sigma2", 0.0) or 0.0)
            r = sig2 if sig2 > 0.0 else 1e-4
        self.r = float(r)
        self.phi = float(phi)
        self._constituents = list(model._constituents)
        self._coef = np.asarray(model._coef, dtype=float).copy()
        p = self._coef.size
        cov = np.asarray(model._covariance, dtype=float)
        if cov.shape != (p, p) or not np.any(cov):
            # Loaded (NOAA-style) models carry no covariance; seed a plausible
            # prior from the amplitudes so the filter has something to shrink.
            cov = np.eye(p)
            cov[0, 0] = max(float(np.var(model._coef)), 1e-2)
            for j in range(1, p):
                cov[j, j] = max(model._coef[j] ** 2, 0.01)
        self._P = cov
        self._surge = 0.0
        self._last_time: datetime | None = None
        self._n_updates = 0
        self._innov: list[float] = []
        self._max_history = int(max_history)
        self._hist_t: list[datetime] = []
        self._hist_y: list[float] = []
        if base_sha256 is not None:
            self._base_sha = base_sha256
        else:
            self._base_sha = str(model.meta.get("data_sha256", "unknown"))
        self._meta = dict(model.meta)

    # -- core recursion ------------------------------------------------------

    def update(self, times: Sequence[datetime], heights) -> UpdateLog:
        """Assimilate a batch of observations (chronological order expected)."""
        times = list(times)
        y = np.asarray(heights, dtype=float).ravel()
        if len(times) != y.size:
            raise ValueError(f"{len(times)} times but {y.size} heights")
        p = self._coef.size
        innovs: list[float] = []
        last_surge = self._surge
        for t, yk in zip(times, y):
            H = _basis_matrix(self._constituents, [t])[0]  # (p,)
            # Predict: state is static, add random-walk process noise.
            dt = 0.0 if self._last_time is None else (
                (t - self._last_time).total_seconds() / 3600.0)
            Q = np.zeros((p, p))
            Q[0, 0] = self.q_mean * max(dt, 0.0)
            Q[1:, 1:] = np.eye(p - 1) * (self.q_coef * max(dt, 0.0))
            P_pred = self._P + Q
            S = float(H @ P_pred @ H + self.r)
            K = P_pred @ H / S
            innov = float(yk - H @ self._coef)
            self._coef = self._coef + K * innov
            self._P = (np.eye(p) - np.outer(K, H)) @ P_pred
            # Surge tracker: the innovation is tide-prediction error; pull the
            # smoothed surge level toward it (EMA with the AR(1) decay rate).
            if self.phi > 0:
                last_surge = self.phi * last_surge + (1.0 - self.phi) * innov
            else:
                last_surge = 0.0
            innovs.append(innov)
            self._last_time = t
            self._n_updates += 1
            self._hist_t.append(t)
            self._hist_y.append(float(yk))
        self._surge = last_surge
        # Trim the rolling observation history (capped for bounded artifacts).
        if len(self._hist_t) > self._max_history:
            self._hist_t = self._hist_t[-self._max_history:]
            self._hist_y = self._hist_y[-self._max_history:]
        self._innov.extend(innovs)
        if len(self._innov) > _MAX_INNOV:
            self._innov = self._innov[-_MAX_INNOV:]
        resid = np.asarray(innovs, dtype=float)
        return UpdateLog(
            n_obs=len(times),
            start=times[0],
            end=times[-1],
            mean_innovation=float(resid.mean()) if resid.size else 0.0,
            rms_innovation=float(math.sqrt(float(np.mean(resid ** 2)))) if resid.size else 0.0,
            last_surge=last_surge,
            n_refit=0,
        )

    def predict(self, times: Sequence[datetime], z: float = 1.96) -> Prediction:
        """Nowcast tide (+ decayed surge) with growing bands.

        The mean tide comes from the assimilated coefficient state; a surge
        contribution ``ξ·φ^{Δt}`` is added with its own (growing) uncertainty,
        and the prediction interval combines parameter covariance, accumulated
        process noise, observation noise, and surge uncertainty.
        """
        times = list(times)
        A = _basis_matrix(self._constituents, times)
        mean = A @ self._coef
        var_mean = np.maximum(
            np.einsum("ij,jk,ik->i", A, self._P, A), 0.0)
        if self._last_time is not None and self.phi > 0:
            dt_h = np.array([
                (t - self._last_time).total_seconds() / 3600.0 for t in times],
                dtype=float)
            surge_term = self._surge * self.phi ** np.maximum(dt_h, 0.0)
            mean = mean + surge_term
            # Surge marginal std and its decay toward marginal variance.
            surge_var = self.r + np.maximum(var_mean.mean(), 1e-6)
            surge_var_series = surge_var * (1.0 - self.phi ** (2.0 * np.maximum(dt_h, 0.0)))
            var_total = var_mean + self.r + surge_var_series
        else:
            var_total = var_mean + self.r
        se = np.sqrt(var_mean)
        half = z * np.sqrt(var_total)
        return Prediction(mean=mean, lower=mean - half, upper=mean + half, se=se)

    # -- state management ----------------------------------------------------

    @property
    def last_time(self) -> datetime | None:
        """Most recent assimilated observation time (None before first update)."""
        return self._last_time

    @property
    def baseline_rmse(self) -> float | None:
        """The base model's own RMSE from its fit metadata (None if unknown)."""
        rmse = self._meta.get("rmse")
        return float(rmse) if rmse is not None else None

    def state(self) -> NowcastState:
        last = self._last_time.isoformat() if self._last_time else ""
        return NowcastState(
            coef=self._coef.copy(),
            covariance=self._P.copy(),
            last_time=last,
            surge_level=self._surge,
            phi=self.phi,
            q_mean=self.q_mean,
            q_coef=self.q_coef,
            r=self.r,
            base_sha256=self._base_sha,
            n_updates=self._n_updates,
            innovations=list(self._innov),
            history=[[t.isoformat(), float(h)] for t, h in zip(self._hist_t, self._hist_y)],
            meta=dict(self._meta),
        )

    @classmethod
    def from_state(cls, model: TideModel, state: NowcastState) -> NowcastEngine:
        eng = cls(
            model, q_mean=state.q_mean, q_coef=state.q_coef,
            r=state.r, phi=state.phi, base_sha256=state.base_sha256,
        )
        eng._coef = np.asarray(state.coef, dtype=float).copy()
        eng._P = np.asarray(state.covariance, dtype=float).copy()
        eng._surge = float(state.surge_level)
        eng._n_updates = int(state.n_updates)
        eng._innov = list(state.innovations)
        eng._meta = dict(state.meta)
        eng._hist_t = [datetime.fromisoformat(row[0]) for row in state.history]
        eng._hist_y = [float(row[1]) for row in state.history]
        if state.last_time:
            eng._last_time = datetime.fromisoformat(state.last_time)
        return eng

    def history_data(self) -> tuple[list[datetime], list[float]]:
        """Return the accumulated observation history (capped) for refitting."""
        return list(self._hist_t), list(self._hist_y)

    def auto_refit(
        self, alpha: float = 0.05, source: str | None = None
    ) -> tuple[TideModel, NowcastEngine]:
        """Re-fit a fresh :class:`TideModel` from the accumulated history.

        Returns the new model and a freshly-seated engine built on it. This is
        the auto-refit triggered by :class:`~tideglass.marea.drift.HealthMonitor`
        when the assimilating model has drifted.

        The fresh engine re-derives its observation noise from the new fit's
        residual variance (rather than inheriting this engine's ``r``), so its
        bands stay calibrated to what the new model actually explains.
        """
        times, heights = self.history_data()
        if not times:
            raise ValueError("no observation history to refit from")
        model = _fit_from_history(
            times, heights,
            station=self._model.station, alpha=alpha, source=source,
        )
        fresh = NowcastEngine(
            model, q_mean=self.q_mean, q_coef=self.q_coef,
            phi=self.phi,
            base_sha256=model.meta.get("data_sha256", "unknown"),
        )
        # Carry the observation history into the new engine so the next drift
        # check still has a window to measure against.
        fresh._hist_t = list(self._hist_t)
        fresh._hist_y = list(self._hist_y)
        fresh._last_time = self._last_time
        return model, fresh

    def to_tide_model(self) -> TideModel:
        """Snapshot the assimilated state back into a :class:`TideModel`."""
        Fit = type(self._model._fits[0])
        fits = []
        for j, c in enumerate(self._constituents):
            a = float(self._coef[1 + 2 * j])
            b = float(self._coef[2 + 2 * j])
            amp = math.hypot(a, b)
            kappa = math.degrees(math.atan2(b, a)) % 360.0
            se_amp = math.sqrt(max(
                self._P[1 + 2 * j, 1 + 2 * j] + self._P[2 + 2 * j, 2 + 2 * j],
                0.0)) / 2.0
            fits.append(Fit(c.name, amp, kappa, se_amp))
        meta = dict(self._meta)
        meta.update({
            "n_obs": int(self._n_updates),
            "n_assimilated": int(self._n_updates),
            "last_update": (self._last_time.isoformat()
                            if self._last_time else None),
            "assimilated_sha256": self._base_sha,
        })
        return TideModel(
            list(self._constituents),
            self._coef.copy(),
            self._P.copy(),
            self.r,
            fits,
            station=self._model.station,
            source="nowcast",
            meta=meta,
        )


def _fit_from_history(
    times: Sequence[datetime], heights,
    station: str | None, alpha: float, source: str | None,
) -> TideModel:
    """Re-fit a fresh :class:`TideModel` from accumulated history (auto-refit)."""
    return TideModel.fit(
        times, heights, alpha=alpha, station=station, source=source,
    )


def save_state(engine: NowcastEngine, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(engine.state().to_dict(), fh, indent=2)


def load_state(model: TideModel, path: str) -> NowcastEngine:
    with open(path) as fh:
        st = NowcastState.from_dict(json.load(fh))
    return NowcastEngine.from_state(model, st)
