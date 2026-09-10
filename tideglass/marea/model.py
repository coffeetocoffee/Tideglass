"""TideModel — the public face of Marea Core.

Orchestrates the engine: ``fit`` runs DCDM selection + the exact SVD solve
on a nodal-modulated basis, ``predict`` evaluates ``height ± σ`` anywhere in
time, and ``load_harmonic`` reuses published (e.g. NOAA) constants.

Prediction model (t_tide-style): with ``θᵢ(t) = Vᵢ(t) + uᵢ(t)`` the nodal-
corrected equilibrium argument of constituent ``i``::

    h(t) = H0 + Σᵢ fᵢ(t)·Aᵢ·cos(θᵢ(t) − κᵢ)

``fit`` solves for ``(Aᵢ, κᵢ)`` on the ``f·cos θ / f·sin θ`` basis, so the
18.6-year nodal modulation is honored in both fitting and prediction rather
than absorbed as a constant.

Uncertainty: ``predict`` returns the 95% *prediction* interval
``mean ± 1.96·√(gCgᵀ + σ²)`` — parameter uncertainty from the solution
covariance plus residual variance — so held-out observations land inside
~95% of the time. ``Prediction.se`` carries the standard error of the mean
tide alone (``√(gCgᵀ)``).

Conventions: datetimes in (naive = UTC), heights in metres, phases in
degrees on the NOAA kappa convention (``cos(V + u − κ)``).
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from tideglass.marea import constituents as CON
from tideglass.marea.astronomy import (
    D2R,
    doodson_args,
    julian_day,
    nodal_factor,
)
from tideglass.marea.selection import Candidate, select
from tideglass.marea.solver import rad_per_hour, solve_matrix

_J2000 = 2451545.0
_Z95 = 1.96


@dataclass(frozen=True)
class Fit:
    """One fitted harmonic constant: name, amplitude (m), phase (°), σ (m)."""

    name: str
    amplitude: float
    phase_deg: float
    sigma_amp: float


@dataclass(frozen=True)
class Prediction:
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    se: np.ndarray  # std error of the mean tide


def _hours_since_j2000(times: Sequence[datetime]) -> np.ndarray:
    return np.array([(julian_day(t) - _J2000) * 24.0 for t in times], dtype=float)


def _design_row(c: CON.Constituent, t: datetime) -> tuple:
    """Modulated ``(f·cos θ, f·sin θ)`` pair for one constituent and time."""
    args = doodson_args(t)
    V = (sum(n * a for n, a in zip(c.doodson, args)) + c.phase0) % 360.0
    f, u = nodal_factor(c, t)
    theta = (V + u) * D2R
    return f * math.cos(theta), f * math.sin(theta)


def _basis_matrix(constituents: Sequence[CON.Constituent],
                   times: Sequence[datetime],
                   kernel: str = "numpy") -> np.ndarray:
    """Tidal design matrix ``[1, cos, sin, …]`` for the given constituents/times.

    ``kernel`` selects the numeric backend (see :mod:`tideglass.marea.kernel`):
    ``"numpy"`` (default, always available) or ``"auto"``/``"numba"`` (fused JIT
    path when Numba is installed). All backends are numerically equivalent.
    """
    from tideglass.marea.kernel import basis_matrix as _kernel_basis

    return _kernel_basis(constituents, times, backend=kernel)


class TideModel:
    """Fitted (or loaded) tide model. Build via :meth:`fit`/:meth:`load_harmonic`."""

    def __init__(
        self,
        constituents: Sequence[CON.Constituent],
        coef: np.ndarray,
        covariance: np.ndarray,
        sigma2: float,
        fits: list[Fit],
        station: str | None = None,
        source: str = "fit",
        meta: dict | None = None,
    ):
        self._constituents = list(constituents)
        self._coef = np.asarray(coef, dtype=float)
        self._covariance = np.asarray(covariance, dtype=float)
        self._sigma2 = float(sigma2)
        self._fits = list(fits)
        self.station = station
        self.source = source
        self.meta = dict(meta or {})
        self._residual = None  # optional learned bias layer (v1.1)

    # -- construction --------------------------------------------------------

    @classmethod
    def fit(
        cls,
        times: Sequence[datetime],
        heights,
        auto_select: bool = True,
        candidates: Sequence[CON.Constituent] | None = None,
        alpha: float = 0.05,
        station: str | None = None,
        source: str | None = None,
        kernel: str = "numpy",
    ) -> TideModel:
        """Fit a model from gauge observations.

        :param auto_select: run DCDM selection over ``candidates`` (default:
            the full catalog); when False, fit ``candidates`` (default: the
            principal 8) directly.
        """
        times = list(times)
        y = np.asarray(heights, dtype=float).ravel()
        if len(times) != y.size:
            raise ValueError(f"{len(times)} times but {y.size} heights")
        if candidates is None:
            candidates = list(CON.CATALOG) if auto_select else CON.principal()
        if auto_select:
            speeds = rad_per_hour([CON.speed(c) for c in candidates])
            res = select(
                _hours_since_j2000(times) - _hours_since_j2000(times)[0],
                y,
                [Candidate(c.name, float(w)) for c, w in zip(candidates, speeds)],
                alpha=alpha,
            )
            by_name = {c.name: c for c in candidates}
            selected = [by_name[n] for n in res.selected]
        else:
            selected = list(candidates)
        if not selected:
            raise ValueError("no constituents selected/fitted")
        A = _basis_matrix(selected, times, kernel=kernel)
        sol = solve_matrix(A, y, [c.name for c in selected])
        by_term = {t.name: t for t in sol.terms}
        fits = [
            Fit(c.name, by_term[c.name].amplitude,
                math.degrees(by_term[c.name].phase) % 360.0,
                by_term[c.name].sigma_amp)
            for c in selected
        ]
        resid = y - A @ sol.coef
        from tideglass.marea.provenance import provenance

        meta = provenance(
            times, heights, source=source, station=station,
            extra={
                "n_obs": len(times),
                "rmse": math.sqrt(float(resid @ resid) / len(times)),
            },
        )
        return cls(
            selected, sol.coef, sol.covariance, sol.sigma2, fits,
            station=station, source="fit", meta=meta,
        )

    @classmethod
    def load_harmonic(cls, source, station: str | None = None) -> TideModel:
        """Load published harmonic constants (NOAA-style JSON).

        Accepts a mapping, a JSON string, or a path to a JSON file::

            {"station": "X", "mean": 0.7,
             "constituents": [{"name": "M2", "amplitude": 1.0, "phase": 30.0}]}

        Loaded models carry no covariance, so prediction bands collapse to
        the mean curve.
        """
        if isinstance(source, (str, os.PathLike)) and os.path.exists(source):
            with open(source) as fh:
                data = json.load(fh)
        elif isinstance(source, str):
            data = json.loads(source)
        else:
            data = dict(source)
        entries = data.get("constituents", [])
        consts = [CON.get(e["name"]) for e in entries]
        coef = np.zeros(1 + 2 * len(consts))
        coef[0] = float(data.get("mean", 0.0))
        fits = []
        for j, e in enumerate(entries):
            kappa = math.radians(float(e["phase"]))
            amp = float(e["amplitude"])
            coef[1 + 2 * j] = amp * math.cos(kappa)
            coef[2 + 2 * j] = amp * math.sin(kappa)
            fits.append(Fit(e["name"], amp, float(e["phase"]) % 360.0, 0.0))
        p = coef.size
        restored_meta = dict(data.get("meta", {}))
        restored_meta.setdefault("source", "harmonic")
        if "station" not in restored_meta and data.get("station"):
            restored_meta["station"] = data.get("station")
        model = cls(
            consts, coef, np.zeros((p, p)), 0.0, fits,
            station=station or data.get("station"), source="harmonic",
            meta=restored_meta or None,
        )
        if restored_meta.get("residual_bias") is not None:
            from tideglass.marea.residual import ResidualModel

            model._residual = ResidualModel.from_dict(
                restored_meta["residual_bias"])
        return model

    # -- use ---------------------------------------------------------------

    def attach_residual(self, residual_model) -> None:
        """Attach a learned bias layer (v1.1) — applied by every ``predict``.

        The table is also pinned into ``meta["residual_bias"]`` so it survives
        the ``to_artifact()`` → ``load_harmonic()`` round-trip.
        """
        self._residual = residual_model
        self.meta["residual_bias"] = residual_model.to_dict()

    @property
    def residual(self):
        """The attached residual-bias layer, or ``None``."""
        return self._residual

    def predict(self, times: Sequence[datetime], z: float = _Z95,
                 kernel: str = "numpy") -> Prediction:
        """Predict ``height ±`` band at ``times`` (95% prediction interval).

        When a residual-bias layer is attached (:meth:`attach_residual`), the
        learned day-of-year correction is applied to the mean (and the band is
        conformally rescaled) before returning.
        """
        times = list(times)
        A = _basis_matrix(self._constituents, times, kernel=kernel)
        mean = A @ self._coef
        var_mean = np.maximum(np.einsum("ij,jk,ik->i", A, self._covariance, A), 0.0)
        se = np.sqrt(var_mean)
        half = z * np.sqrt(var_mean + self._sigma2)
        pred = Prediction(mean=mean, lower=mean - half, upper=mean + half, se=se)
        if self._residual is not None:
            pred = self._residual.apply(pred, times)
        return pred

    def constituents(self) -> list[Fit]:
        """Fitted harmonic constants (name, amplitude, phase°, σ)."""
        return list(self._fits)

    def to_artifact(self) -> dict:
        """JSON-serializable model artifact (round-trips via load_harmonic)."""
        return {
            "station": self.station,
            "units": "m",
            "mean": float(self._coef[0]),
            "constituents": [
                {"name": f.name, "amplitude": f.amplitude, "phase": f.phase_deg}
                for f in self._fits
            ],
            # The versioned provenance in meta is authoritative; fall back to
            # the coarse construction tag only when no provenance was pinned.
            "meta": {"source": self.source, **self.meta},
        }
