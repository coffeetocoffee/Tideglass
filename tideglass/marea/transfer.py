"""Response-function transfer from reference ports — v0.7.

A station with a *short* observation record cannot reliably auto-select its
constituents (DCDM needs many cycles), yet it often sits near a long-observed
*tide reference port*. Classic hydrography gets around this with an **admiralty
response function**: the complex ratio between a target port and its reference

    H_target / H_ref = g · exp(i·Δφ)

is stable across nearby coasts and varies smoothly in space. We estimate that
gain ``g`` and phase lag ``Δφ`` per constituent from a set of well-observed
neighbour stations (each a :class:`~tideglass.marea.model.TideModel`), weighted
by their proximity to the target (inverse-distance on great-circle distance),
and seed the target's harmonic constants from the reference *scaled* by the
transfer. The short record is then used only to **refine**, never to select —
which is exactly what makes transfer invaluable for sparse gauges.

Only the marine-independent harmonic math is touched; the result is a plain
:class:`~tideglass.marea.model.TideModel` for the target, so everything upstream
(prediction, nowcast, drift) works unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from tideglass.marea.model import Fit, TideModel, _basis_matrix


def _haversine_km(lon1, lat1, lon2, lat2) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (lon1, lat1, lon2, lat2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2.0 * math.asin(min(1.0, math.sqrt(a)))


@dataclass
class TransferCoefficients:
    """Per-constituent gain ``g`` and phase shift ``Δφ`` (deg) to apply to ref."""

    gain: dict[str, float]
    phase_shift_deg: dict[str, float]
    sources: list[str]


class ResponseTransfer:
    """Transfer a reference port's tides onto a target location."""

    def __init__(
        self,
        reference: TideModel,
        reference_coords: tuple[float, float],
        neighbors: Sequence[tuple[TideModel, tuple[float, float]]],
        target_coords: tuple[float, float],
        target_station: str | None = None,
    ):
        """Set up the transfer.

        :param reference: the primary well-observed station (a fitted/loaded
            :class:`TideModel`) whose constituent *set* defines the analysis.
        :param reference_coords: ``(lon, lat)`` of the reference port.
        :param neighbors: other well-observed stations ``(TideModel, (lon, lat))``
            used to estimate the spatial variation of the transfer. Should include
            or neighbour the target geographically; the reference itself may be
            repeated here if no others are available (degenerate: gain 1).
        :param target_coords: ``(lon, lat)`` of the (sparse) target station.
        :param target_station: name for the synthesized target model.
        """
        self._ref = reference
        self._ref_coords = reference_coords
        self._target_coords = target_coords
        self._target_station = target_station
        self._neighbors = list(neighbors)
        if not self._neighbors:
            raise ValueError("need at least one neighbour (the reference itself)")
        self._ref_fits = {f.name: f for f in reference.constituents()}

    # -- estimation --------------------------------------------------------

    def coefficients(self, power: float = 2.0) -> TransferCoefficients:
        """Estimate per-constituent gain + phase shift toward the target."""
        tlon, tlat = self._target_coords
        gains: dict[str, float] = {}
        shifts: dict[str, float] = {}
        used: list[str] = []
        for name in self._ref_fits:
            wsum = 0.0
            gw = 0.0
            pw = 0.0
            for model, (lon, lat) in self._neighbors:
                cand = {f.name: f for f in model.constituents()}
                if name not in cand:
                    continue
                d = _haversine_km(lon, lat, tlon, tlat)
                w = 1.0 / (d + 1e-3) ** power
                # Ratio of this neighbour's constant to the reference's.
                g = cand[name].amplitude / max(self._ref_fits[name].amplitude, 1e-9)
                dp = (cand[name].phase_deg - self._ref_fits[name].phase_deg)
                dp = (dp + 180.0) % 360.0 - 180.0
                gw += w * g
                pw += w * dp
                wsum += w
            if wsum <= 0:
                continue
            gains[name] = gw / wsum
            shifts[name] = pw / wsum
            used.append(name)
        return TransferCoefficients(gain=gains, phase_shift_deg=shifts, sources=used)

    def to_tide_model(self, coeffs: TransferCoefficients | None = None) -> TideModel:
        """Build a target :class:`TideModel` from transferred constants."""
        if coeffs is None:
            coeffs = self.coefficients()
        if not coeffs.gain:
            raise ValueError("no constituent transfer could be estimated")
        consts = []
        fits = []
        coef = np.zeros(1 + 2 * len(coeffs.gain))
        for j, name in enumerate(coeffs.gain):
            ref = self._ref_fits[name]
            amp = ref.amplitude * coeffs.gain[name]
            phase = (ref.phase_deg + coeffs.phase_shift_deg[name]) % 360.0
            kappa = math.radians(phase)
            coef[1 + 2 * j] = amp * math.cos(kappa)
            coef[2 + 2 * j] = amp * math.sin(kappa)
            fits.append(Fit(name, amp, phase, 0.0))
            consts.append(_const_for(name))
        meta = {
            "source": "transfer",
            "reference": getattr(self._ref, "station", None),
            "reference_coords": self._reference_coords,
            "target_coords": self._target_coords,
            "transfer_sources": coeffs.sources,
        }
        return TideModel(
            consts, coef, np.zeros((coef.size, coef.size)), 0.0, fits,
            station=self._target_station, source="transfer", meta=meta,
        )

    # -- refinement ---------------------------------------------------------

    def refine(
        self,
        times: Sequence[datetime],
        heights,
        coeffs: TransferCoefficients | None = None,
        z: float = 1.96,
    ) -> TideModel:
        """Refine the transferred constants on a (short) target record.

        The constituent *set* is fixed to the reference's (no DCDM selection),
        so even a handful of days suffices. Returns a :class:`TideModel` with a
        fitted covariance and the record's residual variance as the band scale.
        """
        model = self.to_tide_model(coeffs)
        times = list(times)
        y = np.asarray(heights, dtype=float).ravel()
        A = _basis_matrix(model._constituents, times)
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ sol
        n, p = A.shape
        sigma2 = float(resid @ resid) / max(n - p, 1)
        var = np.maximum(np.einsum("ij,jk,ik->i", A, np.linalg.inv(A.T @ A), A), 0.0)
        fits = []
        for j, c in enumerate(model._constituents):
            a, b = sol[1 + 2 * j], sol[2 + 2 * j]
            amp = math.hypot(a, b)
            ph = math.degrees(math.atan2(b, a)) % 360.0
            se = math.sqrt(max(var[0] * (a ** 2 + b ** 2), 0.0)) / max(n - p, 1) / 2.0
            fits.append(Fit(c.name, amp, ph, se))
        meta = dict(model.meta)
        meta.update({
            "n_obs": n, "rmse": math.sqrt(sigma2),
            "refined": True, "source": "transfer+refine",
        })
        return TideModel(
            list(model._constituents), sol, np.zeros_like(np.outer(sol, sol)),
            sigma2, fits, station=self._target_station,
            source="transfer", meta=meta,
        )

    @property
    def _reference_coords(self):
        return self._ref_coords


def _const_for(name):
    from tideglass.marea import constituents as CON

    return CON.get(name)
