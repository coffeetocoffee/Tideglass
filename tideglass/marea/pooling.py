"""Hierarchical partial pooling for Marea Core — the v0.8 network payoff.

A single cheap sensor with a short, noisy record cannot fit trustworthy
harmonic constants on its own. But stations in a region share the same ocean:
their (``a``, ``b``) coefficients for each constituent scatter around a
regional mean. Partial pooling exploits that with a two-level model, per
constituent and per quadrature component::

    z_ij | theta_ij ~ N(theta_ij, s_ij²)      (station i's own fit + SE)
    theta_ij | mu_j, tau_j² ~ N(mu_j, tau_j²) (regional prior)

with an empirical-Bayes (DerSimonian–Laird) estimate of the between-station
spread ``tau_j²``. The pooled estimate is a precision-weighted compromise::

    z~_ij = w_ij · z_ij + (1 − w_ij) · mu_j,   w_ij = tau_j² / (tau_j² + s_ij²)

Long, clean records (``s² ≪ tau²``) keep ``w ≈ 1`` and are barely touched;
short, noisy ones (``s² ≫ tau²``) shrink hard toward the regional mean — they
*borrow strength* from their neighbours. A constituent the target never
selected is imputed from the prior outright (``w = 0``).

The datum-dependent mean level ``H0`` is never pooled (stations live on
different vertical datums); only the harmonic coefficients are shrunk. The
result is a plain :class:`~tideglass.marea.model.TideModel`, so prediction,
nowcast, and drift machinery work unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from tideglass.marea import constituents as CON
from tideglass.marea.model import Fit, TideModel

_EPS_KM = 1.0  # IDW softening so a station never divides by zero distance


def _haversine_km(lon1, lat1, lon2, lat2) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (lon1, lat1, lon2, lat2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2.0 * math.asin(min(1.0, math.sqrt(a)))


@dataclass(frozen=True)
class PooledConstituent:
    """One constituent after pooling: shrunk amplitude/phase + diagnostics."""

    name: str
    amplitude: float
    phase_deg: float
    sigma_amp: float
    shrinkage: float  # 0 = own data only, 1 = pure regional prior
    tau: float  # between-station std (m), mean over the (a, b) components


class HierarchicalPool:
    """Partial pooling of harmonic constants across a station network."""

    def __init__(
        self,
        models: dict[str, TideModel],
        coords: dict[str, tuple[float, float]] | None = None,
    ):
        """Build the pool from fitted member models.

        :param models: ``{station: TideModel}`` — the well-observed network.
        :param coords: optional ``{station: (lon, lat)}``. When the target's
            coordinates are known, the regional prior becomes the
            inverse-distance-weighted *neighbour* mean (leave-self-out);
            otherwise it is the inverse-variance-weighted network mean.
        """
        if len(models) < 2:
            raise ValueError(
                f"need at least 2 stations to pool, got {len(models)}")
        self._models = dict(models)
        self._coords = dict(coords or {})
        unknown = set(self._coords) - set(self._models)
        if unknown:
            raise ValueError(f"coords for unknown stations: {sorted(unknown)}")
        self._union = self._union_constituents()

    # -- inventory ---------------------------------------------------------

    @property
    def stations(self) -> list[str]:
        return sorted(self._models)

    @property
    def union(self) -> list[CON.Constituent]:
        """Union constituent basis, in canonical catalog order."""
        return list(self._union)

    def _union_constituents(self) -> list[CON.Constituent]:
        seen: set[str] = set()
        for m in self._models.values():
            seen.update(c.name for c in m._constituents)
        ordered = [c for c in CON.CATALOG if c.name in seen]
        extra = sorted(seen - {c.name for c in ordered})
        return ordered + [CON.get(n) for n in extra]

    def _station_ab(self, station: str, name: str):
        """``(a, b, se_a, se_b)`` of one constituent, or ``None`` if absent."""
        m = self._models[station]
        names = [c.name for c in m._constituents]
        if name not in names:
            return None
        j = names.index(name)
        C = m._covariance
        return (
            float(m._coef[1 + 2 * j]),
            float(m._coef[2 + 2 * j]),
            math.sqrt(max(float(C[1 + 2 * j, 1 + 2 * j]), 0.0)),
            math.sqrt(max(float(C[2 + 2 * j, 2 + 2 * j]), 0.0)),
        )

    # -- regional prior ----------------------------------------------------

    def _prior(self, name: str, target: str,
               target_coords: tuple[float, float] | None):
        """Regional prior ``(mu_a, mu_b, tau_a, tau_b)`` for one constituent.

        ``tau`` is the DerSimonian–Laird between-station spread over the
        members carrying the constituent. The mean is inverse-variance
        weighted, or — when ``target_coords`` and network coords are known —
        the inverse-distance-weighted neighbour mean (target excluded).
        """
        have = [(s, self._station_ab(s, name)) for s in self._models]
        have = [(s, ab) for s, ab in have if ab is not None]
        out = {}
        for k in (0, 1):  # a-component, b-component
            z = np.array([ab[k] for _, ab in have])
            s2 = np.array([max(ab[2 + k], 1e-12) ** 2 for _, ab in have])
            w_iv = 1.0 / s2
            mu_iv = float((w_iv @ z) / w_iv.sum())
            if len(have) > 1:
                q = float((w_iv * (z - mu_iv) ** 2).sum())
                den = float(w_iv.sum() - (w_iv**2).sum() / w_iv.sum())
                tau2 = max(0.0, (q - (len(have) - 1)) / den) if den > 0 else 0.0
            else:
                tau2 = 0.0
            if (target_coords is not None and self._coords
                    and len(have) > 1):
                num, den_w = 0.0, 0.0
                for s, ab in have:
                    if s == target or s not in self._coords:
                        continue
                    lon, lat = self._coords[s]
                    d = _haversine_km(target_coords[0], target_coords[1],
                                      lon, lat)
                    wgt = 1.0 / (d + _EPS_KM)
                    num += wgt * ab[k]
                    den_w += wgt
                mu = num / den_w if den_w > 0 else mu_iv
            else:
                mu = mu_iv
            out[k] = (mu, math.sqrt(tau2), len(have))
        return out  # {0: (mu_a, tau_a, n), 1: (mu_b, tau_b, n)}

    # -- pooling -----------------------------------------------------------

    def _pool_one(self, own_ab, prior_k: tuple[float, float, int],
                  is_sole: bool):
        """Shrink one ``(value, se)`` estimate toward its prior.

        Returns ``(pooled_value, pooled_var, shrinkage)``.
        """
        mu, tau, _n = prior_k
        tau2 = tau**2
        if own_ab is None:
            # No information: impute the prior (w = 0).
            return mu, tau2, 1.0
        z0, s0 = own_ab
        s02 = max(s0, 1e-12) ** 2
        if is_sole:
            # Sole contributor: the prior *is* this estimate; keep it whole.
            return z0, s02, 0.0
        w = tau2 / (tau2 + s02)
        return w * z0 + (1.0 - w) * mu, w * s02, 1.0 - w

    def _pool_model(self, own: TideModel, target: str,
                    target_coords: tuple[float, float] | None) -> TideModel:
        consts = self._union
        names = [c.name for c in consts]
        own_names = [c.name for c in own._constituents]
        own_by_name = {}
        for j, n in enumerate(own_names):
            C = own._covariance
            own_by_name[n] = (
                float(own._coef[1 + 2 * j]),
                math.sqrt(max(float(C[1 + 2 * j, 1 + 2 * j]), 0.0)),
                float(own._coef[2 + 2 * j]),
                math.sqrt(max(float(C[2 + 2 * j, 2 + 2 * j]), 0.0)),
            )
        p = 1 + 2 * len(consts)
        coef = np.zeros(p)
        cov = np.zeros((p, p))
        coef[0] = float(own._coef[0])  # H0 stays local (datum-dependent)
        cov[0, 0] = max(float(own._covariance[0, 0]), 0.0)
        fits: list[Fit] = []
        shrink: dict[str, float] = {}
        taus: dict[str, list[float]] = {}
        for j, n in enumerate(names):
            pr = self._prior(n, target, target_coords)
            n_have = pr[0][2]
            vals, vars_, sh = [], [], []
            for k in (0, 1):
                o = (own_by_name[n][2 * k], own_by_name[n][2 * k + 1]) \
                    if n in own_by_name else None
                sole = (n_have == 1 and target in self._models
                        and self._station_ab(target, n) is not None)
                v, vv, s = self._pool_one(o, pr[k], sole)
                vals.append(v)
                vars_.append(max(vv, 0.0))
                sh.append(s)
            a, b = vals
            va, vb = vars_
            coef[1 + 2 * j], coef[2 + 2 * j] = a, b
            cov[1 + 2 * j, 1 + 2 * j] = va
            cov[2 + 2 * j, 2 + 2 * j] = vb
            amp = math.hypot(a, b)
            var_amp = (a * a * va + b * b * vb) / amp**2 if amp > 0 else 0.5 * (va + vb)
            fits.append(Fit(n, amp, math.degrees(math.atan2(b, a)) % 360.0,
                            math.sqrt(max(var_amp, 0.0))))
            shrink[n] = float(np.mean(sh))
            taus[n] = [pr[0][1], pr[1][1]]
        meta = {
            "source": "pooled",
            "network": sorted(self._models),
            "target": target,
            "mean_shrinkage": float(np.mean(list(shrink.values()))) if shrink else 0.0,
            "shrinkage": shrink,
            "tau": taus,
        }
        return TideModel(consts, coef, cov, float(own._sigma2), fits,
                         station=target, source="pooled", meta=meta)

    # -- public API --------------------------------------------------------

    def pool(self, station: str,
             target_coords: tuple[float, float] | None = None) -> TideModel:
        """Return the partially-pooled model for a network member station."""
        if station not in self._models:
            raise ValueError(f"{station!r} is not in the network; "
                             f"use seed_short() for new stations")
        coords = target_coords or self._coords.get(station)
        return self._pool_model(self._models[station], station, coords)

    def seed_short(
        self,
        times: Sequence[datetime],
        heights,
        station: str,
        target_coords: tuple[float, float] | None = None,
    ) -> TideModel:
        """Fit a too-short record on the network basis, then pool it.

        The short record is regressed on the union constituent set *without*
        selection (a sparse gauge cannot select reliably), and the noisy
        coefficients are shrunk toward the regional prior — this is the
        crowdsource payoff: a cheap sensor's constants come mostly from its
        neighbours until its own record earns weight.
        """
        own = TideModel.fit(
            list(times), heights, auto_select=False,
            candidates=self._union, station=station,
        )
        return self._pool_model(own, station, target_coords)

    def report(self, station: str) -> list[PooledConstituent]:
        """Per-constituent pooled estimates with shrinkage diagnostics."""
        pooled = self.pool(station)
        taus = pooled.meta["tau"]
        shrink = pooled.meta["shrinkage"]
        return [
            PooledConstituent(
                name=f.name, amplitude=f.amplitude, phase_deg=f.phase_deg,
                sigma_amp=f.sigma_amp, shrinkage=shrink[f.name],
                tau=float(np.mean(taus[f.name])),
            )
            for f in pooled.constituents()
        ]
