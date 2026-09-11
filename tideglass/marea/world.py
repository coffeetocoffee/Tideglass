"""The living world model — v3.0.

Fuses the kriged regional fields (v0.7) with the federated peer network into a
single **continuously-updating global coastal tide + surge field**: a single
:class:`WorldModel` can predict the water level at *any* ``(lon, lat)`` at *any*
time, even with no local gauge — a cold start anywhere on Earth.

The idea is to stop thinking of a model as "one station plus neighbours" and
start thinking of the installed base as one field. For every location we
**krige the harmonic coefficients themselves** across the network so the tide is
a genuine continuous spatial process (the GP random field from ``marea.krige``,
not a per-station series). Satellite altimetry joins as a *virtual peer*: each
track sample is a scattered SSH observation whose residual (``ssh - tide``)
constrains the non-tidal field offshore, where no gauge exists. Surge is the
precision-blended federated :class:`~tideglass.marea.federation.SurgeResponse`
(v2.3) applied to forecast met forcing.

Dependency-free (numpy only). ``marea`` never imports ``marine``.

Build a model directly::

    wm = WorldModel(models={s: TideModel, ...}, coords={s: (lon, lat), ...},
                    surges={s: SurgeResponse, ...}, altimetry=alt)

or from a store::

    wm = WorldModel.from_artifacts(store_dir, coords="coords.csv")
    wm = WorldModel.from_gauge_store(".tideglass/crowd")

then::

    pred = wm.predict(lon, lat, times, met=(wind_speed, wind_dir, pressure))
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from tideglass.marea import constituents as CON
from tideglass.marea.federation import SurgeResponse, _surge_blend
from tideglass.marea.model import TideModel, _basis_matrix

# ---------------------------------------------------------------------------
# Distance + covariance helpers (local copies; also in krige/spatial/federation)
# ---------------------------------------------------------------------------


def _haversine_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance (km) between coordinate arrays (broadcastable)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _exp_cov(dist_km, range_km, sill, nugget) -> np.ndarray:
    """Exponential covariance; nugget added only for the train matrix (2-D)."""
    c = sill * np.exp(-np.asarray(dist_km, dtype=float) / max(range_km, 1e-9))
    if c.ndim == 2 and nugget > 0.0:
        c = c + nugget * np.eye(c.shape[0])
    return c


def _parse_iso(s: str) -> datetime:
    s = s.strip()
    if s.endswith(("Z", "z")):
        s = s[:-1] + "+00:00"
    t = datetime.fromisoformat(s)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t


# ---------------------------------------------------------------------------
# Ordinary kriging of a single scattered scalar field (precomputed once)
# ---------------------------------------------------------------------------


class _Kriger:
    """Ordinary kriging of a scattered scalar field (GP special case).

    The kriging system depends only on the *station* distances, so it is solved
    once in :meth:`__init__`; :meth:`__call__` then evaluates the prediction and
    its variance at arbitrary query points cheaply. Falls back to inverse-
    distance weighting (and to the global mean for a constant field) when the
    control points are too few or degenerate.
    """

    def __init__(self, station_lons, station_lats, values,
                 range_km: float | None = None, nugget_frac: float = 0.05):
        y = np.asarray(values, dtype=float).ravel()
        self.slon = np.asarray(station_lons, dtype=float).ravel()
        self.slat = np.asarray(station_lats, dtype=float).ravel()
        self.y = y
        self.n = int(y.size)
        self.range_km = float(range_km) if range_km is not None else float("nan")
        self.nugget_frac = nugget_frac
        self.const_val = None
        self.simple = False
        self.Ainv = None
        self.sill = 1.0
        self.nugget = 0.0
        self._setup()

    def _setup(self) -> None:
        if self.n == 0:
            self.const_val = 0.0
            return
        if np.allclose(self.y, self.y[0]):
            self.const_val = float(self.y[0])
            return
        if self.n < 3:
            self.simple = True
            return
        D = _haversine_km(
            self.slon[:, None], self.slat[:, None],
            self.slon[None, :], self.slat[None, :])
        if math.isnan(self.range_km):
            off = D[~np.eye(self.n, dtype=bool)]
            self.range_km = 0.6 * float(off.max()) if off.size else 1.0
        self.sill = float(np.var(self.y))
        if self.sill <= 0.0:
            self.sill = 1e-6
        self.nugget = self.nugget_frac * self.sill
        K = _exp_cov(D, self.range_km, self.sill, self.nugget)
        A = np.zeros((self.n + 1, self.n + 1))
        A[:self.n, :self.n] = K
        A[:self.n, self.n] = 1.0
        A[self.n, :self.n] = 1.0
        self.Ainv = np.linalg.inv(A)

    def __call__(self, glon, glat):
        glon = np.asarray(glon, dtype=float).ravel()
        glat = np.asarray(glat, dtype=float).ravel()
        if self.const_val is not None:
            return (np.full(glon.shape, self.const_val),
                    np.zeros(glon.shape))
        if self.simple:
            return self._idw(glon, glat)
        gdist = _haversine_km(
            glon[None, :], glat[None, :],
            self.slon[:, None], self.slat[:, None])  # (n x nq)
        kvec = _exp_cov(gdist, self.range_km, self.sill, 0.0)  # no nugget
        rhs = np.vstack([kvec, np.ones((1, kvec.shape[1]))])
        lam = self.Ainv @ rhs
        weights = lam[:self.n, :]
        mu = lam[self.n, :]
        pred = (weights.T @ self.y).ravel()
        krig_var = ((self.sill + self.nugget)
                    - np.einsum("ij,ji->i", weights.T, kvec)
                    + mu).ravel()
        return pred, np.maximum(krig_var, 0.0)

    def _idw(self, glon, glat, power=2.0, eps=1e-6):
        nq = glon.size
        pred = np.empty(nq)
        var = np.zeros(nq)
        for i in range(nq):
            d = _haversine_km(
                np.full(self.n, glon[i]), np.full(self.n, glat[i]),
                self.slon, self.slat)
            w = 1.0 / (d + eps) ** power
            wsum = w.sum()
            if wsum <= 0:
                w = np.ones_like(w) / self.n
            else:
                w = w / wsum
            pred[i] = float(w @ self.y)
        return pred, var


# ---------------------------------------------------------------------------
# Satellite altimetry virtual peer
# ---------------------------------------------------------------------------


@dataclass
class AltimetryTrack:
    """A satellite-altimetry track as a virtual peer (offshore SSH samples).

    Each sample is ``(lon, lat, time, ssh)``. The :class:`WorldModel` treats the
    residual ``ssh - tide`` at every sample as a scattered observation of the
    non-tidal field (surge + mean-sea-level anomaly) and kriges it back onto the
    query point, so an otherwise ungauged ocean point is constrained by the
    satellite.
    """

    lons: np.ndarray
    lats: np.ndarray
    times: list[datetime]
    ssh: np.ndarray

    @classmethod
    def from_csv(cls, path: str):
        """Read ``time,lon,lat,ssh`` rows (ISO datetimes, metres)."""
        times: list[datetime] = []
        lons: list[float] = []
        lats: list[float] = []
        ssh: list[float] = []
        with open(path, newline="") as fh:
            rows = list(csv.reader(fh))
        cols = None
        for row in rows:
            if not row or all(not c.strip() for c in row):
                continue
            if cols is None:
                head = [c.strip().lower() for c in row]
                if "time" in head:
                    cols = {k: head.index(k) for k in
                            ("time", "lon", "lat", "ssh")}
                    continue
                cols = {"time": 0, "lon": 1, "lat": 2, "ssh": 3}
            try:
                t = _parse_iso(row[cols["time"]])
                lo = float(row[cols["lon"]])
                la = float(row[cols["lat"]])
                s = float(row[cols["ssh"]])
            except (ValueError, IndexError, KeyError):
                continue
            times.append(t)
            lons.append(lo)
            lats.append(la)
            ssh.append(s)
        if not times:
            raise ValueError(f"no readable time,lon,lat,ssh rows in {path!r}")
        return cls(np.asarray(lons), np.asarray(lats), times,
                  np.asarray(ssh, dtype=float))


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


@dataclass
class WorldPrediction:
    """A prediction of the world tide + surge field at one ``(lon, lat)``."""

    times: list[datetime]
    mean: np.ndarray          # total water level (tide + altimetry + surge)
    lower: np.ndarray
    upper: np.ndarray
    se: np.ndarray            # std error of the mean tide (+ altimetry)
    tide_mean: np.ndarray     # tide + altimetry correction (no surge)
    surge_mean: np.ndarray | None
    surge_sigma: np.ndarray | None
    altimetry_correction: np.ndarray
    components: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The world model
# ---------------------------------------------------------------------------


class WorldModel:
    """A continuously-updating global coastal tide + surge field (v3.0).

    Fuses a network of :class:`TideModel` s (the kriged regional field), optional
    per-station :class:`SurgeResponse` s (the federated surge moat), and an
    optional satellite :class:`AltimetryTrack` (a virtual peer). Predict anywhere.
    """

    def __init__(
        self,
        models: dict[str, TideModel],
        coords: dict[str, tuple[float, float]],
        surges: dict[str, SurgeResponse] | None = None,
        altimetry: AltimetryTrack | None = None,
        range_km: float | None = None,
        nugget_frac: float = 0.05,
        network_sigma2: float | None = None,
    ):
        self.coords = {k: tuple(v) for k, v in coords.items()}
        self.models = {k: m for k, m in models.items() if k in self.coords}
        if not self.models:
            raise ValueError("WorldModel needs at least one model with coordinates")
        self.surges = dict(surges or {})
        self.altimetry = altimetry
        self.range_km = range_km
        self.nugget_frac = nugget_frac
        self.station_names = list(self.models)
        self._slon = np.array([self.coords[s][0] for s in self.station_names], float)
        self._slat = np.array([self.coords[s][1] for s in self.station_names], float)
        self._build_union_and_krigers()
        self.network_sigma2 = (
            float(network_sigma2) if network_sigma2 is not None
            else self._estimate_network_sigma2())
        self.blended_surge = self._blend_surges()

    # -- construction helpers -------------------------------------------------

    def _build_union_and_krigers(self) -> None:
        counts: dict[str, int] = {}
        for m in self.models.values():
            for c in m._constituents:
                counts[c.name] = counts.get(c.name, 0) + 1
        union_names = [c.name for c in CON.CATALOG if counts.get(c.name, 0) >= 2]
        union = [CON.get(n) for n in union_names]

        h0_vals = np.array([self.models[s]._coef[0] for s in self.station_names], float)
        self._h0 = _Kriger(self._slon, self._slat, h0_vals,
                           self.range_km, self.nugget_frac)

        krigers: dict[str, tuple[_Kriger, _Kriger]] = {}
        for c in union:
            a = np.empty(len(self.station_names))
            b = np.empty(len(self.station_names))
            for i, s in enumerate(self.station_names):
                m = self.models[s]
                names = [cc.name for cc in m._constituents]
                if c.name in names:
                    j = names.index(c.name)
                    a[i] = m._coef[1 + 2 * j]
                    b[i] = m._coef[2 + 2 * j]
                else:
                    a[i] = float("nan")
                    b[i] = float("nan")
            if np.isnan(a).any() or np.isnan(b).any():
                pa, pb = self._prior(c.name)
                a = np.where(np.isnan(a), pa, a)
                b = np.where(np.isnan(b), pb, b)
            ka = _Kriger(self._slon, self._slat, a, self.range_km, self.nugget_frac)
            kb = _Kriger(self._slon, self._slat, b, self.range_km, self.nugget_frac)
            krigers[c.name] = (ka, kb)
        self.union = union
        self._krigers = krigers

    def _prior(self, name: str) -> tuple[float, float]:
        """Inverse-variance regional mean ``(a, b)`` for one constituent."""
        za: list[float] = []
        zb: list[float] = []
        wa: list[float] = []
        wb: list[float] = []
        for s in self.station_names:
            m = self.models[s]
            names = [cc.name for cc in m._constituents]
            if name not in names:
                continue
            j = names.index(name)
            C = m._covariance
            saa = max(float(C[1 + 2 * j, 1 + 2 * j]), 1e-12)
            sbb = max(float(C[2 + 2 * j, 2 + 2 * j]), 1e-12)
            za.append(float(m._coef[1 + 2 * j]))
            zb.append(float(m._coef[2 + 2 * j]))
            wa.append(1.0 / saa)
            wb.append(1.0 / sbb)
        if not za:
            return 0.0, 0.0
        va = np.asarray(wa)
        vb = np.asarray(wb)
        return (float((va @ za) / va.sum()), float((vb @ zb) / vb.sum()))

    def _estimate_network_sigma2(self) -> float:
        rmses = [float(m.meta["rmse"]) for m in self.models.values()
                 if m.meta.get("rmse") is not None]
        est = float(np.mean([r * r for r in rmses])) if rmses else 0.0
        return max(est, 1e-3)  # floor: residual non-tidal variance not captured

    def _blend_surges(self) -> SurgeResponse | None:
        responses = list(self.surges.values())
        if not responses:
            return None
        return _surge_blend(responses)

    # -- tide only ------------------------------------------------------------

    def _tide_only(self, lon: float, lat: float, times: Sequence[datetime]):
        """Kriged tidal height and its variance at ``(lon, lat)`` (no surge)."""
        A = _basis_matrix(self.union, list(times))
        p = 1 + 2 * len(self.union)
        coef = np.empty(p)
        var = np.empty(p)
        h0, v0 = self._h0(lon, lat)
        coef[0] = float(h0[0])
        var[0] = float(v0[0])
        for k, c in enumerate(self.union):
            ka, kb = self._krigers[c.name]
            a, va = ka(lon, lat)
            b, vb = kb(lon, lat)
            coef[1 + 2 * k] = float(a[0])
            coef[2 + 2 * k] = float(b[0])
            var[1 + 2 * k] = float(va[0])
            var[2 + 2 * k] = float(vb[0])
        mean = A @ coef
        tide_var = np.maximum(np.einsum("ij,j->i", A * A, var), 0.0)
        return mean, tide_var

    # -- altimetry virtual peer ----------------------------------------------

    def _altimetry_correction(self, lon: float, lat: float):
        alt = self.altimetry
        if alt is None or len(alt.lons) == 0:
            return 0.0, 0.0
        groups: dict[tuple[float, float], list[int]] = {}
        for i in range(len(alt.lons)):
            key = (round(float(alt.lons[i]), 3), round(float(alt.lats[i]), 3))
            groups.setdefault(key, []).append(i)
        locs: list[list[float]] = []
        rvals: list[float] = []
        for (lo, la), idxs in groups.items():
            t = [alt.times[i] for i in idxs]
            ssh = np.array([float(alt.ssh[i]) for i in idxs], dtype=float)
            tmean, _ = self._tide_only(lo, la, t)
            resid = ssh - tmean
            rvals.append(float(np.mean(resid)))
            locs.append([lo, la])
        locs = np.asarray(locs, dtype=float)
        rvals = np.asarray(rvals, dtype=float)
        if locs.shape[0] == 0:
            return 0.0, 0.0
        kr = _Kriger(locs[:, 0], locs[:, 1], rvals,
                     self.range_km, self.nugget_frac)
        corr, cvar = kr(lon, lat)
        return float(corr[0]), max(float(cvar[0]), 0.0)

    # -- public predict -------------------------------------------------------

    def predict(self, lon: float, lat: float, times: Sequence[datetime],
                met=None) -> WorldPrediction:
        """Predict the water level at ``(lon, lat)`` over ``times``.

        :param met: optional ``(wind_speed, wind_dir_deg, pressure_hPa)`` arrays
            (matching ``times``). When given, the blended federated surge response
            is forecast; otherwise a baseline surge sigma (no mean) is reported.
        :returns: :class:`WorldPrediction` with the total level plus components.
        """
        times = list(times)
        tmean, tvar = self._tide_only(lon, lat, times)
        corr, cvar = self._altimetry_correction(lon, lat)

        surge_mean = None
        surge_sigma = None
        if met is not None:
            wind_speed, wind_dir, pressure = met
            if self.blended_surge is not None:
                fc = self.blended_surge.forecast(
                    times, wind_speed, wind_dir, pressure)
                surge_mean = np.asarray(fc.mean, dtype=float)
                surge_sigma = np.asarray(fc.sigma, dtype=float)
            else:
                surge_sigma = np.zeros(len(times))
        elif self.blended_surge is not None:
            surge_mean = np.zeros(len(times))
            surge_sigma = np.full(len(times), self.blended_surge.sigma)

        s_mean = surge_mean if surge_mean is not None else np.zeros(len(times))
        s_sig = surge_sigma if surge_sigma is not None else np.zeros(len(times))

        tide = tmean + corr
        mean = tide + s_mean
        tot_var = tvar + self.network_sigma2 + cvar + s_sig ** 2
        se = np.sqrt(tvar + cvar)
        half = 1.96 * np.sqrt(tot_var)
        return WorldPrediction(
            times=times,
            mean=mean,
            lower=mean - half,
            upper=mean + half,
            se=se,
            tide_mean=tide,
            surge_mean=(s_mean if surge_mean is not None else None),
            surge_sigma=(s_sig if surge_sigma is not None else None),
            altimetry_correction=np.full(len(times), corr),
            components={
                "lon": lon, "lat": lat, "n_stations": len(self.models),
                "constituents": [c.name for c in self.union],
                "altimetry_var": cvar,
                "network_sigma2": self.network_sigma2,
            },
        )

    def predict_field(self, lons, lats, times) -> np.ndarray:
        """Tidal height field (n_grid x n_time) at a grid of points (no surge)."""
        lons = np.asarray(lons, dtype=float).ravel()
        lats = np.asarray(lats, dtype=float).ravel()
        out = np.empty((lons.size, len(times)))
        for i in range(lons.size):
            m, _ = self._tide_only(lons[i], lats[i], times)
            out[i] = m
        return out

    # -- factories ------------------------------------------------------------

    @classmethod
    def from_gauge_store(cls, store_root: str,
                         altimetry: AltimetryTrack | None = None,
                         **kw) -> WorldModel:
        """Build a world field from a crowd :class:`GaugeStore` (has coords)."""
        from tideglass.marea.crowdsource import GaugeStore

        gs = GaugeStore(store_root)
        models = gs.models()
        coords = {g.station: (g.lon, g.lat) for g in gs.stations()}
        surges = {}
        for s in models:
            p = os.path.join(store_root, f"{s}.surge.json")
            if os.path.exists(p):
                try:
                    with open(p) as fh:
                        surges[s] = SurgeResponse.from_dict(json.load(fh))
                except (OSError, ValueError):
                    pass
        return cls(models, coords, surges=surges, altimetry=altimetry, **kw)

    @classmethod
    def from_artifacts(cls, store_dir: str, coords: str | None = None,
                       altimetry: AltimetryTrack | None = None,
                       **kw) -> WorldModel:
        """Build from a store of ``<station>.json`` artifacts.

        Coordinates come from ``coords`` (a ``station,lon,lat`` CSV). If absent,
        the directory is tried as a :class:`GaugeStore`. ``<station>.surge.json``
        files, when present, are loaded as surge responses.
        """
        cmap: dict[str, tuple[float, float]] | None = None
        if coords and os.path.exists(coords):
            with open(coords, newline="") as fh:
                for row in csv.reader(fh):
                    if not row or not row[0].strip():
                        continue
                    head = [c.strip().lower() for c in row]
                    if cmap is None and "station" in head:
                        ci = head.index("station")
                        lo_i = head.index("lon")
                        la_i = head.index("lat")
                        cmap = {}
                        continue
                    if cmap is not None:
                        cmap[row[ci].strip()] = (float(row[lo_i]), float(row[la_i]))
                        continue
                    if len(row) >= 3:
                        cmap = cmap or {}
                        cmap[row[0].strip()] = (float(row[1]), float(row[2]))
        skip = (".surge", ".gpd", ".met", ".nowcast", ".health", ".ops",
                ".ledger", ".trust", ".regimes")
        models: dict[str, TideModel] = {}
        surges: dict[str, SurgeResponse] = {}
        for fn in sorted(os.listdir(store_dir)):
            if not fn.endswith(".json"):
                continue
            base = fn[:-5]
            if any(base.endswith(s) for s in skip):
                continue
            path = os.path.join(store_dir, fn)
            try:
                m = TideModel.load_harmonic(path)
            except (OSError, ValueError, KeyError):
                continue
            if m.station is None:
                continue
            models[m.station] = m
            sp = os.path.join(store_dir, f"{m.station}.surge.json")
            if os.path.exists(sp):
                try:
                    with open(sp) as fh:
                        surges[m.station] = SurgeResponse.from_dict(json.load(fh))
                except (OSError, ValueError):
                    pass
        if cmap is not None:
            coords = {s: cmap[s] for s in models if s in cmap}
        elif models:
            try:
                from tideglass.marea.crowdsource import GaugeStore

                gs = GaugeStore(store_dir)
                coords = {g.station: (g.lon, g.lat) for g in gs.stations()}
            except (OSError, ValueError):
                coords = {}
        else:
            coords = {}
        return cls(models, coords, surges=surges, altimetry=altimetry, **kw)
