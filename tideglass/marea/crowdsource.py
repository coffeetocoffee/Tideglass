"""Crowd-sourced gauge network for Marea Core — the v0.5 network effect.

Cheap consumer-grade sensors (ultrasonic sticks, low-cost pressure logs,
community float gauges) each produce a noisy, often sparse tide series. On its
own a single such gauge is too poor to fit a trustworthy harmonic model. But
every upload is appended to a local :class:`GaugeStore`; once two or more
stations share a time grid, :meth:`GaugeStore.harmonize` borrows strength
across the network through the EOF machinery (``marea.spatial``). The result
is a genuine *network effect*: each new station makes the shared regional field
— and therefore every station's denoised reconstruction — strictly better.

``GaugeStore`` is a tiny JSON directory store: no server, no extra dependency.
Uploads are ``time,height`` CSVs plus a station id and ``(lon, lat)``.

    gs = GaugeStore(".tideglass/crowd")
    gs.add_csv("pier_07.csv", "pier07", lon=-122.34, lat=47.60)
    gs.add_csv("floating_dock.csv", "dock12", lon=-122.31, lat=47.59)
    net = gs.network_effect()   # shows the network effect, per added station
"""

from __future__ import annotations

import csv as _csv
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tideglass.marea.model import TideModel
from tideglass.marea.spatial import harmonize


@dataclass(frozen=True)
class GaugeMeta:
    station: str
    lon: float
    lat: float
    n: int
    source: str


def read_csv(path: str) -> tuple[list, list[float]]:
    """Parse a ``time,height`` upload (ISO datetimes, metres) without CLI deps."""
    from datetime import datetime, timezone

    times, heights = [], []
    with open(path, newline="") as fh:
        for row in _csv.reader(fh):
            if not row or not row[0].strip():
                continue
            s = row[0].strip()
            if s.lower() == "time" or s.lower().endswith("date time"):
                continue  # header
            try:
                t = datetime.fromisoformat(s)
            except ValueError:
                continue
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            try:
                heights.append(float(row[1]))
            except (IndexError, ValueError):
                continue
            times.append(t)
    if not times:
        raise ValueError(f"no readable time,height rows in {path!r}")
    return times, heights


class GaugeStore:
    """A local store of crowd-sourced gauge uploads (JSON + CSV, no server)."""

    def __init__(self, root: str = ".tideglass/crowd"):
        self.root = root
        os.makedirs(root, exist_ok=True)

    # -- ingest --------------------------------------------------------------

    def add(
        self,
        station: str,
        lon: float,
        lat: float,
        times: Sequence,
        heights: Sequence[float],
        source: str = "crowd",
    ) -> int:
        """Persist one gauge's ``(times, heights)`` and its metadata."""
        times = list(times)
        heights = [float(h) for h in heights]
        if len(times) != len(heights):
            raise ValueError(f"{len(times)} times but {len(heights)} heights")
        meta = {"station": station, "lon": float(lon), "lat": float(lat),
                "n": len(times), "source": source}
        with open(os.path.join(self.root, f"{station}.csv"), "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["time", "height"])
            for t, h in zip(times, heights):
                w.writerow([t.isoformat(), f"{h:.4f}"])
        with open(os.path.join(self.root, f"{station}.meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        return len(times)

    def add_csv(
        self,
        path: str,
        station: str,
        lon: float,
        lat: float,
        source: str = "crowd",
    ) -> int:
        """Read a ``time,height`` CSV upload and :meth:`add` it to the store."""
        times, heights = read_csv(path)
        return self.add(station, lon, lat, times, heights, source=source)

    # -- read ----------------------------------------------------------------

    def _meta_paths(self) -> list[str]:
        return [p for p in os.listdir(self.root)
                if p.endswith(".meta.json")]

    def stations(self) -> list[GaugeMeta]:
        out = []
        for p in self._meta_paths():
            with open(os.path.join(self.root, p)) as fh:
                m = json.load(fh)
            out.append(GaugeMeta(m["station"], m["lon"], m["lat"],
                                 m["n"], m["source"]))
        return sorted(out, key=lambda g: g.station)

    @property
    def count(self) -> int:
        return len(self._meta_paths())

    def load(self, station: str) -> tuple[list, list[float], GaugeMeta]:
        meta_path = os.path.join(self.root, f"{station}.meta.json")
        with open(meta_path) as fh:
            m = json.load(fh)
        times, heights = read_csv(os.path.join(self.root, f"{station}.csv"))
        return times, heights, GaugeMeta(
            m["station"], m["lon"], m["lat"], m["n"], m["source"])

    # -- network -------------------------------------------------------------

    def _common_grid(self) -> tuple[list, np.ndarray]:
        """Union hourly grid spanning all stored gauges (interpolated)."""
        from datetime import timedelta, timezone

        lo = None
        hi = None
        for g in self.stations():
            t, _, _ = self.load(g.station)
            lo = t[0] if lo is None else min(lo, t[0])
            hi = t[-1] if hi is None else max(hi, t[-1])
        if lo is None or hi is None:
            raise ValueError("no gauges in store")
        step = timedelta(hours=1)
        grid = []
        cur = lo.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
        hi = hi.astimezone(timezone.utc)
        while cur <= hi:
            grid.append(cur)
            cur += step
        out: dict[str, np.ndarray] = {}
        for g in self.stations():
            t, h, _ = self.load(g.station)
            t_h = np.array([(tt.replace(tzinfo=timezone.utc)
                             - grid[0].replace(tzinfo=timezone.utc)).total_seconds() / 3600.0
                            for tt in t], dtype=float)
            grid_h = np.arange(len(grid), dtype=float)
            out[g.station] = np.interp(grid_h, t_h, np.asarray(h, dtype=float))
        return grid, out

    def harmonize(self, variance_threshold: float = 0.95):
        """EOF-harmonize every stored gauge onto the common grid."""
        if self.count < 2:
            raise ValueError("need at least 2 gauges to form a network")
        _, series = self._common_grid()
        return harmonize(series, variance_threshold=variance_threshold)

    def network_effect(self, variance_threshold: float = 0.95) -> dict:
        """Quantify the network effect: how each added station helps.

        Stations are added in increasing order of record length. For every
        prefix of length ``k`` (``k = 2..N``) we build the EOF network and
        record how many shared modes are needed to explain
        ``variance_threshold`` of the regional variance, plus the total
        explained variance. A working network effect shows ``total_explained``
        rising (and the residual falling) monotonically as stations join.
        """
        if self.count < 2:
            raise ValueError("need at least 2 gauges to form a network")
        gauges = sorted(self.stations(), key=lambda g: g.n)
        _, series = self._common_grid()
        names = [g.station for g in gauges]
        steps = []
        for k in range(2, len(names) + 1):
            prefix = names[:k]
            res = harmonize({s: series[s] for s in prefix},
                            variance_threshold=variance_threshold)
            n_modes = int(np.searchsorted(
                np.cumsum(res.explained), variance_threshold) + 1)
            n_modes = max(1, min(n_modes, res.modes.shape[0]))
            steps.append({
                "n_stations": k,
                "stations": prefix,
                "total_explained": res.total_explained,
                "modes_to_threshold": n_modes,
            })
        return {
            "n_total": len(names),
            "variance_threshold": variance_threshold,
            "steps": steps,
            "gain_total_explained": (
                steps[-1]["total_explained"] - steps[0]["total_explained"]
                if steps else 0.0
            ),
        }

    def models(self) -> dict[str, TideModel]:
        """Fit a TideModel per gauge (for dense, well-sampled uploads)."""
        out = {}
        for g in self.stations():
            t, h, _ = self.load(g.station)
            if len(t) < 48:  # too sparse to fit alone; use the network instead
                continue
            try:
                out[g.station] = TideModel.fit(t, h, station=g.station)
            except ValueError:
                continue
        return out
