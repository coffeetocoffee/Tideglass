"""Content-addressed artifact store — v2.2 reproducibility as a feature.

Provenance (v0.6) records *which* observations produced a model and the
decision ledger (v1.3) records *what* was priced from it, but neither makes a
past model retrievable on demand. :class:`ArtifactStore` closes that gap: every
model artifact is stored under a **content address** — the SHA-256 of its own
canonical JSON — so re-fitting the same data yields the same key (dedup) and an
artifact cannot be tampered with without its key changing. Each entry also
carries its full **lineage**: the observation set that produced it
(``data_sha256`` from the v0.6 provenance block) plus the peer bundles
(v2.1 ``bundle_sha256``) that were folded in and its parent artifact in the
station's refit chain.

``tideglass reproduce <hash>`` then rebuilds any past prediction bit-for-bit —
``hash`` is verified against the stored blob before the model is reconstructed,
so an insurer or port authority can audit exactly what was forecast and prove
it is reproducible.

Store layout (the ``cas/`` dir, or any root passed to the constructor)::

    <root>/<sha256>.json   content-addressed artifact blobs
    <root>/index.json      by-hash refcounts + per-station refit history with lineage

Everything is JSON, numpy-only, and local — consistent with the rest of Marea
Core.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from datetime import datetime, timezone

import numpy as np

from tideglass.marea import constituents as CON
from tideglass.marea.model import Fit, Prediction, TideModel

_DEFAULT_ROOT = ".tideglass/cas"


def artifact_hash(artifact: dict) -> str:
    """Canonical SHA-256 of an artifact — the content-address key.

    Uses ``sort_keys`` + minimal separators so the same logical artifact always
    hashes identically regardless of key order or whitespace. The model must
    round-trip through :meth:`TideModel.to_artifact`, which is already designed
    to be ``json.dump``-safe (``cmd_fit`` serializes it today).
    """
    blob = json.dumps(artifact, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def obs_lineage(meta: dict) -> dict:
    """Extract the observation lineage block from a model's provenance meta.

    Returns ``None`` fields for loaded/published models that carry no v0.6
    provenance, so the store still works for NOAA ``load_harmonic`` artifacts.
    """
    return {
        "data_sha256": meta.get("data_sha256"),
        "obs_start": meta.get("obs_start"),
        "obs_end": meta.get("obs_end"),
        "n_obs": meta.get("n_obs"),
    }


def build_lineage(meta: dict, peers: Sequence[str] | None = None,
                  parent: str | None = None) -> dict:
    """Assemble the lineage block recorded for one stored artifact.

    :param meta: the model's ``meta`` (provenance + source).
    :param peers: ``bundle_sha256`` digests of the peer bundles that were
        folded into this model (empty for an obs-only fit).
    :param parent: content hash of the prior artifact for the same station, so
        the full refit chain is traversable back to the first fit.
    """
    return {
        "obs": obs_lineage(meta),
        "peers": sorted({str(p) for p in (peers or [])}),
        "parent": parent,
        "source": meta.get("source"),
        "fitted_at": meta.get("fitted_at"),
    }


def _peer_bundle_hashes(peer_dir: str | None) -> list[str]:
    """``bundle_sha256`` of every peer bundle in a federation directory.

    ``peer_dir`` is a :class:`~tideglass.marea.federation.GlobalFederation`
    root — bundles live in ``<peer_dir>/peers/*.json``. The federation module is
    imported lazily so ``cas`` never participates in an import cycle.
    """
    if not peer_dir:
        return []
    peers_root = os.path.join(peer_dir, "peers")
    if not os.path.isdir(peers_root):
        return []
    from tideglass.marea.federation import PeerBundle  # lazy, no cycle

    out: list[str] = []
    for fn in sorted(os.listdir(peers_root)):
        if fn.endswith(".json"):
            try:
                b = PeerBundle.load(os.path.join(peers_root, fn))
                out.append(b.manifest()["bundle_sha256"])
            except (OSError, ValueError, KeyError):
                continue
    return sorted(set(out))


def _record(model: TideModel) -> dict:
    """The full content-addressed record for ``model``.

    ``to_artifact()`` intentionally discards the covariance and smears the
    coefficients through amplitude/phase (the public ``load_harmonic`` round
    trip), so reproducing a *fitted* model through it is not bit-for-bit. The
    CAS record keeps the exact fitted state alongside the interoperable
    artifact: ``coef``, ``covariance`` and ``sigma2`` are preserved so the
    model can be rebuilt with coefficients identical to the original solve.
    """
    fits = model.constituents()
    return {
        "artifact": model.to_artifact(),
        "coef": [float(x) for x in model._coef],
        "covariance": [[float(x) for x in row] for row in model._covariance],
        "sigma2": float(model._sigma2),
        "constituents": [f.name for f in fits],
        "fits": [
            {
                "name": f.name,
                "amplitude": f.amplitude,
                "phase_deg": f.phase_deg,
                "sigma_amp": f.sigma_amp,
            }
            for f in fits
        ],
    }


def _model_from_blob(blob: dict) -> TideModel:
    """Rebuild the model from a CAS record, preserving exact coefficients."""
    artifact = blob["artifact"]
    consts = [CON.get(n) for n in blob["constituents"]]
    coef = np.asarray(blob["coef"], dtype=float)
    cov = np.asarray(blob["covariance"], dtype=float)
    fits = [
        Fit(f["name"], float(f["amplitude"]), float(f["phase_deg"]) % 360.0,
            float(f["sigma_amp"]))
        for f in blob["fits"]
    ]
    meta = dict(artifact.get("meta", {}))
    if "station" not in meta and artifact.get("station"):
        meta["station"] = artifact.get("station")
    model = TideModel(
        consts, coef, cov, float(blob["sigma2"]), fits,
        station=artifact.get("station"), source=meta.get("source", "cas"),
        meta=meta,
    )
    if meta.get("residual_bias") is not None:
        from tideglass.marea.residual import ResidualModel

        model._residual = ResidualModel.from_dict(meta["residual_bias"])
    return model


class ArtifactStore:
    """A content-addressed, lineage-tracking store of model artifacts.

    Models are keyed by :func:`artifact_hash` of their canonical JSON, so the
    same model stored twice is deduplicated, and any stored artifact can be
    checked against its key (:meth:`verify`) — the "bit-for-bit reproducible"
    guarantee. The per-station history records every refit with its obs and
    peer lineage, chained through ``parent`` hashes.
    """

    def __init__(self, root: str = _DEFAULT_ROOT,
                 peer_dir: str | None = None):
        """Open (or create) the store at ``root``.

        :param root: directory holding the ``<sha256>.json`` blobs and
            ``index.json``.
        :param peer_dir: optional :class:`GlobalFederation` root to scan for
            peer-bundle lineage (default: none — obs-only lineage).
        """
        self.root = root
        self.peer_dir = peer_dir
        os.makedirs(root, exist_ok=True)
        self._index_path = os.path.join(root, "index.json")
        self._index = self._load_index()

    # -- index plumbing ------------------------------------------------------

    def _blob_path(self, h: str) -> str:
        return os.path.join(self.root, f"{h}.json")

    def _load_index(self) -> dict:
        if os.path.exists(self._index_path):
            try:
                with open(self._index_path) as fh:
                    return json.load(fh)
            except (OSError, ValueError):
                return {"by_hash": {}, "stations": {}}
        return {"by_hash": {}, "stations": {}}

    def _save_index(self) -> None:
        with open(self._index_path, "w") as fh:
            json.dump(self._index, fh, indent=2, sort_keys=True)

    # -- storing -------------------------------------------------------------

    def store(self, model: TideModel, station: str | None = None,
              peer_hashes: Sequence[str] | None = None, *,
              auto_peers: bool = True) -> str:
        """Store ``model`` and return its content address.

        The blob is written only if not already present (content-addressed
        dedup). The station's history records the refit with its obs lineage
        (from ``model.meta``), the peer bundles folded in (explicit
        ``peer_hashes`` or auto-discovered federation when ``auto_peers``), and
        the prior artifact as ``parent``.

        :param station: station id used for the history chain (defaults to
            ``model.station``).
        :param peer_hashes: explicit ``bundle_sha256`` digests; when ``None``
            and ``auto_peers``, they are discovered from ``self.peer_dir``.
        """
        station = station or model.station or "?"
        record = _record(model)
        h = artifact_hash(record)
        if peer_hashes is None and auto_peers:
            peer_hashes = _peer_bundle_hashes(self.peer_dir)
        parent = self.latest(station)
        entry = {
            "hash": h,
            "station": station,
            "stored_at": datetime.now(timezone.utc).isoformat(),
            "lineage": build_lineage(model.meta, peers=peer_hashes, parent=parent),
        }
        by_hash = self._index["by_hash"]
        ref = by_hash.get(h, {"station": station, "refs": 0})
        ref["refs"] = int(ref.get("refs", 0)) + 1
        by_hash[h] = ref

        history = self._index["stations"].setdefault(station, [])
        if h not in {e["hash"] for e in history}:
            history.append(entry)
            self._save_index()
        if not os.path.exists(self._blob_path(h)):
            with open(self._blob_path(h), "w") as fh:
                json.dump(record, fh, indent=2)
        return h

    # -- retrieval -----------------------------------------------------------

    def contains(self, h: str) -> bool:
        """True if ``h`` (full hash) has a stored blob."""
        return os.path.exists(self._blob_path(h))

    def resolve(self, h: str) -> str | None:
        """Resolve a full hash or unambiguous prefix to a full content hash.

        Returns ``None`` when ``h`` is not stored or the prefix is ambiguous
        (mirrors the ergonomics of git SHAs).
        """
        h = h.lower()
        if h in self._index["by_hash"]:
            return h
        if len(h) >= 8:
            matches = [k for k in self._index["by_hash"] if k.startswith(h)]
            if len(matches) == 1:
                return matches[0]
        return None

    def get(self, h: str) -> dict:
        """Return the full CAS record (artifact + exact fitted state) for a hash."""
        if not self.contains(h):
            raise KeyError(f"no artifact with hash {h!r} in {self.root!r}")
        with open(self._blob_path(h)) as fh:
            return json.load(fh)

    def get_artifact(self, h: str) -> dict:
        """Return the interoperable ``to_artifact()`` dict for a hash."""
        return self.get(h)["artifact"]

    def load_model(self, h: str) -> TideModel:
        """Rebuild the :class:`TideModel` for a content hash, exact coefficients.

        Unlike the public ``load_harmonic`` path — which reconstructs from
        amplitude/phase and discards the covariance — this restores the exact
        fitted ``coef``/``covariance``/``sigma2`` so predictions reproduce
        bit-for-bit.
        """
        return _model_from_blob(self.get(h))

    def station_of(self, h: str) -> str | None:
        ref = self._index["by_hash"].get(h)
        return ref["station"] if ref else None

    def latest(self, station: str) -> str | None:
        """Content hash of the most recent stored artifact for ``station``."""
        history = self._index["stations"].get(station, [])
        return history[-1]["hash"] if history else None

    def history(self, station: str) -> list[dict]:
        """Ordered (oldest to newest) refit history for ``station``."""
        return list(self._index["stations"].get(station, []))

    def lineage_of(self, h: str) -> dict | None:
        """The lineage block recorded for a content hash, or ``None``."""
        for history in self._index["stations"].values():
            for e in history:
                if e["hash"] == h:
                    return e.get("lineage")
        return None

    # -- reproducibility -----------------------------------------------------

    def verify(self, h: str) -> bool:
        """Bit-for-bit integrity check: re-hash the stored blob and compare.

        This is the reproducibility proof — if the blob was altered after
        storage its recomputed hash no longer matches ``h``.
        """
        if not self.contains(h):
            return False
        try:
            artifact = self.get(h)
        except (OSError, ValueError):
            return False
        return artifact_hash(artifact) == h

    def reproduce(self, h: str, times: Sequence[datetime],
                  z: float = 1.96) -> Prediction:
        """Rebuild a past prediction bit-for-bit from a content hash.

        The stored model is reconstructed and evaluated at ``times``. For any
        given ``h`` and ``times`` the output is deterministic, so this returns
        exactly the curve that was originally forecast from that artifact.
        """
        model = self.load_model(h)
        return model.predict(times, z=z)