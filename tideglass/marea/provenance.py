"""Data versioning for Marea Core model artifacts.

Every published/inferred model artifact pins the exact observation window and
source it was fit on, plus a content hash of the training data, so a model can
always be audited, reproduced, or detected as stale when new observations
arrive (the v0.6 operational flywheel).

The hash is computed over the *canonicalised* ``(time, height)`` pairs (sorted
by time, fixed-precision text) so that re-fitting the same data reproduces the
identical digest regardless of row order or float formatting.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any


def _version() -> str:
    """Best-effort package version (falls back to ``unknown``)."""
    try:
        from importlib.metadata import version

        return version("tideglass")
    except Exception:  # noqa: BLE001 - version is cosmetic, never fatal
        return "unknown"


def canonical_digest(times: Sequence[datetime], heights) -> str:
    """Deterministic SHA-256 over the observation set.

    Rows are sorted by timestamp and rendered as ``<ISO>;<height>`` with the
    height fixed to 6 decimal places, so reordering or reformatting the input
    does not change the digest.
    """
    pairs = sorted(
        (t, float(h))
        for t, h in zip(times, heights)
    )
    if not pairs:
        return hashlib.sha256(b"<empty>").hexdigest()
    lines = [
        f"{(t.astimezone(timezone.utc) if t.tzinfo else t.replace(tzinfo=timezone.utc)).isoformat()};{h:.6f}"
        .encode("utf-8")
        for t, h in pairs
    ]
    return hashlib.sha256(b"\n".join(lines)).hexdigest()


def provenance(
    times: Sequence[datetime],
    heights,
    source: str | None = None,
    station: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the provenance block for a model fit on ``times``/``heights``.

    :param source: human-readable origin tag (e.g. ``noaa-coops:9414290`` or
        ``csv:obs.csv``); defaults to ``unspecified``.
    :param station: station id/name the data belongs to.
    :param extra: additional keys merged verbatim into the block.
    """
    times = list(times)
    y = list(heights)
    if len(times) != len(y):
        raise ValueError(f"{len(times)} times but {len(y)} heights")
    if not times:
        raise ValueError("cannot build provenance for empty data")
    ordered = sorted(times)
    start = ordered[0]
    end = ordered[-1]
    return {
        "source": source or "unspecified",
        "station": station,
        "obs_start": (start.astimezone(timezone.utc)
                      if start.tzinfo else start.replace(tzinfo=timezone.utc)).isoformat(),
        "obs_end": (end.astimezone(timezone.utc)
                    if end.tzinfo else end.replace(tzinfo=timezone.utc)).isoformat(),
        "n_obs": len(times),
        "data_sha256": canonical_digest(times, y),
        "tideglass_version": _version(),
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    }


def merge_provenance(
    meta: dict[str, Any] | None,
    sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Combine provenance blocks when a model is refined/updated in place.

    Keeps the original ``obs_start`` (the data the model was first fit on) and
    appends the new feed windows to ``refit_history`` so the full lineage is
    recorded. Returns a fresh dict; ``meta`` is not mutated.
    """
    meta = dict(meta or {})
    if sources:
        history = list(meta.get("refit_history", []))
        history.extend(sources)
        meta["refit_history"] = history
        if history:
            meta["source"] = history[-1].get("source", meta.get("source"))
    return meta


def station_window(meta: dict[str, Any]) -> tuple[str, str] | None:
    """Return ``(obs_start, obs_end)`` ISO strings from a provenance block."""
    s, e = meta.get("obs_start"), meta.get("obs_end")
    if s is None or e is None:
        return None
    return s, e
