"""v2.2 tests: content-addressed artifact store and bit-for-bit reproduction."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np

from tideglass import TideModel
from tideglass.marea import constituents as CON
from tideglass.marea.cas import ArtifactStore, _record, artifact_hash

UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _hours(times):
    return np.array([(t - T0).total_seconds() / 3600.0 for t in times], dtype=float)


def _tide(times, consts, noise=0.0, seed=None, m0=0.7):
    h = np.full(len(times), m0)
    for name, (amp, ph) in consts.items():
        c = CON.get(name)
        w = math.radians(CON.speed(c))
        h += amp * np.cos(w * _hours(times) + math.radians(ph))
    if noise > 0:
        h += np.random.default_rng(seed).normal(0.0, noise, len(times))
    return h


def _hourly(start, n):
    return [start + timedelta(hours=i) for i in range(n)]


def _consts(ph=0.0):
    return {"M2": (1.0, ph), "K1": (0.6, ph + 30.0)}


def _fit(times=None, ph=0.0, station="test"):
    times = times or _hourly(T0, 30 * 24)
    model = TideModel.fit(times, _tide(times, _consts(ph), noise=0.01, seed=5),
                          station=station, source=f"csv:test{ph}.csv")
    return model


# -- hashing ------------------------------------------------------------------


def test_artifact_hash_is_content_addressed():
    m = _fit()
    rec = _record(m)
    assert artifact_hash(rec) == artifact_hash(dict(rec))  # same content, same key
    assert len(artifact_hash(rec)) == 64
    # mutating any fitted coefficient changes the key (tamper detection)
    mutated = dict(rec)
    mutated["coef"] = list(rec["coef"])
    mutated["coef"][0] = mutated["coef"][0] + 0.001
    assert artifact_hash(mutated) != artifact_hash(rec)


# -- store + lineage ----------------------------------------------------------


def test_store_dedups_identical_content(tmp_path):
    store = ArtifactStore(str(tmp_path / "cas"))
    m = _fit()
    h1 = store.store(m, station="T")
    h2 = store.store(m, station="T")
    assert h1 == h2
    assert len(store.history("T")) == 1
    assert store.verify(h1)


def test_store_records_obs_lineage(tmp_path):
    store = ArtifactStore(str(tmp_path / "cas"))
    m = _fit(station="SF")
    h = store.store(m)
    li = store.lineage_of(h)
    assert li["obs"]["data_sha256"] == m.meta["data_sha256"]
    assert li["obs"]["obs_start"] == m.meta["obs_start"]
    assert li["obs"]["obs_end"] == m.meta["obs_end"]
    assert li["obs"]["n_obs"] == m.meta["n_obs"]
    assert li["peers"] == []
    assert store.station_of(h) == "SF"


def test_store_chains_parent_across_refits(tmp_path):
    store = ArtifactStore(str(tmp_path / "cas"))
    h1 = store.store(_fit(ph=0.0), station="T")
    h2 = store.store(_fit(ph=15.0), station="T")
    assert h1 != h2
    assert store.latest("T") == h2
    assert store.lineage_of(h2)["parent"] == h1
    assert [e["hash"] for e in store.history("T")] == [h1, h2]


def test_store_records_peer_hashes(tmp_path):
    store = ArtifactStore(str(tmp_path / "cas"))
    h = store.store(_fit(), station="T", peer_hashes=["ab" * 32, "cd" * 32])
    assert sorted(store.lineage_of(h)["peers"]) == ["ab" * 32, "cd" * 32]


# -- reproducibility ----------------------------------------------------------


def test_reproduce_bit_for_bit(tmp_path):
    store = ArtifactStore(str(tmp_path / "cas"))
    m = _fit()
    h = store.store(m, station="T")
    assert store.verify(h)
    times = _hourly(T0 + timedelta(days=40), 48)
    rebuilt = store.reproduce(h, times)
    original = m.predict(times)
    assert np.max(np.abs(rebuilt.mean - original.mean)) == 0.0
    assert np.max(np.abs(rebuilt.lower - original.lower)) == 0.0
    assert np.max(np.abs(rebuilt.upper - original.upper)) == 0.0


def test_verify_detects_tampering(tmp_path):
    store = ArtifactStore(str(tmp_path / "cas"))
    h = store.store(_fit(), station="T")
    blob = store.get(h)
    blob["coef"][0] = blob["coef"][0] + 0.5
    with open(store._blob_path(h), "w") as fh:
        json.dump(blob, fh)
    assert not store.verify(h)


def test_resolve_full_and_prefix(tmp_path):
    store = ArtifactStore(str(tmp_path / "cas"))
    h1 = store.store(_fit(ph=0.0), station="A")
    store.store(_fit(ph=15.0), station="B")
    assert store.resolve(h1) == h1
    assert store.resolve(h1[:12]) == h1
    assert store.resolve(h1[:8]) == h1
    assert store.resolve("0" * 12) is None  # nonexistent key
    assert store.resolve("bad") is None     # too short to resolve


def test_load_model_restores_exact_fit(tmp_path):
    store = ArtifactStore(str(tmp_path / "cas"))
    m = _fit()
    h = store.store(m, station="T")
    times = _hourly(T0 + timedelta(days=30), 24)
    loaded = store.load_model(h)
    # exact coefficients are restored, so means match bit-for-bit (not a
    # polar amplitude/phase round trip like load_harmonic)
    assert np.max(np.abs(loaded.predict(times).mean - m.predict(times).mean)) == 0.0


# -- CLI ----------------------------------------------------------------------


def test_cli_fit_stores_cas_and_reproduce(tmp_path, capsys):
    from tideglass.cli import main

    train = _hourly(T0, 30 * 24)
    csv = tmp_path / "obs.csv"
    with open(csv, "w") as fh:
        fh.write("time,height\n")
        fh.writelines(f"{t.isoformat()},{h:.4f}\n"
                      for t, h in zip(train, _tide(train, _consts(),
                                                   noise=0.01, seed=7)))
    store = str(tmp_path / "store")
    rc = main(["fit", str(csv), "--station", "T", "--store", store,
               "--alpha", "1e-4"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "cas:" in out
    cas = ArtifactStore(str(tmp_path / "store" / "cas"))
    h = cas.latest("T")
    assert h and cas.verify(h)

    capsys.readouterr()
    rc = main(["reproduce", h, "--store", str(tmp_path / "store"),
               "--date", "2024-02-01", "--days", "1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "hash:" in out and "bit-for-bit" in out and "lineage" in out
    rows = [ln for ln in out.splitlines()
            if not ln.startswith("#") and len(ln.split()) == 4
            and ln.split()[0].startswith(("20", "19"))]
    assert len(rows) == 24

    # short (16-hex) prefix resolves and reproduces too
    capsys.readouterr()
    assert main(["reproduce", h[:16], "--store", str(tmp_path / "store")]) == 0


def test_cli_reproduce_missing_hash(tmp_path, capsys):
    from tideglass.cli import main

    assert main(["reproduce", "deadbeef", "--store",
                 str(tmp_path / "empty")]) == 2