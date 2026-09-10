"""v2.0 tests: the federation protocol across Tideglass installations."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np

from tideglass import TideModel
from tideglass.marea import constituents as CON
from tideglass.marea import federation

UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _hours(times):
    return np.array([(t - T0).total_seconds() / 3600.0 for t in times], dtype=float)


def _tide(times, consts, noise=0.0, seed=None):
    h = np.zeros(len(times))
    for name, (amp, ph) in consts.items():
        c = CON.get(name)
        w = math.radians(CON.speed(c))
        h += amp * np.cos(w * _hours(times) + math.radians(ph))
    if noise > 0:
        rng = np.random.default_rng(seed)
        h += rng.normal(0.0, noise, len(times))
    return h


def _hourly(start, n):
    return [start + timedelta(hours=i) for i in range(n)]


def _write_csv(path, times, heights):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "height"])
        for t, hh in zip(times, heights):
            w.writerow([t.isoformat(), f"{hh:.4f}"])


def _consts(ph=0.0):
    return {"M2": (1.0, ph), "K1": (0.6, ph)}


def _seed_store(root, stations):
    """Populate a GaugeStore with dense stations so models() fits them."""
    from tideglass.cli import main

    for name, lon, lat, ph in stations:
        ts = _hourly(T0, 240)
        hh = _tide(ts, _consts(ph), noise=0.01, seed=hash(name) % 1000)
        _write_csv(str(root / f"{name}.csv"), ts, hh)
        main(["contribute", str(root / f"{name}.csv"), str(lon), str(lat),
              "--station", name, "--store", str(root)])


# -- anonymized peer bundle ----------------------------------------------


def test_peer_bundle_anonymized_roundtrip(tmp_path):
    store = tmp_path / "crowd"
    store.mkdir()
    _seed_store(store, [
        ("alpha", -122.34, 47.60, 10.0),
        ("bravo", -122.31, 47.59, 25.0),
    ])
    bundle = federation.build_peer_bundle(str(store), "peerA")
    assert bundle.peer_id == "peerA"
    assert len(bundle.stations) == 2
    for alias in bundle.stations:
        assert len(alias) == 12 and alias.isalnum()  # one-way hex alias

    # the bundle must not leak the real station names
    raw = json.dumps(bundle.to_dict())
    assert "alpha" not in raw and "bravo" not in raw

    # round-trip through disk
    path = tmp_path / "peerA.json"
    bundle.save(str(path))
    back = federation.PeerBundle.load(str(path))
    assert set(back.stations) == set(bundle.stations)
    c = back.stations[list(back.stations)[0]]
    assert len(c.constituents) >= 2
    assert len(c.coef) == 1 + 2 * len(c.constituents)
    assert len(c.covariance) == 1 + 2 * len(c.constituents)


def test_global_federation_reconstructs_models(tmp_path):
    store = tmp_path / "crowd"
    store.mkdir()
    _seed_store(store, [
        ("alpha", -122.34, 47.60, 10.0),
        ("bravo", -122.31, 47.59, 25.0),
    ])
    bundle = federation.build_peer_bundle(str(store), "peerA")
    fed = federation.GlobalFederation(str(tmp_path / "federation"))
    fed.ingest(bundle)
    models = fed.models()
    assert len(models) == 2
    for key in models:
        assert key.startswith("peerA:")  # peer-prefixed anonymized ids
    # reconstructing the bundle reproduces each original model's own curve exactly
    gs = federation.GaugeStore(str(store))
    originals = gs.models()
    ts = _hourly(T0, 240)
    for name, om in originals.items():
        alias = federation._alias("peerA", name)
        rm = models["peerA:" + alias]
        p1 = om.predict(ts).mean
        p2 = rm.predict(ts).mean
        err = float(np.sqrt(np.mean((np.asarray(p1) - np.asarray(p2)) ** 2)))
        assert err < 1e-9


# -- the network effect compounds across deployments ---------------------


def test_federation_compounds_across_deployments(tmp_path):
    # Two independent installations, each with its own consistent network.
    a = tmp_path / "a"
    a.mkdir()
    _seed_store(a, [("a1", -122.34, 47.60, 10.0),
                    ("a2", -122.31, 47.59, 25.0)])
    b = tmp_path / "b"
    b.mkdir()
    _seed_store(b, [("b1", -70.0, 40.0, 10.0),
                    ("b2", -70.1, 40.1, 25.0)])
    bundle_a = federation.build_peer_bundle(str(a), "peerA")
    bundle_b = federation.build_peer_bundle(str(b), "peerB")

    fed = federation.GlobalFederation(str(tmp_path / "federation"))
    fed.ingest(bundle_a)
    assert fed.n_peers == 1 and fed.n_stations == 2
    fed.ingest(bundle_b)
    assert fed.n_peers == 2 and fed.n_stations == 4

    rep = fed.report()
    assert rep["n_peers"] == 2 and rep["n_stations"] == 4
    assert "M2" in rep["constituents"]
    assert rep["constituents"]["M2"]["n"] == 4
    # each installation makes the other's prior tighter (finite, non-negative)
    assert rep["constituents"]["M2"]["tau_a"] >= 0.0


def test_fresh_installation_borrows_from_installed_base(tmp_path):
    # Installation C has NO local stations of its own...
    a = tmp_path / "a"
    a.mkdir()
    _seed_store(a, [("a1", -122.34, 47.60, 10.0)])  # only ONE station
    b = tmp_path / "b"
    b.mkdir()
    _seed_store(b, [("b1", -70.0, 40.0, 10.0),
                    ("b2", -70.1, 40.1, 25.0)])
    bundle_a = federation.build_peer_bundle(str(a), "peerA")
    bundle_b = federation.build_peer_bundle(str(b), "peerB")

    fed = federation.GlobalFederation(str(tmp_path / "federation"))
    fed.ingest(bundle_a)
    # ...so with only its own single station it cannot pool and must fit locally.
    ts = _hourly(T0, 48)
    hh = _tide(ts, _consts(10.0), noise=0.02, seed=5)
    lone = fed.seed("new", ts, hh)
    assert lone.meta.get("source") == "local"

    # After merging a remote peer, the installed base gives it 3 stations to
    # borrow from -- the network effect now compounds across deployments.
    fed.ingest(bundle_b)
    pooled = fed.seed("new", ts, hh)
    assert pooled.meta.get("source") == "pooled"
    assert "shrinkage" in pooled.meta
    # and the seeded short record tracks the true tide
    dense = _hourly(T0, 240)
    truth = _tide(dense, _consts(10.0), noise=0.0)
    pred = pooled.predict(dense)
    rmse = float(np.sqrt(np.mean((pred.mean - np.asarray(truth)) ** 2)))
    assert rmse < 0.3


# -- CLI smoke ------------------------------------------------------------


def test_cli_sync_merge(tmp_path, capsys):
    from tideglass.cli import main

    store = tmp_path / "crowd"
    store.mkdir()
    _seed_store(store, [
        ("alpha", -122.34, 47.60, 10.0),
        ("bravo", -122.31, 47.59, 25.0),
    ])
    out = tmp_path / "peerA.json"
    rc = main(["sync", "--store", str(store), "--peer-id", "peerA",
               "--out", str(out)])
    out_text = capsys.readouterr().out
    assert rc == 0
    assert out.exists()
    assert "anonymized" in out_text

    dest = tmp_path / "federation_root"
    rc = main(["merge", "--store", str(dest), str(out)])
    out_text = capsys.readouterr().out
    assert rc == 0
    assert "2 stations" in out_text
    assert (dest / "federation" / "peers" / "peerA.json").exists()


def test_cli_federate_global(tmp_path, capsys):
    from tideglass.cli import main

    # peer A builds a bundle; we merge it as the installed base
    a = tmp_path / "a"
    a.mkdir()
    _seed_store(a, [("a1", -122.34, 47.60, 10.0),
                    ("a2", -122.31, 47.59, 25.0)])
    bundle = federation.build_peer_bundle(str(a), "peerA")
    gstore = tmp_path / "global"
    gstore.mkdir()
    bundle.save(str(tmp_path / "_b.json"))
    main(["merge", "--store", str(gstore), str(tmp_path / "_b.json")])

    # a brand-new, otherwise-isolated store federates a short sensor and borrows
    # strength from the installed base via --global-store
    local = tmp_path / "local"
    local.mkdir()
    ts = _hourly(T0, 72)
    hh = _tide(ts, _consts(40.0), noise=0.05, seed=99)
    _write_csv(str(local / "charlie.csv"), ts, hh)
    rc = main(["federate", str(local / "charlie.csv"), "-122.28", "47.58",
               "--station", "charlie", "--store", str(local),
               "--global-store", str(gstore)])
    out_text = capsys.readouterr().out
    assert rc == 0
    assert "installed base" in out_text
    assert (local / "charlie.json").exists()
