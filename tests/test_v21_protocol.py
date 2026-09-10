"""v2.1 tests: gossip manifests + deltas, peer trust, and DP noise."""

from __future__ import annotations

import csv
import json
import math
import os
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


def _contrib(alias, consts, var=1e-10, n_obs=10000):
    names = list(consts)
    coef = [0.0]
    for n in names:
        amp, ph = consts[n]
        k = math.radians(ph)
        coef += [amp * math.cos(k), amp * math.sin(k)]
    p = len(coef)
    cov = [[var if i == j and i else 0.0 for j in range(p)] for i in range(p)]
    return federation.StationContribution(
        alias=alias, constituents=names, coef=coef, covariance=cov,
        sigma2=var, n_obs=n_obs, residual_stats={"n_obs": float(n_obs)},
        residual_bias=None)


def _poisoned_bundle():
    # Wrong-but-confident constants: tight covariance, M2 x1.5 / +20 deg.
    b1 = federation._alias("peerB", "bad1")
    b2 = federation._alias("peerB", "bad2")
    stations = {
        b1: _contrib(b1, {"M2": (1.5, 30.0), "K1": (0.9, -10.0)}),
        b2: _contrib(b2, {"M2": (1.5, 32.0), "K1": (0.9, -8.0)}),
    }
    return federation.PeerBundle(
        peer_id="peerB", generated_at="2026-01-01T00:00:00+00:00",
        tideglass_version="2.1.0", stations=stations)


# -- gossip: manifests and deltas ------------------------------------------


def test_bundle_manifest_and_delta(tmp_path):
    store = tmp_path / "crowd"
    store.mkdir()
    _seed_store(store, [
        ("alpha", -122.34, 47.60, 10.0),
        ("bravo", -122.31, 47.59, 25.0),
    ])
    b1 = federation.build_peer_bundle(
        str(store), "peerA", generated_at="2026-01-01T00:00:00+00:00")
    mf = b1.manifest()
    assert mf["peer_id"] == "peerA"
    assert len(mf["stations"]) == 2
    assert len(mf["bundle_sha256"]) == 64

    # a re-export with newer metadata but identical constants is identical
    b2 = federation.build_peer_bundle(
        str(store), "peerA", generated_at="2030-06-01T00:00:00+00:00")
    assert b2.manifest()["bundle_sha256"] == mf["bundle_sha256"]
    assert b1.delta(b2).is_empty

    # a new station shows up as exactly one added alias
    ts = _hourly(T0, 240)
    hh = _tide(ts, _consts(40.0), noise=0.01, seed=77)
    _write_csv(str(store / "charlie.csv"), ts, hh)
    from tideglass.cli import main
    main(["contribute", str(store / "charlie.csv"), "-122.28", "47.58",
          "--station", "charlie", "--store", str(store)])
    b3 = federation.build_peer_bundle(str(store), "peerA")
    d = b2.delta(b3)
    assert d.added == [federation._alias("peerA", "charlie")]
    assert not d.updated and not d.removed and d.unchanged == 2

    # refitting a station with different data marks it updated, not added
    hh2 = _tide(ts, _consts(99.0), noise=0.01, seed=78)
    _write_csv(str(store / "alpha.csv"), ts, hh2)
    main(["contribute", str(store / "alpha.csv"), "-122.34", "47.60",
          "--station", "alpha", "--store", str(store)])
    b4 = federation.build_peer_bundle(str(store), "peerA")
    d2 = b3.delta(b4)
    assert d2.updated == [federation._alias("peerA", "alpha")]
    assert not d2.added and not d2.removed and d2.unchanged == 2


def test_ingest_pulls_only_changed(tmp_path):
    store = tmp_path / "crowd"
    store.mkdir()
    _seed_store(store, [
        ("alpha", -122.34, 47.60, 10.0),
        ("bravo", -122.31, 47.59, 25.0),
    ])
    b1 = federation.build_peer_bundle(
        str(store), "peerA", generated_at="2026-01-01T00:00:00+00:00")
    fed = federation.GlobalFederation(str(tmp_path / "federation"))
    d1 = fed.ingest(b1)
    assert sorted(d1.added) == sorted(b1.stations) and not d1.is_empty
    path = os.path.join(fed.peers_dir, "peerA.json")
    with open(path) as fh:
        saved1 = fh.read()

    # an identical re-export (newer timestamp, same constants): empty delta,
    # and the stored bundle is NOT rewritten
    b2 = federation.build_peer_bundle(
        str(store), "peerA", generated_at="2030-06-01T00:00:00+00:00")
    d2 = fed.ingest(b2)
    assert d2.is_empty
    with open(path) as fh:
        assert fh.read() == saved1

    # one new station: the delta names only it
    from tideglass.cli import main
    ts = _hourly(T0, 240)
    hh = _tide(ts, _consts(40.0), noise=0.01, seed=77)
    _write_csv(str(store / "charlie.csv"), ts, hh)
    main(["contribute", str(store / "charlie.csv"), "-122.28", "47.58",
          "--station", "charlie", "--store", str(store)])
    b3 = federation.build_peer_bundle(str(store), "peerA")
    d3 = fed.ingest(b3)
    assert d3.added == [federation._alias("peerA", "charlie")]
    assert d3.unchanged == 2
    with open(path) as fh:
        assert len(json.load(fh)["stations"]) == 3


# -- lineage survives federation -------------------------------------------


def test_lineage_survives_federation(tmp_path):
    store = tmp_path / "crowd"
    store.mkdir()
    _seed_store(store, [
        ("alpha", -122.34, 47.60, 10.0),
        ("bravo", -122.31, 47.59, 25.0),
    ])
    bundle = federation.build_peer_bundle(str(store), "peerA")
    originals = federation.GaugeStore(str(store)).models()
    for name, m in originals.items():
        c = bundle.stations[federation._alias("peerA", name)]
        assert c.lineage is not None
        assert c.lineage["data_sha256"] == m.meta["data_sha256"]
        assert c.lineage["chain"] == [m.meta["data_sha256"]]

    # save/load round-trips the lineage, and models() restores it into meta
    bundle.save(str(tmp_path / "b.json"))
    back = federation.PeerBundle.load(str(tmp_path / "b.json"))
    for a, c in back.stations.items():
        assert c.lineage == bundle.stations[a].lineage
    fed = federation.GlobalFederation(str(tmp_path / "federation"))
    fed.ingest(bundle)
    for key, m in fed.models().items():
        original = key.split(":", 1)[1]
        c0 = bundle.stations[original]
        assert m.meta["lineage"]["data_sha256"] == c0.lineage["data_sha256"]

    # a refit history extends the chain: old hash, then the current one
    m = list(originals.values())[0]
    m.meta["refit_history"] = [{"data_sha256": "ab" * 32}]
    c2 = federation._contribution_from_model(m, "x")
    assert c2.lineage["chain"] == ["ab" * 32, m.meta["data_sha256"]]


# -- peer-level trust -------------------------------------------------------


def _local_obs(tmp_path, stations):
    out = {}
    for name, ph in stations:
        ts = _hourly(T0, 120)
        hh = _tide(ts, _consts(ph), noise=0.01, seed=1000 + hash(name) % 997)
        p = tmp_path / f"{name}.csv"
        _write_csv(str(p), ts, hh)
        out[name] = (ts, list(hh))
    return out


def _three_peer_federation(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    _seed_store(a, [("a1", -122.34, 47.60, 10.0),
                    ("a2", -122.31, 47.59, 10.0)])
    bundle_a = federation.build_peer_bundle(str(a), "peerA")
    bundle_b = _poisoned_bundle()
    bundle_c = federation.build_peer_bundle(str(a), "peerC")  # duplicate of A
    fed = federation.GlobalFederation(str(tmp_path / "federation"))
    fed.ingest(bundle_a)
    fed.ingest(bundle_b)
    fed.ingest(bundle_c)
    return fed


def test_peer_trust_downweights_poison_and_duplicate(tmp_path):
    fed = _three_peer_federation(tmp_path)
    assert fed.n_peers == 3
    local = _local_obs(tmp_path, [("local1", 10.0), ("local2", 10.0)])
    scores = fed.peer_trust(local)
    assert sorted(scores) == ["peerA", "peerB", "peerC"]

    bad = scores["peerB"]
    # removing the poisoned peer improves YOUR held-out error
    assert bad.rmse_without < bad.rmse_with
    assert bad.relative_gain < 0.0
    assert bad.duplicate_fraction == 0.0
    assert bad.trust <= 0.1

    # the duplicate and its original split one copy's weight between them
    a, c = scores["peerA"], scores["peerC"]
    assert a.duplicate_fraction == 0.5
    assert c.duplicate_fraction == 0.5
    assert abs(a.trust - 0.5) < 0.05
    assert abs(c.trust - 0.5) < 0.05
    assert bad.trust < a.trust

    # trust expands to per-station pooling weights in (0, 1]
    w = fed.trust_weights(scores)
    assert len(w) == fed.n_stations
    assert all(0.0 < v <= 1.0 for v in w.values())
    for key, v in w.items():
        if key.startswith("peerB:"):
            assert v == bad.trust


def test_peer_trust_validation(tmp_path):
    fed = _three_peer_federation(tmp_path)
    local = _local_obs(tmp_path, [("local1", 10.0)])
    try:
        fed.peer_trust({})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty local_obs")
    try:
        fed.peer_trust(local, holdout=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for bad holdout")
    try:
        fed.peer_trust({"short": ([T0], [1.0])})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a too-short record")


# -- optional DP noise ------------------------------------------------------


def test_dp_noisify(tmp_path):
    store = tmp_path / "crowd"
    store.mkdir()
    _seed_store(store, [
        ("alpha", -122.34, 47.60, 10.0),
        ("bravo", -122.31, 47.59, 25.0),
    ])
    bundle = federation.build_peer_bundle(str(store), "peerA")

    noisy = federation.dp_noisify(bundle, 1.0, seed=7)
    dp = noisy.meta["dp"]
    assert dp["epsilon"] == 1.0 and dp["delta"] == 1e-5 and dp["clip_m"] == 5.0
    expect = 5.0 * math.sqrt(2.0 * math.log(1.25 / 1e-5)) / 1.0
    assert dp["sigma_m"] == expect
    # the constants moved (a genuine content change) and the manifest notices
    assert any(
        noisy.stations[a].coef != bundle.stations[a].coef
        for a in bundle.stations)
    assert (noisy.manifest()["bundle_sha256"]
            != bundle.manifest()["bundle_sha256"])
    # the source bundle is untouched
    assert "dp" not in bundle.meta

    # same seed reproduces, a different seed (or epsilon) does not
    again = federation.dp_noisify(bundle, 1.0, seed=7)
    for a in bundle.stations:
        assert again.stations[a].coef == noisy.stations[a].coef
    other = federation.dp_noisify(bundle, 1.0, seed=8)
    assert any(
        other.stations[a].coef != noisy.stations[a].coef
        for a in bundle.stations)
    # looser privacy = less noise
    loose = federation.dp_noisify(bundle, 10.0, seed=7)
    assert loose.meta["dp"]["sigma_m"] < noisy.meta["dp"]["sigma_m"]

    # gentle noise keeps predictions close to the original
    mild = federation.dp_noisify(bundle, 50.0, clip_m=2.0, seed=7)
    ts = _hourly(T0, 48)
    c0 = list(bundle.stations.values())[0]
    m0 = federation._model_from_contribution(c0, "orig")
    m1 = federation._model_from_contribution(
        mild.stations[c0.alias], "mild")
    e0 = np.asarray(m0.predict(ts).mean)
    e1 = np.asarray(m1.predict(ts).mean)
    assert float(np.sqrt(np.mean((e1 - e0) ** 2))) < 1.0

    for kwargs in ({"epsilon": 0.0}, {"epsilon": -1.0},
                   {"delta": 0.0}, {"delta": 1.0},
                   {"clip_m": 0.0}):
        try:
            federation.dp_noisify(bundle, **dict({"epsilon": 1.0}, **kwargs))
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


# -- CLI smoke ---------------------------------------------------------------


def test_cli_sync_merge_peer_delta(tmp_path, capsys):
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
    assert "manifest:" in out_text

    dest = tmp_path / "dest"
    rc = main(["merge", "--store", str(dest), str(out)])
    out_text = capsys.readouterr().out
    assert rc == 0
    assert "2 stations" in out_text
    assert "+2 added" in out_text
    # merging the same bundle again is recognized as unchanged
    rc = main(["merge", "--store", str(dest), str(out)])
    out_text = capsys.readouterr().out
    assert rc == 0
    assert "unchanged" in out_text

    rc = main(["peers", "--store", str(dest)])
    out_text = capsys.readouterr().out
    assert rc == 0
    assert "peer peerA" in out_text
    assert "lineage:" in out_text


def test_cli_peers_missing_store(tmp_path, capsys):
    from tideglass.cli import main

    rc = main(["peers", "--store", str(tmp_path / "empty")])
    assert rc == 2


def test_cli_sync_dp(tmp_path, capsys):
    from tideglass.cli import main

    store = tmp_path / "crowd"
    store.mkdir()
    _seed_store(store, [
        ("alpha", -122.34, 47.60, 10.0),
        ("bravo", -122.31, 47.59, 25.0),
    ])
    plain = tmp_path / "plain.json"
    main(["sync", "--store", str(store), "--peer-id", "peerA",
          "--out", str(plain)])
    capsys.readouterr()
    noisy_p = tmp_path / "noisy.json"
    rc = main(["sync", "--store", str(store), "--peer-id", "peerA",
               "--out", str(noisy_p), "--epsilon", "1.0", "--dp-seed", "7"])
    out_text = capsys.readouterr().out
    assert rc == 0
    assert "dp:" in out_text
    b0 = federation.PeerBundle.load(str(plain))
    b1 = federation.PeerBundle.load(str(noisy_p))
    assert b1.meta["dp"]["epsilon"] == 1.0
    assert any(b1.stations[a].coef != b0.stations[a].coef
               for a in b0.stations)


def test_cli_peers_trust(tmp_path, capsys):
    from tideglass.cli import main

    a = tmp_path / "a"
    a.mkdir()
    _seed_store(a, [("a1", -122.34, 47.60, 10.0),
                    ("a2", -122.31, 47.59, 10.0)])
    bundle_a = federation.build_peer_bundle(str(a), "peerA")
    bundle_c = federation.build_peer_bundle(str(a), "peerC")
    bundle_a.save(str(tmp_path / "pa.json"))
    _poisoned_bundle().save(str(tmp_path / "pb.json"))
    bundle_c.save(str(tmp_path / "pc.json"))

    dest = tmp_path / "dest"
    rc = main(["merge", "--store", str(dest),
               str(tmp_path / "pa.json"),
               str(tmp_path / "pb.json"),
               str(tmp_path / "pc.json")])
    assert rc == 0
    capsys.readouterr()

    local = _local_obs(tmp_path, [("local1", 10.0), ("local2", 10.0)])
    l1, l2 = str(tmp_path / "local1.csv"), str(tmp_path / "local2.csv")
    rc = main(["peers", "--store", str(dest), "--obs", l1, l2])
    out_text = capsys.readouterr().out
    assert rc == 0
    assert "peer peerB" in out_text
    assert "down-weighted" in out_text
