"""Tests for v3.2 — relay discovery and one-command joining."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone

import numpy as np

from tideglass import RelayPeer, TideModel
from tideglass.cli import main as cli_main
from tideglass.marea import constituents as CON
from tideglass.marea import federation
from tideglass.marea.federation import _relay_client

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
        h += np.random.default_rng(seed).normal(0.0, noise, len(times))
    return h


def _hourly(start, n):
    return [start + timedelta(hours=i) for i in range(n)]


def _write_csv(path, times, heights):
    with open(path, "w", newline="") as fh:
        fh.write("time,height\n")
        for t, h in zip(times, heights):
            fh.write(f"{t.isoformat()},{h:.6f}\n")


def _consts(ph=0.0):
    return {"M2": (1.0, ph), "K1": (0.6, ph)}


def _seed_store(root, stations):
    for name, lon, lat, ph in stations:
        ts = _hourly(T0, 240)
        hh = _tide(ts, _consts(ph), noise=0.01, seed=hash(name) % 1000)
        _write_csv(root / f"{name}.csv", ts, hh)
        cli_main([
            "contribute", str(root / f"{name}.csv"), str(lon), str(lat),
            "--station", name, "--store", str(root),
        ])


def _contrib(alias, consts, n_obs=10000):
    names = list(consts)
    coef = [0.0]
    for name in names:
        amp, ph = consts[name]
        k = math.radians(ph)
        coef += [amp * math.cos(k), amp * math.sin(k)]
    p = len(coef)
    return federation.StationContribution(
        alias=alias,
        constituents=names,
        coef=coef,
        covariance=[[1e-8 if i == j and i else 0.0 for j in range(p)]
                    for i in range(p)],
        sigma2=1e-8,
        n_obs=n_obs,
        residual_stats={"rmse": 1e-4, "n_obs": n_obs},
        residual_bias=None,
    )


def test_relay_peer_roundtrip(tmp_path):
    relay = RelayPeer(str(tmp_path))
    store = tmp_path / "source"
    store.mkdir()
    _seed_store(store, [("alpha", -122.34, 47.60, 10.0)])
    bundle = federation.build_peer_bundle(str(store), "peerA")

    digest = relay.push(bundle)
    assert digest == bundle.manifest()["bundle_sha256"]
    assert relay.manifest("peerA") == bundle.manifest()
    assert relay.manifest_all()["peerA"]["peer_id"] == "peerA"
    assert relay.pull("peerA") == bundle
    assert not relay.stale("peerA", 30.0)


def test_relay_staleness_and_prune(tmp_path):
    relay = RelayPeer(str(tmp_path))
    old = federation.PeerBundle(
        peer_id="old",
        generated_at="2000-01-01T00:00:00+00:00",
        tideglass_version="3.0.0",
        stations={"abc123456789": _contrib("abc123456789", _consts())},
    )
    relay.push(old)
    assert relay.stale("old", 30.0)
    assert relay.prune(30.0) == ["old"]
    assert relay.pull("old") is None


def test_relay_client_file_uri_and_http_rejection(tmp_path):
    relay = _relay_client((tmp_path / "relay").as_uri())
    assert os.path.abspath(relay.root) == os.path.abspath(tmp_path / "relay")
    try:
        _relay_client("https://relay.example.test/peers")
    except ValueError as exc:
        assert "not implemented" in str(exc)
    else:
        raise AssertionError("HTTP relay endpoint was accepted")


def test_join_pulls_seeds_and_exports(tmp_path, capsys):
    remote = tmp_path / "remote"
    remote.mkdir()
    _seed_store(remote, [
        ("alpha", -122.34, 47.60, 10.0),
        ("bravo", -122.31, 47.59, 25.0),
    ])
    bundle = federation.build_peer_bundle(str(remote), "peerA")
    relay = RelayPeer(str(tmp_path / "relay"))
    relay.push(bundle)

    local = tmp_path / "local"
    local.mkdir()
    short = _hourly(T0, 24)
    heights = _tide(short, _consts(12.0), noise=0.02, seed=7)
    _write_csv(local / "charlie.csv", short, heights)
    model = TideModel.fit(
        short, heights, auto_select=False, station="charlie",
        source="csv:charlie.csv",
    )
    with open(local / "charlie.json", "w") as fh:
        json.dump(model.to_artifact(), fh)
    state_path = local / "charlie.nowcast.json"
    from tideglass.marea.nowcast import NowcastEngine, save_state

    engine = NowcastEngine(model)
    engine.update(short, heights)
    save_state(engine, state_path)

    coords = tmp_path / "coords.csv"
    with open(coords, "w", newline="") as fh:
        fh.write("station,lon,lat\ncharlie,-122.28,47.58\n")
    rc = cli_main([
        "join", "--relay", (tmp_path / "relay").as_uri(),
        "--peer-id", "peerB", "--store", str(local),
        "--coords", str(coords), "--seed-short-min", "48",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "pulled: 1" in out
    assert "reseeded: 1" in out
    assert "exported: local bundle published" in out

    model = TideModel.load_harmonic(str(local / "charlie.json"))
    assert model.meta.get("source") == "join-reseed"
    assert model.meta.get("pool_source") == "pooled"
    assert model.meta.get("n_obs") == len(short)
    assert model.meta.get("seeded_from") == ["peerA"]
    assert "shrinkage" in model.meta
    assert (local / "charlie.nowcast.json").exists()
    relay_bundle = relay.pull("peerB")
    assert relay_bundle is not None
    assert relay_bundle.peer_id == "peerB"
    assert len(relay_bundle.stations) == 1


def test_join_skips_stale_bundle(tmp_path, capsys):
    relay = RelayPeer(str(tmp_path / "relay"))
    bundle = federation.PeerBundle(
        peer_id="stale",
        generated_at="2000-01-01T00:00:00+00:00",
        tideglass_version="3.0.0",
        stations={"abc123456789": _contrib("abc123456789", _consts())},
    )
    relay.push(bundle)
    store = tmp_path / "store"
    store.mkdir()
    rc = cli_main([
        "join", "--relay", (tmp_path / "relay").as_uri(),
        "--peer-id", "local", "--store", str(store),
        "--max-age-days", "30", "--no-export",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "stale_skipped: 1" in out
    assert "skipped: 1 stale" in out
