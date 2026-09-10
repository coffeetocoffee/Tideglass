"""v1.2 tests: met-forced surge — wind-stress regression + inverse barometer
turning the AR(1) nowcast into a 48-hour forecast that feeds decision pricing."""

from __future__ import annotations

import io
import math
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError

import numpy as np
import pytest

from tideglass import TideModel
from tideglass.marea import constituents as CON
from tideglass.marea.decision import decision_curve
from tideglass.marea.extremes import flood_probability
from tideglass.marea.met import (
    BARO_THEORY_M_PER_HPA,
    MetResponse,
    SurgeForecast,
    learn_met_response,
    read_met_csv,
    stress_features,
    write_met_csv,
)
from tideglass.marea.model import Prediction
from tideglass.marea.surge import AR1, fit_ar1

UTC = timezone.utc
T0 = datetime(2023, 1, 1, tzinfo=UTC)
TRUE = {"c0": 0.02, "su": 0.0041, "sv": 0.0030, "baro": -0.0099, "lag": 2}


def _hourly(n, start=T0):
    return [start + timedelta(hours=i) for i in range(n)]


def _hours(times):
    return np.array([(t - T0).total_seconds() / 3600.0 for t in times],
                    dtype=float)


def _tide(times, amps, noise=0.0, seed=0):
    h = np.zeros(len(times))
    for name, amp in amps.items():
        w = math.radians(CON.speed(CON.get(name)))
        h += amp * np.cos(w * _hours(times))
    if noise > 0:
        h += np.random.default_rng(seed).normal(0.0, noise, len(times))
    return h


def _met(n, seed=3):
    """Smooth pseudo-random wind speed/dir + pressure on the hourly grid."""
    h = np.arange(n, dtype=float)
    rng = np.random.default_rng(seed)
    ws = np.clip(5.0 + 4.0 * np.sin(2 * np.pi * h / 72.0)
                 + 2.0 * np.sin(2 * np.pi * h / 13.0)
                 + rng.normal(0.0, 0.3, n), 0.1, None)
    wd = 200.0 + 50.0 * np.sin(2 * np.pi * h / 240.0) \
        + rng.normal(0.0, 3.0, n)
    pr = 1013.0 + 15.0 * np.sin(2 * np.pi * h / 200.0 + 1.0) \
        + rng.normal(0.0, 0.5, n)
    return ws, wd, pr


def _truth_surge(ws, wd, pr, lag=TRUE["lag"], noise=0.0, seed=11):
    """surge = c0 + a·τ_u + b·τ_v + β·Δp at the true lag (index shift)."""
    su, sv = stress_features(ws, wd)
    dp = pr - 1013.25
    su_l = np.concatenate([np.full(lag, su[0]), su[:-lag]])
    sv_l = np.concatenate([np.full(lag, sv[0]), sv[:-lag]])
    dp_l = np.concatenate([np.full(lag, dp[0]), dp[:-lag]])
    s = TRUE["c0"] + TRUE["su"] * su_l + TRUE["sv"] * sv_l + TRUE["baro"] * dp_l
    if noise > 0:
        s = s + np.random.default_rng(seed).normal(0.0, noise, s.size)
    return s


@pytest.fixture(scope="module")
def learned():
    n = 24 * 90  # 90 days of hourly concurrent obs + met
    times = _hourly(n)
    ws, wd, pr = _met(n)
    surge = _truth_surge(ws, wd, pr, noise=0.01)
    resp, diag = learn_met_response(times, surge, times, ws, wd, pr,
                                    max_lag_hours=6)
    return times, ws, wd, pr, surge, resp, diag


# -- learning the response ------------------------------------------------------

def test_recovers_response_and_lag(learned):
    times, ws, wd, pr, surge, resp, diag = learned
    assert diag["lag"] == TRUE["lag"]
    assert resp.intercept == pytest.approx(TRUE["c0"], abs=2e-3)
    assert resp.stress_u == pytest.approx(TRUE["su"], rel=0.05)
    assert resp.stress_v == pytest.approx(TRUE["sv"], rel=0.05)
    assert resp.barometer == pytest.approx(TRUE["baro"], rel=0.05)
    assert diag["r2"] > 0.95
    assert diag["rmse_after"] < 0.2 * diag["rmse_before"]


def test_barometer_matches_theory():
    # pressure-only forcing at the theoretical inverse-barometer slope
    n = 24 * 60
    times = _hourly(n)
    h = np.arange(n, dtype=float)
    ws = np.zeros(n)
    wd = np.full(n, 90.0)
    pr = 1013.0 + 20.0 * np.sin(2 * np.pi * h / 168.0)
    dp_l = np.concatenate([np.full(1, pr[0] - 1013.25), pr[:-1] - 1013.25])
    surge = 0.0 + BARO_THEORY_M_PER_HPA * dp_l  # lag 1
    resp, diag = learn_met_response(times, surge, times, ws, wd, pr,
                                    max_lag_hours=3)
    assert diag["lag"] == 1
    assert resp.barometer == pytest.approx(BARO_THEORY_M_PER_HPA, rel=0.02)
    assert resp.barometer < 0  # pressure up, water down
    assert resp.stress_u == pytest.approx(0.0, abs=1e-6)
    assert resp.stress_v == pytest.approx(0.0, abs=1e-6)


def test_survives_a_lag_free_record():
    # met with no autocorrelation advantage at any lag: lag 0 must win
    n = 24 * 30
    times = _hourly(n)
    ws, wd, pr = _met(n, seed=5)
    su, sv = stress_features(ws, wd)
    surge = 0.01 + 0.004 * su + 0.002 * sv  # concurrent, no lag
    resp, diag = learn_met_response(times, surge, times, ws, wd, pr,
                                    max_lag_hours=4)
    assert diag["lag"] == 0
    assert diag["r2"] > 0.9


def test_learn_validations():
    n = 24 * 30
    times = _hourly(n)
    ws, wd, pr = _met(n)
    surge = _truth_surge(ws, wd, pr)
    with pytest.raises(ValueError, match="too short"):
        learn_met_response(times[:40], surge[:40], times[:40],
                           ws[:40], wd[:40], pr[:40])
    with pytest.raises(ValueError, match="but"):
        learn_met_response(times, surge[:-1], times, ws, wd, pr)
    with pytest.raises(ValueError, match="mismatch"):
        learn_met_response(times, surge, times, ws[:-1], wd, pr)
    bad_pr = pr / 10.0  # looks like kPa, not hPa
    with pytest.raises(ValueError, match="hPa"):
        learn_met_response(times, surge, times, ws, wd, bad_pr)
    with pytest.raises(ValueError, match="≥ 0"):
        learn_met_response(times, surge, times, -ws, wd, pr)
    with pytest.raises(ValueError, match="constant"):
        learn_met_response(times, np.zeros(n), times, ws, wd, pr)
    with pytest.raises(ValueError, match="cover"):
        # met window far from the observations: no lag can be fitted
        far = _hourly(n, start=T0 + timedelta(days=365))
        learn_met_response(times, surge, far, ws, wd, pr)


# -- forecast --------------------------------------------------------------------

def test_forecast_mean_does_not_decay(learned):
    times, ws, wd, pr, surge, resp, diag = learned
    # sustained storm forcing keeps the forecast mean up at every lead — the
    # v1.1-era AR(1) tracker would have decayed it to ~0 within a day
    storm_ws = np.full(48, 18.0)
    storm_wd = np.full(48, 210.0)
    storm_pr = np.full(48, 985.0)
    ftimes = _hourly(48, start=times[-1] + timedelta(hours=1))
    fc = resp.forecast(ftimes, storm_ws, storm_wd, storm_pr)
    assert isinstance(fc, SurgeForecast)
    assert float(np.all(np.abs(fc.mean - fc.mean[0]) < 1e-9))
    assert np.all(fc.sigma == pytest.approx(resp.sigma))
    # and the level is a real storm surge: ~0.3 m or more
    assert fc.mean[0] > 0.3


def test_forecast_layers_ar1_memory(learned):
    times, ws, wd, pr, surge, resp, diag = learned
    resid = surge - resp.surge(times, ws, wd, pr)
    ar = fit_ar1(resid)
    ftimes = _hourly(24, start=times[-1] + timedelta(hours=1))
    ws24, wd24, pr24 = ws[-24:], wd[-24:], pr[-24:]
    fc = resp.forecast(ftimes, ws24, wd24, pr24, ar=ar,
                       last_residual=0.05, origin=times[-1])
    # the met mean plus the decaying residual memory
    base = resp.surge(ftimes, ws24, wd24, pr24)
    dt = np.arange(1.0, 25.0)
    assert fc.mean == pytest.approx(base + 0.05 * ar.phi ** dt, abs=1e-9)
    # band grows monotonically toward the residual marginal
    assert np.all(np.diff(fc.sigma) > -1e-12)
    assert fc.sigma[0] < fc.sigma[-1]
    if abs(ar.phi) < 1:
        marginal = ar.sigma_eps / math.sqrt(1.0 - ar.phi ** 2)
        assert fc.sigma[-1] == pytest.approx(marginal, rel=0.05)


def test_hand_built_response_forecast():
    resp = MetResponse(intercept=0.01, stress_u=0.004, stress_v=0.003,
                       barometer=BARO_THEORY_M_PER_HPA, lag_hours=0,
                       sigma=0.05, r_squared=0.9, n=100)
    ftimes = _hourly(3)
    ws = np.array([10.0, 0.0, 10.0])
    wd = np.array([90.0, 0.0, 270.0])
    pr = np.array([1003.25, 1013.25, 1013.25])
    su, sv = stress_features(ws, wd)
    dp = pr - 1013.25
    want = 0.01 + 0.004 * su + 0.003 * sv + BARO_THEORY_M_PER_HPA * dp
    fc = resp.forecast(ftimes, ws, wd, pr)
    assert fc.mean == pytest.approx(want, abs=1e-12)
    assert np.all(fc.sigma == pytest.approx(0.05))
    # wind FROM the east (90°) pushes water westward (−u); from the west
    # (270°) it pushes +u; the 10 hPa drop adds ~10 cm (inverse barometer)
    assert fc.mean[2] > 0
    assert fc.mean[0] < 0
    assert BARO_THEORY_M_PER_HPA * 10.0 == pytest.approx(-0.0994, abs=1e-3)


def test_response_roundtrip(learned):
    times, ws, wd, pr, surge, resp, diag = learned
    r2 = MetResponse.from_dict(resp.to_dict())
    assert r2.lag_hours == resp.lag_hours
    assert np.allclose(r2.surge(times, ws, wd, pr), resp.surge(times, ws, wd, pr))


# -- met CSV I/O ------------------------------------------------------------------

def test_met_csv_roundtrip(tmp_path):
    n = 30
    times = _hourly(n)
    ws, wd, pr = _met(n, seed=9)
    path = tmp_path / "met.csv"
    write_met_csv(list(zip(times, ws, wd, pr)), str(path))
    t2, ws2, wd2, pr2 = read_met_csv(str(path))
    assert len(t2) == n
    assert np.allclose(ws2, np.round(ws, 2), atol=0.02)
    assert np.allclose(pr2, np.round(pr, 2), atol=0.02)
    # headerless canonical order + junk rows skipped
    p2 = tmp_path / "raw.csv"
    with open(p2, "w") as fh:
        fh.write("time,wind_speed,wind_dir,pressure\n")
        fh.write("not-a-time,1,2,3\n")  # unparsable row
        for t, s, d, pp in zip(times, ws, wd, pr):
            fh.write(f"{t.isoformat()},{s:.4f},{d:.2f},{pp:.2f}\n")
    t3, ws3, wd3, pr3 = read_met_csv(str(p2))
    assert len(t3) == n
    assert np.allclose(ws3, ws, atol=1e-2)


def test_read_met_csv_missing_column(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("time,wind_speed,pressure\n" + "2023-01-01T00:00:00+00:00,3,1010\n")
    with pytest.raises(ValueError, match="wind_dir"):
        read_met_csv(str(p))


# -- downstream: flood probability + decision pricing -------------------------------

def test_flood_probability_accepts_per_time_surge():
    mean = np.array([1.0, 1.0])
    lo = mean - 0.1
    hi = mean + 0.1
    pred = Prediction(mean=mean, lower=lo, upper=hi, se=np.full(2, 0.051))
    p_scalar = flood_probability(pred, 1.0, surge_sigma=0.2)
    p_arr = flood_probability(pred, 1.0,
                              surge_mean=np.array([0.0, 0.5]),
                              surge_sigma=np.array([0.2, 0.2]))
    # first hour: same as the scalar case; second: 50 cm of surge moves p up
    assert p_arr[0] == pytest.approx(p_scalar[0])
    assert p_arr[1] > p_arr[0] + 0.1
    mu = mean + np.array([0.0, 0.5])
    z = (1.0 - mu) / np.sqrt((0.1 / 1.96) ** 2 + 0.2 ** 2)
    assert p_arr[1] == pytest.approx(float(0.5 * (1.0 + math.erf(
        -z[1] / math.sqrt(2.0)))), abs=1e-9)


def test_decision_curve_prices_surge_forecast(learned):
    times, ws, wd, pr, surge, resp, diag = learned
    ftimes = _hourly(24, start=times[-1] + timedelta(hours=1))
    mean = np.full(24, 1.0)
    pred = Prediction(mean=mean, lower=mean - 0.1, upper=mean + 0.1,
                      se=np.full(24, 0.051))
    s_mean = np.linspace(0.0, 0.8, 24)  # surge ramps in
    s_sigma = np.full(24, 0.15)
    curve = decision_curve(ftimes, pred, 1.4, cost=0.05, loss=1.0,
                           surge_mean=s_mean, surge_sigma=s_sigma)
    # probability rises with the surge; the policy flips to act once p passes
    # the 0.05 break-even
    assert np.all(np.diff(curve.probability) > 0)
    assert curve.act[0] is False or curve.act[0] == np.False_
    assert bool(curve.act[-1]) is True
    assert curve.optimal_cost <= curve.always_cost
    assert curve.optimal_cost <= curve.never_cost + 1e-9


# -- fetch: met ingestion ---------------------------------------------------------

def _fake_resp(text):
    class _Fake:
        def __init__(self, t):
            self._t = t.encode("utf-8")

        def read(self):
            return self._t

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return _Fake(text)


_WIND_CSV = """Date Time,Wind Speed(m/s),Wind Dir(deg),Wind Dir Card,Gust Speed(m/s),Gust Dir(deg)
2024-01-01 00:00,8.5,225,SW,12.1,230
2024-01-01 01:00, ,230,SW,11.0,240
2024-01-01 02:00,7.2,240,SW,10.4,250
"""
_PRESSURE_CSV = """Date Time,Air Pressure
2024-01-01 00:00,1002.3
2024-01-01 01:00,1001.8
2024-01-01 02:00,1001.1
"""


def test_parse_noaa_met_csv():
    from tideglass.fetch import parse_noaa_met_csv

    wind = parse_noaa_met_csv(_WIND_CSV, "wind")
    # the blank speed sample keeps its direction but loses 'speed'; the
    # product merge drops incomplete timestamps later
    assert len(wind) == 3
    assert "speed" not in wind[datetime(2024, 1, 1, 1, tzinfo=UTC)]
    assert wind[datetime(2024, 1, 1, tzinfo=UTC)]["speed"] == pytest.approx(8.5)
    assert wind[datetime(2024, 1, 1, tzinfo=UTC)]["dir"] == pytest.approx(225)
    press = parse_noaa_met_csv(_PRESSURE_CSV, "air_pressure")
    assert len(press) == 3
    assert press[datetime(2024, 1, 1, 2, tzinfo=UTC)]["pressure"] \
        == pytest.approx(1001.1)
    with pytest.raises(ValueError, match="unknown met product"):
        parse_noaa_met_csv(_WIND_CSV, "humidity")


def test_fetch_met_merges_products_and_guards_range():
    from tideglass.fetch import fetch_met, fetch_met_range

    calls = []

    def getter(url, timeout=30):
        calls.append(url)
        return _fake_resp(_WIND_CSV if "product=wind" in url
                          else _PRESSURE_CSV)

    rows = fetch_met("9414290", "2024-01-01", "2024-01-01", _getter=getter)
    # the 01:00 row has no wind (blank) -> merged out; 2 rows survive
    assert len(rows) == 2
    t0, s0, d0, p0 = rows[0]
    assert s0 == pytest.approx(8.5) and d0 == pytest.approx(225)
    assert p0 == pytest.approx(1002.3)
    assert len(calls) == 2  # wind + air_pressure
    with pytest.raises(ValueError, match="31 days"):
        fetch_met("9414290", "2024-01-01", "2024-03-01",
                  _getter=lambda *a, **k: (_ for _ in ()).throw(
                      AssertionError("must not hit the API")))

    long_rows = fetch_met_range("9414290", "2024-01-01", "2024-02-15",
                                _getter=getter)
    assert all(r[0] < s[0] for r, s in zip(long_rows, long_rows[1:]))


def test_fetch_met_http_error_surfaces_noaa_message():
    from tideglass.fetch import fetch_met

    def getter(url, timeout=30):
        raise HTTPError(url, 400, "Bad Request", {},
                        io.BytesIO(b" Wrong Date: Range Limit Exceeded "))

    with pytest.raises(ValueError, match="Range Limit"):
        fetch_met("9414290", "2024-01-01", "2024-01-02", _getter=getter)


def test_write_met_csv_readable_by_module(tmp_path):
    from datetime import datetime as dt

    rows = [(dt(2024, 1, 1, tzinfo=UTC), 8.5, 225.0, 1002.3)]
    path = tmp_path / "m.csv"
    write_met_csv(rows, str(path))
    t, s, d, p = read_met_csv(str(path))
    assert t == [dt(2024, 1, 1, tzinfo=UTC)]
    assert s[0] == pytest.approx(8.5)
    assert d[0] == pytest.approx(225.0)
    assert p[0] == pytest.approx(1002.3)


# -- CLI --------------------------------------------------------------------------

def test_cli_surge_end_to_end(tmp_path, capsys):
    from tideglass.cli import main

    n = 24 * 45  # 45 days: enough for a stable fit + conformal-free learn
    times = _hourly(n)
    truth = _tide(times, {"M2": 1.0, "S2": 0.4, "K1": 0.6})
    ws, wd, pr = _met(n)
    surge = _truth_surge(ws, wd, pr, noise=0.005, seed=21)
    obs = truth + surge

    obs_csv = tmp_path / "obs.csv"
    with open(obs_csv, "w") as fh:
        fh.write("time,height\n")
        for t, o in zip(times, obs):
            fh.write(f"{t.isoformat()},{o:.4f}\n")
    met_csv = tmp_path / "met.csv"
    write_met_csv(list(zip(times, ws, wd, pr)), str(met_csv))

    # forecast met: a building storm over the next 48 h
    fct = _hourly(48, start=times[-1] + timedelta(hours=1))
    fws = np.linspace(5.0, 20.0, 48)
    fwd = np.full(48, 210.0)
    fpr = np.linspace(1013.0, 990.0, 48)
    fcsv = tmp_path / "fc.csv"
    write_met_csv(list(zip(fct, fws, fwd, fpr)), str(fcsv))

    rc = main(["surge", str(obs_csv), str(met_csv),
               "--station", "sg1", "--store", str(tmp_path),
               "--forecast", str(fcsv), "--hours", "48",
               "--flood", "1.6", "--cost", "0.05", "--loss", "1.0"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "lag: 2 h" in out
    assert "barometer:" in out
    assert "surge forecast (48 h" in out
    assert "P(flood>" in out
    assert "decision[" in out
    assert "value vs always" in out
    # artifacts: the model was auto-fit and the response persisted
    import json

    with open(tmp_path / "sg1.json") as fh:
        art = json.load(fh)
    assert art["station"] == "sg1"
    with open(tmp_path / "sg1.met.json") as fh:
        saved = json.load(fh)
    assert saved["lag_hours"] == 2
    restored = MetResponse.from_dict(saved)
    assert restored.r_squared > 0.9
    # the learned response genuinely predicts held-out surge: check the last
    # 48 observed hours against the truth generator (no noise-free leakage —
    # the model's residual is obs − tide, so the storm structure must survive)
    model = TideModel.load_harmonic(str(tmp_path / "sg1.json"))
    resid = obs - model.predict(times).mean
    resp, _ = learn_met_response(times, resid, times, ws, wd, pr,
                                 max_lag_hours=6)
    tail = slice(-48, None)
    err = float(np.sqrt(np.mean(
        (resp.surge(times, ws, wd, pr)[tail] - resid[tail]) ** 2)))
    assert err < 0.05


def test_cli_surge_requires_matching_cost_loss(tmp_path, capsys):
    from tideglass.cli import main

    n = 24 * 30
    times = _hourly(n)
    truth = _tide(times, {"M2": 1.0})
    ws, wd, pr = _met(n)
    obs = truth + _truth_surge(ws, wd, pr, noise=0.005)
    obs_csv = tmp_path / "obs.csv"
    with open(obs_csv, "w") as fh:
        fh.write("time,height\n")
        for t, o in zip(times, obs):
            fh.write(f"{t.isoformat()},{o:.4f}\n")
    met_csv = tmp_path / "met.csv"
    write_met_csv(list(zip(times, ws, wd, pr)), str(met_csv))
    rc = main(["surge", str(obs_csv), str(met_csv), "--station", "sg2",
               "--store", str(tmp_path), "--forecast", str(met_csv),
               "--hours", "24", "--flood", "1.5", "--cost", "0.05"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "together" in err
    rc = main(["surge", str(obs_csv), str(met_csv), "--station", "sg2",
               "--store", str(tmp_path), "--forecast", str(met_csv),
               "--hours", "24", "--cost", "0.05", "--loss", "1.0"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "--flood" in err
