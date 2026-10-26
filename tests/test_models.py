"""Test offline di feature, modelli e backtest su dati simulati."""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from pipeline import analyze, backtest
from pipeline.features import compute_features, garman_klass_var, log_returns
from pipeline.ingest import write_prices
from pipeline.models import anomalies, regimes
from pipeline.models import volatility as vol


# ---------------------------------------------------------------------------
# Dati simulati
# ---------------------------------------------------------------------------

def simulate_garch(n: int, seed: int = 0, omega=0.02, alpha=0.08, beta=0.9) -> pd.Series:
    """Log-rendimenti (in unità, non %) da un GARCH(1,1) con innovazioni t."""
    rng = np.random.default_rng(seed)
    s2 = omega / (1 - alpha - beta)
    r = np.empty(n)
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_t(6) / np.sqrt(1.5)
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    return pd.Series(r / 100, index=pd.bdate_range("2010-01-04", periods=n), name="ret")


def simulate_regimes(n: int, seed: int = 0) -> tuple[pd.Series, np.ndarray]:
    """Due regimi persistenti: calmo (vol 0,7%) e turbolento (vol 3%)."""
    rng = np.random.default_rng(seed)
    state = np.empty(n, dtype=int)
    s = 0
    for t in range(n):
        if rng.random() < (0.005 if s == 0 else 0.02):
            s = 1 - s
        state[t] = s
    sigma = np.where(state == 0, 0.007, 0.03)
    r = rng.normal(0, sigma)
    return pd.Series(r, index=pd.bdate_range("2010-01-04", periods=n)), state


def ohlcv_from_returns(ret: pd.Series, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(ret.cumsum())
    open_ = close.shift(1).fillna(100) * np.exp(rng.normal(0, 0.002, len(ret)))
    spread = np.abs(ret.to_numpy()) + 0.005
    high = np.maximum(open_, close) * (1 + spread / 2)
    low = np.minimum(open_, close) * (1 - spread / 2)
    volume = np.exp(rng.normal(14, 0.3, len(ret)))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume},
                        index=ret.index.rename("date"))


# ---------------------------------------------------------------------------
# Feature
# ---------------------------------------------------------------------------

def test_feature_senza_look_ahead():
    ret = simulate_garch(800, seed=1)
    df, bench = ohlcv_from_returns(ret, 1), ohlcv_from_returns(simulate_garch(800, seed=2), 2)
    f1 = compute_features(df, bench)
    cut = 600
    df2, bench2 = df.copy(), bench.copy()
    df2.iloc[cut:, :4] *= 3.0
    df2.iloc[cut:, 4] *= 10
    bench2.iloc[cut:, :4] *= 0.5
    f2 = compute_features(df2, bench2)
    pd.testing.assert_frame_equal(f1.iloc[:cut], f2.iloc[:cut])


def test_feature_valori_attesi():
    ret = simulate_garch(1500, seed=3)
    df = ohlcv_from_returns(ret, 3)
    f = compute_features(df)
    np.testing.assert_allclose(f["ret"].iloc[1:], log_returns(df["close"]).iloc[1:])
    expected_rv = ret.iloc[-63:].std() * np.sqrt(252)
    assert f["rv_63"].iloc[-1] == pytest.approx(expected_rv, rel=1e-6)
    assert f["mom_1m"].iloc[-1] == pytest.approx(np.log(df["close"].iloc[-1] / df["close"].iloc[-22]))
    assert (f["drawdown"] <= 0).all()
    assert (garman_klass_var(df) >= 0).all()


def test_beta_rolling():
    rng = np.random.default_rng(4)
    rb = pd.Series(rng.normal(0, 0.01, 1000), index=pd.bdate_range("2015-01-01", periods=1000))
    r = 1.5 * rb + rng.normal(0, 0.002, 1000)
    f = compute_features(ohlcv_from_returns(r, 5), ohlcv_from_returns(rb, 6))
    assert f["beta_252"].iloc[-1] == pytest.approx(1.5, abs=0.1)
    assert f["corr_252"].iloc[-1] > 0.9


# ---------------------------------------------------------------------------
# Volatilità
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,h", [("garch", 1), ("garch", 5), ("egarch", 1), ("har", 1), ("har", 5)])
def test_volatilita_senza_look_ahead(model, h):
    ret = simulate_garch(1600, seed=7)
    ret2 = ret.copy()
    ret2.iloc[1300:] *= 5
    if model == "har":
        f1, f2 = vol.forecast_har(ret, h, 1000, 200), vol.forecast_har(ret2, h, 1000, 200)
    else:
        f1, f2 = vol.forecast_arch(ret, h, model, 1000, 200), vol.forecast_arch(ret2, h, model, 1000, 200)
    assert f1.iloc[1000:1300].notna().all()
    np.testing.assert_array_equal(f1.iloc[:1300].to_numpy(), f2.iloc[:1300].to_numpy())


def test_egarch_solo_h1():
    with pytest.raises(ValueError):
        vol.forecast_arch(simulate_garch(1200), 5, "egarch", 1000, 200)


def test_target_realizzato():
    ret = pd.Series([0.01, -0.02, 0.03, 0.0], index=pd.bdate_range("2020-01-01", periods=4))
    t = vol.realized_target(ret, 2)
    assert t["x"].iloc[0] == pytest.approx(0.01)
    assert t["y"].iloc[0] == pytest.approx(0.02 ** 2 + 0.03 ** 2)
    assert np.isnan(t["y"].iloc[-1])


def test_garch_batte_le_baseline_su_dati_garch():
    ret = simulate_garch(3000, seed=11)
    res = backtest.volatility_backtest(ret, use_holdout=True, horizons=(1,),
                                       min_train=1000, refit_every=250)
    m = res["h1"]["models"]
    assert m["garch"]["qlike"] < m["hist_mean"]["qlike"]
    assert m["garch"]["qlike_diff_vs_rw"] < 0
    assert 0.8 < m["garch"]["coverage_90"] < 0.97


# ---------------------------------------------------------------------------
# Metriche e backtest
# ---------------------------------------------------------------------------

def test_qlike_minima_alla_varianza_vera():
    rng = np.random.default_rng(0)
    y = (rng.normal(0, 0.02, 200_000)) ** 2
    losses = {f: backtest.qlike_loss(np.full_like(y, f), y).mean() for f in (2e-4, 4e-4, 8e-4)}
    assert min(losses, key=losses.get) == 4e-4


def test_crps_normale_forma_chiusa():
    rng = np.random.default_rng(1)
    sigma, x = 0.02, 0.013
    sample = rng.normal(0, sigma, 400_000)
    numeric = np.mean(np.abs(sample - x)) - 0.5 * np.mean(np.abs(sample - rng.permutation(sample)))
    assert backtest.crps_normal(np.array([sigma]), np.array([x]))[0] == pytest.approx(numeric, rel=0.02)


def test_holdout_escluso():
    ret = simulate_garch(4300, seed=2)
    ret.index = pd.bdate_range("2010-01-04", periods=len(ret))
    assert ret.index[-1] > pd.Timestamp(backtest.HOLDOUT_START)
    res = backtest.volatility_backtest(ret, horizons=(5,), models=("rw", "hist_mean", "har"),
                                       min_train=1000, refit_every=250)
    last = pd.Timestamp(res["h5"]["last_origin"])
    pos = ret.index.get_loc(last)
    assert ret.index[pos + 5] < pd.Timestamp(backtest.HOLDOUT_START)
    res_all = backtest.volatility_backtest(ret, use_holdout=True, horizons=(5,),
                                           models=("rw", "hist_mean", "har"), min_train=1000, refit_every=250)
    assert res_all["h5"]["n"] > res["h5"]["n"]


def test_registro_configurazioni(tmp_path):
    p = tmp_path / "trials.jsonl"
    a = backtest.record_trial(p, {"x": 1})
    b = backtest.record_trial(p, {"x": 1})
    c = backtest.record_trial(p, {"x": 2})
    assert a["hash"] == b["hash"] and a["n_configurations_tried"] == 1
    assert c["n_configurations_tried"] == 2
    assert len(p.read_text().splitlines()) == 2


# ---------------------------------------------------------------------------
# Regimi
# ---------------------------------------------------------------------------

def test_regimi_riconosce_gli_stati():
    ret, truth = simulate_regimes(2500, seed=3)
    probs, params = regimes.regimes_walk_forward(ret, n_states=2, min_train=756, refit_every=500)
    valid = probs.dropna()
    true = pd.Series(truth, index=ret.index).reindex(valid.index)
    # Ordinati per volatilità: lo stato 1 deve essere quello turbolento.
    accuracy = (valid["state"].astype(int) == true).mean()
    assert accuracy > 0.85
    summary = regimes.regime_summary(ret, probs, params)
    assert summary["available"] and summary["states"][1]["vol_ann"] > summary["states"][0]["vol_ann"]
    np.testing.assert_allclose(valid[["p_calmo", "p_turbolento"]].sum(axis=1), 1.0)


def test_regimi_senza_look_ahead():
    ret, _ = simulate_regimes(1600, seed=5)
    ret2 = ret.copy()
    ret2.iloc[1300:] *= 4
    p1, _ = regimes.regimes_walk_forward(ret, n_states=2, min_train=756, refit_every=300)
    p2, _ = regimes.regimes_walk_forward(ret2, n_states=2, min_train=756, refit_every=300)
    cut = p1.index.get_loc(ret.index[1300])
    pd.testing.assert_frame_equal(p1.iloc[:cut], p2.iloc[:cut])


# ---------------------------------------------------------------------------
# Anomalie
# ---------------------------------------------------------------------------

def test_anomalia_iniettata_e_segnalata():
    ret = simulate_garch(800, seed=8)
    df = ohlcv_from_returns(ret, 8)
    spike = df.index[600]
    df.loc[spike:, ["open", "high", "low", "close"]] *= 1.25   # +22% di log-rendimento in un giorno
    df.loc[df.index[650], "volume"] *= 50
    events = anomalies.detect(df)
    kinds_at = lambda d: set(events.loc[events["date"] == d, "kind"])
    assert "rendimento" in kinds_at(spike)
    assert "volume" in kinds_at(df.index[650])
    # Il giorno successivo allo shock non è anomalo per effetto dello shock stesso.
    assert "rendimento" not in kinds_at(df.index[601])


def test_mad_nulla_non_produce_infiniti():
    x = pd.Series(np.ones(400), index=pd.bdate_range("2020-01-01", periods=400))
    z = anomalies.robust_zscore(x)
    assert not np.isinf(z).any()


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def test_analyze_end_to_end(tmp_path, monkeypatch):
    universe = tmp_path / "u.yml"
    universe.write_text(
        "defaults: {start: 2010-01-01, sources: [yahoo], calendar: XNYS, benchmark: idx}\n"
        "instruments:\n"
        "  - {id: idx, name: Indice, asset_class: index, yahoo: IDX, benchmark: null}\n"
        "  - {id: aaa, name: AAA, asset_class: stock, yahoo: AAA}\n"
    )
    data = tmp_path / "raw"
    rb = simulate_garch(1400, seed=21)
    write_prices(ohlcv_from_returns(rb, 1), data / "idx.parquet")
    write_prices(ohlcv_from_returns(1.2 * rb + simulate_garch(1400, seed=22) * 0.5, 2), data / "aaa.parquet")
    monkeypatch.setattr(backtest, "MIN_TRAIN", 600)
    monkeypatch.setattr(backtest, "REFIT_EVERY", 250)

    out = tmp_path / "results"
    rc = analyze.main(["--universe", str(universe), "--data-dir", str(data), "--results-dir", str(out)])
    assert rc == 0
    a = json.loads((out / "analysis" / "aaa.json").read_text())
    assert a["regime"]["available"] and "beta_63" in a["features_latest"]
    assert "close" not in a["features_latest"]  # nessun prezzo grezzo nei risultati
    bt = json.loads((out / "backtest" / "volatility.json").read_text())
    assert bt["series"] == "backtest" and bt["use_holdout"] is False
    assert set(bt["instruments"]["aaa"]["h1"]["models"]) == set(backtest.MODELS)
    assert "egarch" not in bt["instruments"]["aaa"]["h5"]["models"]
    assert bt["trial"]["n_configurations_tried"] == 1
