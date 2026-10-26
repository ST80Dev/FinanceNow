"""Test offline del modulo sperimentale di direzione."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import direction, patterns
from pipeline.models import direction as dm
from tests.test_models import ohlcv_from_returns, simulate_garch


def test_forward_return_coerente_con_i_pattern():
    ret = simulate_garch(300, seed=1)
    gap = pd.Series(np.random.default_rng(1).normal(0, 0.003, 300), index=ret.index)
    a = dm.forward_return(ret, gap, 5)
    b = patterns.forward_outcomes(ret, 5, gap)["ret"]
    pd.testing.assert_series_equal(a.dropna(), b.dropna(), check_names=False)
    assert a.iloc[-5:].isna().all()


def test_logistica_ritrova_il_segnale():
    rng = np.random.default_rng(0)
    n = 20000
    X = rng.normal(size=(n, 3))
    true = 0.1 + 0.8 * X[:, 0] - 0.5 * X[:, 1]
    y = (rng.random(n) < dm.sigmoid(true)).astype(float)
    beta = dm.fit_logistic(X, y, np.zeros(n), l2=0.0)
    assert beta[1] == pytest.approx(0.8, abs=0.06) and beta[2] == pytest.approx(-0.5, abs=0.06)
    assert abs(beta[3]) < 0.06
    shrunk = dm.fit_logistic(X, y, np.zeros(n), l2=0.5)
    assert abs(shrunk[1]) < abs(beta[1])                     # la penalità restringe


def test_senza_informazione_torna_alla_baseline():
    rng = np.random.default_rng(2)
    n = 40000
    X = rng.normal(size=(n, len(dm.FEATURES)))
    groups = np.where(np.arange(n) % 2, "a", "b")
    y = (rng.random(n) < np.where(groups == "a", 0.6, 0.5)).astype(float)
    m = dm.DirectionModel(X, y, groups, rng.normal(size=n))
    p = m.predict(X, groups)
    dev = np.abs(p - m.base_rate(groups))
    assert dev.mean() < 0.02 and dev.max() < 0.1
    assert m.base_rate(np.array(["a"]))[0] > m.base_rate(np.array(["b"]))[0]
    assert m.base_rate(np.array(["sconosciuto"]))[0] == pytest.approx(m.clim["_all"])


def _per_ticker(k=3, n=2200):
    out = {}
    rb = simulate_garch(n, seed=40)
    bench = ohlcv_from_returns(rb, 40)
    for i in range(k):
        df = ohlcv_from_returns(rb + simulate_garch(n, seed=41 + i) * 0.7, 41 + i)
        out[f"t{i}"] = direction.ticker_inputs(df, bench, None, None)
    return out


def test_walk_forward_senza_look_ahead_e_metriche(monkeypatch):
    per = _per_ticker()
    pan = direction.panel(per, 21)
    fits = []
    real_fit = direction._fit

    def spy(rows, h):
        fits.append(rows["date"].max())
        return real_fit(rows, h)

    monkeypatch.setattr(direction, "_fit", spy)
    pr, _ = direction.walk_forward(pan, 21, None, min_train=800, refit_every=250)
    cal = pd.DatetimeIndex(sorted(pan["date"].unique()))
    starts = pr.groupby(pr["date"].map(lambda d: (cal.get_loc(d) - 800) // 250))["date"].min()
    for train_end, first_pred in zip(fits, starts):
        assert cal.get_loc(train_end) <= cal.get_loc(first_pred) - 21 - direction.EMBARGO
    m = direction.evaluate(pr, 21)
    assert m["n"] == len(pr) and 0.15 < m["brier"]["modello"] < 0.3
    assert m["verdict"] != "vantaggio"                      # dati casuali: nessun vantaggio
    assert 0.6 < m["coverage_80"] < 0.95
    assert set(m["calls"]) == {"ottimista", "pessimista", "neutro"}


def test_cli_scrive_risultati_e_log(tmp_path, monkeypatch):
    from tests.test_pipeline_step3 import _universe, _write_data
    monkeypatch.setattr(direction, "MIN_TRAIN", 600)
    monkeypatch.setattr(direction, "REFIT_EVERY", 250)
    data, results = tmp_path / "raw", tmp_path / "results"
    _write_data(data)
    args = ["--universe", str(_universe(tmp_path)), "--data-dir", str(data), "--results-dir", str(results)]
    assert direction.main(args) == 0
    from pipeline.ingest import read_json
    lat = read_json(results / "direction" / "latest.json")
    assert set(lat["instruments"]) == {"idx", "aaa"}
    h21 = lat["instruments"]["aaa"]["h21"]
    assert 0 < h21["p_up"] < 1 and h21["q10"] < h21["q50"] < h21["q90"]
    bt = read_json(results / "backtest" / "direction.json")
    assert bt["horizons"]["h21"]["n"] > 0 and bt["trial"]["n_configurations_tried"] >= 1
    n1 = len(direction.read_log(results / "live" / direction.FORECASTS_FILE))
    assert direction.main(args + ["--no-backtest"]) == 0
    assert len(direction.read_log(results / "live" / direction.FORECASTS_FILE)) == n1   # append-only, niente doppioni
