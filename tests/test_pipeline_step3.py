"""Test offline del passo 3: serie live, backtest "auto", JSON valido, riepilogo."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pipeline import analyze, backtest, live, summary
from pipeline.ingest import read_json, write_json, write_prices
from tests.test_models import ohlcv_from_returns, simulate_garch


@pytest.fixture
def small_settings(monkeypatch):
    monkeypatch.setattr(backtest, "MIN_TRAIN", 600)
    monkeypatch.setattr(backtest, "REFIT_EVERY", 250)


def _universe(tmp_path):
    p = tmp_path / "u.yml"
    p.write_text(
        "defaults: {start: 2010-01-01, sources: [yahoo], calendar: XNYS, benchmark: idx}\n"
        "instruments:\n"
        "  - {id: idx, name: Indice, asset_class: index, yahoo: IDX, benchmark: null}\n"
        "  - {id: aaa, name: AAA, asset_class: stock, yahoo: AAA}\n"
    )
    return p


def _write_data(data_dir, n=1300):
    rb = simulate_garch(n, seed=31)
    write_prices(ohlcv_from_returns(rb, 1), data_dir / "idx.parquet")
    write_prices(ohlcv_from_returns(1.1 * rb + 0.5 * simulate_garch(n, seed=32), 2), data_dir / "aaa.parquet")


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def test_write_json_niente_nan(tmp_path):
    p = tmp_path / "x.json"
    write_json({"a": float("nan"), "b": [np.float64(np.inf), np.int64(3)], "c": {"d": 1.5}}, p)
    text = p.read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {"a": None, "b": [None, 3], "c": {"d": 1.5}}


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------

def test_latest_forecasts_tutti_i_modelli(small_settings):
    ret = simulate_garch(900, seed=5)
    fc = live.latest_forecasts(ret)
    assert set(fc["origin"]) == {ret.index[-1].date().isoformat()}
    assert set(fc.loc[fc["h"] == 1, "model"]) == set(backtest.MODELS)
    assert set(fc.loc[fc["h"] == 5, "model"]) == set(backtest.MODELS) - {"egarch"}
    assert (fc["forecast_var"] > 0).all()


def test_latest_forecasts_coerente_con_il_walk_forward(small_settings):
    """La previsione live coincide con quella del walk-forward con un blocco all'ultima seduta."""
    ret = simulate_garch(900, seed=6)
    fc = live.latest_forecasts(ret, horizons=(1,), models=("rw", "har"))
    n = len(ret)
    har = backtest.vol.forecast_har(ret, 1, n - 1, 1, n - 1).iloc[-1]
    assert fc.set_index("model").loc["har", "forecast_var"] == pytest.approx(har)


def test_storico_insufficiente_nessuna_previsione():
    assert live.latest_forecasts(simulate_garch(100)).empty


def test_log_append_only(tmp_path):
    path = tmp_path / "log.csv"
    row = pd.DataFrame([{"origin": "2026-10-23", "id": "aaa", "h": 1, "model": "rw",
                         "forecast_var": 1e-4, "model_version": "0.1.0", "created_utc": "t1"}])
    assert live.append_log(path, row) == 1
    # Stessa chiave con valore diverso (es. run ripetuto): la riga registrata non cambia.
    again = row.assign(forecast_var=9e-4, created_utc="t2")
    assert live.append_log(path, again) == 0
    nxt = row.assign(origin="2026-10-26")
    assert live.append_log(path, nxt) == 1
    log_df = live.read_log(path)
    assert len(log_df) == 2
    assert log_df.loc[log_df["origin"] == "2026-10-23", "forecast_var"].iloc[0] == pytest.approx(1e-4)


def test_score_solo_target_realizzati():
    ret = simulate_garch(300, seed=7)
    origins = ret.index[-30:]
    rows = []
    for o in origins:
        for m, v in (("rw", 2e-4), ("garch", 1.5e-4)):
            rows.append({"origin": o.date().isoformat(), "id": "aaa", "h": 1, "model": m, "forecast_var": v})
    res = live.score(pd.DataFrame(rows), {"aaa": ret})
    # L'ultima origine non ha ancora il target realizzato.
    assert res["aaa"]["h1"]["n"] == 29
    assert "qlike_diff_vs_rw" in res["aaa"]["h1"]["models"]["garch"]


def test_live_main_idempotente(tmp_path, small_settings):
    data, out = tmp_path / "raw", tmp_path / "results"
    _write_data(data)
    args = ["--universe", str(_universe(tmp_path)), "--data-dir", str(data), "--results-dir", str(out)]
    assert live.main(args) == 0
    n1 = len(live.read_log(out / "live" / live.FORECASTS_FILE))
    assert n1 == 2 * (len(backtest.MODELS) + len(backtest.MODELS) - 1)
    assert live.main(args) == 0   # secondo run nello stesso giorno: nessuna riga nuova
    assert len(live.read_log(out / "live" / live.FORECASTS_FILE)) == n1
    scores = read_json(out / "live" / live.SCORES_FILE)
    assert scores["series"] == "live" and scores["n_forecasts_logged"] == n1


# ---------------------------------------------------------------------------
# Backtest "auto"
# ---------------------------------------------------------------------------

def test_backtest_auto_non_ricalcola(tmp_path, small_settings, monkeypatch):
    data, out = tmp_path / "raw", tmp_path / "results"
    _write_data(data)
    args = ["--universe", str(_universe(tmp_path)), "--data-dir", str(data), "--results-dir", str(out)]
    calls = []
    real = backtest.volatility_backtest
    monkeypatch.setattr(backtest, "volatility_backtest", lambda *a, **k: calls.append(1) or real(*a, **k))

    assert analyze.main(args) == 0
    assert len(calls) == 2
    assert analyze.main(args) == 0          # auto: niente da rifare
    assert len(calls) == 2
    assert analyze.main(args + ["--backtest", "always"]) == 0
    assert len(calls) == 4
    # Configurazione cambiata: si rifà tutto e il registro conta una configurazione in più.
    monkeypatch.setattr(backtest, "REFIT_EVERY", 200)
    assert analyze.main(args) == 0
    assert len(calls) == 6
    bt = read_json(out / "backtest" / "volatility.json")
    assert bt["trial"]["n_configurations_tried"] == 2
    assert set(bt["instruments"]) == {"idx", "aaa"}


def test_backtest_auto_integra_un_ticker_nuovo(tmp_path, small_settings):
    data, out = tmp_path / "raw", tmp_path / "results"
    _write_data(data)
    base = ["--universe", str(_universe(tmp_path)), "--data-dir", str(data), "--results-dir", str(out)]
    assert analyze.main(base + ["--tickers", "idx"]) == 0
    assert analyze.main(base) == 0
    bt = read_json(out / "backtest" / "volatility.json")
    assert set(bt["instruments"]) == {"idx", "aaa"}


# ---------------------------------------------------------------------------
# Riepilogo
# ---------------------------------------------------------------------------

def test_summary_markdown(tmp_path, small_settings):
    data, out = tmp_path / "raw", tmp_path / "results"
    _write_data(data)
    args = ["--universe", str(_universe(tmp_path)), "--data-dir", str(data), "--results-dir", str(out)]
    analyze.main(args)
    live.main(args)
    write_json({"mode": "daily", "summary": {"updated": 1}, "requests": 2, "max_requests": 200,
                "missing": ["zzz"], "stale": [], "deferred": []}, out / "status" / "ingest.json")
    md = summary.render(out)
    assert "| aaa |" in md and "Senza dati: zzz" in md
    assert "Backtest" in md and "Live" in md
