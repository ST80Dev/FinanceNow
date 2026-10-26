"""Test offline dell'export per la dashboard."""

from __future__ import annotations

import json
import math

import pytest

from pipeline import analyze, backtest, export, live
from pipeline.ingest import load_universe
from tests.test_pipeline_step3 import _universe, _write_data

PRICE_KEYS = {"open", "high", "low", "close", "volume", "adj_close", "price"}


def _all_keys(obj, acc=None):
    acc = set() if acc is None else acc
    if isinstance(obj, dict):
        for k, v in obj.items():
            acc.add(k)
            _all_keys(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _all_keys(v, acc)
    return acc


@pytest.fixture
def exported(tmp_path, monkeypatch):
    monkeypatch.setattr(backtest, "MIN_TRAIN", 600)
    monkeypatch.setattr(backtest, "REFIT_EVERY", 250)
    data, results, out = tmp_path / "raw", tmp_path / "results", tmp_path / "docs_data"
    universe = _universe(tmp_path)
    _write_data(data)
    args = ["--universe", str(universe), "--data-dir", str(data), "--results-dir", str(results)]
    assert analyze.main(args) == 0
    assert live.main(args) == 0
    assert export.main(args + ["--out-dir", str(out)]) == 0
    return out, universe, data, results


def test_file_prodotti_e_json_validi(exported):
    out, *_ = exported
    for name in ("index.json", "models.json", "tickers/aaa.json", "tickers/idx.json"):
        text = (out / name).read_text()
        assert "NaN" not in text and "Infinity" not in text
        json.loads(text)


def test_nessun_prezzo_pubblicato(exported):
    out, *_ = exported
    for p in [out / "index.json", out / "models.json", *(out / "tickers").glob("*.json")]:
        keys = _all_keys(json.loads(p.read_text()))
        assert not (keys & PRICE_KEYS), (p.name, keys & PRICE_KEYS)


def test_serie_allineate_e_regimi(exported):
    out, *_ = exported
    t = json.loads((out / "tickers" / "aaa.json").read_text())
    n = len(t["dates"])
    assert all(len(v) == n for v in t["series"].values())
    states = [s for s in t["series"]["regime_state"] if s is not None]
    assert states and set(states) <= {0, 1, 2}
    assert t["series"]["beta_252"][-1] is not None          # titolo con benchmark
    idx = json.loads((out / "tickers" / "idx.json").read_text())
    assert "beta_252" not in idx["series"]                   # l'indice non ha benchmark


def test_indice_e_live(exported):
    out, *_ = exported
    index = json.loads((out / "index.json").read_text())
    rows = {r["id"]: r for r in index["instruments"]}
    assert set(rows) == {"idx", "aaa"}
    assert rows["aaa"]["regime"] in ("calmo", "normale", "turbolento")
    t = json.loads((out / "tickers" / "aaa.json").read_text())
    vol = t["live_latest"]["vol_ann"]["h1"]
    assert set(vol) == set(backtest.MODELS)
    assert all(0 < v < 5 for v in vol.values())
    models = json.loads((out / "models.json").read_text())
    assert models["backtest"]["available"] and models["live"]["available"]
    assert models["live"]["n_forecasts_logged"] > 0


def test_annualized_vol():
    assert export.annualized_vol(0.0001, 1) == pytest.approx(math.sqrt(0.0001 * 252))
    assert export.annualized_vol(0.0005, 5) == pytest.approx(math.sqrt(0.0001 * 252))
    assert export.annualized_vol(-1, 1) is None


def test_rimuove_strumenti_non_piu_esportati(exported):
    out, universe, data, results = exported
    (out / "tickers" / "vecchio.json").write_text("{}")
    export.export(load_universe(universe), data, results, out)
    assert not (out / "tickers" / "vecchio.json").exists()
