"""Test offline dello studio di eventi (pattern)."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from pipeline import patterns
from pipeline.features import compute_features
from tests.test_models import ohlcv_from_returns, simulate_garch


def test_forward_outcomes_a_mano():
    ret = pd.Series([0.0, 0.10, -0.05, -0.10, 0.02], index=pd.bdate_range("2020-01-01", periods=5))
    o = patterns.forward_outcomes(ret, 2)
    # Origine 0: sedute 1-2 → +0,10 poi -0,05: rendimento 0,05, calo dal massimo exp(-0,05)-1
    assert o["ret"].iloc[0] == pytest.approx(0.05)
    assert o["maxdd"].iloc[0] == pytest.approx(math.expm1(-0.05))
    # Origine 1: sedute 2-3 → -0,05, -0,10: calo totale exp(-0,15)-1
    assert o["maxdd"].iloc[1] == pytest.approx(math.expm1(-0.15))
    assert o["vol"].iloc[1] == pytest.approx(np.std([-0.05, -0.10], ddof=1) * math.sqrt(252))
    assert o.iloc[3:].isna().all().all()          # finestre incomplete


def test_cooldown():
    idx = pd.bdate_range("2020-01-01", periods=100)
    kept = patterns.apply_cooldown(pd.DatetimeIndex([idx[0], idx[5], idx[25], idx[30], idx[60]]), idx, 21)
    assert kept == [idx[0], idx[25], idx[60]]


def _ticker(seed, n=3000):
    ret = simulate_garch(n, seed=seed)
    df = ohlcv_from_returns(ret, seed)
    feats = compute_features(df)
    state = pd.Series(np.where(feats["rv_21"] > feats["rv_21"].median() * 1.5, 2, 1), index=feats.index, dtype=float)
    state.iloc[:30] = np.nan
    return df, feats, state


def test_eventi_senza_look_ahead():
    df, feats, state = _ticker(1)
    e1 = patterns.detect_events(df, feats, state, None)
    df2 = df.copy()
    df2.iloc[2000:, :4] *= np.exp(np.linspace(0, -1, len(df2) - 2000))[:, None]
    feats2 = compute_features(df2)
    state2 = state.copy()
    state2.iloc[2000:] = 2.0
    e2 = patterns.detect_events(df2, feats2, state2, None)
    cut = df.index[2000]
    pd.testing.assert_frame_equal(e1[e1["date"] < cut].reset_index(drop=True), e2[e2["date"] < cut].reset_index(drop=True))


def test_dopo_volatilita_estrema_la_volatilita_resta_alta():
    """Su dati GARCH la volatilità persiste: dopo un picco l'esito deve essere sopra la baseline."""
    per = {}
    for s in range(4):
        df, feats, state = _ticker(10 + s)
        per[f"t{s}"] = {"ret": feats["ret"], "events": patterns.detect_events(df, feats, state, None)}
    res = patterns.study(per, None)
    v = res["stats"]["vol_estrema"]["h21"]
    assert v["n"] >= 10
    assert v["vol_ratio"]["event"] > v["vol_ratio"]["baseline"]
    assert v["vol_ratio"]["significant"]
    assert patterns.describe("vol_estrema", res["stats"]).startswith("storicamente")


def test_cutoff_esclude_le_finestre_successive():
    ret = simulate_garch(600, seed=3)
    cutoff = ret.index[400].date()
    out = patterns.outcome_table(ret, 21, cutoff)
    assert out.index.max() <= ret.index[400 - 21 - 1]


def test_log_live_e_valutazione(tmp_path):
    ret = simulate_garch(300, seed=4)
    stats = {"vol_estrema": {"h21": {"n": 20, "vol_ratio": {"event": 1.5, "baseline": 1.0},
                                     "dd10": {"event": 0.4, "baseline": 0.2}}}}
    path = tmp_path / "log.csv"
    ev = [{"id": "aaa", "type": "vol_estrema", "date": ret.index[100].date().isoformat(), "is_last_session": True},
          {"id": "aaa", "type": "vol_estrema", "date": ret.index[290].date().isoformat(), "is_last_session": True},
          {"id": "aaa", "type": "drawdown_20", "date": ret.index[100].date().isoformat(), "is_last_session": False}]
    assert patterns.log_new_events(path, ev, stats, "t") == 2
    assert patterns.log_new_events(path, ev, stats, "t2") == 0          # append-only, niente doppioni
    sc = patterns.score_log(path, {"aaa": {"ret": ret}})
    assert sc["n_logged"] == 2 and sc["n_evaluated"] == 1              # la seconda finestra non è chiusa
    assert sc["by_type"]["vol_estrema"]["expected_vol_ratio"] == 1.5


def test_describe_pochi_casi():
    assert patterns.describe("vol_estrema", {"vol_estrema": {"h21": {"n": 3}}}) is None


def test_esiti_dall_apertura_del_giorno_dopo():
    """Il gap notturno di t+1 non è catturabile: va escluso da rendimento e calo."""
    idx = pd.bdate_range("2020-01-01", periods=5)
    ret = pd.Series([0.0, -0.10, 0.0, 0.0, 0.0], index=idx)       # tutto il calo di t+1 è...
    gap = pd.Series([0.0, -0.10, 0.0, 0.0, 0.0], index=idx)       # ...un gap di apertura
    o = patterns.forward_outcomes(ret, 2, gap)
    assert o["ret"].iloc[0] == pytest.approx(0.0)
    assert o["maxdd"].iloc[0] == pytest.approx(0.0)
    o2 = patterns.forward_outcomes(ret, 2)                         # misurato dalla chiusura
    assert o2["maxdd"].iloc[0] == pytest.approx(math.expm1(-0.10))


def test_differenze_minime_non_sono_nette():
    idx = pd.bdate_range("2015-01-01", periods=60, freq="W-MON")
    rows = pd.DataFrame({"id": "a", "vol_ratio": 1.0, "dd10": 0.0, "ret": 0.0,
                         "base_vol_ratio": 1.0, "base_dd10": 0.004, "base_ret": 0.0}, index=idx)
    s = patterns.summarize(rows)
    assert not s["dd10"]["significant"]        # -0,4 punti: statisticamente certo ma irrilevante


def test_intervallo_centrato_sulla_media_pubblicata():
    """L'IC deve contenere l'eccesso medio pubblicato anche con settimane di peso diverso."""
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2015-01-01", periods=400)
    # Settimane affollate (es. crollo di mercato) con eccessi più alti: media per evento
    # e media delle medie settimanali differiscono.
    idx = days.repeat(np.where(np.arange(400) % 50 < 5, 6, 1))
    x = rng.normal(0.1, 0.2, len(idx)) + np.where(np.isin(idx, days[np.arange(400) % 50 < 5]), 0.5, 0.0)
    rows = pd.DataFrame({"id": "a", "vol_ratio": 1.0 + x, "dd10": 0.0, "ret": 0.0,
                         "base_vol_ratio": 1.0, "base_dd10": 0.0, "base_ret": 0.0}, index=idx)
    v = patterns.summarize(rows)["vol_ratio"]
    lo, hi = v["ci90"]
    assert lo < v["excess"] < hi


def test_eventi_trimestrali():
    df, feats, state = _ticker(5, n=600)
    idx = df.index
    rx = pd.DataFrame({"first": [idx[100], idx[300]], "last": [idx[100], idx[301]],
                       "reaction": [0.05, -0.02], "reaction_vs_bench": [np.nan, 0.01]})
    ev = patterns.detect_events(df, feats, None, None, rx)
    got = {(t, d) for d, t in zip(ev["date"], ev["type"]) if "trimestrale" in t}
    assert got == {("trimestrale_positiva", idx[100]), ("trimestrale_positiva", idx[301]),   # vs benchmark > 0
                   ("vigilia_trimestrale", idx[95]), ("vigilia_trimestrale", idx[295])}
