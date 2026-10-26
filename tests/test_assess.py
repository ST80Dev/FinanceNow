"""Test delle regole della valutazione automatica."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import assess
from tests.test_digest import ctx
from tests.test_models import simulate_garch

STATS = {"regime_turbolento": {"h21": {"n": 300, "vol_ratio": {"event": 1.3, "baseline": 1.1, "excess": 0.2, "significant": True},
                                       "dd10": {"event": 0.33, "baseline": 0.29, "excess": 0.04, "significant": False},
                                       "ret": {"event": 0.01, "baseline": 0.02, "significant": False}}},
         "drawdown_20": {"h21": {"n": 250, "vol_ratio": {"event": 1.4, "baseline": 1.1, "excess": 0.3, "significant": True},
                                 "dd10": {"event": 0.45, "baseline": 0.35, "excess": 0.1, "significant": True},
                                 "ret": {"event": 0.0, "baseline": 0.02, "significant": False}}}}
DIRECTION = {"h21": {"p_up": 0.66, "p_base": 0.58, "q10": -0.1, "q50": 0.01, "q90": 0.12,
                     "drivers": [{"feature": "mom_12m", "label": "andamento dell'ultimo anno", "effect": 0.2}]},
             "h5": {"p_up": 0.55, "p_base": 0.54, "q10": -0.05, "q50": 0.0, "q90": 0.05, "drivers": []}}


def one(c, current=(), verdicts=None):
    return assess.assess_instrument(c, STATS, set(current), verdicts or {"h21": "nessun vantaggio dimostrato"})


def test_tempesta_e_rischio_alto():
    c = ctx(states=[1] * 10 + [2] * 2, current="turbolento", days=2, pctl=0.92)
    a = one(c)
    assert a["situation"]["key"] == "tempesta"
    assert a["risk"]["level"] == "alto" and a["risk"]["points"] == 3        # vol (2) + evento storico (1)
    assert a["risk"]["evidence"]["event"] == "regime_turbolento"


def test_ripresa_dopo_la_tempesta():
    a = one(ctx(states=[2] * 9 + [1] * 3, current="normale", days=3))
    assert a["situation"]["key"] == "ripresa" and "2 sedute fa" in a["situation"]["why"]


def test_pressione_per_calo_dal_massimo():
    a = one(ctx(feats={"drawdown": -0.25}))
    assert a["situation"]["key"] == "pressione" and "-25%" in a["situation"]["why"]
    assert a["risk"]["evidence"]["event"] == "drawdown_20"


def test_tranquillo_e_rischio_basso():
    a = one(ctx(current="calmo", pctl=0.2))
    assert a["situation"]["key"] == "tranquillo" and a["risk"]["level"] == "basso"
    assert one(ctx(current="calmo", pctl=0.7))["situation"]["key"] == "norma"


def test_trimestrale_alza_il_rischio_e_compare_negli_eventi():
    c = ctx(pctl=0.75)
    c.earnings = {"available": True, "typical_move": 0.05, "typical_day_move": 0.012, "move_ratio": 4,
                  "next": {"date": "2026-10-29", "timing": "dopo la chiusura", "sessions_until": 4}}
    a = one(c)
    assert a["risk"]["level"] == "alto"
    assert a["events"][0]["kind"] == "trimestrale" and "tra 4 sedute" in a["events"][0]["text"]


def test_tendenza_e_previsione():
    c = ctx(feats={"dist_ma200": 0.08, "mom_3m": 0.05, "mom_12m": 0.3})
    c.direction = DIRECTION
    a = one(c)
    assert a["trend"]["key"] == "su"
    assert a["outlook"]["key"] == "favorevole" and not a["outlook"]["reliable"]
    assert "spinge verso l'alto" in a["outlook"]["drivers"][0]
    assert one(c, verdicts={"h21": "vantaggio"})["outlook"]["reliable"]
    assert a["risk"]["range_month"][0] < 0 < a["risk"]["range_month"][1]


def test_build_ordina_per_attenzione_e_mercato():
    cs = [ctx("ndx", cls="index"), ctx("calm", current="calmo", pctl=0.2),
          ctx("hot", states=[1] * 10 + [2] * 2, current="turbolento", days=2, pctl=0.95)]
    out = assess.build(cs, STATS, [], {})
    assert [i["id"] for i in out["instruments"]][0] == "hot"
    assert out["market"]["counts"]["tempesta"] == 1 and "1 in tempesta" in out["market"]["detail"]
    assert "Nasdaq 100" in out["market"]["headline"]
    assert "In tempesta" in assess.to_markdown(out)


def test_log_e_valutazione_dei_giudizi(tmp_path):
    ret = simulate_garch(400, seed=7)
    path = tmp_path / "log.csv"
    a1 = {"instruments": [{"id": "aaa", "as_of": ret.index[100].date().isoformat(), "data_status": "up_to_date",
                           "situation": {"key": "norma"}, "risk": {"level": "alto"}, "trend": {"key": "su"},
                           "outlook": {"key": "favorevole", "p_up": 0.6, "p_base": 0.55}}]}
    a2 = {"instruments": [dict(a1["instruments"][0], as_of=ret.index[-1].date().isoformat())]}
    assert assess.log_assessment(path, a1, "t") == 1
    assert assess.log_assessment(path, a1, "t") == 0
    assert assess.log_assessment(path, a2, "t") == 1
    sc = assess.score_log(path, {"aaa": {"ret": ret, "gap": pd.Series(0.0, index=ret.index)}})
    assert sc["n_logged"] == 2 and sc["n_evaluated"] == 1            # l'ultimo non ha ancora un mese di esito
    assert set(sc["by_risk"]) == {"alto"} and 0 <= sc["by_outlook"]["favorevole"]["up_rate"] <= 1
    assert np.isfinite(sc["by_risk"]["alto"]["vol_ratio"])
