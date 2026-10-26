"""Test della sintesi automatica (regole esplicite su contesti costruiti a mano)."""

from __future__ import annotations

from pipeline import digest
from pipeline.digest import Context

DATES = [f"2026-10-{d:02d}" for d in range(12, 24)]
STATES = [{"label": l} for l in ("calmo", "normale", "turbolento")]


def ctx(id_="aaa", states=None, current="normale", days=30, pctl=0.5, feats=None, anomalies=None, status=None, cls="stock"):
    states = states if states is not None else [1] * len(DATES)
    return Context(
        id=id_, name=id_.upper(), asset_class=cls, as_of=DATES[-1], recent_dates=DATES[-5:],
        regime={"available": True, "current": current, "days_in_state": days,
                "probabilities": {current: 0.9}, "states": STATES},
        regime_states=states, regime_dates=DATES,
        features={"rv_21": 0.3, "rv_5": 0.3, "rv_63": 0.3, "drawdown": -0.05, **(feats or {})},
        rv21_percentile=pctl, anomalies=anomalies or [], data_status=status,
    )


def kinds(sigs):
    return [(s["kind"], s["severity"]) for s in sigs]


def test_nessun_segnale_strumento_tranquillo():
    assert digest.instrument_signals(ctx()) == []


def test_passaggio_a_turbolento_alta():
    c = ctx(states=[1] * 10 + [2] * 2, current="turbolento", days=2)
    sigs = digest.instrument_signals(c)
    assert kinds(sigs) == [("regime", "alta")]
    assert "fase agitata" in sigs[0]["text"] and "1 seduta fa" in sigs[0]["text"]


def test_uscita_da_turbolento_media_e_regime_vecchio_ignorato():
    assert kinds(digest.instrument_signals(ctx(states=[2] * 9 + [1] * 3, current="normale", days=3))) == [("regime", "media")]
    assert digest.instrument_signals(ctx(states=[2] * 2 + [1] * 10, current="normale", days=10)) == []


def test_anomalie_solo_recenti():
    an = [{"date": DATES[-1], "kind": "rendimento", "z": -6.5, "value": -0.08},
          {"date": DATES[0], "kind": "gap", "z": 5.0, "value": 0.04},          # vecchia
          {"date": DATES[-2], "kind": "volume", "z": 4.2, "value": 1.2}]
    sigs = digest.instrument_signals(ctx(anomalies=an))
    assert kinds(sigs) == [("anomalia", "alta"), ("anomalia", "media")]
    assert "×3,3 la mediana" in sigs[1]["text"]


def test_volatilita_e_drawdown():
    sigs = digest.instrument_signals(ctx(pctl=0.97, feats={"rv_5": 0.6, "rv_63": 0.3, "drawdown": -0.25}))
    assert sorted(kinds(sigs)) == [("drawdown", "media"), ("volatilità", "alta"), ("volatilità", "media")]
    assert kinds(digest.instrument_signals(ctx(pctl=0.03))) == [("volatilità", "bassa")]


def test_dati_non_aggiornati():
    assert kinds(digest.instrument_signals(ctx(status="stale")))[0] == ("dati", "alta")


def test_build_ordina_e_raggruppa():
    cs = [ctx("calm"), ctx("hot", pctl=0.97), ctx("dd", feats={"drawdown": -0.3})]
    dg = digest.build(cs, {"missing": ["zzz"]})
    assert [i["id"] for i in dg["instruments"]] == ["hot", "dd"]
    assert dg["quiet"] == ["calm"] and dg["missing"] == ["zzz"]
    assert "HOT" in dg["headline"]
    md = digest.to_markdown(dg)
    assert "Nessuna novità: CALM" in md and "Senza dati: zzz" in md


def test_volatilita_diffusa_detta_una_volta():
    cs = [ctx(f"t{i}", pctl=p) for i, p in enumerate([0.95, 0.93, 0.995, 0.4])]
    dg = digest.build(cs)
    assert dg["market_wide_high_vol"]
    assert "3 strumenti su 4" in dg["market"][0]
    flagged = {i["id"] for i in dg["instruments"]}
    assert flagged == {"t2"}   # solo il caso estremo (>= 99° percentile)


def test_nessun_segnale_headline():
    dg = digest.build([ctx("a"), ctx("b")])
    assert dg["instruments"] == [] and "Nessun segnale" in dg["headline"]


def test_cambio_di_oggi():
    sigs = digest.instrument_signals(ctx(states=[0] * 11 + [2], current="turbolento", days=1))
    assert "oggi" in sigs[0]["text"] and "prima era in fase calma" in sigs[0]["text"]


def test_note_storiche_solo_su_strumenti_segnalati():
    dg = digest.build([ctx("hot", pctl=0.97), ctx("calm"), ctx("calm2")])
    events = [{"id": "hot", "type": "vol_estrema", "date": DATES[-1], "history": "storicamente, ..."},
              {"id": "calm", "type": "vol_estrema", "date": DATES[-1], "history": "storicamente, ..."},
              {"id": "hot", "type": "drawdown_20", "date": DATES[-1], "history": None}]
    digest.attach_history(dg, events, {"vol_estrema": "volatilità estrema"})
    hot = dg["instruments"][0]
    assert [h["type"] for h in hot["history"]] == ["vol_estrema"]
    assert "Volatilità estrema" in digest.to_markdown(dg)


def test_anomalia_nel_giorno_della_trimestrale_e_spiegata():
    an = [{"date": DATES[-1], "kind": "rendimento", "z": -7.0, "value": -0.12, "earnings": True}]
    sigs = digest.instrument_signals(ctx(anomalies=an))
    assert kinds(sigs) == [("anomalia", "media")] and "trimestrale" in sigs[0]["text"]


def test_trimestrale_in_arrivo():
    c = ctx()
    c.earnings = {"available": True, "typical_move": 0.06, "move_ratio": 4.2,
                  "next": {"date": "2026-10-28", "timing": "dopo la chiusura", "sessions_until": 3}}
    sigs = digest.instrument_signals(c)
    assert kinds(sigs) == [("trimestrale", "media")]
    assert "tra 3 sedute" in sigs[0]["text"] and "4 volte" in sigs[0]["text"]
    c.earnings["next"]["sessions_until"] = 15
    assert digest.instrument_signals(c) == []
