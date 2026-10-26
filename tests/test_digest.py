"""Test della sintesi automatica (regole esplicite su contesti costruiti a mano)."""

from __future__ import annotations

from pipeline import digest
from pipeline.digest import Context

DATES = [f"2026-10-{d:02d}" for d in range(12, 24)]
STATES = [{"label": l} for l in ("calmo", "normale", "turbolento")]


def ctx(id_="aaa", states=None, current="normale", days=30, pctl=0.5, feats=None, anomalies=None,
        status=None, cls="stock", events=None):
    states = states if states is not None else [1] * len(DATES)
    return Context(
        id=id_, name=id_.upper(), asset_class=cls, as_of=DATES[-1], recent_dates=DATES[-5:],
        regime={"available": True, "current": current, "days_in_state": days,
                "probabilities": {current: 0.9}, "states": STATES},
        regime_states=states, regime_dates=DATES,
        features={"rv_21": 0.3, "rv_5": 0.3, "rv_63": 0.3, "drawdown": -0.05, **(feats or {})},
        rv21_percentile=pctl, anomalies=anomalies or [], data_status=status, recent_events=events or [],
    )


def kinds(sigs):
    return [(s["kind"], s["severity"]) for s in sigs]


def situations(c, wide=False):
    return [x["kind"] for x in digest.instrument_situations(c, digest.instrument_signals(c, wide), wide)]


# ---------------------------------------------------------------------------
# Novità
# ---------------------------------------------------------------------------

def test_nessun_segnale_strumento_tranquillo():
    c = ctx()
    assert digest.instrument_signals(c) == [] and situations(c) == []


def test_passaggio_a_turbolento_alta():
    c = ctx(states=[1] * 10 + [2] * 2, current="turbolento", days=2)
    sigs = digest.instrument_signals(c)
    assert kinds(sigs) == [("regime", "alta")]
    assert "turbolento" in sigs[0]["text"] and "1 seduta fa" in sigs[0]["text"]
    assert situations(c) == []              # da poche sedute: è una novità, non una situazione


def test_uscita_da_turbolento_media_e_regime_vecchio_ignorato():
    assert kinds(digest.instrument_signals(ctx(states=[2] * 9 + [1] * 3, current="normale", days=3))) == [("regime", "media")]
    assert digest.instrument_signals(ctx(states=[2] * 2 + [1] * 10, current="normale", days=10)) == []


def test_cambio_di_oggi():
    sigs = digest.instrument_signals(ctx(states=[0] * 11 + [2], current="turbolento", days=1))
    assert "oggi" in sigs[0]["text"] and "dal regime calmo" in sigs[0]["text"]


def test_anomalie_raggruppate_per_giorno():
    an = [{"date": DATES[-1], "kind": "range", "z": 4.5, "value": 0.06},
          {"date": DATES[-1], "kind": "rendimento", "z": -6.5, "value": -0.08},
          {"date": DATES[-1], "kind": "rendimento_vs_bench", "z": -5.0, "value": -0.07},
          {"date": DATES[-2], "kind": "volume", "z": 4.2, "value": 1.2},
          {"date": DATES[0], "kind": "gap", "z": 5.0, "value": 0.04}]          # vecchia
    sigs = digest.instrument_signals(ctx(anomalies=an))
    assert kinds(sigs) == [("anomalia", "alta"), ("anomalia", "media")]       # un segnale per giorno
    first = sigs[0]["text"]
    assert first.startswith("movimento anomalo il 23/10")
    # Ordine fisso dei tipi: rendimento, rispetto al benchmark, ..., escursione
    assert first.index("rendimento -8,0%") < first.index("rispetto al benchmark") < first.index("escursione")
    assert "×3,3 la mediana" in sigs[1]["text"]


def test_volatilita_appena_salita_e_novita():
    c = ctx(pctl=0.97, events=[{"type": "vol_estrema", "date": DATES[-2]}])
    sigs = digest.instrument_signals(c)
    assert kinds(sigs) == [("volatilità", "alta")] and "il 22/10" in sigs[0]["text"]
    assert situations(c) == []              # non ripetuta tra le situazioni


def test_discesa_sotto_20_appena_avvenuta_e_novita():
    c = ctx(feats={"drawdown": -0.22}, events=[{"type": "drawdown_20", "date": DATES[-1]}])
    assert kinds(digest.instrument_signals(c)) == [("drawdown", "media")]
    assert situations(c) == []


def test_accelerazione_e_dati():
    assert kinds(digest.instrument_signals(ctx(feats={"rv_5": 0.6, "rv_63": 0.3}))) == [("volatilità", "media")]
    assert kinds(digest.instrument_signals(ctx(status="stale")))[0] == ("dati", "alta")


# ---------------------------------------------------------------------------
# Situazioni in corso
# ---------------------------------------------------------------------------

def test_situazioni_stati_che_durano():
    c = ctx(current="turbolento", days=25, states=[2] * len(DATES), pctl=0.93, feats={"drawdown": -0.3})
    assert digest.instrument_signals(c) == []
    assert situations(c) == ["regime", "drawdown", "volatilità"]
    assert situations(ctx(pctl=0.03)) == ["calma"]


def test_build_novita_situazioni_e_norma():
    cs = [ctx("calm"), ctx("hot", pctl=0.97), ctx("dd", feats={"drawdown": -0.3}),
          ctx("new", states=[1] * 10 + [2] * 2, current="turbolento", days=2)]
    dg = digest.build(cs, {"missing": ["zzz"]})
    assert [i["id"] for i in dg["instruments"]] == ["new"]
    assert [s["id"] for s in dg["situations"]] == ["dd", "hot"]
    assert dg["quiet"] == ["calm"] and dg["missing"] == ["zzz"]
    assert "priorità alta: NEW" in dg["headline"] and "2 situazioni in corso" in dg["headline"]
    md = digest.to_markdown(dg)
    assert "#### Novità" in md and "#### Situazioni in corso" in md
    assert "Nella norma: CALM" in md and "Senza dati: zzz" in md


def test_volatilita_diffusa_detta_una_volta():
    cs = [ctx(f"t{i}", pctl=p) for i, p in enumerate([0.95, 0.93, 0.995, 0.4])]
    dg = digest.build(cs)
    assert dg["market_wide_high_vol"]
    assert "3 strumenti su 4" in dg["market"][0]
    assert {s["id"] for s in dg["situations"]} == {"t2"}   # solo il caso estremo (>= 99° percentile)


def test_nessun_segnale_headline():
    dg = digest.build([ctx("a"), ctx("b")])
    assert dg["instruments"] == [] and dg["situations"] == []
    assert dg["headline"] == "Nessuna novità."


def test_note_storiche_solo_su_strumenti_con_novita():
    hot = ctx("hot", pctl=0.97, events=[{"type": "vol_estrema", "date": DATES[-1]}])
    dg = digest.build([hot, ctx("calm"), ctx("calm2")])
    events = [{"id": "hot", "type": "vol_estrema", "date": DATES[-1], "history": "storicamente, ..."},
              {"id": "calm", "type": "vol_estrema", "date": DATES[-1], "history": "storicamente, ..."},
              {"id": "hot", "type": "drawdown_20", "date": DATES[-1], "history": None}]
    digest.attach_history(dg, events, {"vol_estrema": "volatilità estrema"})
    assert [h["type"] for h in dg["instruments"][0]["history"]] == ["vol_estrema"]
    assert "Volatilità estrema" in digest.to_markdown(dg)
