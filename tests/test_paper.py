"""Test offline del paper trading: pesi obiettivo, ordini, log append-only, config.

Nessuna chiamata di rete e nessun bisogno di alpaca-py (importato solo dal broker).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from pipeline import paper
from pipeline.paper import PaperError, Strategy

ROOT = Path(__file__).resolve().parents[1]


def inst(id_, situation="norma", risk="normale", status="updated", asset_class="stock"):
    return {"id": id_, "asset_class": asset_class, "data_status": status,
            "situation": {"key": situation, "label": situation.capitalize()},
            "risk": {"level": risk}}


SYMBOLS = {"aapl": "AAPL", "msft": "MSFT", "nvda": "NVDA", "amd": "AMD"}


def assessment(*instruments, as_of="2026-11-19"):
    return {"as_of": as_of, "instruments": list(instruments)}


RULE = Strategy("regola_assess", "regola", "assess_rule", "ALPACA_RULE",
                {"allowed_situations": ["norma", "tranquillo", "ripresa"], "block_risk": ["alto"], "max_weight": 0.15})


IND = Strategy("indicazioni", "indicazioni", "indication", "ALPACA_HEDGE", {})
ALLOWED = {"PSQ", "QQQ", "SH", "SPY", "AAPL"}
SIGNALS = {"ndx", "qqq", "spx", "aapl"}
FIRST = {"from": "2026-11-20", "author": "Simone", "bias": "ribassista", "instrument": "PSQ", "max_exposure": 0.2,
         "signal": "ndx", "enter_situations": ["tempesta"], "enter_risk": ["alto"],
         "exit_situations": ["norma", "tranquillo"], "note": "prova"}


def ind(**over):
    return paper.parse_indication({**FIRST, **over}, "x.yml", ALLOWED, SIGNALS)


# --- pesi obiettivo ---------------------------------------------------------

def test_regola_accetta_dati_refreshed_e_up_to_date():
    a = assessment(inst("aapl", status="refreshed"), inst("msft", status="up_to_date"), inst("nvda", status="stale"))
    assert set(paper.target_assess_rule(RULE, a, SYMBOLS).weights) == {"AAPL", "MSFT"}


@pytest.mark.parametrize("sit,risk,held,expected", [
    ("tempesta", "alto", False, 0.2),       # ingresso
    ("tempesta", "normale", False, None),   # tempesta senza rischio alto: non entra
    ("pressione", "alto", False, None),     # pressione: non è un ingresso
    ("pressione", "alto", True, 0.2),       # ma se già dentro resta
    ("ripresa", "normale", True, 0.2),
    ("norma", "normale", True, None),       # esce
    ("tranquillo", "basso", True, None),
])
def test_indicazione_entra_resta_esce(sit, risk, held, expected):
    a = assessment(inst("ndx", sit, risk, asset_class="index"), inst("aapl"), as_of="2026-11-20")
    t = paper.target_indication(ind(), a, {"PSQ"} if held else set())
    assert t.weights.get("PSQ") == expected


def test_indicazione_con_dati_vecchi_non_entra_ma_non_esce():
    a = assessment(inst("ndx", "norma", "normale", status="stale", asset_class="index"), as_of="2026-11-20")
    assert paper.target_indication(ind(), a, set()).weights == {}
    assert paper.target_indication(ind(), a, {"PSQ"}).weights == {"PSQ": 0.2}


def test_indicazione_in_vigore_e_cambio_strumento():
    old, new = ind(), ind(**{"from": "2026-11-27", "bias": "neutrale"})
    assert paper.current_indication([old, new], "2026-11-19") is None          # prima della prima: niente
    assert paper.current_indication([old, new], "2026-11-26") is old
    assert paper.current_indication([old, new], "2026-11-27") is new
    a = assessment(inst("ndx", "tempesta", "alto", asset_class="index"), as_of="2026-11-27")
    assert paper.target_indication(new, a, {"PSQ"}).weights == {}             # neutrale: si chiude PSQ


@pytest.mark.parametrize("over,msg", [
    ({"instrument": "SQQQ"}, "non ammesso"),          # niente leva
    ({"max_exposure": 0.8}, "max_exposure"),
    ({"bias": "boh"}, "bias"),
    ({"from": "20/11/2026"}, "AAAA-MM-GG"),
    ({"signal": "btc"}, "universo"),
    ({"enter_situations": ["panico"]}, "non validi"),
    ({"exit_situations": ["tempesta"]}, "sia di ingresso sia di uscita"),
])
def test_indicazione_non_valida(over, msg):
    with pytest.raises(PaperError, match=msg):
        ind(**over)


def test_indicazioni_del_repo_valide_e_prima_indicazione():
    cfg = paper.load_config(ROOT / "config" / "paper.yml")
    allowed = set(cfg.allowed_instruments) | set(paper.stock_symbols().values())
    signals = {"ndx", "qqq", "spx", *paper.stock_symbols()}
    items = paper.load_indications(ROOT / "config" / "indications", allowed, signals)
    first = items[0]
    assert (first.valid_from, first.instrument, first.max_exposure, first.signal) == ("2026-11-20", "PSQ", 0.2, "ndx")
    assert first.enter_situations == ("tempesta",) and first.enter_risk == ("alto",)
    assert set(first.exit_situations) == {"norma", "tranquillo"}
    assert "SQQQ" not in allowed and "TQQQ" not in allowed


def test_indicazione_usa_la_posizione_del_conto_come_stato(tmp_path):
    class FakeBroker(BaseFake):
        def positions(self): return {"PSQ": (20_000.0, 500.0)}
        def submit(self, plan, coid): return "id", "accettato (accepted)"
    cfg = paper.PaperConfig(0.02, (IND,))
    a = assessment(inst("ndx", "pressione", "alto", asset_class="index"), as_of="2026-11-20")
    res = paper.run_strategy(IND, a, SYMBOLS, cfg, FakeBroker(), tmp_path, send=False, indications=[ind()])
    assert res["weights"] == {"PSQ": 0.2} and res["orders"] == []          # già dentro: nessun ordine
    a = assessment(inst("ndx", "norma", "normale", asset_class="index"), as_of="2026-11-20")
    res = paper.run_strategy(IND, a, SYMBOLS, cfg, FakeBroker(), tmp_path, send=False, indications=[ind()])
    assert [(o.side, o.symbol, o.qty) for o in res["orders"]] == [("sell", "PSQ", 500.0)]


def test_regola_esclude_tempesta_rischio_alto_e_dati_vecchi():
    a = assessment(inst("aapl"), inst("msft", "tempesta"), inst("nvda", "norma", "alto"),
                   inst("amd", status="stale"))
    t = paper.target_assess_rule(RULE, a, SYMBOLS)
    assert list(t.weights) == ["AAPL"]
    assert t.weights["AAPL"] == 0.15          # un solo titolo: vale il tetto, il resto è liquidità


def test_regola_pari_peso_senza_superare_il_tetto():
    ids = ["aapl", "msft", "nvda", "amd"]
    t = paper.target_assess_rule(RULE, assessment(*[inst(i) for i in ids]), SYMBOLS)
    assert set(t.weights) == {"AAPL", "MSFT", "NVDA", "AMD"}
    assert all(w == 0.15 for w in t.weights.values())       # 1/4 = 25% > tetto 15%
    many = {f"x{i}": f"X{i}" for i in range(10)}
    t10 = paper.target_assess_rule(RULE, assessment(*[inst(i) for i in many]), many)
    assert all(abs(w - 0.10) < 1e-12 for w in t10.weights.values())
    assert sum(t10.weights.values()) <= 1.0 + 1e-12


def test_regola_ignora_indici_ed_etf_e_tutto_cash_se_nessuno_passa():
    a = assessment(inst("ndx", asset_class="index"), inst("qqq", asset_class="etf"), inst("aapl", "tempesta"))
    t = paper.target_assess_rule(RULE, a, SYMBOLS)
    assert t.weights == {} and "liquidità" in t.note


def test_buy_hold_un_solo_simbolo():
    s = Strategy("bench_qqq", "b", "buy_hold", "ALPACA_BENCH", {"symbol": "qqq", "weight": 0.99})
    t = paper.compute_target(s, assessment(), SYMBOLS)
    assert t.weights == {"QQQ": 0.99}


# --- ordini -----------------------------------------------------------------

def test_primo_acquisto_per_importo():
    orders = paper.plan_orders(100_000, {}, {"QQQ": 0.99}, 0.02)
    assert [(o.symbol, o.side, o.notional, o.qty) for o in orders] == [("QQQ", "buy", 99_000.0, None)]


def test_scostamento_piccolo_non_genera_ordini():
    # peso attuale 14,5% contro obiettivo 15%: sotto la soglia del 2% del conto
    assert paper.plan_orders(100_000, {"AAPL": (14_500.0, 50.0)}, {"AAPL": 0.15}, 0.02) == []


def test_uscita_completa_a_quantita_e_vendite_prima_degli_acquisti():
    positions = {"MSFT": (15_000.0, 30.5), "AAPL": (15_000.0, 60.0)}
    orders = paper.plan_orders(100_000, positions, {"AAPL": 0.15, "NVDA": 0.15}, 0.02)
    assert [(o.symbol, o.side) for o in orders] == [("MSFT", "sell"), ("NVDA", "buy")]
    assert orders[0].qty == 30.5 and orders[0].notional is None      # chiusura per intero, nessun residuo
    assert orders[1].notional == 15_000.0


def test_riduzione_parziale_vende_importo():
    orders = paper.plan_orders(100_000, {"AAPL": (30_000.0, 100.0)}, {"AAPL": 0.15}, 0.02)
    assert [(o.side, o.notional) for o in orders] == [("sell", 15_000.0)]


def test_conto_non_valido():
    with pytest.raises(PaperError):
        paper.plan_orders(0, {}, {"QQQ": 1.0}, 0.02)


# --- id ordini --------------------------------------------------------------

def test_client_order_id_roundtrip():
    coid = paper.client_order_id("regola_assess", "2026-11-19", "BRK.B")
    assert paper.parse_client_order_id(coid) == ("regola_assess", "2026-11-19", "BRK.B")
    assert paper.parse_client_order_id("altro-ordine") is None
    assert paper.parse_client_order_id("") is None


# --- log append-only --------------------------------------------------------

def test_append_rows_non_duplica_e_non_riscrive(tmp_path):
    path = tmp_path / "equity.csv"
    row = {"as_of": "2026-11-19", "strategy": "bench_qqq", "equity": "100000.00", "cash": "1000.00",
           "invested": "99000.00", "n_positions": 1, "created_utc": "t1"}
    assert paper.append_rows(path, paper.EQUITY_COLUMNS, [row], ("as_of", "strategy")) == 1
    changed = {**row, "equity": "5.00", "created_utc": "t2"}
    assert paper.append_rows(path, paper.EQUITY_COLUMNS, [changed], ("as_of", "strategy")) == 0
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2 and "100000.00" in lines[1]               # la riga originale non cambia
    other = {**row, "as_of": "2026-11-20"}
    assert paper.append_rows(path, paper.EQUITY_COLUMNS, [other], ("as_of", "strategy")) == 1


# --- config e valutazioni ---------------------------------------------------

def test_config_del_repo_valida():
    cfg = paper.load_config(ROOT / "config" / "paper.yml")
    assert {s.id for s in cfg.strategies} == {"bench_qqq", "regola_assess", "indicazioni"}
    prefixes = [s.secret_prefix for s in cfg.strategies]
    assert len(set(prefixes)) == len(prefixes)                        # un conto per strategia


def test_config_rifiuta_id_con_trattino_e_kind_ignoto(tmp_path):
    bad = tmp_path / "p.yml"
    bad.write_text("strategies:\n  - {id: a-b, kind: buy_hold, symbol: QQQ, secret_prefix: X}\n", encoding="utf-8")
    with pytest.raises(PaperError):
        paper.load_config(bad)
    bad.write_text("strategies:\n  - {id: ab, kind: boh, secret_prefix: X}\n", encoding="utf-8")
    with pytest.raises(PaperError):
        paper.load_config(bad)


def test_valutazioni_vecchie_rifiutate(tmp_path):
    f = tmp_path / "assessment.json"
    f.write_text(json.dumps(assessment(inst("aapl"), as_of="2026-11-01")), encoding="utf-8")
    with pytest.raises(PaperError):
        paper.load_assessment(f, today=dt.date(2026, 11, 20))
    ok = paper.load_assessment(f, today=dt.date(2026, 11, 3))
    assert ok["as_of"] == "2026-11-01"


def test_simboli_dell_universo_solo_azioni():
    sym = paper.stock_symbols(ROOT / "config" / "universe.yml")
    assert sym["nvda"] == "NVDA" and "qqq" not in sym and "ndx" not in sym


def test_senza_chiavi_nessun_broker():
    assert paper.broker_for(RULE, env={}) is None


CLOSED_AFTER_19 = paper.MarketState(False, dt.date(2026, 11, 19), dt.datetime(2026, 11, 20, 9, 30))


class BaseFake:
    """Conto paper finto: mercato chiuso dopo la seduta del 19/11, nessun ordine pendente."""
    market_state = CLOSED_AFTER_19
    pending: list = []
    used: set = set()

    def account(self): return 100_000.0, 100_000.0
    def positions(self): return {}
    def market(self): return self.market_state
    def open_orders(self, strategy_id): return list(self.pending)
    def submitted_ids(self, since): return set(self.used)


def test_blocco_mercato_aperto_dati_vecchi_ordini_pendenti():
    assert paper.send_blocker(CLOSED_AFTER_19, "2026-11-19", []) is None
    aperto = paper.MarketState(True, dt.date(2026, 11, 19), dt.datetime(2026, 11, 21, 9, 30))
    assert "aperto" in paper.send_blocker(aperto, "2026-11-19", [])
    assert "non aggiornati" in paper.send_blocker(CLOSED_AFTER_19, "2026-11-18", [])
    assert "in attesa" in paper.send_blocker(CLOSED_AFTER_19, "2026-11-19", ["AAPL"])


def test_bloccato_registra_solo_il_valore_del_conto(tmp_path):
    class FakeBroker(BaseFake):
        pending = ["AAPL"]                                   # es. festivo: ordini di ieri non ancora eseguiti
        def submit(self, *a): raise AssertionError("con ordini pendenti non si invia nulla")
    cfg = paper.PaperConfig(0.02, (RULE,))
    res = paper.run_strategy(RULE, assessment(inst("aapl")), SYMBOLS, cfg, FakeBroker(), tmp_path, send=True)
    assert "in attesa" in res["blocked"] and res["sent"] is False
    assert [p.name for p in tmp_path.iterdir()] == [paper.EQUITY_FILE]


def test_ordine_gia_usato_non_si_reinvia(tmp_path):
    class FakeBroker(BaseFake):
        used = {"fn-regola_assess-2026-11-19-AAPL"}
        def submit(self, *a): raise AssertionError("id già usato: non va reinviato")
    cfg = paper.PaperConfig(0.02, (RULE,))
    res = paper.run_strategy(RULE, assessment(inst("aapl")), SYMBOLS, cfg, FakeBroker(), tmp_path, send=True)
    assert res["sent"] is True and not (tmp_path / paper.ORDERS_FILE).exists()


def test_ordine_rifiutato_segnalato(tmp_path):
    class FakeBroker(BaseFake):
        def submit(self, plan, coid): return "", "rifiutato: insufficient buying power (codice 40310000)"
    cfg = paper.PaperConfig(0.02, (RULE,))
    res = paper.run_strategy(RULE, assessment(inst("aapl")), SYMBOLS, cfg, FakeBroker(), tmp_path, send=True)
    assert res["rejected"] == ["AAPL"]


def test_senza_send_non_scrive_nulla(tmp_path):
    class FakeBroker(BaseFake):
        def submit(self, *a): raise AssertionError("in prova non si devono inviare ordini")
    cfg = paper.PaperConfig(0.02, (RULE,))
    a = assessment(inst("aapl"))
    res = paper.run_strategy(RULE, a, SYMBOLS, cfg, FakeBroker(), tmp_path, send=False)
    assert res["sent"] is False and len(res["orders"]) == 1
    assert list(tmp_path.iterdir()) == []


def test_con_send_registra_decisioni_ordini_equity_e_riprova_senza_duplicati(tmp_path):
    sent = []

    class FakeBroker(BaseFake):
        def submit(self, plan, coid):
            sent.append(coid)
            return "id-1", "accettato (accepted)"
    cfg = paper.PaperConfig(0.02, (RULE,))
    a = assessment(inst("aapl"))
    for _ in range(2):                                        # secondo giro = rilancio dello stesso giorno
        paper.run_strategy(RULE, a, SYMBOLS, cfg, FakeBroker(), tmp_path, send=True)
    for name in (paper.DECISIONS_FILE, paper.ORDERS_FILE, paper.EQUITY_FILE):
        assert len((tmp_path / name).read_text(encoding="utf-8").strip().splitlines()) == 2   # intestazione + 1
    assert sent == ["fn-regola_assess-2026-11-19-AAPL"] * 2   # l'id è stabile: Alpaca rifiuta il duplicato


def test_nessuna_chiave_segreta_nel_codice():
    text = (ROOT / "pipeline" / "paper.py").read_text(encoding="utf-8")
    assert "paper=True" in text and "paper=False" not in text   # mai conto reale


def test_broker_legge_orologio_e_calendario_con_i_modelli_veri_di_alpaca():
    """Usa i modelli di alpaca-py (parsing reale delle risposte) con un client finto, senza rete."""
    pytest.importorskip("alpaca")
    from alpaca.trading.models import Calendar, Clock, Order

    class FakeClient:
        def get_clock(self):     # sabato 21/11/2026, 00:30 a New York
            return Clock(timestamp="2026-11-21T00:30:00-05:00", is_open=False,
                         next_open="2026-11-23T09:30:00-05:00", next_close="2026-11-23T16:00:00-05:00")

        def get_calendar(self, req):
            return [Calendar(date=d, open="09:30", close="16:00") for d in ("2026-11-19", "2026-11-20")]

        def get_orders(self, req):
            base = {"id": "61e69015-8549-4bfd-b9c3-01e75843f47d", "created_at": "2026-11-20T23:00:00Z",
                    "updated_at": "2026-11-20T23:00:00Z", "submitted_at": "2026-11-20T23:00:00Z",
                    "asset_id": "b0b6dd9d-8b9b-48a9-ba46-b9d54906e415", "symbol": "AAPL", "asset_class": "us_equity",
                    "notional": "15000", "filled_qty": "0", "order_class": "simple", "order_type": "market",
                    "type": "market", "side": "buy", "time_in_force": "day", "status": "accepted",
                    "extended_hours": False}
            return [Order(**base, client_order_id="fn-regola_assess-2026-11-20-AAPL"),
                    Order(**{**base, "symbol": "MSFT"}, client_order_id="manuale-123")]

    b = paper.AlpacaPaperBroker.__new__(paper.AlpacaPaperBroker)
    b._client = FakeClient()
    m = b.market()
    assert m.is_open is False and m.last_session == dt.date(2026, 11, 20)
    assert b.open_orders("regola_assess") == ["AAPL"] and b.open_orders("bench_qqq") == []
    assert "fn-regola_assess-2026-11-20-AAPL" in b.submitted_ids(dt.datetime(2026, 11, 1))


def test_json_dashboard_solo_percentuali_e_confronto_col_benchmark(tmp_path):
    bench = Strategy("bench_qqq", "Benchmark", "buy_hold", "ALPACA_BENCH", {"symbol": "QQQ"})
    cfg = paper.PaperConfig(0.02, (bench, RULE))
    eq = [("2026-11-19", "bench_qqq", 100_000, 100_000), ("2026-11-20", "bench_qqq", 101_000, 1_000),
          ("2026-11-19", "regola_assess", 100_000, 100_000), ("2026-11-20", "regola_assess", 100_500, 85_000)]
    paper.append_rows(tmp_path / paper.EQUITY_FILE, paper.EQUITY_COLUMNS,
                      [{"as_of": d, "strategy": s, "equity": e, "cash": c, "invested": e - c, "n_positions": 1,
                        "created_utc": "x"} for d, s, e, c in eq], ("as_of", "strategy"))
    paper.append_rows(tmp_path / paper.DECISIONS_FILE, paper.DECISIONS_COLUMNS,
                      [{"as_of": "2026-11-20", "strategy": "regola_assess", "symbol": "AAPL", "target_weight": "0.15",
                        "reason": "Nella norma, rischio normale", "created_utc": "x"}], ("as_of", "strategy", "symbol"))
    coid = "fn-regola_assess-2026-11-20-AAPL"
    paper.append_rows(tmp_path / paper.ORDERS_FILE, paper.ORDERS_COLUMNS,
                      [{"as_of": "2026-11-20", "strategy": "regola_assess", "symbol": "AAPL", "side": "buy",
                        "notional": "15075.00", "qty": "", "client_order_id": coid, "alpaca_order_id": "id",
                        "result": "accettato (accepted)", "created_utc": "x"}], ("client_order_id",))
    paper.append_rows(tmp_path / paper.FILLS_FILE, paper.FILLS_COLUMNS,
                      [{"order_id": "id", "client_order_id": coid, "as_of": "2026-11-20", "strategy": "regola_assess",
                        "symbol": "AAPL", "side": "buy", "status": "filled", "filled_qty": "55.1",
                        "filled_avg_price": "273.51", "filled_at": "x"}], ("order_id",))
    out = paper.dashboard_payload(cfg, tmp_path, {"regola_assess": {"as_of": "2026-11-20", "n_orders": 1}})
    b, r = out["strategies"]
    assert b["return"] == pytest.approx(0.01) and r["return"] == pytest.approx(0.005)
    assert r["cash_share"] == pytest.approx(0.8458, abs=1e-4) and r["weights"][0]["symbol"] == "AAPL"
    c = out["comparisons"][0]
    assert c["excess"][-1] == pytest.approx(-0.005) and c["verdict"] == "presto"
    o = out["orders"][0]
    assert o["share"] == pytest.approx(0.15) and o["sent"] == "accettato" and o["status"] == "filled"
    text = json.dumps(out)
    for forbidden in ("273.51", "filled_avg_price", "15075", "100500", "100000"):   # niente prezzi né importi
        assert forbidden not in text


def test_indicazioni_fuori_dal_confronto_col_benchmark(tmp_path):
    bench = Strategy("bench_qqq", "Benchmark", "buy_hold", "ALPACA_BENCH", {"symbol": "QQQ"})
    cfg = paper.PaperConfig(0.02, (bench, IND))
    paper.append_rows(tmp_path / paper.EQUITY_FILE, paper.EQUITY_COLUMNS,
                      [{"as_of": "2026-11-20", "strategy": s, "equity": 100_000, "cash": 100_000}
                       for s in ("bench_qqq", "indicazioni")], ("as_of", "strategy"))
    out = paper.dashboard_payload(cfg, tmp_path)
    assert out["comparisons"] == [] and out["strategies"][1]["kind"] == "indication"
    meta = paper.indications_payload(cfg, [ind()], "2026-11-20", ALLOWED, SIGNALS)
    assert meta["current"]["instrument"] == "PSQ" and "SQQQ" not in meta["options"]["instruments"]


def test_yaml_scritto_dalla_pagina_si_rilegge():
    """Stesso formato prodotto da indicationYaml() in docs/app.js (stringhe JSON, liste in linea, null)."""
    text = ('from: 2026-11-27\nauthor: "Simone"\nbias: neutrale\ninstrument: null\nmax_exposure: 0\nsignal: ndx\n'
            'enter_situations: []\nenter_risk: []\nexit_situations: [norma, tranquillo]\nnote: "Pausa: \\"vedo\\" calma."\n')
    import yaml
    i = paper.parse_indication(yaml.safe_load(text), "2026-11-27.yml", ALLOWED, SIGNALS)
    assert i.instrument is None and i.note == 'Pausa: "vedo" calma.'
