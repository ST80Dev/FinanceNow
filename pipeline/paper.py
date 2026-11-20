"""Paper trading su Alpaca: portafogli virtuali che applicano le valutazioni di FinanceNow.

Ogni strategia di `config/paper.yml` ha un proprio conto Alpaca PAPER (chiavi in variabili
d'ambiente `<secret_prefix>_KEY` / `<secret_prefix>_SECRET`, in produzione GitHub Secrets).
Gira a mercato chiuso, dopo `daily.yml`: legge `docs/data/assessment.json`, calcola i pesi
obiettivo, invia ordini a mercato "day" per importo (azioni frazionarie) che Alpaca accoda
ed esegue all'apertura della seduta successiva (stessa convenzione dei pattern: il primo
prezzo a cui si può agire è l'apertura del giorno dopo).

Strategie (passo 6a, solo long, nessun LLM):
- `buy_hold`     un solo simbolo (QQQ) tenuto fisso: il benchmark da battere;
- `assess_rule`  azioni dell'universo con situazione e rischio ammessi da `config/paper.yml`,
                 stesso peso (con tetto), il resto in liquidità;
- `indication`   applica l'indicazione settimanale in vigore (config/indications/<data>.yml), scritta
                 da Simone anche dal modulo della pagina Paper: strumento, esposizione massima,
                 quando entrare e quando uscire secondo la valutazione di un indice.

Serie `paper` (append-only, in `results/paper/`, separata da `backtest` e `live`):
- decisions.csv  pesi obiettivo per data e strategia, con il motivo;
- orders.csv     ordini inviati (importo o quantità, id Alpaca, esito dell'invio);
- fills.csv      esecuzioni lette da Alpaca nei run successivi (quantità e prezzo medio);
- equity.csv     valore del conto a fine seduta per strategia.

Sicurezza: il client è SEMPRE creato con paper=True (nessun parametro per il conto reale) e
senza `--send` il programma non invia nulla.

Uso:
    python -m pipeline.paper                 # prova: mostra pesi e ordini, non invia
    python -m pipeline.paper --send          # invia gli ordini ai conti paper
    python -m pipeline.paper --only bench_qqq
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from pipeline.ingest import DEFAULT_UNIVERSE, ConfigError, load_universe, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "paper.yml"
DEFAULT_ASSESSMENT = ROOT / "docs" / "data" / "assessment.json"
DEFAULT_OUT_DIR = ROOT / "results" / "paper"
DEFAULT_DASHBOARD = ROOT / "docs" / "data" / "paper.json"
MIN_SESSIONS_VERDICT = 63        # sotto i 3 mesi di sedute in comune il confronto col benchmark è solo rumore
RECENT_ORDERS = 40

log = logging.getLogger("financenow.paper")

ID_PREFIX = "fn"                 # prefisso dei client_order_id: riconosce gli ordini di questo programma
MAX_ASSESSMENT_AGE_DAYS = 4      # oltre, le valutazioni sono vecchie (weekend lunghi inclusi)
MIN_ORDER_USD = 1.0
FILLS_LOOKBACK_DAYS = 10

DECISIONS_FILE, ORDERS_FILE, FILLS_FILE, EQUITY_FILE = "decisions.csv", "orders.csv", "fills.csv", "equity.csv"
STOPS_FILE = "stops.csv"
STOPS_COLUMNS = ["indication_from", "strategy", "symbol", "as_of", "loss", "created_utc"]
DECISIONS_COLUMNS = ["as_of", "strategy", "symbol", "target_weight", "reason", "created_utc"]
ORDERS_COLUMNS = ["as_of", "strategy", "symbol", "side", "notional", "qty", "client_order_id",
                  "alpaca_order_id", "result", "created_utc"]
FILLS_COLUMNS = ["order_id", "client_order_id", "as_of", "strategy", "symbol", "side", "status",
                 "filled_qty", "filled_avg_price", "filled_at"]
EQUITY_COLUMNS = ["as_of", "strategy", "equity", "cash", "invested", "n_positions", "created_utc"]


class PaperError(Exception):
    """Errore che impedisce di operare (config o valutazioni non utilizzabili)."""


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

KINDS = ("buy_hold", "assess_rule", "indication")


@dataclass(frozen=True)
class Strategy:
    id: str
    name: str
    kind: str
    secret_prefix: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PaperConfig:
    min_delta: float
    strategies: tuple[Strategy, ...]
    index_instruments: dict = field(default_factory=dict)  # id indice -> {rialzo: ETF, ribasso: ETF inverso 1x}
    extra_stocks: tuple = ()                     # altre azioni del Nasdaq indicabili senza valutazione: {symbol, name}
    github_repo: str = ""                        # owner/nome, per il modulo della pagina che apre GitHub


def load_config(path: Path | str = DEFAULT_CONFIG) -> PaperConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    items = raw.get("strategies") or []
    if not items:
        raise PaperError("paper.yml: nessuna strategia definita")
    strategies, seen = [], set()
    for item in items:
        sid = item.get("id")
        if not sid or "-" in str(sid):
            raise PaperError(f"strategia senza id valido (niente trattini): {item!r}")
        if sid in seen:
            raise PaperError(f"id strategia duplicato: {sid}")
        seen.add(sid)
        if item.get("kind") not in KINDS:
            raise PaperError(f"{sid}: kind sconosciuto {item.get('kind')!r} (ammessi: {KINDS})")
        if not item.get("secret_prefix"):
            raise PaperError(f"{sid}: manca 'secret_prefix'")
        params = {k: v for k, v in item.items() if k not in ("id", "name", "kind", "secret_prefix")}
        if item["kind"] == "buy_hold" and not params.get("symbol"):
            raise PaperError(f"{sid}: buy_hold richiede 'symbol'")
        strategies.append(Strategy(sid, item.get("name", sid), item["kind"], item["secret_prefix"], params))
    min_delta = float(raw.get("min_delta", 0.02))
    if not 0 <= min_delta < 0.5:
        raise PaperError("paper.yml: min_delta deve stare tra 0 e 0,5")
    return PaperConfig(min_delta=min_delta, strategies=tuple(strategies),
                       index_instruments={str(k): {d: (str(v).upper() if v else None) for d, v in (m or {}).items()}
                                          for k, m in (raw.get("index_instruments") or {}).items()},
                       extra_stocks=tuple(raw.get("extra_stocks") or ()),
                       github_repo=str(raw.get("github_repo", "")))


def load_assessment(path: Path | str = DEFAULT_ASSESSMENT, today: dt.date | None = None) -> dict:
    data = read_json(Path(path))
    if not data or "instruments" not in data or "as_of" not in data:
        raise PaperError(f"valutazioni non trovate o incomplete: {path}")
    today = today or dt.datetime.now(dt.timezone.utc).date()
    as_of = dt.date.fromisoformat(data["as_of"])
    age = (today - as_of).days
    if age > MAX_ASSESSMENT_AGE_DAYS:
        raise PaperError(f"valutazioni vecchie: as_of {as_of} ({age} giorni fa); non opero")
    return data


# ---------------------------------------------------------------------------
# Pesi obiettivo (funzioni pure, testate offline)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Target:
    weights: dict[str, float]            # simbolo -> peso sul valore del conto (0..1)
    reasons: dict[str, str]              # simbolo -> motivo in parole semplici
    note: str = ""


def target_buy_hold(strategy: Strategy) -> Target:
    symbol = str(strategy.params["symbol"]).upper()
    weight = float(strategy.params.get("weight", 0.99))   # un filo sotto 1: margine per gli arrotondamenti
    return Target({symbol: weight}, {symbol: "benchmark: tenuto sempre"}, "benchmark")


STALE_STATUSES = ("stale", "no_data")


def is_fresh(inst: dict, as_of: str) -> bool:
    """Valutazione calcolata su dati della seduta `as_of` (updated, refreshed o up_to_date vanno tutti bene)."""
    return inst.get("data_status") not in STALE_STATUSES and inst.get("as_of", as_of) == as_of


def target_assess_rule(strategy: Strategy, assessment: dict, symbols_by_id: dict[str, str]) -> Target:
    """Azioni con situazione ammessa, rischio non bloccato e dati aggiornati, a pari peso (con tetto)."""
    p = strategy.params
    allowed = set(p.get("allowed_situations", ["norma", "tranquillo", "ripresa"]))
    blocked_risk = set(p.get("block_risk", ["alto"]))
    max_weight = float(p.get("max_weight", 0.15))
    eligible: dict[str, str] = {}
    n_stocks = 0
    for inst in assessment["instruments"]:
        symbol = symbols_by_id.get(inst["id"])
        if symbol is None:              # indici, ETF o strumenti non negoziabili: fuori dalla regola
            continue
        n_stocks += 1
        if not is_fresh(inst, assessment["as_of"]):
            continue
        sit = inst["situation"]["key"]
        risk = inst["risk"]["level"]
        if sit in allowed and risk not in blocked_risk:
            eligible[symbol] = f"{inst['situation']['label']}, rischio {risk}"
    if not eligible:
        return Target({}, {}, f"nessuno dei {n_stocks} titoli rispetta la regola: tutto in liquidità")
    weight = min(max_weight, 1.0 / len(eligible))
    return Target({s: weight for s in eligible}, eligible,
                  f"{len(eligible)} titoli su {n_stocks}, peso {weight:.1%} ciascuno")


# ---------------------------------------------------------------------------
# Indicazioni settimanali (config/indications/<data>.yml): una per file, mai modificate dopo
# ---------------------------------------------------------------------------

DEFAULT_INDICATIONS = ROOT / "config" / "indications"
DIRECTIONS = ("rialzo", "ribasso")
SITUATIONS = ("tempesta", "pressione", "ripresa", "norma", "tranquillo")
RISKS = ("alto", "normale", "basso")
MAX_VIEW_EXPOSURE = 0.5          # per riga
MAX_SHORT_EXPOSURE = 0.10        # per riga short (ribasso su un'azione): decisione di Simone del 20/11/2026
SHORT_STOP_LOSS = 0.15           # short che perde più del 15% si chiude da solo e non rientra con la stessa indicazione
WHEN = ("segnale", "sempre")     # segnale = entra/esce secondo la valutazione; sempre = tieni la quota finché non cambio
MAX_TOTAL_EXPOSURE = 1.0         # somma delle righe: niente leva
MAX_VIEWS = 6


@dataclass(frozen=True)
class Tradable:
    """Cosa si può indicare: un indice o un'azione dell'universo, con lo strumento per ogni direzione."""
    id: str
    name: str
    kind: str                        # index | etf | stock
    up: str | None                   # strumento comprato per il rialzo
    down: str | None                 # strumento per il ribasso: ETF inverso 1x (indici) o l'azione stessa (short)
    down_note: str = ""
    short: bool = False              # True: il ribasso è una vendita allo scoperto dell'azione
    has_signal: bool = True          # False: niente valutazione notturna, solo righe "sempre"


def tradables(universe_path: Path | str, cfg: "PaperConfig") -> dict[str, Tradable]:
    """Indici con i loro ETF, azioni dell'universo (con valutazione) e altre azioni del Nasdaq (senza)."""
    out = {}
    for i in load_universe(universe_path):
        if not i.active:
            continue
        if i.asset_class == "stock":
            sym = i.symbols.get("yahoo")
            out[i.id] = Tradable(i.id, i.name, "stock", sym, sym, short=True)
        elif i.id in cfg.index_instruments:
            m = cfg.index_instruments[i.id]
            out[i.id] = Tradable(i.id, i.name, i.asset_class, m.get("rialzo"), m.get("ribasso"))
    known = {t.up for t in out.values()}
    for e in cfg.extra_stocks:
        sym = str(e["symbol"]).upper()
        if sym in known:
            continue
        out[sym.lower()] = Tradable(sym.lower(), str(e["name"]), "stock", sym, sym, short=True, has_signal=False)
    return out


@dataclass(frozen=True)
class View:
    """Una riga dell'indicazione: su cosa, in che direzione, quanto, quando entrare e uscire."""
    target: str
    direction: str
    instrument: str
    max_exposure: float
    signal: str                      # di default il titolo stesso
    enter_situations: tuple[str, ...]
    enter_risk: tuple[str, ...]
    exit_situations: tuple[str, ...]
    when: str = "segnale"
    short: bool = False

    @property
    def sign(self) -> float:
        return -1.0 if self.short else 1.0

    def as_dict(self) -> dict:
        return {"target": self.target, "direction": self.direction, "instrument": self.instrument,
                "max_exposure": self.max_exposure, "signal": self.signal, "when": self.when, "short": self.short,
                "enter_situations": list(self.enter_situations), "enter_risk": list(self.enter_risk),
                "exit_situations": list(self.exit_situations)}


@dataclass(frozen=True)
class Indication:
    valid_from: str                  # in vigore dalla valutazione di questa data in poi
    author: str
    note: str
    views: tuple[View, ...]          # nessuna riga = tutto in liquidità
    file: str

    def as_dict(self) -> dict:
        return {"from": self.valid_from, "author": self.author, "note": self.note, "file": self.file,
                "views": [v.as_dict() for v in self.views]}


def _str_list(raw: dict, key: str, allowed: tuple[str, ...], where: str) -> tuple[str, ...]:
    vals = raw.get(key) or []
    if isinstance(vals, str):
        vals = [vals]
    bad = [v for v in vals if v not in allowed]
    if bad:
        raise PaperError(f"{where}: {key} non validi {bad} (ammessi: {', '.join(allowed)})")
    return tuple(vals)


def parse_view(raw: dict, where: str, catalog: dict[str, Tradable], signals: set[str]) -> View:
    if not isinstance(raw, dict):
        raise PaperError(f"{where}: riga non valida")
    target = str(raw.get("target", ""))
    t = catalog.get(target)
    if t is None:
        raise PaperError(f"{where}: {target!r} non si può indicare (ammessi: {', '.join(sorted(catalog))})")
    direction = raw.get("direction")
    if direction not in DIRECTIONS:
        raise PaperError(f"{where}: direzione {direction!r} non valida (rialzo o ribasso)")
    instrument = t.up if direction == "rialzo" else t.down
    if not instrument:
        raise PaperError(f"{where}: {t.name} al {direction}: {t.down_note or 'non disponibile'}")
    short = direction == "ribasso" and t.short
    when = raw.get("when", "segnale")
    if when not in WHEN:
        raise PaperError(f"{where}: 'when' deve essere segnale o sempre")
    if when == "segnale" and not t.has_signal:
        raise PaperError(f"{where}: {t.name} non ha la valutazione notturna: usa when: sempre")
    try:
        exposure = float(raw.get("max_exposure", 0))
    except (TypeError, ValueError):
        raise PaperError(f"{where}: max_exposure deve essere un numero") from None
    cap = MAX_SHORT_EXPOSURE if short else MAX_VIEW_EXPOSURE
    if not 0 < exposure <= cap + 1e-9:
        raise PaperError(f"{where}: max_exposure deve stare tra 0 e {cap}" + (" (limite per gli short)" if short else ""))
    if when == "sempre":
        return View(target, direction, instrument, exposure, target, (), (), (), "sempre", short)
    signal = str(raw.get("signal") or target)
    if signal not in signals:
        raise PaperError(f"{where}: segnale {signal!r} non è nell'universo")
    enter_sit = _str_list(raw, "enter_situations", SITUATIONS, where)
    enter_risk = _str_list(raw, "enter_risk", RISKS, where) or RISKS
    exit_sit = _str_list(raw, "exit_situations", SITUATIONS, where)
    if not enter_sit:
        raise PaperError(f"{where}: servono le situazioni di ingresso (enter_situations)")
    if set(enter_sit) & set(exit_sit):
        raise PaperError(f"{where}: una situazione non può essere sia di ingresso sia di uscita")
    return View(target, direction, instrument, exposure, signal, enter_sit, enter_risk, exit_sit, "segnale", short)


def parse_indication(raw: dict, file: str, catalog: dict[str, Tradable], signals: set[str]) -> Indication:
    where = f"indicazione {file}"
    if not isinstance(raw, dict):
        raise PaperError(f"{where}: formato non valido")
    try:
        valid_from = dt.date.fromisoformat(str(raw.get("from"))).isoformat()
    except ValueError:
        raise PaperError(f"{where}: 'from' deve essere una data AAAA-MM-GG") from None
    rows = raw.get("views") or []
    if not isinstance(rows, list) or len(rows) > MAX_VIEWS:
        raise PaperError(f"{where}: 'views' deve essere un elenco di al massimo {MAX_VIEWS} righe")
    views = tuple(parse_view(r, f"{where}, riga {n}", catalog, signals) for n, r in enumerate(rows, 1))
    instruments = [v.instrument for v in views]
    if len(instruments) != len(set(instruments)):
        raise PaperError(f"{where}: due righe usano lo stesso strumento ({', '.join(instruments)})")
    if sum(v.max_exposure for v in views) > MAX_TOTAL_EXPOSURE + 1e-9:
        raise PaperError(f"{where}: la somma delle quote supera il 100% del conto")
    return Indication(valid_from, str(raw.get("author", "")), str(raw.get("note", "") or "").strip(), views, file)


def load_indications(directory: Path, catalog: dict[str, Tradable], signals: set[str]) -> list[Indication]:
    """Tutte le indicazioni, dalla più vecchia. Un file non valido blocca il conto (errore esplicito)."""
    out = []
    for path in sorted(Path(directory).glob("*.yml")):
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        out.append(parse_indication(raw, path.name, catalog, signals))
    dates = [i.valid_from for i in out]
    if len(dates) != len(set(dates)):
        raise PaperError("due indicazioni con la stessa data 'from': tenerne una sola")
    return sorted(out, key=lambda i: i.valid_from)


def current_indication(indications: list[Indication], as_of: str) -> Indication | None:
    valid = [i for i in indications if i.valid_from <= as_of]
    return valid[-1] if valid else None


def view_weight(view: View, assessment: dict, held: bool) -> tuple[float, str]:
    """Peso della riga e motivo. Entra se il segnale è in una situazione E con un rischio di ingresso;
    se è già dentro resta finché il segnale non arriva in una situazione di uscita; segnale senza dati
    aggiornati: non entra e, se è dentro, resta."""
    if view.when == "sempre":
        return view.max_exposure, "tenuto finché l'indicazione non cambia"
    inst = next((i for i in assessment["instruments"] if i["id"] == view.signal), None)
    if inst is None or not is_fresh(inst, assessment["as_of"]):
        return (view.max_exposure if held else 0.0), f"{view.signal}: dati non aggiornati, {'mantengo' if held else 'non entro'}"
    sit, risk = inst["situation"]["key"], inst["risk"]["level"]
    label = f"{inst.get('name', view.signal)} {inst['situation']['label'].lower()}, rischio {risk}"
    if sit in view.enter_situations and risk in view.enter_risk:
        return view.max_exposure, f"{label}: {'ingresso' if not held else 'segnale ancora attivo'}"
    if held and sit not in view.exit_situations:
        return view.max_exposure, f"{label}: mantengo"
    return 0.0, f"{label}: fuori"


def target_indication(ind: Indication | None, assessment: dict, held_symbols: set[str],
                      stopped: set[str] | None = None) -> Target:
    """Applica l'indicazione in vigore riga per riga; il resto in liquidità. Strumenti non più indicati si chiudono.

    Peso negativo = short. `stopped`: short chiusi dallo stop con questa indicazione, che non rientrano.
    """
    stopped = stopped or set()
    if ind is None:
        return Target({}, {}, "nessuna indicazione in vigore: tutto in liquidità")
    head = f"indicazione del {ind.valid_from}"
    if not ind.views:
        return Target({}, {}, f"{head}: nessuna posizione, tutto in liquidità")
    weights, reasons, parts = {}, {}, []
    for v in ind.views:
        if v.instrument in stopped:
            parts.append(f"{v.target} al {v.direction} → fermo (stop del {SHORT_STOP_LOSS:.0%} scattato)")
            continue
        w, why = view_weight(v, assessment, v.instrument in held_symbols)
        what = ("short " if v.short else "") + v.instrument
        parts.append(f"{v.target} al {v.direction} {'→ ' + what + f' {w:.0%}' if w else '→ fuori'}")
        if w > 0:
            weights[v.instrument] = v.sign * w
            reasons[v.instrument] = f"{v.target} al {v.direction}{' (short)' if v.short else ''}: {why}"
    return Target(weights, reasons, f"{head}: " + "; ".join(parts))


def stock_symbols(universe_path: Path | str = DEFAULT_UNIVERSE) -> dict[str, str]:
    """id interno -> simbolo negoziabile (solo azioni attive; il simbolo yfinance coincide con quello US)."""
    return {i.id: i.symbols["yahoo"] for i in load_universe(universe_path)
            if i.asset_class == "stock" and i.active and "yahoo" in i.symbols}


def compute_target(strategy: Strategy, assessment: dict, symbols_by_id: dict[str, str],
                   positions: dict[str, tuple[float, float]] | None = None,
                   indications: list[Indication] | None = None, stopped: set[str] | None = None) -> Target:
    if strategy.kind == "buy_hold":
        return target_buy_hold(strategy)
    if strategy.kind == "assess_rule":
        return target_assess_rule(strategy, assessment, symbols_by_id)
    if strategy.kind == "indication":
        held = {sym for sym, (_, qty) in (positions or {}).items() if qty != 0}
        return target_indication(current_indication(indications or [], assessment["as_of"]), assessment, held, stopped)
    raise PaperError(f"kind non gestito: {strategy.kind}")


# ---------------------------------------------------------------------------
# Ordini (funzione pura)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrderPlan:
    symbol: str
    side: str                    # "buy" | "sell"
    notional: float | None       # importo in dollari (None se si vende tutto a quantità)
    qty: float | None            # quantità (solo per chiudere una posizione per intero)


def plan_orders(equity: float, positions: dict[str, tuple[float, float]], weights: dict[str, float],
                min_delta: float, prices: dict[str, float] | None = None) -> list[OrderPlan]:
    """Ordini per portare il conto ai pesi obiettivo.

    positions: simbolo -> (valore di mercato, quantità); quantità negativa = posizione short.
    Peso negativo = short (solo indicazioni al ribasso su un'azione): quote intere, prezzo dalla posizione
    o da `prices`. Le vendite vengono prima degli acquisti. Scostamenti sotto `min_delta` del conto non si
    toccano; le posizioni fuori dall'obiettivo si chiudono per intero a quantità. Passare da long a short
    (o viceversa) sullo stesso titolo richiede due sere: prima si chiude, poi si apre.
    """
    if equity <= 0:
        raise PaperError(f"valore del conto non valido: {equity}")
    prices = prices or {}
    sells: list[OrderPlan] = []
    buys: list[OrderPlan] = []
    for symbol, (value, qty) in sorted(positions.items()):
        w = weights.get(symbol, 0.0)
        if qty > 0 and w <= 0:
            sells.append(OrderPlan(symbol, "sell", None, qty))             # chiude il long
        elif qty < 0 and w >= 0:
            buys.append(OrderPlan(symbol, "buy", None, abs(qty)))           # ricompra lo short
    for symbol, weight in sorted(weights.items()):
        value, qty = positions.get(symbol, (0.0, 0.0))
        if weight > 0:
            if qty < 0:
                continue                                                    # prima si chiude lo short
            delta = weight * equity - value
            if abs(delta) / equity < min_delta or abs(delta) < MIN_ORDER_USD:
                continue
            order = OrderPlan(symbol, "buy" if delta > 0 else "sell", round(abs(delta), 2), None)
            (buys if delta > 0 else sells).append(order)
        elif weight < 0:
            if qty > 0:
                continue                                                    # prima si chiude il long
            price = abs(value / qty) if qty else prices.get(symbol)
            if not price or price <= 0:
                continue                                                    # senza prezzo niente short
            target_qty = -int(abs(weight) * equity // price)                # quote intere
            delta_qty = target_qty - int(round(qty))
            if delta_qty == 0 or abs(delta_qty) * price / equity < min_delta:
                continue
            if delta_qty < 0:
                sells.append(OrderPlan(symbol, "sell", None, float(-delta_qty)))   # apre/aumenta lo short
            else:
                buys.append(OrderPlan(symbol, "buy", None, float(delta_qty)))      # riduce lo short
    return sells + buys


@dataclass(frozen=True)
class MarketState:
    is_open: bool                # mercato aperto adesso (seduta regolare)
    last_session: dt.date        # ultima seduta regolare già chiusa
    next_open: dt.datetime       # prossima apertura (ora di New York)


def send_blocker(market: MarketState, as_of: str, open_orders: list[str]) -> str | None:
    """Motivo per NON inviare ordini stasera, oppure None se si può.

    - mercato aperto: un ordine a mercato verrebbe eseguito subito, non all'apertura successiva;
    - valutazioni non della seduta appena chiusa (daily in ritardo o fallito): si deciderebbe su dati vecchi;
    - ordini di questa strategia ancora in attesa (es. festivo: quelli di ieri non sono ancora eseguiti):
      ripianificare sopra ordini pendenti raddoppierebbe le posizioni.
    """
    if market.is_open:
        return "mercato aperto: gli ordini si inviano solo a mercato chiuso"
    if dt.date.fromisoformat(as_of) != market.last_session:
        return (f"valutazioni al {as_of}, ma l'ultima seduta chiusa è il {market.last_session}: "
                "non opero su dati non aggiornati")
    if open_orders:
        return f"{len(open_orders)} ordini ancora in attesa di esecuzione ({', '.join(sorted(open_orders))})"
    return None


def client_order_id(strategy_id: str, as_of: str, symbol: str) -> str:
    return f"{ID_PREFIX}-{strategy_id}-{as_of}-{symbol}"


def parse_client_order_id(coid: str) -> tuple[str, str, str] | None:
    """(strategia, as_of, simbolo) da un nostro client_order_id, altrimenti None."""
    parts = (coid or "").split("-")
    if len(parts) < 6 or parts[0] != ID_PREFIX:
        return None
    return parts[1], "-".join(parts[2:5]), "-".join(parts[5:])


# ---------------------------------------------------------------------------
# Log append-only
# ---------------------------------------------------------------------------

def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_rows(path: Path, columns: list[str], rows: list[dict], key: tuple[str, ...]) -> int:
    """Aggiunge solo righe con chiave nuova; le righe già registrate non si modificano mai."""
    existing = {tuple(r.get(k, "") for k in key) for r in _read_rows(path)}
    new = []
    for r in rows:
        k = tuple(str(r.get(c, "")) for c in key)
        if k not in existing:
            existing.add(k)
            new.append(r)
    if not new:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        if write_header:
            w.writeheader()
        w.writerows(new)
    return len(new)


# ---------------------------------------------------------------------------
# Broker (alpaca-py importato solo qui: i test non ne hanno bisogno)
# ---------------------------------------------------------------------------

class AlpacaPaperBroker:
    """Sottile involucro su un conto Alpaca PAPER. Nessuna via per il conto reale."""

    def __init__(self, key: str, secret: str):
        from alpaca.trading.client import TradingClient  # import tardivo

        self._client = TradingClient(key, secret, paper=True)
        self._key, self._secret = key, secret

    def account(self) -> tuple[float, float]:
        acc = self._client.get_account()
        return float(acc.equity), float(acc.cash)

    def positions(self) -> dict[str, tuple[float, float]]:
        return {p.symbol: (float(p.market_value), float(p.qty)) for p in self._client.get_all_positions()}

    def position_losses(self) -> dict[str, float]:
        """Perdita/guadagno non realizzato in frazione (es. -0.15) per simbolo."""
        return {p.symbol: float(p.unrealized_plpc or 0) for p in self._client.get_all_positions()}

    def short_prices(self, symbols: list[str]) -> dict[str, float]:
        """Ultimo prezzo dei titoli da vendere allo scoperto (feed IEX, incluso nei conti paper), solo se shortabili."""
        if not symbols:
            return {}
        from alpaca.data.enums import DataFeed
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest

        ok = []
        for sym in symbols:
            asset = self._client.get_asset(sym)
            if getattr(asset, "shortable", False) and getattr(asset, "easy_to_borrow", False):
                ok.append(sym)
            else:
                log.warning("%s non è vendibile allo scoperto su Alpaca: niente short", sym)
        if not ok:
            return {}
        data = StockHistoricalDataClient(self._key, self._secret)
        trades = data.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=ok, feed=DataFeed.IEX))
        return {sym: float(t.price) for sym, t in trades.items() if t and t.price}

    def market(self) -> MarketState:
        """Orologio e calendario di Alpaca: festività e chiusure anticipate già incluse."""
        from alpaca.trading.requests import GetCalendarRequest

        clock = self._client.get_clock()
        now = clock.timestamp
        days = self._client.get_calendar(GetCalendarRequest(start=now.date() - dt.timedelta(days=10),
                                                            end=now.date()))
        closed = [d.date for d in days if _aware(d.close, now) <= now]
        if not closed:
            raise PaperError("calendario di Alpaca senza sedute chiuse negli ultimi 10 giorni")
        return MarketState(bool(clock.is_open), max(closed), clock.next_open)

    def open_orders(self, strategy_id: str) -> list[str]:
        """Simboli con ordini di questa strategia non ancora chiusi (inviati, accodati, parziali)."""
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        out = []
        for o in self._client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)):
            parsed = parse_client_order_id(o.client_order_id)
            if parsed and parsed[0] == strategy_id:
                out.append(parsed[2])
        return out

    def submitted_ids(self, since: dt.datetime) -> set[str]:
        """client_order_id già usati (ordini di qualsiasi stato): un rilancio non li reinvia."""
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        orders = self._client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, after=since, limit=500))
        return {o.client_order_id for o in orders if o.client_order_id}

    def submit(self, plan: OrderPlan, coid: str) -> tuple[str, str]:
        """Invia un ordine a mercato 'day'. Ritorna (id Alpaca, esito). Un id già usato = già inviato."""
        from alpaca.common.exceptions import APIError
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        req = MarketOrderRequest(
            symbol=plan.symbol,
            side=OrderSide.BUY if plan.side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            client_order_id=coid,
            **({"notional": plan.notional} if plan.notional is not None else {"qty": plan.qty}),
        )
        try:
            order = self._client.submit_order(req)
        except APIError as e:
            msg = _api_message(e)
            if "client_order_id" in msg.lower():
                return "", "già inviato (client_order_id presente)"
            return "", f"rifiutato: {msg}"
        return str(order.id), f"accettato ({getattr(order.status, 'value', order.status)})"

    def recent_closed_orders(self, since: dt.datetime) -> list[dict]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        orders = self._client.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, after=since, limit=500))
        out = []
        for o in orders:
            parsed = parse_client_order_id(o.client_order_id)
            if parsed is None:
                continue
            strategy, as_of, symbol = parsed
            out.append({
                "order_id": str(o.id), "client_order_id": o.client_order_id, "as_of": as_of,
                "strategy": strategy, "symbol": symbol,
                "side": getattr(o.side, "value", str(o.side)),
                "status": getattr(o.status, "value", str(o.status)),
                "filled_qty": str(o.filled_qty or ""),
                "filled_avg_price": str(o.filled_avg_price or ""),
                "filled_at": o.filled_at.isoformat() if o.filled_at else "",
            })
        return out


def _aware(t: dt.datetime, ref: dt.datetime) -> dt.datetime:
    """Il calendario può tornare orari senza fuso: sono ore di New York, come l'orologio."""
    return t if t.tzinfo is not None else t.replace(tzinfo=ref.tzinfo)


def _api_message(e: Exception) -> str:
    """Messaggio leggibile di un APIError (JSON con code/message), senza mai riportare le chiavi."""
    try:
        return f"{e.message} (codice {e.code})"          # type: ignore[attr-defined]
    except Exception:
        return str(e)[:300]


def broker_for(strategy: Strategy, env=os.environ) -> AlpacaPaperBroker | None:
    key, secret = env.get(f"{strategy.secret_prefix}_KEY"), env.get(f"{strategy.secret_prefix}_SECRET")
    if not key or not secret:
        return None
    return AlpacaPaperBroker(key, secret)


# ---------------------------------------------------------------------------
# Esecuzione
# ---------------------------------------------------------------------------

def run_strategy(strategy: Strategy, assessment: dict, symbols_by_id: dict[str, str], cfg: PaperConfig,
                 broker, out_dir: Path, send: bool, indications: list[Indication] | None = None) -> dict:
    as_of = assessment["as_of"]
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    stopped: set[str] = set()
    new_stops: list[dict] = []
    if broker is not None:
        equity, cash = broker.account()
        positions = broker.positions()
        if strategy.kind == "indication":
            stopped, new_stops = short_stops(strategy, indications, assessment, positions, broker, out_dir, now)
    target = compute_target(strategy, assessment, symbols_by_id, positions if broker is not None else None,
                            indications, stopped)
    print(f"\n[{strategy.id}] {strategy.name}: {target.note}")
    for s, w in sorted(target.weights.items()):
        print(f"    {s:6s} peso {w:6.1%}  ({target.reasons.get(s, '')})")

    result = {"strategy": strategy.id, "weights": target.weights, "orders": [], "sent": False}
    if broker is None:
        print(f"    (nessun conto: mancano {strategy.secret_prefix}_KEY / {strategy.secret_prefix}_SECRET)")
        return result

    to_short = [sym for sym, w in target.weights.items() if w < 0 and positions.get(sym, (0.0, 0.0))[1] == 0]
    prices = broker.short_prices(sorted(to_short)) if to_short and hasattr(broker, "short_prices") else {}
    for sym in to_short:
        if sym not in prices:
            print(f"    {sym}: niente short stasera (non shortabile o prezzo non disponibile)")
    orders = plan_orders(equity, positions, target.weights, cfg.min_delta, prices)
    result["orders"] = orders
    print(f"    conto: valore {equity:,.2f}, liquidità {cash:,.2f}, posizioni {len(positions)}")
    for o in orders:
        what = f"{o.notional:,.2f} $" if o.notional is not None else f"{o.qty:g} azioni a quote"
        print(f"    ordine: {o.side:4s} {o.symbol:6s} {what}")
    if not orders:
        print("    nessun ordine: il conto è già vicino ai pesi obiettivo")
    market = broker.market()
    blocker = send_blocker(market, as_of, broker.open_orders(strategy.id))
    print(f"    mercato: {'aperto' if market.is_open else 'chiuso'}, ultima seduta chiusa {market.last_session}, "
          f"prossima apertura {market.next_open:%Y-%m-%d %H:%M} (New York)")
    if blocker:
        print(f"    NON invio ordini: {blocker}")
        result["blocked"] = blocker
    if not send:
        return result

    # Il valore del conto si registra sempre (anche quando non si opera): serve alla curva della strategia.
    invested = sum(v for v, _ in positions.values())
    append_rows(out_dir / EQUITY_FILE, EQUITY_COLUMNS,
                [{"as_of": as_of, "strategy": strategy.id, "equity": f"{equity:.2f}", "cash": f"{cash:.2f}",
                  "invested": f"{invested:.2f}", "n_positions": len(positions), "created_utc": now}],
                ("as_of", "strategy"))
    if blocker:
        return result
    if new_stops:
        append_rows(out_dir / STOPS_FILE, STOPS_COLUMNS, new_stops, ("indication_from", "strategy", "symbol"))

    decisions = [{"as_of": as_of, "strategy": strategy.id, "symbol": s, "target_weight": f"{w:.6f}",
                  "reason": target.reasons.get(s, ""), "created_utc": now} for s, w in sorted(target.weights.items())]
    if not decisions:      # anche "tutto in liquidità" è una decisione da registrare
        decisions = [{"as_of": as_of, "strategy": strategy.id, "symbol": "CASH", "target_weight": "1.000000",
                      "reason": target.note, "created_utc": now}]
    append_rows(out_dir / DECISIONS_FILE, DECISIONS_COLUMNS, decisions, ("as_of", "strategy", "symbol"))

    already = broker.submitted_ids(dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=FILLS_LOOKBACK_DAYS))
    order_rows = []
    for o in orders:
        coid = client_order_id(strategy.id, as_of, o.symbol)
        if coid in already:
            print(f"    -> {o.side} {o.symbol}: già inviato in un run precedente, salto")
            continue
        order_id, outcome = broker.submit(o, coid)
        print(f"    -> {o.side} {o.symbol}: {outcome}")
        order_rows.append({"as_of": as_of, "strategy": strategy.id, "symbol": o.symbol, "side": o.side,
                           "notional": "" if o.notional is None else f"{o.notional:.2f}",
                           "qty": "" if o.qty is None else o.qty, "client_order_id": coid,
                           "alpaca_order_id": order_id, "result": outcome, "created_utc": now})
    append_rows(out_dir / ORDERS_FILE, ORDERS_COLUMNS, order_rows, ("client_order_id",))
    result["sent"] = True
    if any(r["result"].startswith("rifiutato") for r in order_rows):
        result["rejected"] = [r["symbol"] for r in order_rows if r["result"].startswith("rifiutato")]
    return result


def short_stops(strategy: Strategy, indications: list[Indication] | None, assessment: dict,
                positions: dict[str, tuple[float, float]], broker, out_dir: Path, now: str) -> tuple[set[str], list[dict]]:
    """Short da tenere fermi: già fermati con l'indicazione in vigore, più quelli che ora perdono oltre lo stop."""
    cur = current_indication(indications or [], assessment["as_of"])
    if cur is None:
        return set(), []
    stopped = {r["symbol"] for r in _read_rows(out_dir / STOPS_FILE)
               if r["indication_from"] == cur.valid_from and r["strategy"] == strategy.id}
    losses = broker.position_losses() if hasattr(broker, "position_losses") else {}
    new = []
    for sym, (_, qty) in sorted(positions.items()):
        loss = losses.get(sym, 0.0)
        if qty < 0 and loss <= -SHORT_STOP_LOSS and sym not in stopped:
            print(f"    STOP: lo short su {sym} perde {loss:.1%}, si chiude e non rientra con questa indicazione")
            stopped.add(sym)
            new.append({"indication_from": cur.valid_from, "strategy": strategy.id, "symbol": sym,
                        "as_of": assessment["as_of"], "loss": f"{loss:.4f}", "created_utc": now})
    return stopped, new


def record_fills(broker, out_dir: Path) -> int:
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=FILLS_LOOKBACK_DAYS)
    rows = broker.recent_closed_orders(since)
    return append_rows(out_dir / FILLS_FILE, FILLS_COLUMNS, rows, ("order_id",))


# ---------------------------------------------------------------------------
# JSON per la dashboard (#/paper): solo percentuali, nessun prezzo
# ---------------------------------------------------------------------------

def _f(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def dashboard_payload(cfg: PaperConfig, out_dir: Path, last_run: dict | None = None) -> dict:
    """Andamento di ogni conto paper rispetto al suo primo valore, pesi attuali, ordini recenti.

    Niente importi assoluti né prezzi di esecuzione: andamenti in %, ordini in % del conto.
    """
    equity = _read_rows(out_dir / EQUITY_FILE)
    decisions = _read_rows(out_dir / DECISIONS_FILE)
    orders = _read_rows(out_dir / ORDERS_FILE)
    fills = {r["client_order_id"]: r for r in _read_rows(out_dir / FILLS_FILE)}
    eq_by = {}
    for r in equity:
        v = _f(r.get("equity"))
        if v and v > 0:
            eq_by.setdefault(r["strategy"], {})[r["as_of"]] = (v, _f(r.get("cash")) or 0.0)

    strategies = []
    for st in cfg.strategies:
        series = dict(sorted(eq_by.get(st.id, {}).items()))
        dates = list(series)
        first = series[dates[0]][0] if dates else None
        rets = [round(series[d][0] / first - 1, 6) for d in dates] if first else []
        mine = [r for r in decisions if r["strategy"] == st.id]
        last_dec = max((r["as_of"] for r in mine), default=None)
        weights = [{"symbol": r["symbol"], "weight": _f(r["target_weight"]), "reason": r["reason"]}
                   for r in mine if r["as_of"] == last_dec]
        last = series[dates[-1]] if dates else None
        strategies.append({
            "id": st.id, "name": st.name, "kind": st.kind, "is_benchmark": st.kind == "buy_hold",
            "start": dates[0] if dates else None, "as_of": dates[-1] if dates else None, "n_sessions": len(dates),
            "return": rets[-1] if rets else None,
            "day_return": round(rets[-1] - rets[-2], 6) if len(rets) > 1 else None,
            "cash_share": round(last[1] / last[0], 4) if last else None,
            "decision_as_of": last_dec, "weights": weights,
            "series": {"dates": dates, "returns": rets},
            "last_run": (last_run or {}).get(st.id),
        })

    bench = next((s for s in strategies if s["is_benchmark"] and s["n_sessions"]), None)
    comparisons = []
    for s in strategies:
        # Le indicazioni (es. copertura ribassista) non hanno lo scopo di battere QQQ: confronto solo per le regole.
        if s["is_benchmark"] or s["kind"] == "indication" or not bench or not s["n_sessions"]:
            continue
        a, b = eq_by[s["id"]], eq_by[bench["id"]]
        common = sorted(set(a) & set(b))
        if not common:
            continue
        a0, b0 = a[common[0]][0], b[common[0]][0]
        excess = [round((a[d][0] / a0) - (b[d][0] / b0), 6) for d in common]
        n = len(common)
        if n < MIN_SESSIONS_VERDICT:
            verdict = "presto"
        else:
            verdict = "avanti" if excess[-1] > 0 else "indietro"
        comparisons.append({"strategy": s["id"], "benchmark": bench["id"], "n_sessions": n,
                            "dates": common, "excess": excess, "verdict": verdict})

    equity_at = {(r["strategy"], r["as_of"]): _f(r.get("equity")) for r in equity}
    recent = []
    for r in sorted(orders, key=lambda r: (r["as_of"], r["strategy"], r["symbol"]), reverse=True)[:RECENT_ORDERS]:
        eq = equity_at.get((r["strategy"], r["as_of"]))
        notional = _f(r.get("notional"))
        fill = fills.get(r["client_order_id"])
        recent.append({
            "as_of": r["as_of"], "strategy": r["strategy"], "symbol": r["symbol"], "side": r["side"],
            "share": round(notional / eq, 4) if notional and eq else None,
            "close_all": not r.get("notional"),
            "sent": r["result"].split(":")[0].split(" (")[0],
            "status": fill["status"] if fill else None,
        })
    return {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "min_sessions_verdict": MIN_SESSIONS_VERDICT, "strategies": strategies,
            "comparisons": comparisons, "orders": recent}


def indications_payload(cfg: PaperConfig, indications: list[Indication], as_of: str, catalog: dict[str, Tradable],
                        error: str | None = None) -> dict:
    """Indicazione in vigore, storico e opzioni del modulo della pagina (che apre GitHub già compilato)."""
    cur = current_indication(indications, as_of)
    order = {"index": 0, "etf": 1, "stock": 2}
    return {"current": cur.as_dict() if cur else None,
            "history": [i.as_dict() for i in reversed(indications)],
            "error": error, "github_repo": cfg.github_repo, "path": "config/indications",
            "options": {"targets": [{"id": t.id, "name": t.name, "kind": t.kind, "up": t.up, "down": t.down,
                                     "down_note": t.down_note, "short": t.short, "has_signal": t.has_signal}
                                    for t in sorted(catalog.values(), key=lambda t: (order.get(t.kind, 9), not t.has_signal, t.name))],
                        "situations": list(SITUATIONS), "risks": list(RISKS),
                        "max_view_exposure": MAX_VIEW_EXPOSURE, "max_short_exposure": MAX_SHORT_EXPOSURE,
                        "short_stop_loss": SHORT_STOP_LOSS, "max_views": MAX_VIEWS}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Paper trading su Alpaca (solo conti paper).")
    ap.add_argument("--send", action="store_true", help="invia gli ordini (senza: solo prova)")
    ap.add_argument("--only", help="id di una sola strategia")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--assessment", type=Path, default=DEFAULT_ASSESSMENT)
    ap.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--dashboard", type=Path, default=DEFAULT_DASHBOARD)
    ap.add_argument("--indications", type=Path, default=DEFAULT_INDICATIONS)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        cfg = load_config(args.config)
        assessment = load_assessment(args.assessment)
        symbols_by_id = stock_symbols(args.universe)
        signals = {i.id for i in load_universe(args.universe) if i.active}
        catalog = tradables(args.universe, cfg)
    except (PaperError, ConfigError) as e:
        print(f"ERRORE: {e}", file=sys.stderr)
        return 1
    indications: list[Indication] = []
    indications_error = None
    try:
        indications = load_indications(args.indications, catalog, signals)
    except (PaperError, yaml.YAMLError) as e:       # blocca solo il conto delle indicazioni
        indications_error = str(e)

    strategies = [s for s in cfg.strategies if not args.only or s.id == args.only]
    if not strategies:
        print(f"ERRORE: strategia non trovata: {args.only}", file=sys.stderr)
        return 1

    print(f"Valutazioni al {assessment['as_of']} — {'INVIO ORDINI' if args.send else 'PROVA (nessun ordine inviato)'}")
    failed = False
    last_run: dict[str, dict] = {}
    for strategy in strategies:
        try:
            broker = broker_for(strategy)
            if broker is None and args.send:
                print(f"\nATTENZIONE [{strategy.id}]: invio richiesto ma mancano le chiavi del conto paper")
            if strategy.kind == "indication" and indications_error:
                raise PaperError(f"indicazioni non valide, il conto non opera: {indications_error}")
            res = run_strategy(strategy, assessment, symbols_by_id, cfg, broker, args.out_dir, args.send, indications)
            last_run[strategy.id] = {"as_of": assessment["as_of"], "blocked": res.get("blocked"),
                                     "n_orders": len(res["orders"]), "has_account": broker is not None}
            if res.get("rejected"):
                failed = True
                log.error("%s: ordini rifiutati da Alpaca: %s", strategy.id, ", ".join(res["rejected"]))
            if broker is not None and args.send:
                n = record_fills(broker, args.out_dir)
                if n:
                    print(f"    esecuzioni nuove registrate: {n}")
        except Exception as e:       # una strategia in errore non ferma le altre
            failed = True
            last_run[strategy.id] = {"as_of": assessment["as_of"], "error": True}
            log.error("%s: %s", strategy.id, e)
    if args.send:
        payload = dashboard_payload(cfg, args.out_dir, last_run)
        payload["indications"] = indications_payload(cfg, indications, assessment["as_of"], catalog, indications_error)
        write_json(payload, args.dashboard)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
