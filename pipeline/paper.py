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
                 stesso peso (con tetto), il resto in liquidità.

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

from pipeline.ingest import DEFAULT_UNIVERSE, ConfigError, load_universe, read_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "paper.yml"
DEFAULT_ASSESSMENT = ROOT / "docs" / "data" / "assessment.json"
DEFAULT_OUT_DIR = ROOT / "results" / "paper"

log = logging.getLogger("financenow.paper")

ID_PREFIX = "fn"                 # prefisso dei client_order_id: riconosce gli ordini di questo programma
MAX_ASSESSMENT_AGE_DAYS = 4      # oltre, le valutazioni sono vecchie (weekend lunghi inclusi)
MIN_ORDER_USD = 1.0
FILLS_LOOKBACK_DAYS = 10

DECISIONS_FILE, ORDERS_FILE, FILLS_FILE, EQUITY_FILE = "decisions.csv", "orders.csv", "fills.csv", "equity.csv"
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

KINDS = ("buy_hold", "assess_rule")


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
    return PaperConfig(min_delta=min_delta, strategies=tuple(strategies))


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
        if inst.get("data_status") != "updated":
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


def stock_symbols(universe_path: Path | str = DEFAULT_UNIVERSE) -> dict[str, str]:
    """id interno -> simbolo negoziabile (solo azioni attive; il simbolo yfinance coincide con quello US)."""
    return {i.id: i.symbols["yahoo"] for i in load_universe(universe_path)
            if i.asset_class == "stock" and i.active and "yahoo" in i.symbols}


def compute_target(strategy: Strategy, assessment: dict, symbols_by_id: dict[str, str]) -> Target:
    if strategy.kind == "buy_hold":
        return target_buy_hold(strategy)
    if strategy.kind == "assess_rule":
        return target_assess_rule(strategy, assessment, symbols_by_id)
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
                min_delta: float) -> list[OrderPlan]:
    """Ordini per portare il conto ai pesi obiettivo.

    positions: simbolo -> (valore di mercato, quantità). Le vendite vengono prima degli acquisti.
    Scostamenti sotto `min_delta` del conto non si toccano (meno rotazione inutile); le posizioni
    fuori dall'obiettivo si chiudono per intero a quantità, senza residui.
    """
    if equity <= 0:
        raise PaperError(f"valore del conto non valido: {equity}")
    sells: list[OrderPlan] = []
    buys: list[OrderPlan] = []
    for symbol, (value, qty) in sorted(positions.items()):
        if weights.get(symbol, 0.0) <= 0 and qty > 0:
            sells.append(OrderPlan(symbol, "sell", None, qty))
    for symbol, weight in sorted(weights.items()):
        if weight <= 0:
            continue
        current = positions.get(symbol, (0.0, 0.0))[0]
        delta = weight * equity - current
        if abs(delta) / equity < min_delta or abs(delta) < MIN_ORDER_USD:
            continue
        order = OrderPlan(symbol, "buy" if delta > 0 else "sell", round(abs(delta), 2), None)
        (buys if delta > 0 else sells).append(order)
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

    def account(self) -> tuple[float, float]:
        acc = self._client.get_account()
        return float(acc.equity), float(acc.cash)

    def positions(self) -> dict[str, tuple[float, float]]:
        return {p.symbol: (float(p.market_value), float(p.qty)) for p in self._client.get_all_positions()}

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
                 broker, out_dir: Path, send: bool) -> dict:
    as_of = assessment["as_of"]
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    target = compute_target(strategy, assessment, symbols_by_id)
    print(f"\n[{strategy.id}] {strategy.name}: {target.note}")
    for s, w in sorted(target.weights.items()):
        print(f"    {s:6s} peso {w:6.1%}  ({target.reasons.get(s, '')})")

    result = {"strategy": strategy.id, "weights": target.weights, "orders": [], "sent": False}
    if broker is None:
        print(f"    (nessun conto: mancano {strategy.secret_prefix}_KEY / {strategy.secret_prefix}_SECRET)")
        return result

    equity, cash = broker.account()
    positions = broker.positions()
    orders = plan_orders(equity, positions, target.weights, cfg.min_delta)
    result["orders"] = orders
    print(f"    conto: valore {equity:,.2f}, liquidità {cash:,.2f}, posizioni {len(positions)}")
    for o in orders:
        what = f"{o.notional:,.2f} $" if o.notional is not None else f"{o.qty} azioni (chiusura)"
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


def record_fills(broker, out_dir: Path) -> int:
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=FILLS_LOOKBACK_DAYS)
    rows = broker.recent_closed_orders(since)
    return append_rows(out_dir / FILLS_FILE, FILLS_COLUMNS, rows, ("order_id",))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Paper trading su Alpaca (solo conti paper).")
    ap.add_argument("--send", action="store_true", help="invia gli ordini (senza: solo prova)")
    ap.add_argument("--only", help="id di una sola strategia")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--assessment", type=Path, default=DEFAULT_ASSESSMENT)
    ap.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        cfg = load_config(args.config)
        assessment = load_assessment(args.assessment)
        symbols_by_id = stock_symbols(args.universe)
    except (PaperError, ConfigError) as e:
        print(f"ERRORE: {e}", file=sys.stderr)
        return 1

    strategies = [s for s in cfg.strategies if not args.only or s.id == args.only]
    if not strategies:
        print(f"ERRORE: strategia non trovata: {args.only}", file=sys.stderr)
        return 1

    print(f"Valutazioni al {assessment['as_of']} — {'INVIO ORDINI' if args.send else 'PROVA (nessun ordine inviato)'}")
    failed = False
    for strategy in strategies:
        try:
            broker = broker_for(strategy)
            if broker is None and args.send:
                print(f"\nATTENZIONE [{strategy.id}]: invio richiesto ma mancano le chiavi del conto paper")
            res = run_strategy(strategy, assessment, symbols_by_id, cfg, broker, args.out_dir, args.send)
            if res.get("rejected"):
                failed = True
                log.error("%s: ordini rifiutati da Alpaca: %s", strategy.id, ", ".join(res["rejected"]))
            if broker is not None and args.send:
                n = record_fills(broker, args.out_dir)
                if n:
                    print(f"    esecuzioni nuove registrate: {n}")
        except Exception as e:       # una strategia in errore non ferma le altre
            failed = True
            log.error("%s: %s", strategy.id, e)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
