"""Sintesi automatica del giorno: cosa merita attenzione, senza dover aprire ogni scheda.

Regole esplicite e descrittive (nessun consiglio operativo). Ogni segnale ha una gravità:
`alta` > `media` > `bassa`. Gli strumenti sono ordinati per gravità; quelli senza segnali
finiscono in un'unica riga "nessuna novità".

Segnali per strumento (finestra "recente" = ultime RECENT_SESSIONS sedute):
- cambio di regime recente (verso turbolento = alta; uscita da turbolento = media; altro = bassa);
- anomalie recenti (|z| >= 6 alta, altrimenti media);
- volatilità a 21 sedute estrema rispetto al proprio storico (>= 95° percentile alta,
  >= 90° media; <= 5° percentile bassa: calma insolita);
- volatilità in accelerazione: 5 sedute >= 1,5 volte quella a 63 sedute (media);
- drawdown dal massimo <= -20% (media);
- trimestrale nelle prossime SOON_EARNINGS sedute (entro 5 = media, altrimenti bassa);
- dati non aggiornati o mancanti (alta).
Un'anomalia nel giorno di reazione alla trimestrale è "spiegata": al massimo gravità media.

Contro il rumore: se almeno metà degli strumenti ha la volatilità >= 90° percentile, la cosa
viene detta una volta sola a livello di mercato e per i singoli strumenti restano solo i
casi >= 99° percentile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

RECENT_SESSIONS = 5
SOON_EARNINGS = 10
MARKET_WIDE_SHARE = 0.5
SEVERITY_RANK = {"alta": 3, "media": 2, "bassa": 1}
ANOMALY_LABELS = {
    "rendimento": "rendimento",
    "gap": "gap di apertura",
    "range": "escursione high-low",
    "volume": "volume",
    "rendimento_vs_bench": "rendimento rispetto al benchmark",
}
REGIME_LEVEL = {"calmo": 0, "normale": 1, "turbolento": 2}
# Nomi in parole semplici delle fasi (regimi) stimate dal modello.
PHASE = {"calmo": "calma", "normale": "normale", "turbolento": "agitata"}


@dataclass
class Context:
    """Quello che serve per valutare uno strumento (tutto già calcolato altrove)."""
    id: str
    name: str
    asset_class: str
    as_of: str | None
    recent_dates: list[str]                 # ultime RECENT_SESSIONS date di seduta
    regime: dict                            # regime_summary
    regime_states: list[int | None]         # stato per seduta, allineato alle date complete
    regime_dates: list[str]
    features: dict                          # features_latest
    rv21_percentile: float | None           # percentile della vol. 21g nel proprio storico (0-1)
    anomalies: list[dict] = field(default_factory=list)   # anomalie recenti (dict con date/kind/z/value)
    data_status: str | None = None
    description: str | None = None
    earnings: dict = field(default_factory=dict)           # riassunto trimestrali (pipeline.earnings.summary)
    direction: dict = field(default_factory=dict)          # previsioni di direzione di oggi, per orizzonte


def _pct(v: float, digits: int = 0, signed: bool = False) -> str:
    s = f"{v * 100:.{digits}f}".replace(".", ",") + "%"
    return ("+" + s) if signed and v > 0 else s


def _num(v: float, digits: int = 1) -> str:
    return f"{v:.{digits}f}".replace(".", ",")


def _fmt_date(iso: str) -> str:
    y, m, d = iso[:10].split("-")
    return f"{d}/{m}"


def _signal(severity: str, kind: str, text: str) -> dict:
    return {"severity": severity, "kind": kind, "text": text}


def regime_change(ctx: Context) -> dict | None:
    reg = ctx.regime or {}
    if not reg.get("available"):
        return None
    # days_in_state conta le sedute nel regime attuale, oggi compreso: il cambio è
    # avvenuto `days - 1` sedute fa.
    days = reg.get("days_in_state")
    states = [s for s in ctx.regime_states if s is not None]
    if not days or days > RECENT_SESSIONS or len(states) <= days:
        return None
    labels = [s["label"] for s in reg.get("states", [])]
    current = reg.get("current")
    previous = labels[states[-days - 1]] if states[-days - 1] < len(labels) else None
    if previous is None or previous == current:
        return None
    ago = days - 1
    when = "oggi" if ago == 0 else ("1 seduta fa" if ago == 1 else f"{ago} sedute fa")
    up = REGIME_LEVEL.get(current, 0) > REGIME_LEVEL.get(previous, 0)
    if current == "turbolento":
        sev = "alta"
    elif previous == "turbolento":
        sev = "media"
    else:
        sev = "bassa"
    prob = reg.get("probabilities", {}).get(current)
    prob_txt = f" (il modello ne è sicuro al {_pct(prob)})" if prob is not None else ""
    if current == "turbolento":
        text = f"è entrato in una fase agitata, con movimenti ampi e nervosi, {when}{prob_txt}; prima era in fase {PHASE.get(previous, previous)}."
    elif previous == "turbolento":
        text = f"è uscito dalla fase agitata {when}: ora è in fase {PHASE.get(current, current)}{prob_txt}."
    else:
        verb = "è passato" if up else "è tornato"
        text = f"{verb} a una fase {PHASE.get(current, current)} {when}{prob_txt}; prima era in fase {PHASE.get(previous, previous)}."
    return _signal(sev, "regime", text)


def anomaly_signals(ctx: Context) -> list[dict]:
    recent = set(ctx.recent_dates)
    out = []
    for a in ctx.anomalies:
        if a.get("date") not in recent or a.get("z") is None:
            continue
        z, kind, value = a["z"], a["kind"], a.get("value")
        if kind == "volume" and value is not None:
            val = f"×{math.exp(value):.1f} la mediana".replace(".", ",")
        elif value is not None:
            val = _pct(value, 1, signed=True)
        else:
            val = "n.d."
        sev = "alta" if abs(z) >= 6 else "media"
        times = f"{abs(z):.0f}"
        why = " Era il giorno di reazione alla trimestrale: movimento spiegato dai conti." if a.get("earnings") else ""
        if a.get("earnings") and sev == "alta":
            sev = "media"
        out.append(_signal(sev, "anomalia",
                           f"giorno insolito il {_fmt_date(a['date'])} per {ANOMALY_LABELS.get(kind, kind)}: {val}, "
                           f"circa {times} volte lo scarto tipico dell'ultimo anno.{why}"))
    return out


def volatility_signals(ctx: Context, market_wide: bool = False) -> list[dict]:
    f, out = ctx.features, []
    p, rv21 = ctx.rv21_percentile, f.get("rv_21")
    if market_wide and p is not None and p < 0.99:
        p = None  # già detto a livello di mercato
    if p is not None and rv21 is not None:
        month = _pct(rv21 / math.sqrt(12))
        if p >= 0.90:
            out.append(_signal("alta" if p >= 0.95 else "media", "volatilità",
                               f"si muove molto più del solito: oscillazione tipica di ±{month} al mese, "
                               f"più ampia che nel {round(p * 100)}% della sua storia."))
        elif p <= 0.05:
            out.append(_signal("bassa", "volatilità",
                               f"insolitamente calmo: oscillazione tipica di ±{month} al mese, "
                               f"più bassa che nel {round((1 - p) * 100)}% della sua storia."))
    rv5, rv63 = f.get("rv_5"), f.get("rv_63")
    if rv5 and rv63 and rv63 > 0 and rv5 / rv63 >= 1.5:
        out.append(_signal("media", "volatilità",
                           f"si sta agitando: nell'ultima settimana si muove {_num(rv5 / rv63)} volte più che negli ultimi 3 mesi."))
    return out


def drawdown_signal(ctx: Context) -> dict | None:
    dd = ctx.features.get("drawdown")
    if dd is not None and dd <= -0.20:
        need = 1 / (1 + dd) - 1
        return _signal("media", "drawdown", f"è a {_pct(dd)} dal suo massimo storico (per tornarci servirebbe {_pct(need, signed=True)}).")
    return None


def data_signal(ctx: Context) -> dict | None:
    if ctx.data_status in ("stale", "no_data"):
        return _signal("alta", "dati", f"dati non aggiornati: ultima seduta disponibile {ctx.as_of}.")
    return None


def earnings_signal(ctx: Context) -> dict | None:
    """Trimestrale nelle prossime sedute: giorni con movimenti di solito molto più ampi."""
    e = ctx.earnings or {}
    nxt = e.get("next")
    if not e.get("available") or not nxt or nxt.get("sessions_until") is None:
        return None
    n = nxt["sessions_until"]
    if n > SOON_EARNINGS:
        return None
    when = "domani" if n <= 1 else f"tra {n} sedute"
    move = e.get("typical_move")
    move_txt = (f": di solito quel giorno si muove di ±{_pct(math.expm1(move))}, "
                f"{_num(e['move_ratio'], 0)} volte un giorno normale") if move and e.get("move_ratio") else ""
    return _signal("media" if n <= 5 else "bassa", "trimestrale",
                   f"trimestrale in arrivo ({_fmt_date(nxt['date'])}, {nxt.get('timing', 'orario n.d.')}), reazione {when}{move_txt}.")


def instrument_signals(ctx: Context, market_wide: bool = False) -> list[dict]:
    signals = [s for s in (data_signal(ctx), regime_change(ctx), earnings_signal(ctx)) if s]
    signals += anomaly_signals(ctx)
    signals += volatility_signals(ctx, market_wide)
    if (d := drawdown_signal(ctx)):
        signals.append(d)
    return sorted(signals, key=lambda s: -SEVERITY_RANK[s["severity"]])


def market_lines(contexts: list[Context], benchmark_id: str = "ndx") -> list[str]:
    lines = []
    by_id = {c.id: c for c in contexts}
    bench = by_id.get(benchmark_id)
    if bench and bench.regime.get("available"):
        lines.append(f"{bench.name}: fase {PHASE.get(bench.regime['current'], bench.regime['current'])} "
                     f"da {bench.regime['days_in_state']} sedute, oscillazione tipica ±"
                     f"{_pct(bench.features.get('rv_21', 0) / math.sqrt(12))} al mese.")
    stocks = [c for c in contexts if c.asset_class == "stock"]
    if stocks:
        turb = [c.id.upper() for c in stocks if c.regime.get("current") == "turbolento"]
        calm = [c for c in stocks if c.regime.get("current") == "calmo"]
        txt = f"Azioni: {len(turb)} su {len(stocks)} in fase agitata"
        txt += f" ({', '.join(turb)})" if turb else ""
        txt += f", {len(calm)} in fase calma."
        lines.append(txt)
        pcts = sorted(c.rv21_percentile for c in stocks if c.rv21_percentile is not None)
        if pcts:
            med = pcts[len(pcts) // 2]
            lines.append(f"Il titolo tipico oggi si muove più che nel {round(med * 100)}% della sua storia (50% = nella media).")
    return lines


def market_wide_high_vol(contexts: list[Context]) -> tuple[bool, int, int]:
    pcts = [c.rv21_percentile for c in contexts if c.rv21_percentile is not None]
    high = sum(p >= 0.90 for p in pcts)
    return (bool(pcts) and high / len(pcts) >= MARKET_WIDE_SHARE, high, len(pcts))


def build(contexts: list[Context], data_status: dict | None = None) -> dict:
    instruments = []
    quiet = []
    wide, n_high, n_tot = market_wide_high_vol(contexts)
    market = market_lines(contexts)
    if wide:
        market.insert(0, f"Mercato agitato quasi ovunque: {n_high} strumenti su {n_tot} si muovono più che nel 90% "
                         f"della loro storia (singolarmente segnalati solo i casi estremi).")
    for ctx in contexts:
        sigs = instrument_signals(ctx, wide)
        if not sigs:
            quiet.append(ctx.id)
            continue
        instruments.append({
            "id": ctx.id,
            "name": ctx.name,
            "max_severity": sigs[0]["severity"],
            "signals": sigs,
        })
    instruments.sort(key=lambda i: (-SEVERITY_RANK[i["max_severity"]], -len(i["signals"]), i["id"]))
    counts = {s: sum(1 for i in instruments for x in i["signals"] if x["severity"] == s) for s in SEVERITY_RANK}
    missing = (data_status or {}).get("missing") or []
    as_of = max((c.as_of for c in contexts if c.as_of), default=None)
    if instruments:
        top = ", ".join(i["id"].upper() for i in instruments if i["max_severity"] == "alta")
        headline = (f"{len(instruments)} strumenti con segnali"
                    + (f"; priorità alta: {top}" if top else "; nessuna priorità alta") + ".")
    else:
        headline = "Nessun segnale rilevante: tutti gli strumenti nella norma."
    return {
        "as_of": as_of,
        "headline": headline,
        "market": market,
        "market_wide_high_vol": wide,
        "counts": counts,
        "instruments": instruments,
        "quiet": quiet,
        "missing": missing,
        "rules": __doc__.split("Segnali per strumento")[1].strip() if __doc__ else "",
    }


def attach_history(dg: dict, current_events: list[dict], labels: dict[str, str]) -> dict:
    """Aggiunge a ogni strumento della sintesi le note "storicamente…" degli eventi recenti."""
    by_id = {i["id"]: i for i in dg["instruments"]}
    for e in current_events:
        if not e.get("history"):
            continue
        inst = by_id.get(e["id"])
        if inst is None:
            continue  # nessun segnale oggi per lo strumento: la nota da sola non basta a segnalarlo
        notes = inst.setdefault("history", [])
        if not any(n["type"] == e["type"] for n in notes):
            notes.append({"type": e["type"], "event": labels.get(e["type"], e["type"]),
                          "date": e["date"], "text": e["history"]})
    return dg


def to_markdown(digest: dict) -> str:
    icons = {"alta": "🔴", "media": "🟠", "bassa": "⚪"}
    lines = [f"### Sintesi del {digest.get('as_of') or '–'}", "", f"**{digest['headline']}**", ""]
    lines += [f"- {m}" for m in digest.get("market", [])]
    if digest.get("missing"):
        lines.append(f"- 🔴 Senza dati: {', '.join(digest['missing'])}")
    lines.append("")
    for inst in digest["instruments"]:
        lines.append(f"**{inst['name']} ({inst['id'].upper()})**")
        lines += [f"- {icons[s['severity']]} {s['text'][0].upper()}{s['text'][1:]}" for s in inst["signals"]]
        lines += [f"  - _{h['event'].capitalize()} ({h['date']}): {h['text']}_" for h in inst.get("history", [])]
        lines.append("")
    if digest.get("quiet"):
        lines.append(f"Nessuna novità: {', '.join(i.upper() for i in digest['quiet'])}.")
    return "\n".join(lines)
