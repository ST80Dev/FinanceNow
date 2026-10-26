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
- dati non aggiornati o mancanti (alta).

Contro il rumore: se almeno metà degli strumenti ha la volatilità >= 90° percentile, la cosa
viene detta una volta sola a livello di mercato e per i singoli strumenti restano solo i
casi >= 99° percentile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

RECENT_SESSIONS = 5
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


def _pct(v: float, digits: int = 0, signed: bool = False) -> str:
    s = f"{v * 100:.{digits}f}".replace(".", ",") + "%"
    return ("+" + s) if signed and v > 0 else s


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
    verb = "è passato" if up else "è tornato"
    prob = reg.get("probabilities", {}).get(current)
    prob_txt = f" (probabilità attuale {_pct(prob)})" if prob is not None else ""
    return _signal(sev, "regime", f"{verb} al regime {current}{prob_txt} {when}, dal regime {previous}.")


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
        z_txt = f"{z:+.1f}".replace(".", ",")
        out.append(_signal(sev, "anomalia",
                           f"anomalia di {ANOMALY_LABELS.get(kind, kind)} il {_fmt_date(a['date'])}: {val} (z {z_txt})."))
    return out


def volatility_signals(ctx: Context, market_wide: bool = False) -> list[dict]:
    f, out = ctx.features, []
    p, rv21 = ctx.rv21_percentile, f.get("rv_21")
    if market_wide and p is not None and p < 0.99:
        p = None  # già detto a livello di mercato
    if p is not None and rv21 is not None:
        pct_txt = f"{round(p * 100)}° percentile del proprio storico"
        if p >= 0.95:
            out.append(_signal("alta", "volatilità", f"volatilità a 21 sedute {_pct(rv21)}, al {pct_txt}."))
        elif p >= 0.90:
            out.append(_signal("media", "volatilità", f"volatilità a 21 sedute {_pct(rv21)}, al {pct_txt}."))
        elif p <= 0.05:
            out.append(_signal("bassa", "volatilità", f"volatilità insolitamente bassa: {_pct(rv21)}, al {pct_txt}."))
    rv5, rv63 = f.get("rv_5"), f.get("rv_63")
    if rv5 and rv63 and rv63 > 0 and rv5 / rv63 >= 1.5:
        out.append(_signal("media", "volatilità",
                           f"volatilità in accelerazione: {_pct(rv5)} sulle ultime 5 sedute contro {_pct(rv63)} a 63 sedute."))
    return out


def drawdown_signal(ctx: Context) -> dict | None:
    dd = ctx.features.get("drawdown")
    if dd is not None and dd <= -0.20:
        return _signal("media", "drawdown", f"drawdown di {_pct(dd)} dal massimo storico.")
    return None


def data_signal(ctx: Context) -> dict | None:
    if ctx.data_status in ("stale", "no_data"):
        return _signal("alta", "dati", f"dati non aggiornati: ultima seduta disponibile {ctx.as_of}.")
    return None


def instrument_signals(ctx: Context, market_wide: bool = False) -> list[dict]:
    signals = [s for s in (data_signal(ctx), regime_change(ctx)) if s]
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
        lines.append(f"{bench.name}: regime {bench.regime['current']} da {bench.regime['days_in_state']} sedute, "
                     f"volatilità a 21 sedute {_pct(bench.features.get('rv_21', 0))}.")
    stocks = [c for c in contexts if c.asset_class == "stock"]
    if stocks:
        turb = [c.id.upper() for c in stocks if c.regime.get("current") == "turbolento"]
        calm = [c for c in stocks if c.regime.get("current") == "calmo"]
        txt = f"Azioni: {len(turb)} su {len(stocks)} in regime turbolento"
        txt += f" ({', '.join(turb)})" if turb else ""
        txt += f", {len(calm)} in regime calmo."
        lines.append(txt)
        pcts = sorted(c.rv21_percentile for c in stocks if c.rv21_percentile is not None)
        if pcts:
            med = pcts[len(pcts) // 2]
            lines.append(f"Volatilità dei titoli: mediana al {round(med * 100)}° percentile del rispettivo storico.")
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
        market.insert(0, f"Volatilità elevata su gran parte dell'universo: {n_high} strumenti su {n_tot} "
                         f"sopra il 90° percentile del proprio storico (singolarmente segnalati solo i casi estremi).")
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
        lines.append("")
    if digest.get("quiet"):
        lines.append(f"Nessuna novità: {', '.join(i.upper() for i in digest['quiet'])}.")
    return "\n".join(lines)
