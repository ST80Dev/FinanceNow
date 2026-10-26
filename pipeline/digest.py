"""Sintesi automatica del giorno: cosa merita attenzione, senza dover aprire ogni scheda.

Regole esplicite e descrittive (nessun consiglio operativo). Due livelli:
- NOVITÀ: ciò che è cambiato nelle ultime RECENT_SESSIONS sedute, con gravità
  `alta` > `media` > `bassa`; gli strumenti sono ordinati per gravità;
- SITUAZIONI IN CORSO: stati che durano (non nuovi), in forma compatta, così non
  ricompaiono ogni sera come se fossero novità.
Gli strumenti senza novità né situazioni finiscono in un'unica riga.

Novità per strumento (finestra "recente" = ultime RECENT_SESSIONS sedute):
- cambio di regime recente (verso turbolento = alta; uscita da turbolento = media; altro = bassa);
- movimenti anomali, raggruppati per giorno (|z| massimo >= 6 alta, altrimenti media);
- volatilità a 21 sedute appena salita oltre il 95° percentile del proprio storico (alta);
- volatilità in accelerazione: 5 sedute >= 1,5 volte quella a 63 sedute (media);
- discesa appena avvenuta sotto -20% dal massimo (media);
- dati non aggiornati o mancanti (alta).

Situazioni in corso:
- regime turbolento da più di RECENT_SESSIONS sedute;
- drawdown dal massimo <= -20%;
- volatilità a 21 sedute >= 90° percentile del proprio storico, o <= 5° (calma insolita).

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
    "rendimento_vs_bench": "rispetto al benchmark",
    "gap": "gap di apertura",
    "range": "escursione high-low",
    "volume": "volume",
}
ANOMALY_ORDER = list(ANOMALY_LABELS)
SITUATION_RANK = {"regime": 3, "drawdown": 2, "volatilità": 2, "calma": 1}
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
    recent_events: list[dict] = field(default_factory=list)  # eventi recenti di pipeline.patterns (type, date)


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


def _anomaly_part(a: dict) -> str:
    kind, value, z = a["kind"], a.get("value"), a["z"]
    if kind == "volume" and value is not None:
        val = f"×{math.exp(value):.1f} la mediana".replace(".", ",")
    elif value is not None:
        val = _pct(value, 1, signed=True)
    else:
        val = "n.d."
    return f"{ANOMALY_LABELS.get(kind, kind)} {val} (z {z:+.1f})".replace(".", ",")


def anomaly_signals(ctx: Context) -> list[dict]:
    """Un segnale per giorno: le anomalie dello stesso giorno sono di solito un unico movimento."""
    recent = set(ctx.recent_dates)
    by_day: dict[str, list[dict]] = {}
    for a in ctx.anomalies:
        if a.get("date") in recent and a.get("z") is not None:
            by_day.setdefault(a["date"], []).append(a)
    out = []
    for day in sorted(by_day, reverse=True):
        items = sorted(by_day[day], key=lambda a: ANOMALY_ORDER.index(a["kind"]) if a["kind"] in ANOMALY_ORDER else 99)
        zmax = max(abs(a["z"]) for a in items)
        sev = "alta" if zmax >= 6 else "media"
        parts = ", ".join(_anomaly_part(a) for a in items)
        out.append(_signal(sev, "anomalia", f"movimento anomalo il {_fmt_date(day)}: {parts}."))
    return out


def _recent_event(ctx: Context, event_type: str) -> dict | None:
    hits = [e for e in ctx.recent_events if e.get("type") == event_type]
    return max(hits, key=lambda e: e["date"]) if hits else None


def _vol_visible(ctx: Context, market_wide: bool) -> float | None:
    """Percentile da riportare per lo strumento (None se già detto a livello di mercato)."""
    p = ctx.rv21_percentile
    if market_wide and p is not None and p < 0.99:
        return None
    return p


def volatility_signals(ctx: Context, market_wide: bool = False) -> list[dict]:
    f, out = ctx.features, []
    p, rv21 = _vol_visible(ctx, market_wide), f.get("rv_21")
    ev = _recent_event(ctx, "vol_estrema")
    if ev and p is not None and rv21 is not None and p >= 0.90:
        out.append(_signal("alta", "volatilità",
                           f"volatilità salita oltre il 95° percentile del proprio storico il {_fmt_date(ev['date'])} "
                           f"(ora {_pct(rv21)}, {round(p * 100)}° percentile)."))
    rv5, rv63 = f.get("rv_5"), f.get("rv_63")
    if rv5 and rv63 and rv63 > 0 and rv5 / rv63 >= 1.5:
        out.append(_signal("media", "volatilità",
                           f"volatilità in accelerazione: {_pct(rv5)} sulle ultime 5 sedute contro {_pct(rv63)} a 63 sedute."))
    return out


def drawdown_signal(ctx: Context) -> dict | None:
    dd = ctx.features.get("drawdown")
    ev = _recent_event(ctx, "drawdown_20")
    if ev and dd is not None and dd <= -0.20:
        return _signal("media", "drawdown",
                       f"sceso sotto −20% dal massimo il {_fmt_date(ev['date'])} (ora {_pct(dd)}).")
    return None


def instrument_situations(ctx: Context, signals: list[dict], market_wide: bool = False) -> list[dict]:
    """Stati che durano. Non ripete ciò che è già tra le novità di oggi."""
    news = {s["kind"] for s in signals}
    out = []
    reg = ctx.regime or {}
    days = reg.get("days_in_state")
    if reg.get("current") == "turbolento" and days and days > RECENT_SESSIONS:
        out.append({"kind": "regime", "text": f"regime turbolento da {days} sedute"})
    dd = ctx.features.get("drawdown")
    if dd is not None and dd <= -0.20 and "drawdown" not in news:
        out.append({"kind": "drawdown", "text": f"drawdown {_pct(dd)} dal massimo"})
    p, rv21 = _vol_visible(ctx, market_wide), ctx.features.get("rv_21")
    if p is not None and rv21 is not None and not any(s["kind"] == "volatilità" and "percentile" in s["text"] for s in signals):
        if p >= 0.90:
            out.append({"kind": "volatilità", "text": f"volatilità al {round(p * 100)}° percentile del proprio storico ({_pct(rv21)})"})
        elif p <= 0.05:
            out.append({"kind": "calma", "text": f"volatilità insolitamente bassa ({round(p * 100)}° percentile, {_pct(rv21)})"})
    return sorted(out, key=lambda x: -SITUATION_RANK[x["kind"]])


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
    instruments, situations, quiet = [], [], []
    wide, n_high, n_tot = market_wide_high_vol(contexts)
    market = market_lines(contexts)
    if wide:
        market.insert(0, f"Volatilità elevata su gran parte dell'universo: {n_high} strumenti su {n_tot} "
                         f"sopra il 90° percentile del proprio storico (singolarmente segnalati solo i casi estremi).")
    for ctx in contexts:
        sigs = instrument_signals(ctx, wide)
        sits = instrument_situations(ctx, sigs, wide)
        if sigs:
            instruments.append({"id": ctx.id, "name": ctx.name, "max_severity": sigs[0]["severity"], "signals": sigs})
        if sits:
            situations.append({"id": ctx.id, "name": ctx.name, "items": sits})
        if not sigs and not sits:
            quiet.append(ctx.id)
    instruments.sort(key=lambda i: (-SEVERITY_RANK[i["max_severity"]], -len(i["signals"]), i["id"]))
    situations.sort(key=lambda i: (-max(SITUATION_RANK[x["kind"]] for x in i["items"]), -len(i["items"]), i["id"]))
    counts = {s: sum(1 for i in instruments for x in i["signals"] if x["severity"] == s) for s in SEVERITY_RANK}
    missing = (data_status or {}).get("missing") or []
    as_of = max((c.as_of for c in contexts if c.as_of), default=None)
    if instruments:
        top = ", ".join(i["id"].upper() for i in instruments if i["max_severity"] == "alta")
        headline = (f"Novità su {len(instruments)} strumenti"
                    + (f" (priorità alta: {top})" if top else ", nessuna con priorità alta"))
    else:
        headline = "Nessuna novità"
    headline += (f"; {len(situations)} situazioni in corso." if situations else ".")
    return {
        "as_of": as_of,
        "headline": headline,
        "market": market,
        "market_wide_high_vol": wide,
        "counts": counts,
        "instruments": instruments,
        "situations": situations,
        "quiet": quiet,
        "missing": missing,
        "rules": __doc__.split("Novità per strumento")[1].strip() if __doc__ else "",
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
    if digest["instruments"]:
        lines += ["#### Novità", ""]
    for inst in digest["instruments"]:
        lines.append(f"**{inst['name']} ({inst['id'].upper()})**")
        lines += [f"- {icons[s['severity']]} {s['text'][0].upper()}{s['text'][1:]}" for s in inst["signals"]]
        lines += [f"  - _{h['event'].capitalize()} ({h['date']}): {h['text']}_" for h in inst.get("history", [])]
        lines.append("")
    if digest.get("situations"):
        lines += ["#### Situazioni in corso", ""]
        lines += [f"- **{s['id'].upper()}**: {' · '.join(x['text'] for x in s['items'])}" for s in digest["situations"]]
        lines.append("")
    if digest.get("quiet"):
        lines.append(f"Nella norma: {', '.join(i.upper() for i in digest['quiet'])}.")
    return "\n".join(lines)
