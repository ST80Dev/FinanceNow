"""Valutazione automatica: per ogni strumento un giudizio in parole semplici.

Mette insieme ciò che le altre parti misurano (fase/regime, volatilità, anomalie, cali,
trimestrali, pattern storici, modello di direzione) e risponde a quattro domande:

1. **Situazione** — com'è messo adesso? (una sola etichetta, la prima regola vera vince)
   - `tempesta`   fase agitata in corso (regime turbolento);
   - `ripresa`    uscito dalla fase agitata da al massimo RECOVERY_SESSIONS sedute;
   - `pressione`  calo di almeno il 20% dal massimo, oppure agitazione in aumento
                  (volatilità ≥ 90% della propria storia, accelerazione, crollo anomalo recente);
   - `tranquillo` fase calma e volatilità non sopra il 60% della propria storia;
   - `norma`      tutto il resto.
2. **Rischio nel prossimo mese** — punti: volatilità rispetto alla propria storia
   (≥ 90%: +2, ≥ 70%: +1, ≤ 30%: −1) + 1 se è attivo un evento che STORICAMENTE è stato
   seguito da più agitazione o più cali (differenza netta nei pattern) + 1 se la trimestrale
   arriva entro SOON_SESSIONS sedute. ≥ 2 = alto, ≤ −1 = basso, altrimenti normale.
3. **Direzione** — la tendenza attuale (descrittiva: media a 200 sedute e ultimi 3 mesi) e
   la previsione sperimentale del modello di direzione, dichiarata affidabile solo se nel
   backtest ha battuto la frequenza storica di salita in modo netto.
4. **Prossimi eventi** — trimestrale in arrivo, reazione all'ultima trimestrale.

È un giudizio descrittivo con regole esplicite, non un consiglio operativo. Ogni sera viene
registrato (results/live/assessment_log.csv) e dopo un mese confrontato con quello che è
successo davvero (results/live/assessment_scores.json): è la prova di quanto vale.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pipeline import patterns
from pipeline.digest import PHASE, Context, instrument_signals

RECOVERY_SESSIONS = 10
SOON_SESSIONS = 10
TILT = 0.05
HORIZON = 21

SITUATIONS = {
    "tempesta": "In tempesta",
    "pressione": "Sotto pressione",
    "ripresa": "In ripresa",
    "norma": "Nella norma",
    "tranquillo": "Tranquillo",
}
SITUATION_WEIGHT = {"tempesta": 4, "pressione": 3, "ripresa": 2, "norma": 1, "tranquillo": 0}
RISK_WEIGHT = {"alto": 2, "normale": 1, "basso": 0}
# Eventi "di stato" ancora attivi oggi -> tipo di evento dei pattern con cui confrontarli.
ACTIVE_EVENT_LABELS = {
    "regime_turbolento": "fase agitata",
    "drawdown_20": "calo oltre il 20% dal massimo",
    "vol_estrema": "agitazione estrema",
    "vol_accelerazione": "agitazione in aumento",
    "crollo_anomalo": "crollo anomalo recente",
    "trimestrale_negativa": "reazione negativa all'ultima trimestrale",
    "trimestrale_positiva": "reazione positiva all'ultima trimestrale",
}

LOG_FILE = "assessment_log.csv"
SCORES_FILE = "assessment_scores.json"
LOG_COLUMNS = ["date", "id", "situation", "risk", "trend", "outlook", "p_up", "p_base", "created_utc"]


def _pct(v: float, digits: int = 0, signed: bool = False) -> str:
    s = f"{v * 100:.{digits}f}".replace(".", ",") + "%"
    return ("+" + s) if signed and v > 0 else s


# ---------------------------------------------------------------------------
# Situazione
# ---------------------------------------------------------------------------

def _previous_regime(ctx: Context) -> str | None:
    reg = ctx.regime or {}
    days = reg.get("days_in_state")
    states = [s for s in ctx.regime_states if s is not None]
    labels = [s["label"] for s in reg.get("states", [])]
    if not days or len(states) <= days:
        return None
    prev = states[-days - 1]
    return labels[prev] if prev < len(labels) else None


def active_events(ctx: Context, current_types: set[str]) -> list[str]:
    """Eventi dei pattern che descrivono la situazione di oggi."""
    f, out = ctx.features, []
    if (ctx.regime or {}).get("current") == "turbolento":
        out.append("regime_turbolento")
    if (f.get("drawdown") or 0) <= patterns.DRAWDOWN_LEVEL:
        out.append("drawdown_20")
    if (ctx.rv21_percentile or 0) >= patterns.VOL_QUANTILE:
        out.append("vol_estrema")
    rv5, rv63 = f.get("rv_5"), f.get("rv_63")
    if rv5 and rv63 and rv5 / rv63 >= patterns.ACCEL_RATIO:
        out.append("vol_accelerazione")
    for t in ("crollo_anomalo", "trimestrale_negativa", "trimestrale_positiva"):
        if t in current_types:
            out.append(t)
    return out


def situation(ctx: Context, actives: list[str]) -> str:
    reg = ctx.regime or {}
    current = reg.get("current")
    if current == "turbolento":
        return "tempesta"
    if _previous_regime(ctx) == "turbolento" and (reg.get("days_in_state") or 99) <= RECOVERY_SESSIONS:
        return "ripresa"
    if set(actives) & {"drawdown_20", "vol_estrema", "vol_accelerazione", "crollo_anomalo"} or \
            (ctx.rv21_percentile or 0) >= 0.90:
        return "pressione"
    if current == "calmo" and (ctx.rv21_percentile is None or ctx.rv21_percentile <= 0.60):
        return "tranquillo"
    return "norma"


# ---------------------------------------------------------------------------
# Rischio
# ---------------------------------------------------------------------------

def historical_effect(event: str, stats: dict) -> dict | None:
    """Esito storico a un mese dopo l'evento, se diverso in modo netto da un giorno qualsiasi."""
    s = (stats or {}).get(event, {}).get(f"h{HORIZON}")
    if not s or s.get("n", 0) < 10:
        return None
    v, d = s.get("vol_ratio", {}), s.get("dd10", {})
    worse_vol = v.get("significant") and v.get("excess", 0) > 0
    worse_dd = d.get("significant") and d.get("excess", 0) > 0
    return {"event": event, "n": s["n"], "vol_event": v.get("event"), "vol_base": v.get("baseline"),
            "dd10_event": d.get("event"), "dd10_base": d.get("baseline"),
            "ret_event": s.get("ret", {}).get("event"), "ret_base": s.get("ret", {}).get("baseline"),
            "ret_significant": bool(s.get("ret", {}).get("significant")),
            "riskier": bool(worse_vol or worse_dd), "worse_dd": bool(worse_dd)}


def risk(ctx: Context, actives: list[str], stats: dict) -> dict:
    p = ctx.rv21_percentile
    reasons, points = [], 0
    if p is not None:
        if p >= 0.90:
            points += 2
            reasons.append(f"si muove più che nel {round(p * 100)}% della sua storia")
        elif p >= 0.70:
            points += 1
            reasons.append(f"si muove più del solito (più che nel {round(p * 100)}% della sua storia)")
        elif p <= 0.30:
            points -= 1
            reasons.append("si muove meno del solito")
    effects = [e for e in (historical_effect(a, stats) for a in actives) if e]
    risky = [e for e in effects if e["riskier"]]
    evidence = None
    if risky:
        points += 1
        evidence = max(risky, key=lambda e: ((e["dd10_event"] or 0) - (e["dd10_base"] or 0), e["vol_event"] or 0))
        reasons.append(f"in passato, dopo una situazione simile ({ACTIVE_EVENT_LABELS[evidence['event']]}), "
                       f"il mese seguente è stato più movimentato del solito")
    nxt = (ctx.earnings or {}).get("next") or {}
    if nxt.get("sessions_until") is not None and nxt["sessions_until"] <= SOON_SESSIONS:
        points += 1
        reasons.append("trimestrale in arrivo: di solito il giorno dopo i conti si muove molto")
    level = "alto" if points >= 2 else ("basso" if points <= -1 else "normale")
    rv21 = ctx.features.get("rv_21")
    out = {"level": level, "points": points, "reasons": reasons,
           "month_move": rv21 / math.sqrt(12) if rv21 else None,
           "vol_percentile": p, "evidence": evidence}
    h21 = (ctx.direction or {}).get("h21")
    if h21:
        out["range_month"] = [math.expm1(h21["q10"]), math.expm1(h21["q90"])]
    return out


# ---------------------------------------------------------------------------
# Direzione
# ---------------------------------------------------------------------------

def trend(features: dict) -> dict:
    d200, m3, m12 = features.get("dist_ma200"), features.get("mom_3m"), features.get("mom_12m")
    if d200 is None or m3 is None:
        return {"key": "nd", "label": "non disponibile"}
    if d200 > 0.02 and m3 > 0:
        key, label = "su", "In salita"
    elif d200 < -0.02 and m3 < 0:
        key, label = "giu", "In discesa"
    else:
        key, label = "laterale", "Senza direzione chiara"
    out = {"key": key, "label": label, "ret_3m": math.expm1(m3)}
    if m12 is not None:
        out["ret_12m"] = math.expm1(m12)
    out["vs_ma200"] = math.expm1(d200)
    return out


def outlook(ctx: Context, verdicts: dict) -> dict | None:
    fc = (ctx.direction or {}).get(f"h{HORIZON}")
    if not fc:
        return None
    tilt = fc["p_up"] - fc["p_base"]
    key = "favorevole" if tilt >= TILT else ("sfavorevole" if tilt <= -TILT else "neutro")
    label = {"favorevole": "Più favorevole del solito", "sfavorevole": "Meno favorevole del solito",
             "neutro": "Come al solito"}[key]
    verdict = verdicts.get(f"h{HORIZON}", "non valutabile")
    drivers = []
    for d in fc.get("drivers", []):
        push = "spinge verso l'alto" if d["effect"] > 0 else "spinge verso il basso"
        drivers.append(f"{d['label']} ({push})")
    out = {"key": key, "label": label, "p_up": fc["p_up"], "p_base": fc["p_base"], "drivers": drivers,
           "reliable": verdict == "vantaggio", "verdict": verdict}
    h5 = (ctx.direction or {}).get("h5")
    if h5:
        out["p_up_week"], out["p_base_week"] = h5["p_up"], h5["p_base"]
    return out


# ---------------------------------------------------------------------------
# Eventi
# ---------------------------------------------------------------------------

def upcoming_events(ctx: Context) -> list[dict]:
    e = ctx.earnings or {}
    out = []
    if not e.get("available"):
        return out
    nxt = e.get("next")
    if nxt and nxt.get("sessions_until") is not None:
        n = nxt["sessions_until"]
        when = "domani" if n <= 1 else f"tra {n} sedute"
        text = f"Trimestrale il {nxt['date'][8:10]}/{nxt['date'][5:7]} ({nxt.get('timing', 'orario n.d.')}): reazione {when}."
        if e.get("typical_move"):
            text += (f" Di solito quel giorno si muove di ±{_pct(math.expm1(e['typical_move']))}"
                     f" (un giorno qualsiasi ±{_pct(math.expm1(e['typical_day_move']), 1)}).")
        out.append({"kind": "trimestrale", "sessions_until": n, "date": nxt["date"], "text": text,
                    "soon": n <= SOON_SESSIONS})
    last = e.get("last")
    if last and last.get("sessions_ago") is not None and last["sessions_ago"] <= 5:
        vs = last.get("reaction_vs_bench")
        text = f"Reazione all'ultima trimestrale ({last['session'][8:10]}/{last['session'][5:7]}): {_pct(math.expm1(last['reaction']), 1, True)}"
        if vs is not None:
            text += f", {_pct(math.expm1(vs), 1, True)} rispetto al Nasdaq 100"
        out.append({"kind": "trimestrale_passata", "sessions_ago": last["sessions_ago"], "text": text + "."})
    return out


# ---------------------------------------------------------------------------
# Insieme
# ---------------------------------------------------------------------------

def assess_instrument(ctx: Context, stats: dict, current_types: set[str], verdicts: dict) -> dict:
    actives = active_events(ctx, current_types)
    sit = situation(ctx, actives)
    rk = risk(ctx, actives, stats)
    tr = trend(ctx.features)
    ol = outlook(ctx, verdicts)
    events = upcoming_events(ctx)
    why = _why(ctx, sit)
    attention = SITUATION_WEIGHT[sit] + RISK_WEIGHT[rk["level"]] + \
        (1 if any(e.get("soon") for e in events) else 0) + (5 if ctx.data_status in ("stale", "no_data") else 0)
    return {
        "id": ctx.id, "name": ctx.name, "description": ctx.description, "asset_class": ctx.asset_class,
        "as_of": ctx.as_of, "data_status": ctx.data_status,
        "situation": {"key": sit, "label": SITUATIONS[sit], "why": why},
        "risk": rk, "trend": tr, "outlook": ol, "events": events,
        "active_events": [{"type": a, "label": ACTIVE_EVENT_LABELS[a]} for a in actives],
        "signals": instrument_signals(ctx),
        "attention": attention,
    }


def _why(ctx: Context, sit: str) -> str:
    reg, f = ctx.regime or {}, ctx.features
    days = reg.get("days_in_state")
    dd = f.get("drawdown")
    if sit == "tempesta":
        return f"movimenti ampi e nervosi da {days} sedute" if days else "movimenti ampi e nervosi"
    if sit == "ripresa":
        ago = (days or 1) - 1
        txt = "oggi" if ago == 0 else ("1 seduta fa" if ago == 1 else f"{ago} sedute fa")
        return f"uscito dalla fase agitata {txt}" + (f", ancora a {_pct(dd)} dal massimo" if dd is not None and dd <= -0.1 else "")
    if sit == "pressione":
        parts = []
        if dd is not None and dd <= patterns.DRAWDOWN_LEVEL:
            parts.append(f"a {_pct(dd)} dal massimo")
        if (ctx.rv21_percentile or 0) >= 0.90:
            parts.append("si muove molto più del solito")
        rv5, rv63 = f.get("rv_5"), f.get("rv_63")
        if rv5 and rv63 and rv5 / rv63 >= patterns.ACCEL_RATIO:
            parts.append("agitazione in aumento nell'ultima settimana")
        return ", ".join(parts) or "segnali di tensione"
    if sit == "tranquillo":
        return f"fase calma da {days} sedute, movimenti contenuti" if days else "movimenti contenuti"
    phase = PHASE.get(reg.get("current"), "n.d.")
    return f"fase {phase}, nessun segnale particolare"


def build(contexts: list[Context], stats: dict, current_events: list[dict], verdicts: dict,
          benchmark_id: str = "ndx") -> dict:
    by_id: dict[str, set] = {}
    for e in current_events:
        by_id.setdefault(e["id"], set()).add(e["type"])
    items = [assess_instrument(c, stats, by_id.get(c.id, set()), verdicts) for c in contexts]
    items.sort(key=lambda a: (-a["attention"], a["id"]))
    stocks = [a for a in items if a["asset_class"] == "stock"]
    counts = {k: sum(a["situation"]["key"] == k for a in stocks) for k in SITUATIONS}
    risk_counts = {k: sum(a["risk"]["level"] == k for a in stocks) for k in RISK_WEIGHT}
    bench = next((a for a in items if a["id"] == benchmark_id), None)
    return {"market": market_view(bench, counts, risk_counts, len(stocks)), "instruments": items,
            "situation_labels": SITUATIONS, "horizon": HORIZON}


def market_view(bench: dict | None, counts: dict, risk_counts: dict, n_stocks: int) -> dict:
    if bench is None:
        return {"headline": "Benchmark non disponibile.", "counts": counts, "risk_counts": risk_counts}
    sit = bench["situation"]
    head = f"Mercato (Nasdaq 100): {sit['label'].lower()} — {sit['why']}."
    parts = []
    if counts.get("tempesta"):
        parts.append(f"{counts['tempesta']} in tempesta")
    if counts.get("pressione"):
        parts.append(f"{counts['pressione']} sotto pressione")
    if counts.get("ripresa"):
        parts.append(f"{counts['ripresa']} in ripresa")
    calm = counts.get("tranquillo", 0) + counts.get("norma", 0)
    detail = (f"Titoli: {', '.join(parts)}; {calm} su {n_stocks} tranquilli o nella norma."
              if parts else f"Tutti i {n_stocks} titoli tranquilli o nella norma.")
    return {"headline": head, "detail": detail, "situation": sit, "risk": bench["risk"]["level"],
            "trend": bench["trend"], "counts": counts, "risk_counts": risk_counts}


# ---------------------------------------------------------------------------
# Serie live dei giudizi
# ---------------------------------------------------------------------------

def log_assessment(path, assessment: dict, created: str) -> int:
    rows = []
    for a in assessment["instruments"]:
        if not a.get("as_of") or a.get("data_status") in ("stale", "no_data"):
            continue
        ol = a.get("outlook") or {}
        rows.append({"date": a["as_of"], "id": a["id"], "situation": a["situation"]["key"],
                     "risk": a["risk"]["level"], "trend": a["trend"]["key"], "outlook": ol.get("key", ""),
                     "p_up": ol.get("p_up"), "p_base": ol.get("p_base"), "created_utc": created})
    if not rows:
        return 0
    new = pd.DataFrame(rows, columns=LOG_COLUMNS)
    if path.exists():
        old = pd.read_csv(path, dtype={"date": str, "id": str})
        seen = set(zip(old["date"], old["id"]))
        new = new[[k not in seen for k in zip(new["date"], new["id"])]]
    if new.empty:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, mode="a", header=not path.exists(), index=False, float_format="%.6g")
    return len(new)


def score_log(path, returns: dict[str, dict]) -> dict:
    """Esiti a un mese dei giudizi registrati: per livello di rischio e per previsione di direzione.

    returns[id] = {"ret": log-rendimenti, "gap": salti di apertura}.
    """
    if not path.exists():
        return {"n_logged": 0, "n_evaluated": 0}
    lg = pd.read_csv(path, dtype={"date": str, "id": str, "outlook": str})
    rows = []
    for r in lg.itertuples():
        d = returns.get(r.id)
        if d is None:
            continue
        out = patterns.outcome_table(d["ret"].dropna(), HORIZON, None, d.get("gap"))
        day = pd.Timestamp(r.date)
        if day in out.index:
            o = out.loc[day]
            rows.append({"risk": r.risk, "situation": r.situation, "outlook": r.outlook if isinstance(r.outlook, str) else "",
                         "vol_ratio": o["vol_ratio"], "dd10": o["dd10"], "up": float(o["ret"] > 0)})
    res = {"n_logged": int(len(lg)), "first_date": lg["date"].min() if len(lg) else None,
           "n_evaluated": len(rows), "horizon": HORIZON}
    if rows:
        ev = pd.DataFrame(rows)
        for col in ("risk", "situation", "outlook"):
            res[f"by_{col}"] = {k: {"n": int(len(g)), "vol_ratio": float(g["vol_ratio"].mean()),
                                    "dd10": float(g["dd10"].mean()), "up_rate": float(g["up"].mean())}
                                for k, g in ev.groupby(col) if k}
    return res


def to_markdown(assessment: dict) -> str:
    icons = {"tempesta": "🔴", "pressione": "🟠", "ripresa": "🔵", "norma": "⚪", "tranquillo": "🟢"}
    m = assessment["market"]
    lines = [f"**{m['headline']}**", "", m.get("detail", ""), "",
             "| Strumento | Situazione | Rischio (mese) | Tendenza | Direzione (sperimentale) | Eventi |",
             "|---|---|---|---|---|---|"]
    for a in assessment["instruments"]:
        ol = a.get("outlook") or {}
        ev = "; ".join(e["text"] for e in a["events"] if e.get("soon") or e["kind"] == "trimestrale_passata")
        lines.append(f"| {a['name']} ({a['id'].upper()}) | {icons[a['situation']['key']]} {a['situation']['label']}: "
                     f"{a['situation']['why']} | {a['risk']['level']} | {a['trend']['label']} | "
                     f"{ol.get('label', '–')}{'' if ol.get('reliable') else ' (non affidabile)' if ol else ''} | {ev or '–'} |")
    return "\n".join(lines)
