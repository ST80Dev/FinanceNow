"""Pattern: cosa è successo storicamente DOPO certi eventi, confrontato con un giorno qualsiasi.

È uno studio di eventi descrittivo, non una previsione. Per ogni tipo di evento:
1. si trovano tutti i casi passati, usando a ogni data solo i dati disponibili fino a quella data;
2. si misura cosa è successo nelle H sedute successive (H = 5, 21, 63);
3. si confronta con la stessa misura su TUTTE le sedute dello stesso titolo (baseline);
4. si usano solo finestre che finiscono prima del periodo di test finale (HOLDOUT_START).

Esiti misurati sulle H sedute successive:
- `vol_ratio`  volatilità realizzata / volatilità tipica del titolo (mediana storica): >1 = più agitato del normale;
- `dd10`       1 se dentro la finestra c'è stato un calo di almeno il 10% da un massimo, altrimenti 0;
- `ret`        log-rendimento cumulato medio (dato debole: rumore e bias di selezione dell'universo).

Rendimento e calo si misurano dall'APERTURA della seduta successiva all'evento: l'analisi
gira a mercato chiuso, quindi è il primo prezzo a cui si potrebbe agire.

Un risultato è "netto" se l'intervallo al 90% esclude lo zero E la differenza supera una soglia
minima di rilevanza (MATERIAL). Incertezza: gli eventi della stessa settimana (es. crollo di mercato che colpisce tutti i titoli)
non sono indipendenti, quindi l'intervallo al 90% si calcola sulle medie per settimana.

Serie live: ogni evento nuovo (datato all'ultima seduta) viene registrato con l'esito atteso in
results/live/pattern_log.csv; quando la finestra si chiude si confronta con l'esito realizzato
(results/live/pattern_scores.json). Le due serie restano separate.

Uso:
    python -m pipeline.patterns
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import backtest
from pipeline.analyze import DEFAULT_RESULTS_DIR
from pipeline.features import TRADING_DAYS, compute_features
from pipeline.ingest import (DEFAULT_DATA_DIR, DEFAULT_UNIVERSE, ConfigError, load_universe, read_json,
                             read_prices, write_json)
from pipeline.models import anomalies

log = logging.getLogger("financenow.patterns")

HORIZONS = (5, 21, 63)
MAIN_HORIZON = 21
COOLDOWN = 21
RECENT_SESSIONS = 5
VOL_QUANTILE = 0.95
ACCEL_RATIO = 1.5
DRAWDOWN_LEVEL = -0.20
ANOMALY_Z = 4.0
DD_THRESHOLD = -0.10
MIN_HISTORY = 252
Z90 = 1.645
# Differenza minima perché un risultato conti come "netto": oltre alla significatività
# statistica serve una differenza rilevante in pratica.
MATERIAL = {"vol_ratio": 0.05, "dd10": 0.02, "ret": 0.005}

EVENT_TYPES = {
    "regime_turbolento": "ingresso nel regime turbolento",
    "uscita_turbolento": "uscita dal regime turbolento",
    "crollo_anomalo": "crollo anomalo (rendimento con z ≤ −4)",
    "balzo_anomalo": "balzo anomalo (rendimento con z ≥ +4)",
    "volume_anomalo": "volume anomalo (z ≥ 4)",
    "vol_estrema": "volatilità a 21 sedute oltre il 95° percentile dello storico fino a quel giorno",
    "vol_accelerazione": "volatilità in accelerazione (5 sedute ≥ 1,5 volte quella a 63)",
    "drawdown_20": "discesa sotto −20% dal massimo",
}

LOG_FILE = "pattern_log.csv"
SCORES_FILE = "pattern_scores.json"


# ---------------------------------------------------------------------------
# Eventi
# ---------------------------------------------------------------------------

def _crossing_up(x: pd.Series, thr) -> pd.Series:
    """True il giorno in cui x supera la soglia (il giorno prima non la superava)."""
    above = x > thr
    return above & ~above.shift(1, fill_value=False) & x.notna()


def apply_cooldown(dates: pd.DatetimeIndex, index: pd.DatetimeIndex, sessions: int = COOLDOWN) -> list:
    """Tiene un evento solo se sono passate almeno `sessions` sedute dal precedente tenuto."""
    pos = index.get_indexer(dates)
    kept, last = [], -10 ** 9
    for d, p in sorted(zip(dates, pos), key=lambda t: t[1]):
        if p - last >= sessions:
            kept.append(d)
            last = p
    return kept


def detect_events(df: pd.DataFrame, feats: pd.DataFrame, regime_state: pd.Series | None,
                  anomaly_events: pd.DataFrame | None) -> pd.DataFrame:
    """Eventi (date, type) usando a ogni data solo informazioni disponibili fino a quella data."""
    idx = feats.index
    raw: dict[str, pd.DatetimeIndex] = {}

    rv21 = feats["rv_21"]
    thr = rv21.expanding(min_periods=MIN_HISTORY).quantile(VOL_QUANTILE).shift(1)
    raw["vol_estrema"] = idx[_crossing_up(rv21, thr).to_numpy()]
    ratio = feats["rv_5"] / feats["rv_63"].where(feats["rv_63"] > 0)
    raw["vol_accelerazione"] = idx[_crossing_up(ratio, ACCEL_RATIO).to_numpy()]
    raw["drawdown_20"] = idx[_crossing_up(-feats["drawdown"], -DRAWDOWN_LEVEL).to_numpy()]

    if regime_state is not None and regime_state.notna().any():
        s = regime_state.reindex(idx)
        prev = s.shift(1)
        valid = s.notna() & prev.notna()
        raw["regime_turbolento"] = idx[(valid & (s == 2) & (prev != 2)).to_numpy()]
        raw["uscita_turbolento"] = idx[(valid & (s < 2) & (prev == 2)).to_numpy()]

    if anomaly_events is not None and len(anomaly_events):
        a = anomaly_events
        raw["crollo_anomalo"] = pd.DatetimeIndex(a.loc[(a["kind"] == "rendimento") & (a["z"] <= -ANOMALY_Z), "date"])
        raw["balzo_anomalo"] = pd.DatetimeIndex(a.loc[(a["kind"] == "rendimento") & (a["z"] >= ANOMALY_Z), "date"])
        raw["volume_anomalo"] = pd.DatetimeIndex(a.loc[(a["kind"] == "volume") & (a["z"] >= ANOMALY_Z), "date"])

    rows = [(d, t) for t, dates in raw.items() for d in apply_cooldown(dates, idx)]
    return pd.DataFrame(rows, columns=["date", "type"]).sort_values(["date", "type"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Esiti
# ---------------------------------------------------------------------------

def forward_outcomes(ret: pd.Series, h: int, gap: pd.Series | None = None) -> pd.DataFrame:
    """Per ogni seduta t: esiti sulle sedute t+1..t+h (NaN se la finestra non è completa).

    Il segnale è noto solo a mercato chiuso: il primo prezzo a cui si può agire è l'apertura
    di t+1. Con `gap` = log(open / close precedente), rendimento e calo massimo si misurano
    dall'apertura di t+1 (il gap notturno non è catturabile); la volatilità resta close-to-close.
    """
    r = ret.fillna(0.0).to_numpy()
    n = len(r)
    out = pd.DataFrame(index=ret.index, columns=["vol", "ret", "maxdd"], dtype=float)
    if n <= h:
        return out
    c = np.concatenate([[0.0], np.cumsum(r)])            # c[t+1] = log-prezzo relativo dopo la seduta t
    windows = np.lib.stride_tricks.sliding_window_view(c[1:], h + 1)   # riga t: c[t+1..t+h+1]
    windows = windows[: n - h]
    path = windows - windows[:, :1]                         # cumulato da fine seduta t
    g = np.zeros(n - h)
    if gap is not None:
        g = gap.reindex(ret.index).fillna(0.0).to_numpy()[1: n - h + 1]   # gap di apertura di t+1
        path[:, 1:] -= g[:, None]                           # cumulato dall'apertura di t+1
    runmax = np.maximum.accumulate(path, axis=1)
    maxdd = np.expm1((path - runmax).min(axis=1))
    fwd = np.lib.stride_tricks.sliding_window_view(r[1:], h)[: n - h]
    out.iloc[: n - h, 0] = fwd.std(axis=1, ddof=1) * math.sqrt(TRADING_DAYS) if h > 1 else np.abs(fwd[:, 0]) * math.sqrt(TRADING_DAYS)
    out.iloc[: n - h, 1] = fwd.sum(axis=1) - g
    out.iloc[: n - h, 2] = maxdd
    return out


def outcome_table(ret: pd.Series, h: int, cutoff: dt.date | None, gap: pd.Series | None = None) -> pd.DataFrame:
    """Esiti normalizzati per titolo, solo per finestre chiuse prima del cutoff."""
    o = forward_outcomes(ret, h, gap)
    if cutoff is not None:
        pos_cut = int((ret.index < pd.Timestamp(cutoff)).sum())
        o.iloc[max(0, pos_cut - h):] = np.nan                   # la finestra deve finire prima del cutoff
    o = o.dropna()
    typical = o["vol"].median()
    return pd.DataFrame({
        "vol_ratio": o["vol"] / typical if typical > 0 else np.nan,
        "dd10": (o["maxdd"] <= DD_THRESHOLD).astype(float),
        "ret": o["ret"],
    }, index=o.index)


# ---------------------------------------------------------------------------
# Statistiche
# ---------------------------------------------------------------------------

def _clustered_ci(excess: pd.Series) -> tuple[float | None, float | None, int]:
    """Intervallo al 90% della media dell'eccesso, con cluster per settimana."""
    if excess.empty:
        return None, None, 0
    week = excess.index.to_period("W")
    clusters = excess.groupby(week).mean()
    k = len(clusters)
    if k < 5:
        return None, None, k
    m, se = float(clusters.mean()), float(clusters.std(ddof=1) / math.sqrt(k))
    return m - Z90 * se, m + Z90 * se, k


def summarize(event_rows: pd.DataFrame) -> dict:
    """event_rows: colonne vol_ratio, dd10, ret + base_* (baseline del titolo) indicizzate per data."""
    n = len(event_rows)
    if n == 0:
        return {"n": 0}
    res = {"n": n, "n_instruments": int(event_rows["id"].nunique())}
    for m in ("vol_ratio", "dd10", "ret"):
        cond = event_rows[m]
        base = event_rows[f"base_{m}"]
        excess = (cond - base)
        lo, hi, k = _clustered_ci(excess)
        res[m] = {
            "event": float(cond.mean()),
            "baseline": float(base.mean()),
            "excess": float(excess.mean()),
            "ci90": [lo, hi] if lo is not None else None,
            "significant": bool(lo is not None and (lo > 0 or hi < 0) and abs(excess.mean()) >= MATERIAL[m]),
        }
        res["n_weeks"] = k
    return res


def study(per_ticker: dict[str, dict], cutoff: dt.date | None) -> dict:
    """per_ticker[id] = {"ret": Series, "events": DataFrame(date, type)} -> statistiche per tipo e orizzonte."""
    stats: dict = {t: {} for t in EVENT_TYPES}
    by_ticker: dict = {t: {} for t in EVENT_TYPES}
    for h in HORIZONS:
        rows = []
        for tid, d in per_ticker.items():
            out = outcome_table(d["ret"], h, cutoff, d.get("gap"))
            if out.empty:
                continue
            base = {"base_vol_ratio": out["vol_ratio"].mean(), "base_dd10": out["dd10"].mean(),
                    "base_ret": out["ret"].mean()}
            ev = d["events"]
            ev = ev[ev["date"].isin(out.index)]
            if ev.empty:
                continue
            o = out.loc[ev["date"]].reset_index(drop=True)
            o["type"], o["id"], o["date"] = ev["type"].to_numpy(), tid, ev["date"].to_numpy()
            for k, v in base.items():
                o[k] = v
            rows.append(o)
        if not rows:
            continue
        allrows = pd.concat(rows, ignore_index=True).set_index("date")
        for t in EVENT_TYPES:
            sub = allrows[allrows["type"] == t]
            stats[t][f"h{h}"] = summarize(sub)
            if h == MAIN_HORIZON:
                by_ticker[t] = {tid: int(c) for tid, c in sub["id"].value_counts().items()}
    return {"stats": stats, "events_by_instrument": by_ticker}


# ---------------------------------------------------------------------------
# Testo per la sintesi
# ---------------------------------------------------------------------------

def _p(v: float, signed=False) -> str:
    s = f"{v * 100:.0f}%"
    return ("+" + s) if signed and v > 0 else s


def describe(event_type: str, stats: dict) -> str | None:
    """Frase su cosa è successo di solito dopo questo tipo di evento (orizzonte principale)."""
    s = stats.get(event_type, {}).get(f"h{MAIN_HORIZON}")
    if not s or s.get("n", 0) < 10:
        return None
    v, d = s["vol_ratio"], s["dd10"]
    parts = []
    if v["significant"]:
        parts.append(f"volatilità pari a {v['event']:.1f} volte quella tipica (giorno qualsiasi: {v['baseline']:.1f})".replace(".", ","))
    if d["significant"]:
        parts.append(f"calo di almeno il 10% nel {_p(d['event'])} dei casi (giorno qualsiasi: {_p(d['baseline'])})")
    head = (f"storicamente, dopo eventi simili ({s['n']} casi su {s['n_instruments']} strumenti, "
            f"{s.get('n_weeks', 0)} settimane diverse), nelle {MAIN_HORIZON} sedute successive "
            f"(misurate dall'apertura del giorno dopo): ")
    if not parts:
        return head + "nessuna differenza netta di volatilità o rischio rispetto a un giorno qualsiasi."
    return head + "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# Live: registrazione e verifica
# ---------------------------------------------------------------------------

def log_new_events(path: Path, current: list[dict], stats: dict, created: str) -> int:
    """Registra gli eventi datati all'ultima seduta, con l'esito atteso. Append-only."""
    rows = []
    for e in current:
        if not e.get("is_last_session"):
            continue
        s = stats.get(e["type"], {}).get(f"h{MAIN_HORIZON}", {})
        if not s.get("n"):
            continue
        rows.append({
            "date": e["date"], "id": e["id"], "type": e["type"], "h": MAIN_HORIZON,
            "exp_vol_ratio": s["vol_ratio"]["event"], "base_vol_ratio": s["vol_ratio"]["baseline"],
            "exp_dd10": s["dd10"]["event"], "base_dd10": s["dd10"]["baseline"],
            "n_hist": s["n"], "model_version": backtest.MODEL_VERSION, "created_utc": created,
        })
    if not rows:
        return 0
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path, dtype={"date": str, "id": str, "type": str})
        keys = set(zip(old["date"], old["id"], old["type"]))
        new = new[[k not in keys for k in zip(new["date"], new["id"], new["type"])]]
    if new.empty:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, mode="a", header=not path.exists(), index=False, float_format="%.6g")
    return len(new)


def score_log(path: Path, per_ticker: dict[str, dict]) -> dict:
    """Esiti realizzati degli eventi registrati la cui finestra si è chiusa."""
    if not path.exists():
        return {"n_logged": 0, "n_evaluated": 0, "by_type": {}}
    lg = pd.read_csv(path, dtype={"date": str, "id": str, "type": str})
    evaluated = []
    for r in lg.itertuples():
        d = per_ticker.get(r.id)
        if d is None:
            continue
        out = outcome_table(d["ret"].dropna(), int(r.h), None, d.get("gap"))
        day = pd.Timestamp(r.date)
        if day not in out.index:
            continue
        o = out.loc[day]
        evaluated.append({"type": r.type, "vol_ratio": o["vol_ratio"], "dd10": o["dd10"],
                          "exp_vol_ratio": r.exp_vol_ratio, "base_vol_ratio": r.base_vol_ratio,
                          "exp_dd10": r.exp_dd10, "base_dd10": r.base_dd10})
    by_type = {}
    if evaluated:
        ev = pd.DataFrame(evaluated)
        for t, g in ev.groupby("type"):
            by_type[t] = {
                "n": int(len(g)),
                "realized_vol_ratio": float(g["vol_ratio"].mean()),
                "expected_vol_ratio": float(g["exp_vol_ratio"].mean()),
                "baseline_vol_ratio": float(g["base_vol_ratio"].mean()),
                "realized_dd10": float(g["dd10"].mean()),
                "expected_dd10": float(g["exp_dd10"].mean()),
                "baseline_dd10": float(g["base_dd10"].mean()),
            }
    return {"n_logged": int(len(lg)), "n_evaluated": len(evaluated), "by_type": by_type}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect(instruments, data_dir: Path, results_dir: Path) -> dict[str, dict]:
    """Per ogni strumento: rendimenti ed eventi (con dati e risultati già calcolati)."""
    prices = {}
    per_ticker = {}
    for inst in instruments:
        df = prices.setdefault(inst.id, read_prices(data_dir / f"{inst.id}.parquet"))
        if df is None or len(df) < MIN_HISTORY:
            continue
        bench = None
        if inst.benchmark:
            bench = prices.setdefault(inst.benchmark, read_prices(data_dir / f"{inst.benchmark}.parquet"))
        feats = compute_features(df, bench)
        analysis = read_json(results_dir / "analysis" / f"{inst.id}.json")
        hist = analysis.get("regime_history", {})
        state = (pd.Series(hist["state"], index=pd.DatetimeIndex(hist["dates"]), dtype=float)
                 if hist.get("dates") else None)
        events = detect_events(df, feats, state, anomalies.detect(df, bench))
        gap = np.log(df["open"] / df["close"].shift(1))
        per_ticker[inst.id] = {"ret": feats["ret"], "gap": gap, "events": events}
    return per_ticker


def current_events(per_ticker: dict[str, dict]) -> list[dict]:
    out = []
    for tid, d in per_ticker.items():
        idx = d["ret"].index
        recent = set(idx[-RECENT_SESSIONS:])
        for e in d["events"].itertuples():
            if e.date in recent:
                out.append({"id": tid, "type": e.type, "date": e.date.date().isoformat(),
                            "is_last_session": e.date == idx[-1]})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pipeline.patterns", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    try:
        instruments = load_universe(args.universe)
    except ConfigError as e:
        log.error("configurazione non valida: %s", e)
        return 2

    per_ticker = collect(instruments, args.data_dir, args.results_dir)
    if not per_ticker:
        log.warning("nessun dato: pattern non calcolati")
        return 1
    result = study(per_ticker, backtest.HOLDOUT_START)
    cur = current_events(per_ticker)
    created = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for e in cur:
        e["history"] = describe(e["type"], result["stats"])
    out_dir = args.results_dir / "patterns"
    write_json({
        "series": "backtest",
        "generated": created,
        "note": "studio di eventi sullo storico, finestre chiuse prima del periodo di test finale",
        "holdout_start": backtest.HOLDOUT_START.isoformat(),
        "horizons": list(HORIZONS),
        "main_horizon": MAIN_HORIZON,
        "event_types": EVENT_TYPES,
        **result,
    }, out_dir / "stats.json")
    write_json({"generated": created, "events": cur}, out_dir / "current.json")

    live_dir = args.results_dir / "live"
    added = log_new_events(live_dir / LOG_FILE, cur, result["stats"], created)
    scores = score_log(live_dir / LOG_FILE, per_ticker)
    write_json({"series": "live", "generated": created, "horizon": MAIN_HORIZON, **scores}, live_dir / SCORES_FILE)
    n_ev = sum(len(v["events"]) for v in per_ticker.values())
    log.info("pattern: %d eventi storici, %d recenti, %d nuovi nel log live", n_ev, len(cur), added)
    return 0


if __name__ == "__main__":
    sys.exit(main())
