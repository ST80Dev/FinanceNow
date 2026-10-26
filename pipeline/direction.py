"""Direzione del prezzo (SPERIMENTALE): previsione, backtest walk-forward e serie live.

Domanda: "nelle prossime h sedute il titolo salirà?" (h = 5 e 21). Il modello
(pipeline/models/direction.py) parte dalla frequenza storica di salita del titolo e la
corregge con feature di andamento, volatilità, regime e trimestrali.

Baseline obbligatorie:
- `freq_storica`  frequenza di salita del titolo fino a quel giorno (la "media storica");
- `moneta`        50% sempre (random walk senza deriva).

Validazione:
- walk-forward: ri-stima ogni REFIT_EVERY sedute su tutte le righe il cui esito era già noto
  (t + h + EMBARGO sedute prima della data di stima: niente sovrapposizione con il futuro);
- solo origini con esito chiuso prima del periodo di test finale (HOLDOUT_START);
- metriche: Brier (più basso = meglio), skill rispetto alla frequenza storica, log-loss,
  t di Diebold-Mariano sulla differenza di Brier (media per data, errori HAC),
  calibrazione, esito quando il modello si discosta dalla baseline, copertura dell'intervallo;
- configurazione registrata in results/backtest/trials.jsonl.

File:
- results/backtest/direction.json      metriche walk-forward (serie `backtest`);
- results/direction/latest.json        previsioni di oggi, con i fattori che le muovono;
- results/live/direction_forecasts.csv log append-only delle previsioni serali (serie `live`);
- results/live/direction_scores.json   valutazione delle previsioni live con esito già noto.

Uso:
    python -m pipeline.direction
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

from pipeline import backtest, earnings
from pipeline.analyze import DEFAULT_RESULTS_DIR
from pipeline.features import compute_features
from pipeline.ingest import (DEFAULT_DATA_DIR, DEFAULT_UNIVERSE, ConfigError, load_universe, read_json,
                             read_prices, write_json)
from pipeline.models import direction as dm

log = logging.getLogger("financenow.direction")

MODEL_VERSION = "0.1.0"
HORIZONS = (5, 21)
MIN_TRAIN = 1260          # ~5 anni di calendario prima della prima previsione
REFIT_EVERY = 63
EMBARGO = 5
MIN_HISTORY = 300
TILT = 0.05               # scostamento dalla baseline oltre il quale il modello "si sbilancia"
CAL_EDGES = (0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 1.0)
FORECASTS_FILE = "direction_forecasts.csv"
SCORES_FILE = "direction_scores.json"
LOG_KEY = ["origin", "id", "h"]
LOG_COLUMNS = LOG_KEY + ["p_up", "p_base", "q10", "q50", "q90", "model_version", "created_utc"]


def config() -> dict:
    return {
        "module": "direzione",
        "target": "segno del log-rendimento dall'apertura di t+1 alla chiusura di t+h",
        "horizons": list(HORIZONS),
        "model": "logistica L2 panel con offset = frequenza storica del titolo",
        "features": list(dm.FEATURES),
        "l2": dm.L2,
        "clim_shrink": dm.CLIM_SHRINK,
        "min_train": MIN_TRAIN,
        "refit_every": REFIT_EVERY,
        "embargo": EMBARGO,
        "quantiles": list(dm.QUANTILES),
        "holdout_start": backtest.HOLDOUT_START.isoformat(),
        "model_version": MODEL_VERSION,
    }


# ---------------------------------------------------------------------------
# Dati
# ---------------------------------------------------------------------------

def ticker_inputs(df: pd.DataFrame, bench: pd.DataFrame | None, p_turb: pd.Series | None,
                  announcements: list | None) -> dict:
    """Tutto ciò che serve per costruire le righe del panel di un titolo."""
    feats = compute_features(df, bench)
    gap = np.log(df["open"] / df["close"].shift(1))
    win = earnings.reaction_windows(announcements or [], df.index)
    rx = earnings.reactions(win, df["close"], bench["close"] if bench is not None else None) if announcements else None
    first = list(win["first"].dropna())
    upcoming = list(win.loc[win["first"].isna(), "expected_session"])
    return {"feats": feats, "gap": gap, "p_turb": p_turb, "earn_first": first, "earn_upcoming": upcoming,
            "earn_rx": rx}


def panel(per_ticker: dict[str, dict], h: int) -> pd.DataFrame:
    """Righe (data, id): feature, rendimento futuro, volatilità di oggi."""
    frames = []
    for tid, d in per_ticker.items():
        X = dm.feature_frame(d["feats"], d["p_turb"], h, d["earn_first"], d["earn_rx"], d["earn_upcoming"])
        X["fwd"] = dm.forward_return(d["feats"]["ret"], d["gap"], h)
        X["rv21"] = d["feats"]["rv_21"]
        X["id"] = tid
        X = X[X["rv21"].notna() & d["feats"]["mom_12m"].notna()]
        frames.append(X.reset_index(names="date"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["date", "id"]).reset_index(drop=True)


def _fit(rows: pd.DataFrame, h: int) -> dm.DirectionModel:
    X = rows[list(dm.FEATURES)].to_numpy(float)
    y = (rows["fwd"] > 0).to_numpy(float)
    z = rows["fwd"].to_numpy() / dm.sigma_h(rows["rv21"].to_numpy(), h)
    return dm.DirectionModel(X, y, rows["id"].to_numpy(), z)


def _predict(model: dm.DirectionModel, rows: pd.DataFrame, h: int) -> pd.DataFrame:
    X = rows[list(dm.FEATURES)].to_numpy(float)
    g = rows["id"].to_numpy()
    q = model.interval(dm.sigma_h(rows["rv21"].to_numpy(), h))
    return pd.DataFrame({"date": rows["date"].to_numpy(), "id": g, "p": model.predict(X, g),
                         "base": model.base_rate(g), "q10": q[:, 0], "q50": q[:, 1], "q90": q[:, 2],
                         "fwd": rows["fwd"].to_numpy()})


# ---------------------------------------------------------------------------
# Walk-forward e metriche
# ---------------------------------------------------------------------------

def walk_forward(pan: pd.DataFrame, h: int, cutoff: dt.date | None,
                 min_train: int | None = None, refit_every: int | None = None) -> tuple[pd.DataFrame, dm.DirectionModel | None]:
    min_train = MIN_TRAIN if min_train is None else min_train
    refit_every = REFIT_EVERY if refit_every is None else refit_every
    cal = pd.DatetimeIndex(sorted(pan["date"].unique()))
    pos = pd.Series(np.arange(len(cal)), index=cal)
    rpos = pos.reindex(pan["date"]).to_numpy()
    # Ultima origine valutabile: esito (t+h) chiuso prima del cutoff.
    n_ok = len(cal) - h if cutoff is None else int((cal < pd.Timestamp(cutoff)).sum()) - h
    last_eval = n_ok - 1
    preds, model = [], None
    for k in range(min_train, last_eval + 1, refit_every):
        train = pan[(rpos <= k - h - EMBARGO) & pan["fwd"].notna()]
        if len(train) < 500:
            continue
        model = _fit(train, h)
        test = pan[(rpos >= k) & (rpos <= min(k + refit_every - 1, last_eval)) & pan["fwd"].notna()]
        if len(test):
            preds.append(_predict(model, test, h))
    return (pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()), model


def _brier(p, y):
    return (p - y) ** 2


def _logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def evaluate(pr: pd.DataFrame, h: int) -> dict:
    """Metriche sulle previsioni con esito noto (colonne p, base, fwd, q10, q90, date, id)."""
    pr = pr.dropna(subset=["p", "base", "fwd"])
    if pr.empty:
        return {"n": 0}
    y = (pr["fwd"] > 0).to_numpy(float)
    p, b = pr["p"].to_numpy(), pr["base"].to_numpy()
    bm, bb, bc = _brier(p, y), _brier(b, y), _brier(np.full_like(p, 0.5), y)
    d = pd.Series(bm - bb).groupby(pr["date"].to_numpy()).mean().to_numpy()
    out = {
        "n": int(len(pr)),
        "n_dates": int(pr["date"].nunique()),
        "first_origin": pd.Timestamp(pr["date"].min()).date().isoformat(),
        "last_origin": pd.Timestamp(pr["date"].max()).date().isoformat(),
        "up_rate": float(y.mean()),
        "brier": {"modello": float(bm.mean()), "freq_storica": float(bb.mean()), "moneta": float(bc.mean())},
        "logloss": {"modello": float(_logloss(p, y).mean()), "freq_storica": float(_logloss(b, y).mean()),
                    "moneta": math.log(2)},
        "skill_vs_freq_storica": float(1 - bm.mean() / bb.mean()) if bb.mean() > 0 else None,
        "dm_vs_freq_storica": backtest.diebold_mariano(d, lags=max(h, 5)),
        "accuracy": {"modello": float(((p > 0.5) == (y > 0)).mean()), "freq_storica": float(((b > 0.5) == (y > 0)).mean())},
    }
    tilt = p - b
    simple = np.expm1(pr["fwd"].to_numpy())
    for name, mask in (("ottimista", tilt >= TILT), ("pessimista", tilt <= -TILT), ("neutro", np.abs(tilt) < TILT)):
        out.setdefault("calls", {})[name] = (
            {"n": int(mask.sum()), "share": float(mask.mean()), "up_rate": float(y[mask].mean()),
             "base_rate": float(b[mask].mean()), "mean_ret": float(simple[mask].mean())} if mask.any() else {"n": 0})
    cal = []
    for lo, hi in zip(CAL_EDGES[:-1], CAL_EDGES[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() >= 30:
            cal.append({"from": lo, "to": hi, "n": int(m.sum()), "p_mean": float(p[m].mean()), "up_rate": float(y[m].mean())})
    out["calibration"] = cal
    if {"q10", "q90"} <= set(pr.columns):
        f = pr["fwd"].to_numpy()
        out["coverage_80"] = float(((f >= pr["q10"].to_numpy()) & (f <= pr["q90"].to_numpy())).mean())
    out["verdict"] = verdict(out)
    return out


def verdict(m: dict) -> str:
    """'vantaggio' solo se meglio della frequenza storica E statisticamente netto."""
    s, t = m.get("skill_vs_freq_storica"), m.get("dm_vs_freq_storica")
    if s is None or t is None or not math.isfinite(t):
        return "non valutabile"
    if s > 0 and t < -1.96:
        return "vantaggio"
    if s < 0 and t > 1.96:
        return "peggio della baseline"
    return "nessun vantaggio dimostrato"


def per_instrument(pr: pd.DataFrame, h: int) -> dict:
    out = {}
    for tid, g in pr.groupby("id"):
        m = evaluate(g, h)
        out[tid] = {k: m.get(k) for k in ("n", "up_rate", "skill_vs_freq_storica", "dm_vs_freq_storica",
                                          "accuracy", "coverage_80", "verdict")}
    return out


# ---------------------------------------------------------------------------
# Previsioni di oggi e serie live
# ---------------------------------------------------------------------------

def latest(pan: pd.DataFrame, h: int) -> tuple[pd.DataFrame, dm.DirectionModel | None, pd.DataFrame]:
    """Modello stimato su tutte le righe con esito noto, applicato all'ultima seduta di ogni titolo."""
    cal = pd.DatetimeIndex(sorted(pan["date"].unique()))
    if len(cal) <= h + EMBARGO:
        return pd.DataFrame(), None, pd.DataFrame()
    cut = cal[len(cal) - 1 - h - EMBARGO]
    train = pan[(pan["date"] <= cut) & pan["fwd"].notna()]
    if len(train) < 500:
        return pd.DataFrame(), None, pd.DataFrame()
    model = _fit(train, h)
    last = pan.sort_values("date").groupby("id").tail(1)
    return _predict(model, last, h), model, last


def drivers(model: dm.DirectionModel, row: pd.Series, top: int = 3) -> list[dict]:
    x = row[list(dm.FEATURES)].to_numpy(float)
    contrib = model.contributions(x)
    best = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:top]
    return [{"feature": k, "label": dm.FEATURES[k], "effect": round(v, 4)} for k, v in best if abs(v) >= 0.01]


def read_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(path, dtype={"origin": str, "id": str, "model_version": str})


def append_log(path: Path, new: pd.DataFrame) -> int:
    old = read_log(path)
    if new.empty:
        return 0
    seen = set(zip(old["origin"].astype(str), old["id"].astype(str), old["h"].astype(int))) if len(old) else set()
    mask = [(str(o), str(i), int(h)) not in seen for o, i, h in zip(new["origin"], new["id"], new["h"])]
    fresh = new.loc[mask, LOG_COLUMNS]
    if fresh.empty:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh.to_csv(path, mode="a", header=not path.exists(), index=False, float_format="%.6g")
    return len(fresh)


def score_log(lg: pd.DataFrame, per_ticker: dict[str, dict]) -> dict:
    out = {"n_logged": int(len(lg)), "first_origin": lg["origin"].min() if len(lg) else None, "horizons": {}}
    for h, g in lg.groupby("h"):
        rows = []
        for r in g.itertuples():
            d = per_ticker.get(r.id)
            if d is None:
                continue
            fwd = dm.forward_return(d["feats"]["ret"], d["gap"], int(h))
            day = pd.Timestamp(r.origin)
            if day in fwd.index and pd.notna(fwd[day]):
                rows.append({"date": day, "id": r.id, "p": r.p_up, "base": r.p_base, "q10": r.q10, "q90": r.q90,
                             "fwd": float(fwd[day])})
        out["horizons"][f"h{int(h)}"] = evaluate(pd.DataFrame(rows), int(h)) if rows else {"n": 0}
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect(instruments, data_dir: Path, results_dir: Path) -> dict[str, dict]:
    prices: dict = {}

    def load(key):
        if key not in prices:
            prices[key] = read_prices(data_dir / f"{key}.parquet")
        return prices[key]

    per_ticker = {}
    for inst in instruments:
        df = load(inst.id)
        if df is None or len(df) < MIN_HISTORY:
            continue
        bench = load(inst.benchmark) if inst.benchmark else None
        hist = read_json(results_dir / "analysis" / f"{inst.id}.json").get("regime_history", {})
        key = next((k for k in hist if k.startswith("p_")), None)
        p_turb = (pd.Series(hist[key], index=pd.DatetimeIndex(hist["dates"]), dtype=float)
                  if key and hist.get("dates") else None)
        per_ticker[inst.id] = ticker_inputs(df, bench, p_turb, earnings.load(data_dir, inst.id))
    return per_ticker


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pipeline.direction", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--no-backtest", action="store_true", help="solo previsioni di oggi e serie live")
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
        log.warning("nessun dato: direzione non calcolata")
        return 1
    created = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    cfg = config()
    bt: dict = {}
    today: dict = {}
    batches = []
    for h in HORIZONS:
        pan = panel(per_ticker, h)
        if pan.empty:
            continue
        if not args.no_backtest:
            pr, last_model = walk_forward(pan, h, backtest.HOLDOUT_START)
            if len(pr):
                m = evaluate(pr, h)
                m["instruments"] = per_instrument(pr, h)
                if last_model is not None:
                    m["coefficients"] = {k: round(float(b), 4) for k, b in zip(dm.FEATURES, last_model.beta[1:])}
                bt[f"h{h}"] = m
                log.info("direzione h=%d: %d previsioni walk-forward, skill vs frequenza storica %+.4f (t %.2f) → %s",
                         h, m["n"], m["skill_vs_freq_storica"] or 0, m["dm_vs_freq_storica"], m["verdict"])
            else:
                bt[f"h{h}"] = {"n": 0, "note": "storico insufficiente"}
        pred, model, last_rows = latest(pan, h)
        for (_, p), (_, row) in zip(pred.iterrows(), last_rows.iterrows()):
            origin = pd.Timestamp(p["date"]).date().isoformat()
            today.setdefault(p["id"], {})[f"h{h}"] = {
                "origin": origin, "p_up": float(p["p"]), "p_base": float(p["base"]),
                "q10": float(p["q10"]), "q50": float(p["q50"]), "q90": float(p["q90"]),
                "drivers": drivers(model, row),
            }
            batches.append({"origin": origin, "id": p["id"], "h": h, "p_up": p["p"], "p_base": p["base"],
                            "q10": p["q10"], "q50": p["q50"], "q90": p["q90"],
                            "model_version": MODEL_VERSION, "created_utc": created})

    if bt:
        trial = backtest.record_trial(args.results_dir / "backtest" / "trials.jsonl", cfg, note="direzione")
        write_json({"series": "backtest", "generated": created, "config": cfg, "trial": trial,
                    "feature_labels": dm.FEATURES, "horizons": bt}, args.results_dir / "backtest" / "direction.json")
    write_json({"generated": created, "model_version": MODEL_VERSION, "experimental": True,
                "feature_labels": dm.FEATURES, "instruments": today},
               args.results_dir / "direction" / "latest.json")
    live_dir = args.results_dir / "live"
    added = append_log(live_dir / FORECASTS_FILE, pd.DataFrame(batches))
    scores = score_log(read_log(live_dir / FORECASTS_FILE), per_ticker)
    write_json({"series": "live", "generated": created,
                "note": "previsioni di direzione registrate da daily.yml prima di conoscere l'esito; mai ricalcolate",
                **scores}, live_dir / SCORES_FILE)
    log.info("direzione: %d previsioni live nuove", added)
    return 0


if __name__ == "__main__":
    sys.exit(main())
