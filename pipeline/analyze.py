"""Analisi dell'universo: feature, regimi, anomalie e backtest della volatilità.

Legge i dati grezzi da data/raw/ (prodotti da pipeline.ingest) e scrive SOLO risultati
elaborati, mai prezzi grezzi:
- results/analysis/<id>.json   stato descrittivo: feature recenti, regime, anomalie;
- results/backtest/volatility.json   metriche walk-forward per modello e orizzonte;
- results/backtest/trials.jsonl      registro delle configurazioni provate.

Uso:
    python -m pipeline.analyze [--tickers nvda,aapl] [--backtest auto|always|never] [--use-holdout]

Con `--backtest auto` (default) il backtest si rifà solo per gli strumenti che non lo hanno
o quando la configurazione è cambiata: il run serale non ripete ogni giorno lo stesso calcolo.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import math
import sys
import time
from pathlib import Path

import pandas as pd

from pipeline import backtest
from pipeline.features import compute_features
from pipeline.ingest import (DEFAULT_DATA_DIR, DEFAULT_UNIVERSE, ROOT, ConfigError, load_universe,
                             read_json, read_prices, write_json)
from pipeline.models import anomalies, regimes

log = logging.getLogger("financenow.analyze")

DEFAULT_RESULTS_DIR = ROOT / "results"
N_STATES = 3
RECENT_ANOMALIES = 50
MIN_SESSIONS = 300


def _num(x, digits: int = 5):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return round(float(x), digits)


def analyze_instrument(inst, df: pd.DataFrame, bench: pd.DataFrame | None) -> dict:
    feats = compute_features(df, bench)
    latest = feats.iloc[-1]
    ret = feats["ret"]

    probs, params = regimes.regimes_walk_forward(ret, n_states=N_STATES)
    regime = regimes.regime_summary(ret, probs, params)
    valid = probs.dropna()
    turbulent = f"p_{regimes.LABELS[N_STATES][-1]}"

    events = anomalies.detect(df, bench)
    last_year = df.index[-1] - pd.Timedelta(days=365)
    return {
        "id": inst.id,
        "name": inst.name,
        "asset_class": inst.asset_class,
        "benchmark": inst.benchmark,
        "model_version": backtest.MODEL_VERSION,
        "as_of": df.index[-1].date().isoformat(),
        "data_period": {"first": df.index[0].date().isoformat(),
                        "last": df.index[-1].date().isoformat(), "sessions": len(df)},
        "features_latest": {k: _num(v) for k, v in latest.items() if _num(v) is not None},
        "regime": regime,
        "regime_history": {
            "note": "probabilità filtrate (solo dati fino a ogni data), parametri ri-stimati ogni anno",
            "dates": [d.date().isoformat() for d in valid.index],
            "state": valid["state"].astype(int).tolist(),
            turbulent: [round(float(p), 3) for p in valid[turbulent]],
        },
        "anomalies": {
            "method": f"z-score robusto mediana/MAD su {anomalies.WINDOW} sedute precedenti",
            "threshold": anomalies.THRESHOLD,
            "count_by_kind": events["kind"].value_counts().to_dict(),
            "count_last_365_days": int((events["date"] > last_year).sum()),
            "recent": [
                {"date": r.date.date().isoformat(), "kind": r.kind, "z": _num(r.z, 2), "value": _num(r.value, 5)}
                for r in events.tail(RECENT_ANOMALIES).itertuples()
            ],
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pipeline.analyze", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", help="id separati da virgola (default: tutto l'universo)")
    ap.add_argument("--backtest", choices=["auto", "always", "never"], default="auto",
                    help="auto: solo se manca o se la configurazione è cambiata (default)")
    ap.add_argument("--use-holdout", action="store_true",
                    help="valuta ANCHE il periodo di test finale (da usare una volta sola, a sviluppo concluso)")
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
    by_id = {i.id: i for i in instruments}
    if args.tickers:
        wanted = [t.strip() for t in args.tickers.split(",") if t.strip()]
        unknown = [t for t in wanted if t not in by_id]
        if unknown:
            log.error("ticker non presenti nell'universo: %s", unknown)
            return 2
        instruments = [by_id[t] for t in wanted]
    if args.use_holdout:
        log.warning("ATTENZIONE: il backtest include il periodo di test finale (dal %s)",
                    backtest.HOLDOUT_START)

    prices = {}
    for inst in instruments:
        for key in {inst.id, inst.benchmark} - {None}:
            if key not in prices:
                prices[key] = read_prices(args.data_dir / f"{key}.parquet")

    bt_path = args.results_dir / "backtest" / "volatility.json"
    config = backtest.backtest_config()
    previous = read_json(bt_path)
    same_config = (previous.get("config") == config and previous.get("use_holdout") == args.use_holdout)
    # Risultati già validi per la configurazione corrente: si tengono e si integrano.
    kept = dict(previous.get("instruments", {})) if same_config else {}
    bt_results, done, skipped = {}, 0, []
    for inst in instruments:
        df = prices.get(inst.id)
        if df is None or len(df) < MIN_SESSIONS:
            log.warning("%s: dati assenti o insufficienti, salto", inst.id)
            skipped.append(inst.id)
            continue
        bench = prices.get(inst.benchmark) if inst.benchmark else None
        t0 = time.monotonic()
        result = analyze_instrument(inst, df, bench)
        write_json(result, args.results_dir / "analysis" / f"{inst.id}.json")
        reg = result["regime"]
        log.info("%-6s regime=%s (%s gg) anomalie ultimo anno=%d [%.1fs]", inst.id,
                 reg.get("current"), reg.get("days_in_state"),
                 result["anomalies"]["count_last_365_days"], time.monotonic() - t0)
        if args.backtest == "always" or (args.backtest == "auto" and inst.id not in kept):
            t0 = time.monotonic()
            ret = compute_features(df)["ret"]
            bt_results[inst.id] = backtest.volatility_backtest(ret, use_holdout=args.use_holdout)
            log.info("%-6s backtest volatilità [%.1fs]", inst.id, time.monotonic() - t0)
        done += 1

    if bt_results:
        merged = {**kept, **bt_results}
        trial = backtest.record_trial(args.results_dir / "backtest" / "trials.jsonl", config)
        write_json({
            "series": "backtest",
            "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "config": config,
            "use_holdout": args.use_holdout,
            "trial": trial,
            "aggregate": backtest.aggregate(merged),
            "instruments": merged,
        }, bt_path)
        log.info("backtest aggiornato per %d strumenti; configurazioni provate finora: %d",
                 len(bt_results), trial["n_configurations_tried"])
    elif args.backtest != "never":
        log.info("backtest già aggiornato per la configurazione corrente: nessun ricalcolo")

    if skipped:
        log.warning("senza dati sufficienti: %s", ", ".join(skipped))
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
