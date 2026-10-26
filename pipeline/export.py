"""Export dei JSON leggeri per la dashboard (docs/data/, servita da GitHub Pages).

Scrive SOLO dati derivati, mai prezzi o livelli (licenze, repo pubblico):
- docs/data/index.json          panoramica: una riga per strumento + stato dei dati;
- docs/data/tickers/<id>.json   serie storiche derivate (volatilità, drawdown, beta, regime),
                                statistiche dei regimi, anomalie recenti, ultime previsioni live;
- docs/data/models.json         volatilità: `backtest` e `live` in due blocchi separati;
- docs/data/digest.json         sintesi del giorno (pipeline/digest.py), anche in results/digest/latest.md;
- docs/data/patterns.json       pattern: statistiche storiche (`backtest`) e verifica `live`, separate.

Uso:
    python -m pipeline.export
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
import sys
from pathlib import Path

import pandas as pd

from pipeline import backtest, digest, live, patterns
from pipeline.analyze import DEFAULT_RESULTS_DIR
from pipeline.features import TRADING_DAYS, compute_features
from pipeline.ingest import (DEFAULT_DATA_DIR, DEFAULT_UNIVERSE, ROOT, ConfigError, json_safe,
                             load_universe, read_json, read_prices)

log = logging.getLogger("financenow.export")

DEFAULT_OUT_DIR = ROOT / "docs" / "data"
SCHEMA_VERSION = 1
SERIES = ("rv_21", "rv_63", "drawdown", "beta_252", "corr_252")
SUMMARY_FEATURES = ("rv_21", "rv_63", "drawdown", "mom_1m", "mom_3m", "mom_12m",
                    "dist_ma200", "beta_252", "corr_252", "vol_z")


def write_compact(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(obj), separators=(",", ":"), ensure_ascii=False, allow_nan=False),
                   encoding="utf-8")
    os.replace(tmp, path)


def _round(values, digits: int = 4) -> list:
    return [None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), digits)
            for v in values]


def annualized_vol(var: float, h: int) -> float | None:
    """Varianza della somma su h sedute -> volatilità annualizzata."""
    if var is None or not math.isfinite(var) or var <= 0:
        return None
    return math.sqrt(var / h * TRADING_DAYS)


def latest_live(log_df: pd.DataFrame, tid: str) -> dict | None:
    rows = log_df[log_df["id"] == tid]
    if rows.empty:
        return None
    origin = rows["origin"].max()
    rows = rows[rows["origin"] == origin]
    out: dict = {"origin": origin, "model_version": rows["model_version"].iloc[0], "vol_ann": {}}
    for r in rows.itertuples():
        out["vol_ann"].setdefault(f"h{int(r.h)}", {})[r.model] = annualized_vol(float(r.forecast_var), int(r.h))
    return out


def ticker_payload(inst, df: pd.DataFrame, bench: pd.DataFrame | None, analysis: dict,
                   live_latest: dict | None) -> dict:
    feats = compute_features(df, bench)
    dates = [d.date().isoformat() for d in feats.index]
    series = {name: _round(feats[name].tolist()) for name in SERIES if name in feats}
    hist = analysis.get("regime_history", {})
    turb_key = next((k for k in hist if k.startswith("p_")), None)
    state_by_date = dict(zip(hist.get("dates", []), hist.get("state", [])))
    p_by_date = dict(zip(hist.get("dates", []), hist.get(turb_key, []))) if turb_key else {}
    series["regime_state"] = [state_by_date.get(d) for d in dates]
    series["p_turbolento"] = [p_by_date.get(d) for d in dates]
    return {
        "schema": SCHEMA_VERSION,
        "id": inst.id,
        "name": inst.name,
        "asset_class": inst.asset_class,
        "benchmark": inst.benchmark,
        "as_of": analysis.get("as_of"),
        "data_period": analysis.get("data_period"),
        "features_latest": analysis.get("features_latest", {}),
        "regime": analysis.get("regime", {}),
        "anomalies": analysis.get("anomalies", {}),
        "live_latest": live_latest,
        "dates": dates,
        "series": series,
    }


def digest_context(inst, payload: dict, status: dict | None) -> digest.Context:
    rv21 = [v for v in payload["series"].get("rv_21", []) if v is not None]
    current = payload["features_latest"].get("rv_21")
    pctl = (sum(v <= current for v in rv21) / len(rv21)) if rv21 and current is not None else None
    return digest.Context(
        id=inst.id, name=inst.name, asset_class=inst.asset_class, as_of=payload.get("as_of"),
        recent_dates=payload["dates"][-digest.RECENT_SESSIONS:],
        regime=payload.get("regime", {}),
        regime_states=payload["series"].get("regime_state", []),
        regime_dates=payload["dates"],
        features=payload.get("features_latest", {}),
        rv21_percentile=pctl,
        anomalies=payload.get("anomalies", {}).get("recent", []),
        data_status=(status or {}).get("status"),
    )


def summary_row(inst, analysis: dict, status: dict | None, live_latest: dict | None) -> dict:
    feats = analysis.get("features_latest", {})
    reg = analysis.get("regime", {})
    recent = analysis.get("anomalies", {}).get("recent", [])
    current = reg.get("current")
    forecast = (live_latest or {}).get("vol_ann", {}).get("h1", {})
    return {
        "id": inst.id,
        "name": inst.name,
        "asset_class": inst.asset_class,
        "as_of": analysis.get("as_of"),
        "regime": current,
        "regime_prob": reg.get("probabilities", {}).get(current) if current else None,
        "days_in_state": reg.get("days_in_state"),
        **{k: feats.get(k) for k in SUMMARY_FEATURES},
        "anomalies_12m": analysis.get("anomalies", {}).get("count_last_365_days"),
        "last_anomaly": recent[-1] if recent else None,
        "vol_forecast_h1": {m: forecast.get(m) for m in ("garch", "egarch", "har", "rw") if m in forecast},
        "data_status": (status or {}).get("status"),
    }


def models_payload(results_dir: Path) -> dict:
    bt = read_json(results_dir / "backtest" / "volatility.json")
    lv = read_json(results_dir / "live" / live.SCORES_FILE)

    def slim(per_instrument: dict) -> dict:
        keep = ("qlike_diff_vs_rw", "dm_vs_rw", "mse_ratio_vs_rw", "coverage_90")
        return {tid: {h: {"n": r.get("n"), "first_origin": r.get("first_origin"), "last_origin": r.get("last_origin"),
                          "models": {m: {k: v for k, v in met.items() if k in keep}
                                     for m, met in r.get("models", {}).items()}}
                      for h, r in res.items()}
                for tid, res in per_instrument.items()}

    return {
        "schema": SCHEMA_VERSION,
        "backtest": {
            "available": bool(bt),
            "generated": bt.get("generated"),
            "holdout_start": bt.get("config", {}).get("holdout_start"),
            "use_holdout": bt.get("use_holdout"),
            "config": bt.get("config"),
            "n_configurations_tried": bt.get("trial", {}).get("n_configurations_tried"),
            "aggregate": bt.get("aggregate", {}),
            "instruments": slim(bt.get("instruments", {})),
        },
        "live": {
            "available": bool(lv),
            "generated": lv.get("generated"),
            "n_forecasts_logged": lv.get("n_forecasts_logged", 0),
            "first_origin": lv.get("first_origin"),
            "aggregate": lv.get("aggregate", {}),
            "instruments": slim(lv.get("instruments", {})),
        },
    }


def export(instruments, data_dir: Path, results_dir: Path, out_dir: Path) -> dict:
    status = read_json(results_dir / "status" / "ingest.json")
    log_df = live.read_log(results_dir / "live" / live.FORECASTS_FILE)
    prices: dict[str, pd.DataFrame | None] = {}

    def load(key):
        if key not in prices:
            prices[key] = read_prices(data_dir / f"{key}.parquet")
        return prices[key]

    rows, exported, contexts = [], [], []
    for inst in instruments:
        analysis = read_json(results_dir / "analysis" / f"{inst.id}.json")
        df = load(inst.id)
        if not analysis or df is None or df.empty:
            log.warning("%s: analisi o dati mancanti, escluso dalla dashboard", inst.id)
            continue
        bench = load(inst.benchmark) if inst.benchmark else None
        ll = latest_live(log_df, inst.id) if len(log_df) else None
        payload = ticker_payload(inst, df, bench, analysis, ll)
        write_compact(payload, out_dir / "tickers" / f"{inst.id}.json")
        inst_status = status.get("instruments", {}).get(inst.id)
        rows.append(summary_row(inst, analysis, inst_status, ll))
        contexts.append(digest_context(inst, payload, inst_status))
        exported.append(inst.id)

    index = {
        "schema": SCHEMA_VERSION,
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "as_of": max((r["as_of"] for r in rows if r["as_of"]), default=None),
        "model_version": backtest.MODEL_VERSION,
        "data_status": {k: status.get(k, []) for k in ("missing", "stale", "deferred")} if status else None,
        "last_download": status.get("run_finished") if status else None,
        "instruments": rows,
    }
    write_compact(index, out_dir / "index.json")
    write_compact(models_payload(results_dir), out_dir / "models.json")
    dg = digest.build(contexts, status)
    current = read_json(results_dir / "patterns" / "current.json").get("events", [])
    digest.attach_history(dg, current, patterns.EVENT_TYPES)
    pstats = read_json(results_dir / "patterns" / "stats.json")
    pscores = read_json(results_dir / "live" / patterns.SCORES_FILE)
    write_compact({
        "schema": SCHEMA_VERSION,
        "backtest": {k: pstats.get(k) for k in ("generated", "holdout_start", "horizons", "main_horizon",
                                                  "event_types", "stats", "events_by_instrument")} if pstats else None,
        "live": pscores or None,
        "current": current,
    }, out_dir / "patterns.json")
    dg["generated"] = index["generated"]
    write_compact(dg, out_dir / "digest.json")
    md_path = results_dir / "digest" / "latest.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(digest.to_markdown(dg) + "\n", encoding="utf-8")
    # Rimuove i file di strumenti non più esportati (es. tolti dall'universo).
    for p in (out_dir / "tickers").glob("*.json"):
        if p.stem not in exported:
            p.unlink()
    return index


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pipeline.export", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    try:
        instruments = load_universe(args.universe)
    except ConfigError as e:
        log.error("configurazione non valida: %s", e)
        return 2
    index = export(instruments, args.data_dir, args.results_dir, args.out_dir)
    log.info("dashboard: %d strumenti esportati in %s", len(index["instruments"]), args.out_dir)
    return 0 if index["instruments"] else 1


if __name__ == "__main__":
    sys.exit(main())
