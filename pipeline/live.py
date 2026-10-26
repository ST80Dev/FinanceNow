"""Serie `live`: previsioni di volatilità registrate ogni sera e valutate quando arriva il dato.

A differenza del backtest, qui ogni previsione viene SCRITTA prima che il futuro sia noto
(da daily.yml) e non viene mai ricalcolata: è l'unica prova pulita.

- results/live/volatility_forecasts.csv  log append-only: una riga per
  (origine, strumento, orizzonte, modello), con data di creazione e versione del modello;
- results/live/volatility_scores.json    metriche sulle previsioni il cui target è già
  realizzato (stesse metriche del backtest, tenute rigorosamente separate).

Uso:
    python -m pipeline.live [--tickers nvda,aapl]
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import backtest
from pipeline.analyze import DEFAULT_RESULTS_DIR
from pipeline.features import log_returns
from pipeline.ingest import DEFAULT_DATA_DIR, DEFAULT_UNIVERSE, ConfigError, load_universe, read_prices, write_json
from pipeline.models import volatility as vol

log = logging.getLogger("financenow.live")

FORECASTS_FILE = "volatility_forecasts.csv"
SCORES_FILE = "volatility_scores.json"
KEY = ["origin", "id", "h", "model"]
COLUMNS = KEY + ["forecast_var", "model_version", "created_utc"]


def latest_forecasts(ret: pd.Series, horizons=None, models=None) -> pd.DataFrame:
    """Previsioni con origine nell'ultima seduta disponibile, parametri stimati su tutto lo storico.

    Usa le stesse funzioni del backtest con un solo blocco che termina all'ultima seduta.
    """
    horizons = backtest.HORIZONS if horizons is None else horizons
    models = backtest.MODELS if models is None else models
    ret = ret.dropna()
    n = len(ret)
    if n < backtest.MIN_TRAIN:
        return pd.DataFrame(columns=["origin", "h", "model", "forecast_var"])
    rows = []
    for h in horizons:
        fc = vol.all_forecasts(ret, h, min_train=n - 1, refit_every=1, last_origin=n - 1, models=models)
        last = fc.iloc[-1]
        for model, value in last.items():
            if np.isfinite(value) and value > 0:
                rows.append({"origin": ret.index[-1].date().isoformat(), "h": h,
                             "model": model, "forecast_var": float(value)})
    return pd.DataFrame(rows)


def read_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype={"origin": str, "id": str, "model": str, "model_version": str})


def append_log(path: Path, new: pd.DataFrame) -> int:
    """Aggiunge solo le righe con chiave nuova: le previsioni già registrate non si toccano."""
    existing = read_log(path)
    if new.empty:
        return 0
    seen = set(map(tuple, existing[KEY].astype(str).to_numpy())) if len(existing) else set()
    mask = [tuple(map(str, k)) not in seen for k in new[KEY].to_numpy()]
    fresh = new.loc[mask, COLUMNS]
    if fresh.empty:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh.to_csv(path, mode="a", header=not path.exists(), index=False, float_format="%.10g")
    return len(fresh)


def score(log_df: pd.DataFrame, returns: dict[str, pd.Series]) -> dict:
    """Metriche per strumento e orizzonte sulle previsioni con target già realizzato."""
    per_instrument: dict[str, dict] = {}
    for (tid, h), grp in log_df.groupby(["id", "h"]):
        ret = returns.get(tid)
        if ret is None:
            continue
        ret = ret.dropna()
        wide = grp.pivot_table(index="origin", columns="model", values="forecast_var", aggfunc="first")
        wide.index = pd.DatetimeIndex(pd.to_datetime(wide.index))
        target = vol.realized_target(ret, int(h)).reindex(wide.index)
        res = backtest.evaluate(wide, target, int(h))
        per_instrument.setdefault(tid, {})[f"h{int(h)}"] = res
    return per_instrument


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pipeline.live", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", help="id separati da virgola (default: tutto l'universo)")
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
    if args.tickers:
        wanted = {t.strip() for t in args.tickers.split(",") if t.strip()}
        instruments = [i for i in instruments if i.id in wanted]

    live_dir = args.results_dir / "live"
    log_path = live_dir / FORECASTS_FILE
    created = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    returns: dict[str, pd.Series] = {}
    batches = []
    for inst in instruments:
        df = read_prices(args.data_dir / f"{inst.id}.parquet")
        if df is None or df.empty:
            log.warning("%s: nessun dato, nessuna previsione", inst.id)
            continue
        ret = log_returns(df["close"])
        returns[inst.id] = ret
        fc = latest_forecasts(ret)
        if fc.empty:
            log.warning("%s: storico insufficiente per le previsioni", inst.id)
            continue
        fc.insert(1, "id", inst.id)
        fc["model_version"] = backtest.MODEL_VERSION
        fc["created_utc"] = created
        batches.append(fc)

    added = append_log(log_path, pd.concat(batches, ignore_index=True)) if batches else 0
    log.info("previsioni live registrate: %d nuove", added)

    log_df = read_log(log_path)
    per_instrument = score(log_df, returns) if len(log_df) else {}
    write_json({
        "series": "live",
        "generated": created,
        "note": "previsioni registrate da daily.yml prima di conoscere il risultato; mai ricalcolate",
        "n_forecasts_logged": int(len(log_df)),
        "first_origin": log_df["origin"].min() if len(log_df) else None,
        "aggregate": backtest.aggregate(per_instrument),
        "instruments": per_instrument,
    }, live_dir / SCORES_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
