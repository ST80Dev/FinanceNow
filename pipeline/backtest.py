"""Backtest walk-forward della volatilità con baseline obbligatorie.

- Le origini di previsione sono valutate solo se TUTTO il target (t+1..t+h) cade
  prima di HOLDOUT_START: il periodo finale resta un test mai toccato durante lo
  sviluppo. Usarlo richiede `use_holdout=True` ed è registrato nei risultati.
- Tutti i modelli sono confrontati sulle stesse origini (intersezione).
- Ogni configurazione provata viene registrata in results/backtest/trials.jsonl
  (rischio overfitting da test multipli).

Metriche (per orizzonte h):
- `qlike`  media di y/f + log f (Patton 2011, robusta a proxy rumorosi; più bassa = meglio);
- `qlike_diff_vs_rw`  differenza di QLIKE rispetto alla baseline rw (negativa = meglio);
- `dm_vs_rw`  statistica t di Diebold-Mariano sulla differenza di QLIKE (errori HAC);
- `mse_ratio_vs_rw`  MSE della varianza diviso MSE della baseline rw;
- `crps`  CRPS di una previsiva normale N(0, f) sulla somma dei rendimenti;
- `coverage_90`  frazione di somme di rendimenti dentro l'intervallo centrale al 90%.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from pipeline.models import volatility as vol

MODEL_VERSION = "0.1.0"
# Inizio del periodo di test finale (circa l'ultimo anno al 26/10/2026). Data FISSA:
# non va spostata in avanti per "fare spazio" a nuovi esperimenti.
HOLDOUT_START = dt.date(2025, 11, 3)
HORIZONS = (1, 5)
MIN_TRAIN = 1000          # ~4 anni prima della prima previsione
REFIT_EVERY = 63          # ri-stima trimestrale
MODELS = ("rw", "hist_mean", "har", "garch", "egarch")
BASELINES = ("rw", "hist_mean")


def backtest_config() -> dict:
    return {
        "target": "varianza della somma dei log-rendimenti a h sedute",
        "horizons": list(HORIZONS),
        "models": list(MODELS),
        "min_train": MIN_TRAIN,
        "refit_every": REFIT_EVERY,
        "rw_window": vol.RW_WINDOW,
        "har_lags": list(vol.HAR_LAGS),
        "arch": "p=1,q=1, mean=Zero, dist=t",
        "holdout_start": HOLDOUT_START.isoformat(),
        "model_version": MODEL_VERSION,
    }


# ---------------------------------------------------------------------------
# Metriche
# ---------------------------------------------------------------------------

def qlike_loss(f: np.ndarray, y: np.ndarray) -> np.ndarray:
    return y / f + np.log(f)


def crps_normal(sigma: np.ndarray, x: np.ndarray) -> np.ndarray:
    """CRPS in forma chiusa per una previsiva N(0, sigma^2)."""
    z = x / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def diebold_mariano(d: np.ndarray, lags: int) -> float:
    """Statistica t della media di d con varianza di lungo periodo Newey-West."""
    n = len(d)
    if n < 10:
        return float("nan")
    u = d - d.mean()
    lrv = u @ u / n
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        lrv += 2 * w * (u[lag:] @ u[:-lag]) / n
    if lrv <= 0:
        return float("nan")
    return float(d.mean() / np.sqrt(lrv / n))


def evaluate(forecasts: pd.DataFrame, target: pd.DataFrame, h: int) -> dict:
    """Metriche di ogni modello sulle origini comuni a tutti i modelli."""
    data = pd.concat([forecasts, target], axis=1).dropna()
    data = data[(data[forecasts.columns] > 0).all(axis=1)]
    if data.empty:
        return {"n": 0}
    y, x = data["y"].to_numpy(), data["x"].to_numpy()
    rw_loss = qlike_loss(data["rw"].to_numpy(), y) if "rw" in data else None
    rw_mse = np.mean((data["rw"].to_numpy() - y) ** 2) if "rw" in data else None
    z90 = norm.ppf(0.95)
    out = {
        "n": int(len(data)),
        "first_origin": data.index[0].date().isoformat(),
        "last_origin": data.index[-1].date().isoformat(),
        "models": {},
    }
    for m in forecasts.columns:
        f = data[m].to_numpy()
        loss = qlike_loss(f, y)
        sigma = np.sqrt(f)
        res = {
            "qlike": float(loss.mean()),
            "mse_ratio_vs_rw": float(np.mean((f - y) ** 2) / rw_mse) if rw_mse else None,
            "crps": float(crps_normal(sigma, x).mean()),
            "coverage_90": float(np.mean(np.abs(x) <= z90 * sigma)),
        }
        if rw_loss is not None and m != "rw":
            d = loss - rw_loss
            res["qlike_diff_vs_rw"] = float(d.mean())
            res["dm_vs_rw"] = diebold_mariano(d, lags=max(h, 5))
        out["models"][m] = res
    return out


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

def last_origin_before(index: pd.DatetimeIndex, h: int, cutoff: dt.date | None) -> int:
    """Ultima posizione di origine il cui target t+1..t+h è tutto prima di `cutoff`."""
    if cutoff is None:
        return len(index) - 1 - h
    before = int((index < pd.Timestamp(cutoff)).sum())  # sedute prima del cutoff
    return before - 1 - h


def volatility_backtest(ret: pd.Series, use_holdout: bool = False,
                        horizons=None, models=None,
                        min_train: int | None = None, refit_every: int | None = None) -> dict:
    # Default letti al momento della chiamata (non della definizione).
    horizons = HORIZONS if horizons is None else horizons
    models = MODELS if models is None else models
    min_train = MIN_TRAIN if min_train is None else min_train
    refit_every = REFIT_EVERY if refit_every is None else refit_every
    ret = ret.dropna()
    cutoff = None if use_holdout else HOLDOUT_START
    results = {}
    for h in horizons:
        last = last_origin_before(ret.index, h, cutoff)
        if last < min_train + 50:
            results[f"h{h}"] = {"n": 0, "note": "storico insufficiente"}
            continue
        fc = vol.all_forecasts(ret, h, min_train, refit_every, last_origin=last, models=models)
        fc = fc.iloc[: last + 1]
        target = vol.realized_target(ret, h).iloc[: last + 1]
        results[f"h{h}"] = evaluate(fc, target, h)
    return results


def aggregate(per_instrument: dict[str, dict]) -> dict:
    """Riassunto tra strumenti: media della differenza di QLIKE vs rw e vittorie."""
    agg: dict = {}
    for res in per_instrument.values():
        for hkey, r in res.items():
            for m, met in r.get("models", {}).items():
                if "qlike_diff_vs_rw" not in met:
                    continue
                a = agg.setdefault(hkey, {}).setdefault(m, {"diffs": [], "wins": 0, "significant_wins": 0})
                a["diffs"].append(met["qlike_diff_vs_rw"])
                a["wins"] += met["qlike_diff_vs_rw"] < 0
                a["significant_wins"] += (met.get("dm_vs_rw") or 0) < -1.96
    for hkey, ms in agg.items():
        for m, a in ms.items():
            diffs = a.pop("diffs")
            a.update(n_instruments=len(diffs), mean_qlike_diff_vs_rw=float(np.mean(diffs)))
    return agg


def record_trial(path: Path, config: dict, note: str = "") -> dict:
    """Registra una configurazione (una sola volta per hash) e restituisce il conteggio."""
    blob = json.dumps(config, sort_keys=True)
    digest = hashlib.sha256(blob.encode()).hexdigest()[:12]
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = []
    if path.exists():
        seen = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not any(t["hash"] == digest for t in seen):
        entry = {"hash": digest, "first_run": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                 "note": note, "config": config}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        seen.append(entry)
    return {"hash": digest, "n_configurations_tried": len(seen)}
