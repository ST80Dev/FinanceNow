"""Previsione della volatilità con walk-forward: GARCH, EGARCH, HAR e baseline.

Cosa si prevede: la varianza della somma dei log-rendimenti nelle prossime h sedute,
in unità giornaliere (non annualizzata). Previsione fatta a fine seduta t, con dati
fino a t compreso. Proxy realizzato per la valutazione: somma dei rendimenti al
quadrato da t+1 a t+h (rumoroso ma non distorto; con QLIKE/MSE il ranking dei
modelli resta corretto, Patton 2011).

Modelli:
- `rw`        baseline "random walk" della volatilità: varianza media delle ultime 21 sedute;
- `hist_mean` baseline media storica: media di tutti i rendimenti al quadrato fino a t;
- `har`       HAR-RV (Corsi 2009) su rendimenti al quadrato giornalieri/settimanali/mensili,
              regressione diretta sull'orizzonte h;
- `garch`     GARCH(1,1) con innovazioni t di Student (`arch`);
- `egarch`    EGARCH(1,1) t di Student, solo h=1 (oltre serve simulazione).

I parametri si stimano ogni `refit_every` sedute su finestra espandente; tra una stima
e l'altra i parametri restano fissi e il filtro usa solo dati passati.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

RW_WINDOW = 21
HIST_MIN_PERIODS = 63
HAR_LAGS = (1, 5, 22)
SCALE = 100.0  # arch lavora meglio con rendimenti in punti percentuali


def realized_target(ret: pd.Series, h: int) -> pd.DataFrame:
    """Per ogni origine t: somma dei rendimenti (x) e dei quadrati (y) da t+1 a t+h."""
    r = ret.fillna(0.0)
    x = r.rolling(h).sum().shift(-h)
    y = (r ** 2).rolling(h).sum().shift(-h)
    return pd.DataFrame({"x": x, "y": y})


def forecast_rw(ret: pd.Series, h: int) -> pd.Series:
    return h * (ret ** 2).rolling(RW_WINDOW, min_periods=RW_WINDOW).mean()


def forecast_hist_mean(ret: pd.Series, h: int) -> pd.Series:
    return h * (ret ** 2).expanding(min_periods=HIST_MIN_PERIODS).mean()


def _blocks(n: int, min_train: int, refit_every: int, last_origin: int):
    """Blocchi [i0, i1] di origini; i parametri si stimano con dati fino a i0 compreso."""
    i0 = min_train
    while i0 <= last_origin:
        i1 = min(i0 + refit_every - 1, last_origin)
        yield i0, i1
        i0 = i1 + 1


def har_design(ret: pd.Series) -> pd.DataFrame:
    sq = ret ** 2
    return pd.DataFrame(
        {f"rv_{lag}": sq.rolling(lag, min_periods=lag).mean() for lag in HAR_LAGS}, index=ret.index
    )


def forecast_har(ret: pd.Series, h: int, min_train: int, refit_every: int,
                 last_origin: int | None = None) -> pd.Series:
    """HAR diretto: y(t, h) = b0 + b1 rv_1(t) + b2 rv_5(t) + b3 rv_22(t)."""
    ret = ret.dropna()
    n = len(ret)
    last_origin = n - 1 if last_origin is None else last_origin
    X = har_design(ret).to_numpy()
    X = np.column_stack([np.ones(n), X])
    y = realized_target(ret, h)["y"].to_numpy()
    out = np.full(n, np.nan)
    for i0, i1 in _blocks(n, min_train, refit_every, last_origin):
        # Solo righe il cui target è già osservato alla data i0 (t + h <= i0).
        rows = np.arange(0, i0 - h + 1)
        ok = rows[np.isfinite(X[rows]).all(axis=1) & np.isfinite(y[rows])]
        if len(ok) < 50:
            continue
        beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        floor = 0.01 * np.mean(y[ok])
        out[i0:i1 + 1] = np.maximum(X[i0:i1 + 1] @ beta, floor)
    return pd.Series(out, index=ret.index, name="har")


def forecast_arch(ret: pd.Series, h: int, kind: str, min_train: int, refit_every: int,
                  last_origin: int | None = None) -> pd.Series:
    """GARCH/EGARCH(1,1) t di Student, walk-forward a finestra espandente."""
    from arch import arch_model

    if kind == "egarch" and h > 1:
        raise ValueError("EGARCH: solo h=1 (le previsioni multi-step richiedono simulazione)")
    ret = ret.dropna()
    n = len(ret)
    last_origin = n - 1 if last_origin is None else last_origin
    y = ret.to_numpy() * SCALE
    out = np.full(n, np.nan)
    start_values = None
    for i0, i1 in _blocks(n, min_train, refit_every, last_origin):
        # Il modello vede solo i dati fino alla fine del blocco: le previsioni con
        # origine t usano il filtro fino a t, i parametri sono stimati fino a i0.
        am = arch_model(y[: i1 + 1], mean="Zero", vol="EGARCH" if kind == "egarch" else "GARCH",
                        p=1, o=0, q=1, dist="t", rescale=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = am.fit(last_obs=i0 + 1, disp="off", starting_values=start_values,
                             options={"maxiter": 500})
            except Exception:
                res = am.fit(last_obs=i0 + 1, disp="off", options={"maxiter": 500})
        if not np.all(np.isfinite(res.params)):
            continue
        start_values = res.params.to_numpy()
        fc = res.forecast(start=i0, horizon=h, reindex=False)
        var = fc.variance.to_numpy().sum(axis=1)[: i1 - i0 + 1]
        out[i0:i1 + 1] = var / SCALE ** 2
    return pd.Series(out, index=ret.index, name=kind)


def all_forecasts(ret: pd.Series, h: int, min_train: int, refit_every: int,
                  last_origin: int | None = None,
                  models: tuple[str, ...] = ("rw", "hist_mean", "har", "garch", "egarch")) -> pd.DataFrame:
    """Previsioni di tutti i modelli richiesti, indicizzate per data di origine."""
    ret = ret.dropna()
    cols = {}
    for m in models:
        if m == "rw":
            cols[m] = forecast_rw(ret, h)
        elif m == "hist_mean":
            cols[m] = forecast_hist_mean(ret, h)
        elif m == "har":
            cols[m] = forecast_har(ret, h, min_train, refit_every, last_origin)
        elif m in ("garch", "egarch"):
            if m == "egarch" and h > 1:
                continue
            cols[m] = forecast_arch(ret, h, m, min_train, refit_every, last_origin)
        else:
            raise ValueError(f"modello sconosciuto: {m}")
    return pd.DataFrame(cols, index=ret.index)
