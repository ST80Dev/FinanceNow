"""Anomalie con z-score robusti (mediana/MAD) su finestra trailing.

z(t) = (x(t) - mediana) / (1.4826 * MAD), con mediana e MAD calcolate sulle `window`
sedute PRECEDENTI a t: il valore anomalo non "gonfia" la propria scala e non c'è
look-ahead. Il fattore 1.4826 rende la MAD confrontabile con la deviazione standard
per dati gaussiani.

Serie controllate:
- `rendimento`          log-rendimento close-to-close;
- `gap`                 log(open / close precedente);
- `range`               log(high / low) della seduta;
- `volume`              log-volume (il valore riportato è log(volume / mediana precedente));
- `rendimento_vs_bench` rendimento in eccesso rispetto al benchmark, al netto del beta
                        a 63 sedute stimato fino al giorno prima.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 252
MIN_PERIODS = 126
THRESHOLD = 4.0
MAD_SCALE = 1.4826


def _mad(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan
    return float(np.median(np.abs(a - np.median(a))))


def robust_zscore(x: pd.Series, window: int = WINDOW, min_periods: int = MIN_PERIODS) -> pd.Series:
    past = x.shift(1).rolling(window, min_periods=min_periods)
    med = past.median()
    mad = past.apply(_mad, raw=True) * MAD_SCALE
    return (x - med) / mad.where(mad > 0)


def anomaly_inputs(df: pd.DataFrame, bench: pd.DataFrame | None = None) -> pd.DataFrame:
    ret = np.log(df["close"]).diff()
    out = pd.DataFrame({
        "rendimento": ret,
        "gap": np.log(df["open"] / df["close"].shift(1)),
        "range": np.log(df["high"] / df["low"]),
        "volume": np.log(df["volume"].where(df["volume"] > 0)),
    }, index=df.index)
    if bench is not None:
        rb = np.log(bench["close"]).diff().reindex(df.index)
        cov = ret.rolling(63, min_periods=63).cov(rb)
        var = rb.rolling(63, min_periods=63).var()
        beta = (cov / var.where(var > 0)).shift(1)  # stimato con dati fino a t-1
        out["rendimento_vs_bench"] = ret - beta * rb
    return out


def detect(df: pd.DataFrame, bench: pd.DataFrame | None = None,
           threshold: float = THRESHOLD, window: int = WINDOW) -> pd.DataFrame:
    """Eventi con |z| >= threshold: una riga per (data, serie)."""
    inputs = anomaly_inputs(df, bench)
    z = pd.DataFrame({c: robust_zscore(inputs[c], window) for c in inputs.columns}, index=df.index)
    # Il volume non si pubblica in valore assoluto (dato grezzo): si riporta il
    # log-rapporto rispetto alla mediana delle sedute precedenti.
    vol_median = inputs["volume"].shift(1).rolling(window, min_periods=MIN_PERIODS).median()
    inputs = inputs.assign(volume=inputs["volume"] - vol_median)
    long = z.stack().rename("z").reset_index()
    long.columns = ["date", "kind", "z"]
    long = long[long["z"].abs() >= threshold].copy()
    long["value"] = [inputs.at[d, k] for d, k in zip(long["date"], long["kind"])]
    return long.sort_values(["date", "kind"]).reset_index(drop=True)
