"""Feature descrittive calcolate da OHLCV giornaliero.

Regole:
- si lavora su log-rendimenti, mai sui livelli di prezzo (i livelli servono solo
  come input di rapporti: rendimenti, range, distanze da medie mobili);
- ogni feature al giorno t usa solo dati fino a t compreso (finestre trailing):
  nessun look-ahead;
- volatilità annualizzate con 252 sedute/anno.

Le colonne prodotte sono documentate in FEATURE_DOCS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
VOL_WINDOWS = (5, 21, 63)
MOMENTUM_WINDOWS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
MA_WINDOWS = (50, 200)
ROLLING_STATS_WINDOW = 63
VOLUME_Z_WINDOW = 63
ATR_WINDOW = 14
BETA_WINDOWS = (63, 252)

FEATURE_DOCS = {
    "ret": "log-rendimento close-to-close",
    "rv_{w}": "volatilità realizzata annualizzata su w sedute (dev. std. dei log-rendimenti)",
    "park_{w}": "volatilità di Parkinson annualizzata (range high-low)",
    "gk_{w}": "volatilità di Garman-Klass annualizzata (OHLC)",
    "atr_pct": "ATR di Wilder a 14 sedute diviso per il close",
    "mom_{p}": "momentum: somma dei log-rendimenti su 1/3/6/12 mesi",
    "dist_ma{w}": "log(close / media mobile semplice a w sedute)",
    "vol_z": "z-score del log-volume rispetto alle 63 sedute precedenti",
    "skew_63 / kurt_63": "asimmetria e curtosi in eccesso rolling dei log-rendimenti",
    "drawdown": "close / massimo storico - 1",
    "beta_{w} / corr_{w}": "beta e correlazione rolling vs benchmark",
    "rs_{p}": "forza relativa: momentum del titolo meno momentum del benchmark",
}


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def realized_vol(ret: pd.Series, window: int) -> pd.Series:
    return ret.rolling(window, min_periods=window).std() * np.sqrt(TRADING_DAYS)


def parkinson_vol(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    hl2 = np.log(high / low) ** 2
    var = hl2.rolling(window, min_periods=window).mean() / (4 * np.log(2))
    return np.sqrt(var * TRADING_DAYS)


def garman_klass_var(df: pd.DataFrame) -> pd.Series:
    """Varianza giornaliera di Garman-Klass (non annualizzata)."""
    hl = np.log(df["high"] / df["low"])
    co = np.log(df["close"] / df["open"])
    return (0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2).clip(lower=0)


def garman_klass_vol(df: pd.DataFrame, window: int) -> pd.Series:
    var = garman_klass_var(df).rolling(window, min_periods=window).mean()
    return np.sqrt(var * TRADING_DAYS)


def atr_pct(df: pd.DataFrame, window: int = ATR_WINDOW) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return atr / df["close"]


def volume_zscore(volume: pd.Series, window: int = VOLUME_Z_WINDOW) -> pd.Series:
    """z-score del log-volume rispetto alle `window` sedute PRECEDENTI (oggi escluso)."""
    lv = np.log(volume.where(volume > 0))
    past = lv.shift(1).rolling(window, min_periods=window)
    sd = past.std()
    return (lv - past.mean()) / sd.where(sd > 0)


def drawdown(close: pd.Series) -> pd.Series:
    return close / close.cummax() - 1


def compute_features(df: pd.DataFrame, bench: pd.DataFrame | None = None) -> pd.DataFrame:
    """Feature per un titolo. `bench` è l'OHLCV del benchmark (stesso calendario)."""
    close = df["close"]
    ret = log_returns(close)
    out = pd.DataFrame(index=df.index)
    out["ret"] = ret
    for w in VOL_WINDOWS:
        out[f"rv_{w}"] = realized_vol(ret, w)
    for w in (21, 63):
        out[f"park_{w}"] = parkinson_vol(df["high"], df["low"], w)
        out[f"gk_{w}"] = garman_klass_vol(df, w)
    out["atr_pct"] = atr_pct(df)
    for name, w in MOMENTUM_WINDOWS.items():
        out[f"mom_{name}"] = ret.rolling(w, min_periods=w).sum()
    for w in MA_WINDOWS:
        out[f"dist_ma{w}"] = np.log(close / close.rolling(w, min_periods=w).mean())
    out["vol_z"] = volume_zscore(df["volume"])
    out[f"skew_{ROLLING_STATS_WINDOW}"] = ret.rolling(ROLLING_STATS_WINDOW).skew()
    out[f"kurt_{ROLLING_STATS_WINDOW}"] = ret.rolling(ROLLING_STATS_WINDOW).kurt()
    out["drawdown"] = drawdown(close)

    if bench is not None:
        # Allineamento sulle date comuni: un giorno mancante nel benchmark non
        # viene "riempito", la coppia di rendimenti semplicemente non esiste.
        rb = log_returns(bench["close"]).reindex(df.index)
        pair = pd.concat([ret, rb], axis=1, keys=["r", "b"])
        for w in BETA_WINDOWS:
            cov = pair["r"].rolling(w, min_periods=w).cov(pair["b"])
            var = pair["b"].rolling(w, min_periods=w).var()
            out[f"beta_{w}"] = cov / var.where(var > 0)
            out[f"corr_{w}"] = pair["r"].rolling(w, min_periods=w).corr(pair["b"])
        for name in ("3m", "12m"):
            w = MOMENTUM_WINDOWS[name]
            out[f"rs_{name}"] = out[f"mom_{name}"] - rb.rolling(w, min_periods=w).sum()
    return out
