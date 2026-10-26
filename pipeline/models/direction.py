"""Direzione del prezzo (SPERIMENTALE): probabilità di salita nelle prossime h sedute.

Modello: regressione logistica con penalità L2, stimata su tutti i titoli insieme (panel),
su feature standardizzate. Parte dalla frequenza storica di salita del titolo (offset in
scala logit): il modello deve spiegare solo lo SCOSTAMENTO dalla baseline, quindi se le
feature non contengono informazione la previsione torna alla baseline.

Target: log-rendimento dall'APERTURA della seduta successiva alla chiusura di t+h (l'analisi
gira a mercato chiuso: il primo prezzo a cui si può agire è l'apertura del giorno dopo).

Intervallo plausibile: quantili empirici (10°, 50°, 90°) del rendimento a h sedute diviso per
la volatilità del momento (rv_21 scalata su h), moltiplicati per la volatilità di oggi.

Tutte le feature al giorno t usano solo dati fino a t (le date delle trimestrali sono
annunciate in anticipo). Ogni valore mancante vale 0 dopo la standardizzazione (= media).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from pipeline.features import TRADING_DAYS

L2 = 0.02                 # penalità relativa al numero di righe (ridge su feature standardizzate)
CLIP = 5.0
CLIM_SHRINK = 252         # frequenza di salita del titolo "tirata" verso quella di tutti (righe equivalenti)
QUANTILES = (0.10, 0.50, 0.90)

FEATURES = {
    "mom_1m": "andamento dell'ultimo mese",
    "mom_3m": "andamento degli ultimi 3 mesi",
    "mom_12m": "andamento dell'ultimo anno",
    "ret_5": "andamento dell'ultima settimana",
    "dist_ma50": "distanza dalla media mobile a 50 sedute",
    "dist_ma200": "distanza dalla media mobile a 200 sedute",
    "drawdown": "distanza dal massimo storico",
    "vol_rel": "volatilità rispetto alla sua mediana dell'ultimo anno",
    "rs_3m": "forza rispetto al Nasdaq 100 a 3 mesi",
    "rs_12m": "forza rispetto al Nasdaq 100 a 12 mesi",
    "vol_z5": "volumi dell'ultima settimana rispetto al solito",
    "p_turb": "probabilità di fase agitata",
    "earn_soon": "trimestrale nelle prossime sedute",
    "earn_rx": "reazione all'ultima trimestrale (se recente)",
}
EARNINGS_RECENT = 63


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------------
# Feature
# ---------------------------------------------------------------------------

def feature_frame(feats: pd.DataFrame, p_turb: pd.Series | None, h: int,
                  earn_first: list[pd.Timestamp] | None = None,
                  earn_rx: pd.DataFrame | None = None,
                  upcoming: list | None = None) -> pd.DataFrame:
    """Feature del modello per un titolo (indice = sedute).

    feats: output di features.compute_features; p_turb: probabilità filtrata del regime turbolento;
    earn_first: prime sedute di reazione alle trimestrali già avvenute; upcoming: date (dt.date)
    stimate delle reazioni future; earn_rx: reazioni (colonne last, reaction, reaction_vs_bench).
    """
    idx = feats.index
    out = pd.DataFrame(index=idx)
    for c in ("mom_1m", "mom_3m", "mom_12m", "dist_ma50", "dist_ma200", "drawdown"):
        out[c] = feats[c]
    out["ret_5"] = feats["ret"].rolling(5, min_periods=5).sum()
    med = feats["rv_21"].rolling(TRADING_DAYS, min_periods=126).median()
    out["vol_rel"] = np.log(feats["rv_21"] / med.where(med > 0))
    for c in ("rs_3m", "rs_12m"):
        out[c] = feats[c] if c in feats else np.nan
    out["vol_z5"] = feats["vol_z"].rolling(5, min_periods=3).mean()
    out["p_turb"] = p_turb.reindex(idx) if p_turb is not None else np.nan

    # Trimestrale entro h sedute (date note in anticipo).
    soon = np.zeros(len(idx))
    targets = sorted(set(pd.DatetimeIndex(earn_first or []).normalize()) |
                     {pd.Timestamp(d) for d in (upcoming or [])})
    if targets:
        days = idx.normalize()
        t_arr = pd.DatetimeIndex(targets)
        nxt = t_arr.searchsorted(days, side="right")            # prossima reazione strettamente dopo t
        pos = days.searchsorted(t_arr)                           # posizione (anche oltre la fine)
        for i in range(len(idx)):
            k = nxt[i]
            if k < len(t_arr):
                gap = pos[k] - i if pos[k] < len(idx) else np.busday_count(days[i].date(), t_arr[k].date())
                soon[i] = 1.0 if 0 < gap <= h else 0.0
    out["earn_soon"] = soon

    # Reazione all'ultima trimestrale, se avvenuta nelle ultime EARNINGS_RECENT sedute.
    rx_feat = np.zeros(len(idx))
    if earn_rx is not None and len(earn_rx):
        signed = earn_rx["reaction_vs_bench"].where(earn_rx["reaction_vs_bench"].notna(), earn_rx["reaction"])
        s = pd.Series(signed.to_numpy(), index=pd.DatetimeIndex(earn_rx["last"])).sort_index()
        s = s[~s.index.duplicated(keep="last")].reindex(idx)
        last_val = s.ffill(limit=EARNINGS_RECENT - 1)
        rx_feat = last_val.fillna(0.0).to_numpy()
    out["earn_rx"] = rx_feat
    return out[list(FEATURES)]


def forward_return(ret: pd.Series, gap: pd.Series, h: int) -> pd.Series:
    """Log-rendimento dall'apertura di t+1 alla chiusura di t+h (NaN se non ancora noto)."""
    r = ret.fillna(0.0)
    fwd = r[::-1].rolling(h, min_periods=h).sum()[::-1].shift(-1)   # somma di ret t+1..t+h
    g = gap.reindex(ret.index).fillna(0.0).shift(-1)
    out = fwd - g
    out.iloc[len(out) - h:] = np.nan
    return out


# ---------------------------------------------------------------------------
# Stima
# ---------------------------------------------------------------------------

class Standardizer:
    def __init__(self, X: np.ndarray):
        with warnings.catch_warnings():          # colonne tutte mancanti (es. forza relativa degli indici)
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mean = np.nanmean(X, axis=0)
            sd = np.nanstd(X, axis=0)
        self.sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        self.mean = np.where(np.isfinite(self.mean), self.mean, 0.0)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mean) / self.sd
        Z = np.clip(Z, -CLIP, CLIP)
        return np.where(np.isfinite(Z), Z, 0.0)


def fit_logistic(Z: np.ndarray, y: np.ndarray, offset: np.ndarray, l2: float = L2,
                 max_iter: int = 50) -> np.ndarray:
    """Newton-Raphson con penalità L2 (intercetta non penalizzata). Restituisce [b0, b...]."""
    n, k = Z.shape
    A = np.hstack([np.ones((n, 1)), Z])
    P = np.eye(k + 1) * (l2 * n)
    P[0, 0] = 0.0
    beta = np.zeros(k + 1)
    for _ in range(max_iter):
        p = sigmoid(offset + A @ beta)
        w = p * (1 - p)
        grad = A.T @ (p - y) + P @ beta
        H = (A * w[:, None]).T @ A + P
        step = np.linalg.solve(H, grad)
        beta -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def climatology(y: np.ndarray, groups: np.ndarray) -> dict:
    """Frequenza di salita per titolo, tirata verso quella di tutti (pochi dati = più vicina al totale)."""
    pooled = float(y.mean()) if len(y) else 0.5
    out = {"_all": pooled}
    for g in np.unique(groups):
        m = groups == g
        out[g] = (float(y[m].sum()) + CLIM_SHRINK * pooled) / (int(m.sum()) + CLIM_SHRINK)
    return out


class DirectionModel:
    """Modello stimato su un insieme di righe (titolo, origine) con target realizzato."""

    def __init__(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, z_resid: np.ndarray):
        self.std = Standardizer(X)
        self.clim = climatology(y, groups)
        off = logit(np.array([self.clim[g] for g in groups]))
        self.beta = fit_logistic(self.std(X), y, off)
        zr = z_resid[np.isfinite(z_resid)]
        self.q = np.quantile(zr, QUANTILES) if len(zr) > 50 else np.array([-1.28, 0.0, 1.28])

    def base_rate(self, groups: np.ndarray) -> np.ndarray:
        return np.array([self.clim.get(g, self.clim["_all"]) for g in groups])

    def predict(self, X: np.ndarray, groups: np.ndarray) -> np.ndarray:
        Z = self.std(X)
        return sigmoid(logit(self.base_rate(groups)) + self.beta[0] + Z @ self.beta[1:])

    def contributions(self, x: np.ndarray) -> dict:
        """Contributo di ogni feature al logit (per spiegare la previsione di oggi)."""
        z = self.std(x[None, :])[0]
        return {name: float(b * v) for name, b, v in zip(FEATURES, self.beta[1:], z)}

    def interval(self, sigma_h: np.ndarray) -> np.ndarray:
        """Quantili del log-rendimento a h sedute: colonne = QUANTILES."""
        return sigma_h[:, None] * self.q[None, :]


def sigma_h(rv21: np.ndarray, h: int) -> np.ndarray:
    return rv21 * math.sqrt(h / TRADING_DAYS)
