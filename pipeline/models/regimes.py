"""Regimi di mercato con HMM gaussiano (hmmlearn) su rendimenti e volatilità.

Input per ogni seduta t: log-rendimento e log della volatilità realizzata a 21 sedute
(entrambi noti a fine seduta t). Gli stati sono ordinati per volatilità crescente,
così l'etichetta è stabile tra una stima e l'altra (0 = calmo, ultimo = turbolento).

Nessun look-ahead:
- i parametri si ri-stimano ogni `refit_every` sedute con i soli dati disponibili;
- la standardizzazione usa media e deviazione standard del solo campione di stima;
- le probabilità sono FILTRATE (algoritmo forward: usano dati fino a t), non quelle
  "smoothed" di hmmlearn (`predict_proba`), che userebbero anche il futuro.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from pipeline.features import TRADING_DAYS, realized_vol

LABELS = {2: ["calmo", "turbolento"], 3: ["calmo", "normale", "turbolento"]}
SEEDS = (0, 1, 2)


@dataclass
class HMMParams:
    start: np.ndarray        # (k,)
    trans: np.ndarray        # (k, k)
    means: np.ndarray        # (k, d) nello spazio standardizzato
    covs: np.ndarray         # (k, d, d)
    mu: np.ndarray           # (d,) media usata per standardizzare
    sd: np.ndarray           # (d,) dev. std usata per standardizzare
    loglik: float

    @property
    def k(self) -> int:
        return len(self.start)


def regime_inputs(ret: pd.Series) -> pd.DataFrame:
    """Colonne: ret, logvol (log della volatilità realizzata annualizzata a 21 sedute)."""
    rv = realized_vol(ret, 21)
    return pd.DataFrame({"ret": ret, "logvol": np.log(rv.where(rv > 0))}).dropna()


def fit_hmm(X: np.ndarray, n_states: int, seeds=SEEDS) -> HMMParams:
    from hmmlearn.hmm import GaussianHMM

    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd
    best = None
    for seed in seeds:
        model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200,
                            tol=1e-4, random_state=seed, min_covar=1e-4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model.fit(Z)
                score = model.score(Z)
            except Exception:
                continue
        if np.isfinite(score) and (best is None or score > best[0]):
            best = (score, model)
    if best is None:
        raise RuntimeError("HMM: nessuna stima riuscita")
    score, m = best
    order = np.argsort(m.means_[:, 1])  # per volatilità crescente
    return HMMParams(
        start=m.startprob_[order],
        trans=m.transmat_[np.ix_(order, order)],
        means=m.means_[order],
        covs=m.covars_[order],
        mu=mu, sd=sd, loglik=float(score),
    )


def filtered_probs(params: HMMParams, X: np.ndarray) -> np.ndarray:
    """P(stato_t | dati fino a t) con l'algoritmo forward normalizzato a ogni passo."""
    Z = (X - params.mu) / params.sd
    k = params.k
    log_em = np.column_stack([
        multivariate_normal(params.means[j], params.covs[j], allow_singular=True).logpdf(Z)
        for j in range(k)
    ])
    # Riscalare ogni riga non cambia le probabilità normalizzate ed evita underflow.
    em = np.exp(log_em - log_em.max(axis=1, keepdims=True))
    out = np.empty((len(Z), k))
    alpha = params.start * em[0]
    alpha /= alpha.sum()
    out[0] = alpha
    for t in range(1, len(Z)):
        alpha = (alpha @ params.trans) * em[t]
        s = alpha.sum()
        alpha = alpha / s if s > 0 else np.full(k, 1 / k)
        out[t] = alpha
    return out


def regimes_walk_forward(ret: pd.Series, n_states: int = 3, min_train: int = 756,
                         refit_every: int = 252) -> tuple[pd.DataFrame, HMMParams | None]:
    """Probabilità filtrate walk-forward. Restituisce (probabilità + stato, ultimi parametri)."""
    inp = regime_inputs(ret.dropna())
    X = inp.to_numpy()
    n = len(X)
    cols = [f"p_{lab}" for lab in LABELS[n_states]]
    probs = np.full((n, n_states), np.nan)
    params = None
    i0 = min_train
    while i0 < n:
        i1 = min(i0 + refit_every - 1, n - 1)
        try:
            params = fit_hmm(X[: i0 + 1], n_states)
        except RuntimeError:
            i0 = i1 + 1
            continue
        # Forward dall'inizio fino a fine blocco con i parametri di questo blocco.
        probs[i0:i1 + 1] = filtered_probs(params, X[: i1 + 1])[i0:i1 + 1]
        i0 = i1 + 1
    df = pd.DataFrame(probs, index=inp.index, columns=cols)
    valid = df.notna().all(axis=1)
    df["state"] = pd.Series(np.nan, index=df.index)
    df.loc[valid, "state"] = df.loc[valid, cols].to_numpy().argmax(axis=1)
    return df, params


def regime_summary(ret: pd.Series, probs: pd.DataFrame, params: HMMParams | None) -> dict:
    """Stato attuale e statistiche descrittive per stato."""
    valid = probs.dropna()
    if valid.empty or params is None:
        return {"available": False}
    labels = LABELS[params.k]
    state = valid["state"].astype(int)
    current = int(state.iloc[-1])
    run = (state.iloc[::-1] != current).to_numpy()
    days_in_state = int(np.argmax(run)) if run.any() else len(state)
    r = ret.reindex(state.index)
    per_state = []
    for j, lab in enumerate(labels):
        rj = r[state == j]
        p_stay = float(params.trans[j, j])
        per_state.append({
            "label": lab,
            "share_of_days": round(float((state == j).mean()), 4),
            "ret_ann": round(float(rj.mean() * TRADING_DAYS), 4) if len(rj) else None,
            "vol_ann": round(float(rj.std() * np.sqrt(TRADING_DAYS)), 4) if len(rj) > 1 else None,
            "expected_duration_days": round(1 / (1 - p_stay), 1) if p_stay < 1 else None,
        })
    return {
        "available": True,
        "as_of": state.index[-1].date().isoformat(),
        "current": labels[current],
        "probabilities": {lab: round(float(valid.iloc[-1][f"p_{lab}"]), 4) for lab in labels},
        "days_in_state": days_in_state,
        "states": per_state,
        "transition_matrix": np.round(params.trans, 4).tolist(),
    }
