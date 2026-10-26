# FinanceNow — istruzioni per Claude Code

Sistema automatico di **analisi del comportamento** di azioni e indici di borsa.
Obiettivo primario: **descrivere e misurare** (regimi, volatilità, anomalie, correlazioni, rischio),
non prevedere il prezzo. I moduli predittivi sono sperimentali e vanno sempre confrontati con baseline banali.

## Modo di lavorare
- Lingua: italiano (codice, commenti e nomi di variabili possono essere in inglese).
- **Conferma a ogni passo**: prima di cambi strutturali (nuove cartelle, nuove dipendenze, schema dati,
  workflow) proporre e attendere conferma. Lavorare un passo della roadmap alla volta.
- Frontend: **vanilla JS/HTML/CSS, nessun framework**. Librerie da CDN solo se necessarie
  (es. DuckDB-WASM, una libreria di grafici leggera).
- Backend/pipeline: Python 3.12.

## Architettura (decisa)
Tutto su GitHub, nessun VPS, nessun Supabase.
- **GitHub Actions** esegue la pipeline Python:
  - `backfill.yml` — manuale (`workflow_dispatch`): scarica lo storico completo dei ticker in `config/universe.yml`.
  - `daily.yml` — cron serale (dopo la chiusura dei mercati USA): aggiornamento incrementale, ricalcolo modelli,
    export JSON, commit automatico dei risultati. Se un ticker nuovo non ha storico, fa backfill solo per quello.
- **GitHub Pages** serve la dashboard statica da `docs/`, che legge solo file statici (JSON / Parquet).
- Repo **pubblico**: Actions gratuito; i cron si disattivano dopo 60 giorni senza attività (i commit automatici
  lo tengono vivo); gli orari cron possono slittare (irrilevante con dati EOD).

## Dati
- Granularità: **giornaliera (OHLCV)**, prezzi aggiustati per split/dividendi.
- Fonte primaria: **Stooq**. Dal 2026 richiede API key → secret `STOOQ_API_KEY`. Mai la chiave nel codice.
  Stooq ha quote di download: il backfill va **diluito** (un ticker alla volta, pause tra richieste,
  eventualmente su più run) e deve essere riprendibile.
- Fallback: **yfinance** (può essere bloccato a intermittenza dagli IP dei runner GitHub).
  Pipeline: primaria → secondaria → segnalare i ticker rimasti senza dati (non fallire tutto il run).
- **Licenze**: repo pubblico ⇒ **i dati grezzi NON vanno committati**. Stanno nella **cache di Actions**
  (se persa, il workflow rifà il backfill). Nel repo finiscono solo risultati elaborati
  (regimi, volatilità, anomalie, JSON per la dashboard).
- Formato interno: Parquet (pandas/polars, DuckDB per query).
- Trappole da evitare: survivorship bias, look-ahead, fusi orari/festività/calendari diversi.

## Analisi
Lavorare sempre su **log-rendimenti**, mai sui livelli di prezzo.
- Feature: volatilità realizzata multi-finestra, Parkinson/Garman-Klass, ATR; momentum 1/3/6/12 mesi,
  distanza da medie mobili; z-score volume; beta e correlazioni rolling vs indice; forza relativa;
  skew/kurtosis rolling; drawdown.
- Moduli iniziali:
  1. **Volatilità**: GARCH/EGARCH (`arch`), HAR-RV.
  2. **Regimi**: HMM 2–3 stati su rendimenti e volatilità (`hmmlearn`).
  3. **Anomalie**: z-score robusti (mediana/MAD); eventuale isolation forest.
- Baseline obbligatorie per ogni previsione: random walk, media storica.
- Sperimentale e più avanti: LightGBM su target ben definiti; modelli foundation (es. Mistral Timestral)
  come confronto, soprattutto sulla volatilità.

## Validazione
- **Walk-forward**, niente k-fold casuale (per ML: purged CV con embargo).
- Tenere traccia del numero di test/configurazioni provate (rischio overfitting da test multipli).
- Un periodo di test finale mai toccato durante lo sviluppo.
- Metriche: MASE, CRPS (per quantili), calibrazione degli intervalli; QLIKE/MSE per la volatilità.
- **Due serie separate**, sempre distinte anche nella dashboard:
  - `backtest` — walk-forward retroattivo sullo storico;
  - `live` — log delle previsioni fatte da `daily.yml`, con data e versione del modello. È l'unica prova pulita.

## Struttura prevista
```
.github/workflows/daily.yml
.github/workflows/backfill.yml
pipeline/
  ingest.py        # download OHLCV (Stooq + fallback yfinance), incrementale, riprendibile
  features.py
  models/
    volatility.py
    regimes.py
    anomalies.py
  export.py        # JSON leggeri per la dashboard
config/universe.yml
results/           # output elaborati committati (backtest/, live/)
docs/              # GitHub Pages: index.html, app.js, style.css, data/
requirements.txt
```

## Universo iniziale (da confermare)
Una decina di indici/ETF: S&P 500, Nasdaq 100, FTSE MIB, Euro Stoxx 50, DAX, settori USA principali,
oro, Treasury lunghi; più qualche titolo singolo.

## Roadmap (un passo alla volta, conferma alla fine di ciascuno)
1. Struttura cartelle, `universe.yml`, `requirements.txt`, `ingest.py` con backfill + fallback + incrementale.
2. `features.py` e i tre moduli (volatilità, regimi, anomalie) + backtest walk-forward con baseline.
3. Workflow Actions (`backfill.yml`, `daily.yml`) con cache dei dati grezzi e commit dei risultati.
4. Dashboard su Pages (vanilla JS).
5. Più avanti: moduli predittivi sperimentali, alert (Telegram/email).

## Avvertenza
Strumento di analisi personale, non consulenza finanziaria.
