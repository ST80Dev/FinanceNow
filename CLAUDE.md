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
- Fonte primaria: **yfinance** (decisione del 26/10/2026). Può essere bloccato a intermittenza dagli IP
  dei runner GitHub: retry con backoff; un ticker non aggiornato resta con lo storico in cache ("stale")
  e viene recuperato al run successivo grazie alla sovrapposizione dell'incrementale.
- **Stooq**: senza API key, da GitHub Actions risponde con una pagina anti-bot JavaScript invece del CSV
  (test del 26/10/2026) ⇒ non usabile senza chiave. Resta supportato come sorgente di riserva e si attiva
  da solo se esiste il secret `STOOQ_API_KEY`. Mai la chiave nel codice o nei log. Non aggirare protezioni
  anti-bot. Stooq ha quote di download: con la chiave il backfill va diluito (budget di richieste per run).
- Pipeline: sorgenti in ordine di priorità → segnalare i ticker rimasti senza dati (non fallire tutto il run).
  Il download è un ticker alla volta, con pause, riprendibile. La sorgente di un ticker è fissa:
  non si uniscono serie di fonti diverse (aggiustamenti diversi).
- Barre incomplete: la seduta in corso si scarta finché il mercato non è chiuso (dopo le 17:00 di New York).
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

### Scelte del passo 2 (implementate)
- Target volatilità: varianza della somma dei log-rendimenti a h ∈ {1, 5} sedute, proxy realizzato =
  somma dei rendimenti al quadrato. Modelli: GARCH(1,1)-t, EGARCH(1,1)-t (solo h=1), HAR-RV;
  baseline `rw` (varianza delle ultime 21 sedute) e `hist_mean` (media storica). Stima su finestra
  espandente, ri-stima ogni 63 sedute, prima previsione dopo 1000 sedute.
- **Test finale**: dal **2025-11-03** in poi (`HOLDOUT_START` in `pipeline/backtest.py`), escluso di default.
  Data fissa: non spostarla. Usarlo (`--use-holdout`) una sola volta, a sviluppo concluso.
- Registro delle configurazioni provate: `results/backtest/trials.jsonl` (una riga per configurazione).
- Regimi: HMM gaussiano 3 stati (calmo/normale/turbolento) su rendimento e log-volatilità a 21 sedute;
  probabilità **filtrate** (forward), parametri ri-stimati ogni 252 sedute: nessun look-ahead.
- Anomalie: z-score robusto (mediana/MAD sulle 252 sedute precedenti), soglia |z| ≥ 4, su rendimento,
  gap, range, volume e rendimento in eccesso vs benchmark (beta stimato fino al giorno prima).
- Nei risultati non finiscono mai prezzi grezzi: solo feature derivate, stati, eventi, metriche.

### Scelte del passo 3 (implementate)
- `daily.yml`: cron `30 22 * * 1-5` (22:30 UTC = sempre dopo le 17:00 di New York) + avvio manuale.
  Sequenza: ripristino cache → `ingest daily` (backfill automatico dei ticker senza storico) → salvataggio
  cache → `analyze --backtest auto` → `live` → riepilogo nel job summary → commit di `results/`.
  Se il download fallisce per tutti i ticker, i risultati si calcolano sui dati in cache e il run termina rosso.
- `backfill.yml`: solo manuale (ticker, `--force`, budget di richieste); non committa nulla.
- Cache dei dati grezzi: chiave `raw-data-v1-<run_id>`, ripristino per prefisso `raw-data-v1-` (l'ultima salvata).
  Persa la cache, il primo `daily` riscarica tutto da solo. I due workflow condividono un gruppo di
  `concurrency`, quindi non girano mai insieme.
- Serie `live`: `results/live/volatility_forecasts.csv` (append-only, chiave origine+strumento+orizzonte+modello,
  con versione del modello e ora di creazione) e `results/live/volatility_scores.json` (valutazione dei soli
  target già realizzati). Le previsioni live stimano i parametri su tutto lo storico disponibile la sera stessa.
- Backtest `auto`: si rifà solo se manca, se cambia la configurazione o per un ticker nuovo.
- `results/status/ingest.json`: esito dell'ultimo download (ticker senza dati / non aggiornati / rimandati).
- I cron girano solo sul branch di default (`main`).

## Struttura prevista
```
.github/workflows/daily.yml
.github/workflows/backfill.yml
pipeline/
  ingest.py        # download OHLCV (yfinance, Stooq con chiave), incrementale, riprendibile
  features.py
  models/
    volatility.py
    regimes.py
    anomalies.py
  backtest.py      # walk-forward della volatilità, metriche, test finale, registro configurazioni
  analyze.py       # CLI: feature + regimi + anomalie + backtest → results/
  live.py          # previsioni live registrate ogni sera e loro valutazione
  summary.py       # riepilogo Markdown per il job summary di Actions
  export.py        # JSON leggeri per la dashboard
config/universe.yml
results/           # output elaborati committati (analysis/, backtest/, live/, status/)
docs/              # GitHub Pages: index.html, app.js, style.css, data/
scripts/           # strumenti diagnostici (es. stooq_probe.py)
tests/             # test offline (pytest, nessuna chiamata di rete)
data/raw/          # dati grezzi Parquet + _state.json/_report.json — NON committati (cache Actions)
requirements.txt
```

## Universo (confermato, `config/universe.yml`)
Mercato USA, centrato sul Nasdaq. Riferimenti: Nasdaq 100 (benchmark di default), QQQ, S&P 500.
Azioni: le 10 più capitalizzate del Nasdaq al 26/10/2026 (SpaceX esclusa per storico insufficiente,
sostituita da AMD): NVDA, AAPL, GOOGL, MSFT, AMZN, META, AVGO, TSLA, MU, AMD. Dati dal 2010-01-01.
Lista **congelata** (non ricalcolata in automatico). Bias di selezione: sono i vincitori di oggi ⇒
usarli per analisi descrittive per titolo, non per conclusioni su rendimenti attesi.

## Roadmap (un passo alla volta, conferma alla fine di ciascuno)
1. Struttura cartelle, `universe.yml`, `requirements.txt`, `ingest.py` con backfill + fallback + incrementale.
2. `features.py` e i tre moduli (volatilità, regimi, anomalie) + backtest walk-forward con baseline.
3. Workflow Actions (`backfill.yml`, `daily.yml`) con cache dei dati grezzi e commit dei risultati.
4. Dashboard su Pages (vanilla JS).
5. Dati di contesto (da proporre dopo la dashboard, richiesta del 26/10/2026). Oggi il sistema usa SOLO
   prezzi e volumi OHLCV: nessun dato di bilancio, macro, notizie o calendario. Candidati:
   - date delle trimestrali, per distinguere le anomalie "spiegate" (giorno di utili) da quelle inattese;
   - VIX e tassi dei Treasury come contesto di mercato (regimi, volatilità).
6. Più avanti: moduli predittivi sperimentali, alert (Telegram/email).

## Avvertenza
Strumento di analisi personale, non consulenza finanziaria.
