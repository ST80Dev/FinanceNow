# FinanceNow — istruzioni per Claude Code

Sistema automatico di **analisi del comportamento** di azioni e indici di borsa.
Obiettivo (aggiornato il 26/10/2026): l'app **valuta da sola** la situazione di ogni titolo e la propone
in parole semplici (situazione, rischio, direzione, prossimi eventi), per un appassionato che conosce i termini
ma non i meccanismi. Base: **descrivere e misurare** (regimi, volatilità, anomalie, correlazioni, rischio).
Direzione futura richiesta: un sistema di previsione da affinare su rischio, direzione del prezzo e lettura dei
prossimi eventi. I moduli predittivi restano sperimentali, sempre confrontati con baseline banali, e la dashboard
dice chiaramente quando una previsione non batte la baseline ("non affidabile").

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

### Scelte del passo 4 (implementate)
- `pipeline/export.py` scrive JSON compatti in `docs/data/` (`index.json`, `models.json`, `tickers/<id>.json`);
  `daily.yml` lo esegue dopo `live` e `patterns` e committa anche `docs/data/`.
- Nella dashboard **nessun prezzo né volume assoluto**: solo grandezze derivate (volatilità, drawdown, beta,
  momentum, regimi, z-score; per le anomalie di volume il rapporto con la mediana). Un test lo verifica.
- Frontend `docs/` in vanilla JS senza librerie: grafici SVG scritti a mano, routing via hash
  (`#/`, `#/t/<id>`, `#/modelli`, `#/metodo`), tema chiaro/scuro automatico, layout mobile.
- `backtest` e `live` in pannelli separati con etichetta e colore propri (viola / verde acqua).
- GitHub Pages: Settings → Pages → "Deploy from a branch" → `main` / `docs`. URL: https://st80dev.github.io/FinanceNow/
- **Sintesi automatica** (`pipeline/digest.py`, richiesta del 26/10/2026): ogni sera regole esplicite selezionano
  cosa merita attenzione (cambi di regime recenti, anomalie delle ultime 5 sedute, volatilità estrema rispetto
  allo storico del titolo, volatilità in accelerazione, drawdown ≤ -20%, dati non aggiornati), con gravità
  alta/media/bassa. Volatilità alta diffusa (≥ metà dell'universo) detta una volta sola a livello di mercato.
  Esce in cima alla dashboard (`docs/data/digest.json`), nel job summary e in `results/digest/latest.md`.
- **Alert solo in pagina** (decisione del 26/10/2026): nessuna notifica esterna (email/Telegram/issue).
- **Pattern** (`pipeline/patterns.py`, richiesta del 26/10/2026): studio di eventi descrittivo, non previsione.
  Eventi (rilevati con soli dati fino al giorno stesso, cooldown 21 sedute): ingresso/uscita dal regime turbolento,
  crollo/balzo anomalo, volume anomalo, volatilità oltre il 95° percentile dello storico fino a quel giorno,
  volatilità in accelerazione, discesa sotto -20%. Esiti a 5/21/63 sedute: volatilità rispetto alla tipica del
  titolo, probabilità di calo ≥ 10%, rendimento (dato debole). Confronto con un giorno qualsiasi dello stesso titolo,
  solo finestre chiuse prima del test finale. "Netto" = IC 90% (cluster per settimana) esclude lo zero E differenza
  oltre una soglia minima. **Rendimento e calo misurati dall'apertura del giorno dopo**: l'analisi gira a mercato
  chiuso, il primo prezzo a cui si può agire è l'apertura successiva. Serie live separata:
  `results/live/pattern_log.csv` (eventi nuovi con esito atteso) e `results/live/pattern_scores.json`.
  Le note "storicamente…" compaiono nella sintesi sotto gli strumenti con segnali.
- Uso EOD: i risultati sono pronti dopo la chiusura USA (circa 23:30–00:30 ora italiana); servono per decidere
  prima della seduta successiva, non per operare intraday (servirebbero dati intraday, altra architettura).

### Scelte del passo 5 (implementate, richiesta del 26/10/2026)
- **Linguaggio semplice**: ogni grandezza ha un tooltip (`data-tip`, mouse e tocco) con cosa significa e come leggerla;
  glossario nella pagina **Guida** (`#/guida`). Nomi completi dei titoli + descrizione (`description` in `universe.yml`).
  Fasi del regime chiamate calma / normale / agitata; volatilità mostrata come oscillazione tipica al mese (σ/√12);
  andamenti in % semplice (non log). Il "Metodo" tecnico resta per i dettagli.
- **Trimestrali** (`pipeline/earnings.py`): date da yfinance (`get_earnings_dates`, pagina calendario Yahoo), solo azioni,
  in `data/raw/_earnings/<id>.json` (cache, non committate), riscaricate al massimo una volta a settimana o quando la
  prossima data è passata. Giorno di reazione: prima delle 9:30 NY → stessa seduta; dopo → seduta successiva; orario
  ignoto (mezzanotte) → finestra di 2 sedute datata all'ultima. Anomalie nei giorni di reazione marcate "spiegate".
  Eventi nei pattern: vigilia (5 sedute prima), reazione positiva/negativa vs benchmark.
- **Direzione** (`pipeline/direction.py`, `pipeline/models/direction.py`, SPERIMENTALE): P(log-rendimento da apertura t+1
  a chiusura t+h > 0), h ∈ {5, 21}. Logistica L2 (0,02·n) panel con offset = frequenza storica di salita del titolo
  (shrink 252 righe), 14 feature (andamento, medie mobili, drawdown, volatilità relativa, forza relativa, volumi,
  prob. fase agitata, trimestrali). Walk-forward: prima previsione dopo 1260 sedute, ri-stima ogni 63, embargo 5.
  Baseline `freq_storica` e `moneta`; metriche Brier, log-loss, Diebold-Mariano, calibrazione, copertura dell'intervallo
  10–90%. Verdetto "vantaggio" solo se skill > 0 e t < −1,96. Primo risultato (v0.1.0): **nessun vantaggio** a 5 e 21
  sedute. Serie live: `results/live/direction_forecasts.csv` / `direction_scores.json`.
- **Valutazione** (`pipeline/assess.py`, chiamata da `export.py`): situazione (tempesta / pressione / ripresa / norma /
  tranquillo), rischio nel prossimo mese (alto / normale / basso a punti: volatilità rispetto alla propria storia,
  +1 se un evento attivo ha effetto storico netto nei pattern, +1 se trimestrale entro 10 sedute), tendenza descrittiva,
  direzione del modello con etichetta di affidabilità, prossimi eventi. Panoramica = verdetti ordinati per attenzione.
  Serie live dei giudizi: `results/live/assessment_log.csv` / `assessment_scores.json` (esiti a 21 sedute per livello).
- Pattern: IC al 90% con errore standard robusto ai cluster per settimana, centrato sulla stessa media pubblicata.
- Registro configurazioni condiviso tra moduli: `results/backtest/trials.jsonl` (campo `module` per la direzione).

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
    direction.py
  backtest.py      # walk-forward della volatilità, metriche, test finale, registro configurazioni
  analyze.py       # CLI: feature + regimi + anomalie + backtest → results/
  live.py          # previsioni live registrate ogni sera e loro valutazione
  summary.py       # riepilogo Markdown per il job summary di Actions
  export.py        # JSON leggeri per la dashboard
  digest.py        # sintesi del giorno: segnali con gravità, per dashboard e job summary
  patterns.py      # studio di eventi: cosa è successo dopo eventi simili (storico + verifica live)
  earnings.py      # date delle trimestrali, giorno di reazione, mossa tipica
  direction.py     # direzione sperimentale: walk-forward vs baseline, previsioni di oggi, serie live
  assess.py        # valutazione automatica: situazione, rischio, direzione, prossimi eventi
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
5. Linguaggio semplice, trimestrali, direzione sperimentale, valutazione automatica (fatto, vedi sopra).
6. Da proporre: affinare il sistema di previsione (rischio, direzione, prossimi eventi) — ogni modifica è una nuova
   configurazione nel registro; contesto di mercato (VIX, tassi dei Treasury, calendario delle riunioni Fed);
   verifica storica delle regole di rischio; più avanti LightGBM con purged CV, alert (Telegram/email).

## Avvertenza
Strumento di analisi personale, non consulenza finanziaria.
