# FinanceNow

Analisi automatica del comportamento di azioni e indici di borsa: regimi di mercato, volatilità,
anomalie, correlazioni e rischio, aggiornati ogni giorno.

- **Pipeline**: Python su GitHub Actions (dati giornalieri, aggiornamento serale)
- **Dati**: OHLCV giornaliero da yfinance; Stooq solo come riserva opzionale (richiede API key)
- **Dashboard**: pagina statica in vanilla JS su GitHub Pages
- **Validazione**: backtest walk-forward separato dal log delle previsioni live

Lo stato del progetto e le specifiche tecniche sono in [`CLAUDE.md`](CLAUDE.md).

> Strumento di analisi personale. Non costituisce consulenza finanziaria.
