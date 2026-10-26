"""Download OHLCV giornaliero (prezzi aggiustati) per l'universo in config/universe.yml.

Sorgenti in ordine di priorità configurabile (default: yfinance, poi Stooq solo se
esiste il secret STOOQ_API_KEY). Caratteristiche:

- backfill completo per i ticker senza storico, incrementale per gli altri;
- un ticker alla volta, pausa tra le richieste, budget di richieste per run;
- stato salvato dopo ogni ticker: un run interrotto riparte da dove era arrivato;
- incrementale con sovrapposizione: se i prezzi già salvati sono cambiati
  (dividendo/split ⇒ nuovo aggiustamento) si riscarica tutto lo storico del ticker;
- la sorgente di un ticker è "appiccicosa": non si mescolano dati di fonti diverse;
- la seduta in corso (mercato non ancora chiuso) viene scartata;
- un ticker che fallisce non fa fallire il run: finisce nel report.

I dati grezzi (data/raw/) NON vanno committati: vivono nella cache di Actions.

Uso:
    python -m pipeline.ingest backfill [--tickers nvda,aapl] [--force]
    python -m pipeline.ingest daily
    python -m pipeline.ingest check
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import logging
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger("financenow.ingest")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIVERSE = ROOT / "config" / "universe.yml"
DEFAULT_DATA_DIR = ROOT / "data" / "raw"
STATE_FILE = "_state.json"
REPORT_FILE = "_report.json"

COLUMNS = ["open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close"]

# Calendario -> (fuso della borsa, ora locale dopo cui la barra del giorno è considerata definitiva).
CALENDARS = {
    "XNYS": ("America/New_York", dt.time(17, 0)),
}

# Finestra di sovrapposizione dell'incrementale (giorni di calendario ≈ 14 sedute).
OVERLAP_CALENDAR_DAYS = 21
# Differenza relativa massima sui close sovrapposti prima di riscaricare tutto.
# Sotto questa soglia l'effetto sui log-rendimenti è trascurabile.
ADJUSTMENT_TOLERANCE = 1e-4
# Tolleranza per barre "impossibili" (high < max(open, close) ecc.): entro questa
# soglia è arrotondamento e si corregge, oltre la barra viene scartata.
BAR_TOLERANCE = 0.005
# Log-rendimento giornaliero oltre cui si segnala (senza correggere) un salto sospetto.
JUMP_THRESHOLD = 0.25
# Dopo quanti ticker consecutivi falliti una sorgente viene disattivata per il run
# (es. yfinance bloccato dagli IP del runner): evita di insistere inutilmente.
MAX_CONSECUTIVE_FAILURES = 3


# ---------------------------------------------------------------------------
# Universo
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Instrument:
    id: str
    name: str
    asset_class: str
    symbols: dict[str, str]          # sorgente -> simbolo
    sources: tuple[str, ...]         # ordine di priorità
    start: dt.date
    calendar: str
    currency: str
    region: str
    benchmark: str | None
    active: bool
    listed_since: dt.date | None = None

    def symbol_for(self, source: str) -> str | None:
        return self.symbols.get(source)


class ConfigError(Exception):
    pass


KNOWN_SOURCES = ("yahoo", "stooq")
ASSET_CLASSES = ("index", "etf", "stock")


def _as_date(value, field: str, inst_id: str) -> dt.date:
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError as e:
        raise ConfigError(f"{inst_id}: '{field}' non è una data valida: {value!r}") from e


def load_universe(path: Path | str = DEFAULT_UNIVERSE) -> list[Instrument]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    defaults = raw.get("defaults") or {}
    items = raw.get("instruments") or []
    if not items:
        raise ConfigError("universe.yml: nessuno strumento definito")

    instruments: list[Instrument] = []
    for item in items:
        cfg = {**defaults, **item}
        inst_id = cfg.get("id")
        if not inst_id:
            raise ConfigError(f"strumento senza 'id': {item!r}")
        for field in ("name", "asset_class", "start", "calendar", "sources"):
            if cfg.get(field) in (None, "", []):
                raise ConfigError(f"{inst_id}: campo obbligatorio mancante '{field}'")
        if cfg["asset_class"] not in ASSET_CLASSES:
            raise ConfigError(f"{inst_id}: asset_class sconosciuta {cfg['asset_class']!r}")
        if cfg["calendar"] not in CALENDARS:
            raise ConfigError(f"{inst_id}: calendario non supportato {cfg['calendar']!r}")
        sources = tuple(cfg["sources"])
        unknown = [s for s in sources if s not in KNOWN_SOURCES]
        if unknown:
            raise ConfigError(f"{inst_id}: sorgenti sconosciute {unknown}")
        symbols = {s: cfg[s] for s in KNOWN_SOURCES if cfg.get(s)}
        if not any(s in symbols for s in sources):
            raise ConfigError(f"{inst_id}: nessun simbolo per le sorgenti {list(sources)}")
        instruments.append(
            Instrument(
                id=inst_id,
                name=cfg["name"],
                asset_class=cfg["asset_class"],
                symbols=symbols,
                sources=sources,
                start=_as_date(cfg["start"], "start", inst_id),
                calendar=cfg["calendar"],
                currency=cfg.get("currency", ""),
                region=cfg.get("region", ""),
                benchmark=cfg.get("benchmark"),
                active=bool(cfg.get("active", True)),
                listed_since=(
                    _as_date(cfg["listed_since"], "listed_since", inst_id)
                    if cfg.get("listed_since") else None
                ),
            )
        )

    ids = [i.id for i in instruments]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ConfigError(f"id duplicati: {dupes}")
    for inst in instruments:
        if inst.benchmark is not None and inst.benchmark not in ids:
            raise ConfigError(f"{inst.id}: benchmark inesistente {inst.benchmark!r}")
    return instruments


# ---------------------------------------------------------------------------
# Sorgenti
# ---------------------------------------------------------------------------

class SourceError(Exception):
    """Errore di download per un singolo simbolo."""


class TransientError(SourceError):
    """Errore probabilmente temporaneo: si riprova."""


class NoDataError(SourceError):
    """La sorgente non ha dati per il simbolo."""


class SourceDisabled(SourceError):
    """La sorgente non è utilizzabile per il resto del run (quota, blocco, chiave)."""


class Source:
    name = "base"

    def available(self) -> tuple[bool, str]:
        """(utilizzabile, motivo se non lo è)."""
        return True, ""

    def fetch(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        """OHLCV giornaliero aggiustato tra start ed end inclusi.

        Restituisce un DataFrame indicizzato per data di borsa (datetime64 senza
        fuso, normalizzato a mezzanotte) con colonne COLUMNS.
        """
        raise NotImplementedError


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Uniforma nomi colonna, indice e tipi."""
    df = df.rename(columns=lambda c: str(c).strip().lower())
    if "date" in df.columns:
        df = df.set_index("date")
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    if idx.tz is not None:
        # L'ora locale della borsa determina la data della seduta.
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "date"
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[COLUMNS].astype("float64")


class YahooSource(Source):
    name = "yahoo"

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    def fetch(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        import yfinance as yf

        try:
            df = yf.Ticker(symbol).history(
                start=start.isoformat(),
                end=(end + dt.timedelta(days=1)).isoformat(),  # 'end' di yfinance è esclusivo
                interval="1d",
                auto_adjust=True,
                actions=False,
                timeout=self.timeout,
            )
        except Exception as e:  # yfinance solleva tipi eterogenei
            raise TransientError(f"yfinance: {type(e).__name__}: {e}") from e
        if df is None or df.empty:
            # yfinance non distingue "simbolo inesistente" da "bloccato": si riprova.
            raise TransientError("yfinance: risposta vuota (simbolo errato o fonte bloccata)")
        return normalize_frame(df)


class StooqSource(Source):
    name = "stooq"
    URL = "https://stooq.com/q/d/l/"
    # Nome del parametro della chiave: da verificare quando sarà disponibile una chiave.
    KEY_PARAM = "apikey"

    def __init__(self, api_key: str | None, timeout: float = 30.0, session=None):
        self.api_key = api_key or None
        self.timeout = timeout
        self._session = session

    def available(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "STOOQ_API_KEY assente (senza chiave Stooq restituisce una pagina anti-bot)"
        return True, ""

    def _redact(self, text: str) -> str:
        return text.replace(self.api_key, "***") if self.api_key else text

    def fetch(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        import requests

        session = self._session or requests
        params = {"s": symbol, "i": "d", "d1": start.strftime("%Y%m%d"), "d2": end.strftime("%Y%m%d")}
        if self.api_key:
            params[self.KEY_PARAM] = self.api_key
        try:
            resp = session.get(self.URL, params=params, timeout=self.timeout,
                               headers={"User-Agent": "FinanceNow/ingest"})
        except Exception as e:  # mai loggare l'URL completo: contiene la chiave
            raise TransientError(self._redact(f"stooq: {type(e).__name__}: {e}")) from None
        if resp.status_code >= 500 or resp.status_code == 429:
            raise TransientError(f"stooq: HTTP {resp.status_code}")
        if resp.status_code >= 400:
            raise SourceDisabled(f"stooq: HTTP {resp.status_code}")
        return self.parse(resp.text)

    @staticmethod
    def parse(text: str) -> pd.DataFrame:
        body = text.strip()
        low = body.lower()
        if not body:
            raise TransientError("stooq: risposta vuota")
        if "exceeded the daily hits limit" in low:
            raise SourceDisabled("stooq: quota giornaliera esaurita")
        if low.startswith("no data") or low == "brak danych":
            raise NoDataError("stooq: nessun dato per il simbolo")
        if "<html" in low[:500] or low.startswith("<!doctype"):
            raise SourceDisabled("stooq: pagina HTML (anti-bot/login) invece del CSV")
        df = pd.read_csv(io.StringIO(body))
        if "Date" not in df.columns or "Close" not in df.columns:
            raise SourceDisabled(f"stooq: formato CSV inatteso: {list(df.columns)}")
        return normalize_frame(df)


def build_sources() -> dict[str, Source]:
    return {
        "yahoo": YahooSource(),
        "stooq": StooqSource(api_key=os.environ.get("STOOQ_API_KEY")),
    }


# ---------------------------------------------------------------------------
# Validazione
# ---------------------------------------------------------------------------

def last_complete_session(calendar: str, now: dt.datetime) -> dt.date:
    """Ultima data la cui barra giornaliera può considerarsi definitiva."""
    tz_name, cutoff = CALENDARS[calendar]
    local = now.astimezone(ZoneInfo(tz_name))
    return local.date() if local.time() >= cutoff else local.date() - dt.timedelta(days=1)


def validate(df: pd.DataFrame, last_session: dt.date) -> tuple[pd.DataFrame, list[str]]:
    """Pulisce un OHLCV normalizzato. Restituisce (df pulito, avvisi)."""
    warnings: list[str] = []
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    n = len(df)
    df = df[df.index <= pd.Timestamp(last_session)]
    if len(df) < n:
        warnings.append(f"scartate {n - len(df)} barre successive a {last_session} (seduta non conclusa)")

    n = len(df)
    df = df.dropna(subset=PRICE_COLUMNS)
    if len(df) < n:
        warnings.append(f"scartate {n - len(df)} barre con prezzi mancanti")

    n = len(df)
    df = df[(df[PRICE_COLUMNS] > 0).all(axis=1)]
    if len(df) < n:
        warnings.append(f"scartate {n - len(df)} barre con prezzi <= 0")

    body_hi = df[["open", "close"]].max(axis=1)
    body_lo = df[["open", "close"]].min(axis=1)
    bad = (df["high"] < body_hi * (1 - BAR_TOLERANCE)) | (df["low"] > body_lo * (1 + BAR_TOLERANCE)) \
        | (df["high"] < df["low"] * (1 - BAR_TOLERANCE))
    if bad.any():
        dates = ", ".join(d.date().isoformat() for d in df.index[bad][:5])
        warnings.append(f"scartate {int(bad.sum())} barre incoerenti (high/low): {dates}")
        df = df[~bad]
    df = df.copy()
    # Piccole incoerenze da arrotondamento: si allarga high/low per contenere open/close.
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    df.loc[df["volume"] < 0, "volume"] = np.nan

    jumps = np.log(df["close"]).diff().abs() > JUMP_THRESHOLD
    if jumps.any():
        dates = ", ".join(d.date().isoformat() for d in df.index[jumps][:5])
        warnings.append(f"{int(jumps.sum())} salti > {JUMP_THRESHOLD:.0%} in log-rendimento (da verificare): {dates}")
    return df, warnings


# ---------------------------------------------------------------------------
# Persistenza (scritture atomiche)
# ---------------------------------------------------------------------------

def write_prices(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.reset_index().to_parquet(tmp, index=False)
    os.replace(tmp, path)


def read_prices(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path).set_index("date")
    df.index = pd.DatetimeIndex(df.index)
    return df


def _json_safe(obj):
    """NaN/inf -> null (JSON valido per la dashboard), tipi numpy -> tipi Python."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(_json_safe(obj), indent=2, ensure_ascii=False, default=str, allow_nan=False)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

class BudgetExhausted(Exception):
    pass


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Ingestor:
    def __init__(
        self,
        instruments: list[Instrument],
        sources: dict[str, Source],
        data_dir: Path = DEFAULT_DATA_DIR,
        *,
        max_requests: int = 200,
        pause: float = 2.0,
        retries: int = 3,
        backoff: float = 5.0,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], dt.datetime] = _utcnow,
    ):
        self.instruments = instruments
        self.sources = sources
        self.data_dir = Path(data_dir)
        self.max_requests = max_requests
        self.pause = pause
        self.retries = retries
        self.backoff = backoff
        self.sleep = sleep
        self.now = now
        self.requests = 0
        self.disabled: dict[str, str] = {}
        self._failures: dict[str, int] = {}
        for name, src in sources.items():
            ok, reason = src.available()
            if not ok:
                self.disabled[name] = reason
        self.state_path = self.data_dir / STATE_FILE
        self.state = read_json(self.state_path)
        self.state.setdefault("version", 1)
        self.state.setdefault("instruments", {})

    # -- utilità -------------------------------------------------------------

    def _price_path(self, inst: Instrument) -> Path:
        return self.data_dir / f"{inst.id}.parquet"

    def _usable_sources(self, inst: Instrument) -> list[str]:
        return [s for s in inst.sources
                if s in self.sources and s not in self.disabled and inst.symbol_for(s)]

    def _fetch(self, source: str, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Scarica con retry e backoff. Ogni tentativo conta nel budget."""
        last_err: Exception | None = None
        for attempt in range(self.retries):
            if source in self.disabled:
                raise SourceDisabled(self.disabled[source])
            if self.requests >= self.max_requests:
                raise BudgetExhausted()
            if attempt:
                delay = self.backoff * 3 ** (attempt - 1) * (1 + random.random() * 0.5)
                log.info("  retry %d/%d tra %.0fs", attempt, self.retries - 1, delay)
                self.sleep(delay)
            self.requests += 1
            try:
                df = self.sources[source].fetch(symbol, start, end)
                self._failures[source] = 0
                return df
            except NoDataError:
                raise
            except TransientError as e:
                last_err = e
                log.warning("  %s %s: %s", source, symbol, e)
            except SourceDisabled as e:
                self.disabled[source] = str(e)
                log.warning("  %s disattivata per questo run: %s", source, e)
                raise
        self._failures[source] = self._failures.get(source, 0) + 1
        if self._failures[source] >= MAX_CONSECUTIVE_FAILURES:
            self.disabled[source] = f"{self._failures[source]} ticker consecutivi falliti"
            log.warning("  %s disattivata per questo run: %s", source, self.disabled[source])
        raise SourceError(f"{source}: {self.retries} tentativi falliti; ultimo errore: {last_err}")

    # -- logica per strumento --------------------------------------------------

    def update_instrument(self, inst: Instrument, force: bool = False) -> dict:
        now = self.now()
        last_session = last_complete_session(inst.calendar, now)
        entry = self.state["instruments"].get(inst.id, {})
        existing = read_prices(self._price_path(inst))

        source = entry.get("source")
        needs_full = (
            force
            or existing is None
            or existing.empty
            or source not in inst.sources
            or entry.get("symbol") != inst.symbol_for(source)
            or entry.get("start") != inst.start.isoformat()
        )
        result: dict = {"id": inst.id, "warnings": []}
        entry["last_attempt"] = now.isoformat(timespec="seconds")

        if needs_full:
            result.update(self._backfill(inst, last_session))
            if result["status"] == "no_data" and existing is not None:
                result["status"] = "stale"  # lo storico precedente resta valido
        else:
            result.update(self._incremental(inst, entry, existing, last_session))

        if result["status"] in ("backfilled", "updated", "up_to_date", "refreshed"):
            df = result.pop("df")
            if result["status"] != "up_to_date":
                write_prices(df, self._price_path(inst))
            entry.update(
                source=result["source"],
                symbol=inst.symbol_for(result["source"]),
                start=inst.start.isoformat(),
                first_date=df.index[0].date().isoformat(),
                last_date=df.index[-1].date().isoformat(),
                rows=len(df),
                last_success=now.isoformat(timespec="seconds"),
                error=None,
            )
            result.update(rows=len(df), first_date=entry["first_date"], last_date=entry["last_date"])
        else:
            result.pop("df", None)
            entry["error"] = result.get("error")
            if existing is not None:
                result.update(rows=len(existing), last_date=entry.get("last_date"))
        entry["status"] = result["status"]
        self.state["instruments"][inst.id] = entry
        write_json(self.state, self.state_path)  # stato salvato dopo ogni ticker
        return result

    def _backfill(self, inst: Instrument, last_session: dt.date) -> dict:
        errors = []
        for source in self._usable_sources(inst):
            symbol = inst.symbol_for(source)
            log.info("  backfill da %s (%s) dal %s", source, symbol, inst.start)
            try:
                raw = self._fetch(source, symbol, inst.start, last_session)
            except BudgetExhausted:
                raise
            except SourceError as e:
                errors.append(str(e))
                continue
            df, warnings = validate(raw, last_session)
            if df.empty:
                errors.append(f"{source}: nessuna barra valida dopo la validazione")
                continue
            return {"status": "backfilled", "source": source, "df": df,
                    "new_rows": len(df), "warnings": warnings}
        if not self._usable_sources(inst):
            errors.append("nessuna sorgente utilizzabile: " + ", ".join(
                f"{s} ({self.disabled.get(s, 'simbolo mancante')})" for s in inst.sources))
        return {"status": "no_data", "error": " | ".join(errors)}

    def _incremental(self, inst: Instrument, entry: dict, existing: pd.DataFrame,
                     last_session: dt.date) -> dict:
        source = entry["source"]
        symbol = inst.symbol_for(source)
        last_date = existing.index[-1].date()
        if last_date >= last_session:
            return {"status": "up_to_date", "source": source, "df": existing, "new_rows": 0}
        if source in self.disabled:
            # La sorgente del ticker non è disponibile: non si mescolano fonti, si resta "stale".
            return {"status": "stale", "source": source, "error": self.disabled[source]}

        start = last_date - dt.timedelta(days=OVERLAP_CALENDAR_DAYS)
        log.info("  incrementale da %s (%s) dal %s", source, symbol, start)
        try:
            raw = self._fetch(source, symbol, start, last_session)
        except BudgetExhausted:
            raise
        except SourceError as e:
            return {"status": "stale", "source": source, "error": str(e)}
        new, warnings = validate(raw, last_session)

        window = existing[existing.index >= pd.Timestamp(start)]
        common = window.index.intersection(new.index)
        if len(window) and not len(common):
            reason = "nessuna seduta in comune nella finestra di sovrapposizione"
        elif len(common):
            rel = (new.loc[common, "close"] / existing.loc[common, "close"] - 1).abs().max()
            reason = (f"close sovrapposti diversi (max {rel:.2e}): nuovo aggiustamento"
                      if rel > ADJUSTMENT_TOLERANCE else None)
        else:
            reason = None

        if reason:
            log.info("  %s -> riscarico lo storico completo", reason)
            try:
                raw = self._fetch(source, symbol, inst.start, last_session)
            except BudgetExhausted:
                raise
            except SourceError as e:
                return {"status": "stale", "source": source,
                        "error": f"{reason}; riscaricamento fallito: {e}"}
            df, w2 = validate(raw, last_session)
            if df.empty:
                return {"status": "stale", "source": source,
                        "error": f"{reason}; riscaricamento senza barre valide"}
            return {"status": "refreshed", "source": source, "df": df,
                    "new_rows": int((df.index > existing.index[-1]).sum()),
                    "warnings": [reason, *w2]}

        added = new[new.index > existing.index[-1]]
        if added.empty:
            return {"status": "up_to_date", "source": source, "df": existing,
                    "new_rows": 0, "warnings": warnings}
        df = pd.concat([existing, added])
        return {"status": "updated", "source": source, "df": df,
                "new_rows": len(added), "warnings": warnings}

    # -- run -----------------------------------------------------------------

    def _select(self, tickers: list[str] | None) -> list[Instrument]:
        if not tickers:
            return self.instruments
        known = {i.id: i for i in self.instruments}
        missing = [t for t in tickers if t not in known]
        if missing:
            raise ConfigError(f"ticker non presenti nell'universo: {missing}")
        return [known[t] for t in tickers]

    def run(self, mode: str, tickers: list[str] | None = None, force: bool = False) -> dict:
        if mode not in ("backfill", "daily"):
            raise ValueError(mode)
        started = self.now()
        results: dict[str, dict] = {}
        selected = self._select(tickers)
        budget_hit = False
        first = True
        for inst in selected:
            if not inst.active:
                results[inst.id] = {"id": inst.id, "status": "inactive"}
                continue
            has_data = self._price_path(inst).exists()
            if mode == "backfill" and has_data and not force:
                results[inst.id] = {"id": inst.id, "status": "skipped",
                                    "note": "storico già presente (usa --force per riscaricare)"}
                continue
            if budget_hit:
                results[inst.id] = {"id": inst.id, "status": "deferred"}
                continue
            if not first:
                self.sleep(self.pause * (1 + random.random() * 0.5))
            first = False
            log.info("%s (%s)", inst.id, inst.name)
            try:
                results[inst.id] = self.update_instrument(inst, force=force)
            except BudgetExhausted:
                budget_hit = True
                results[inst.id] = {"id": inst.id, "status": "deferred"}
                log.warning("budget di %d richieste esaurito: i ticker rimanenti al prossimo run",
                            self.max_requests)
                continue
            r = results[inst.id]
            log.info("  -> %s%s", r["status"], f" ({r['error']})" if r.get("error") else "")
            for w in r.get("warnings", []):
                log.warning("  ! %s", w)

        summary: dict[str, int] = {}
        for r in results.values():
            summary[r["status"]] = summary.get(r["status"], 0) + 1
        report = {
            "run_started": started.isoformat(timespec="seconds"),
            "run_finished": self.now().isoformat(timespec="seconds"),
            "mode": mode,
            "requests": self.requests,
            "max_requests": self.max_requests,
            "disabled_sources": self.disabled,
            "summary": summary,
            "missing": sorted(i for i, r in results.items() if r["status"] == "no_data"),
            "stale": sorted(i for i, r in results.items() if r["status"] == "stale"),
            "deferred": sorted(i for i, r in results.items() if r["status"] == "deferred"),
            "instruments": results,
        }
        write_json(report, self.data_dir / REPORT_FILE)
        return report

    def check(self, tickers: list[str] | None = None, days: int = 14) -> list[dict]:
        """Verifica simboli e sorgenti scaricando pochi giorni, senza scrivere nulla."""
        out = []
        end = self.now().date()
        start = end - dt.timedelta(days=days)
        first = True
        for inst in self._select(tickers):
            for source in inst.sources:
                symbol = inst.symbol_for(source)
                row = {"id": inst.id, "source": source, "symbol": symbol}
                if not symbol:
                    row["result"] = "simbolo non definito"
                elif source in self.disabled:
                    row["result"] = f"sorgente disattivata: {self.disabled[source]}"
                else:
                    if not first:
                        self.sleep(self.pause)
                    first = False
                    try:
                        df = self._fetch(source, symbol, start, end)
                        row["result"] = (f"ok: {len(df)} barre, ultima {df.index[-1].date()}"
                                         if len(df) else "risposta senza barre")
                    except BudgetExhausted:
                        row["result"] = "budget esaurito"
                    except SourceError as e:
                        row["result"] = f"errore: {e}"
                out.append(row)
                log.info("%-6s %-6s %-10s %s", row["id"], source, symbol or "-", row["result"])
        return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pipeline.ingest", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["backfill", "daily", "check"])
    ap.add_argument("--tickers", help="id separati da virgola (default: tutto l'universo)")
    ap.add_argument("--force", action="store_true", help="backfill: riscarica anche chi ha già lo storico")
    ap.add_argument("--max-requests", type=int, default=200, help="budget di richieste per run")
    ap.add_argument("--pause", type=float, default=2.0, help="secondi di pausa tra un ticker e l'altro")
    ap.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    if not args.verbose:
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        warnings.filterwarnings("ignore", module="yfinance")

    try:
        instruments = load_universe(args.universe)
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] if args.tickers else None
        ing = Ingestor(instruments, build_sources(), args.data_dir,
                       max_requests=args.max_requests, pause=args.pause)
        for name, reason in ing.disabled.items():
            log.info("sorgente %s non attiva: %s", name, reason)
        if args.mode == "check":
            rows = ing.check(tickers)
            return 0 if any(r["result"].startswith("ok") for r in rows) else 1
        report = ing.run(args.mode, tickers, force=args.force)
    except ConfigError as e:
        log.error("configurazione non valida: %s", e)
        return 2

    log.info("riepilogo: %s | richieste %d/%d", report["summary"], report["requests"], report["max_requests"])
    for key, label in (("missing", "SENZA DATI"), ("stale", "non aggiornati"), ("deferred", "rimandati")):
        if report[key]:
            log.warning("%s: %s", label, ", ".join(report[key]))
    attempted = [r for r in report["instruments"].values()
                 if r["status"] not in ("inactive", "skipped", "deferred")]
    ok = [r for r in attempted if r["status"] in ("backfilled", "updated", "up_to_date", "refreshed")]
    # Il run fallisce solo se è stato tentato qualcosa e non è andato a buon fine niente.
    return 1 if attempted and not ok else 0


if __name__ == "__main__":
    sys.exit(main())
