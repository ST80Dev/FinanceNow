"""Trimestrali: date di pubblicazione dei risultati e reazione del prezzo.

Perché servono: molte anomalie di prezzo e volume cadono nel giorno dopo i conti. Sapere
quando arriva la prossima trimestrale, e quanto si muove di solito il titolo in quel giorno,
distingue i movimenti "spiegati" da quelli inattesi e anticipa i giorni ad alto rischio.

Sorgente: yfinance (pagina calendario di Yahoo), solo per le azioni. Le date restano con i
dati grezzi in data/raw/_earnings/<id>.json (cache di Actions, non committate): nel repo e
nella dashboard finiscono solo grandezze derivate (prossima data, reazioni in %).

Giorno di reazione (il primo prezzo che "vede" i conti):
- annuncio prima dell'apertura (prima delle 9:30 di New York) → seduta dello stesso giorno;
- annuncio a mercato aperto o dopo la chiusura → seduta successiva;
- orario sconosciuto (Yahoo indica mezzanotte) → finestra di due sedute (giorno stesso e
  successivo), datata all'ultima: nessuna informazione usata prima che sia nota.
Reazione = log-rendimento dalla chiusura prima della finestra alla chiusura dell'ultima
seduta della finestra (comprende il salto di apertura).

Le date delle trimestrali sono annunciate settimane prima: usarle in anticipo non è look-ahead.

Uso:
    python -m pipeline.earnings [--force] [--tickers nvda,aapl]
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import math
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from pipeline.ingest import (DEFAULT_DATA_DIR, DEFAULT_UNIVERSE, ConfigError, load_universe, read_json,
                             write_json)

log = logging.getLogger("financenow.earnings")

NY = ZoneInfo("America/New_York")
SUBDIR = "_earnings"
REFRESH_DAYS = 7          # riscarica al massimo una volta a settimana...
LIMIT = 100               # ...circa 25 anni di trimestri (massimo consentito da Yahoo)
PAUSE_SECONDS = 2.0
OPEN_TIME = dt.time(9, 30)
HISTORY_ROWS = 12         # reazioni recenti esportate per la dashboard
SOON_SESSIONS = 10        # "trimestrale in arrivo"


# ---------------------------------------------------------------------------
# Download e file
# ---------------------------------------------------------------------------

def path_for(data_dir: Path, inst_id: str) -> Path:
    return data_dir / SUBDIR / f"{inst_id}.json"


def fetch(symbol: str, limit: int = LIMIT) -> list[dict]:
    """Date passate e future da Yahoo. Solleva eccezione se la fonte non risponde."""
    import yfinance as yf

    df = yf.Ticker(symbol).get_earnings_dates(limit=limit)
    if df is None or df.empty:
        raise RuntimeError(f"{symbol}: nessuna data di trimestrale (fonte vuota o bloccata)")
    out = []
    for ts, row in df.iterrows():
        ts = pd.Timestamp(ts)
        ts = ts.tz_localize(NY) if ts.tzinfo is None else ts.tz_convert(NY)

        def num(key):
            v = row.get(key)
            return float(v) if v is not None and pd.notna(v) else None

        out.append({"ts": ts.isoformat(), "eps_estimate": num("EPS Estimate"),
                    "eps_reported": num("Reported EPS"), "surprise_pct": num("Surprise(%)")})
    return sorted(out, key=lambda r: r["ts"])


def needs_refresh(stored: dict, today: dt.date, force: bool = False) -> bool:
    if force or not stored.get("dates"):
        return True
    fetched = dt.datetime.fromisoformat(stored["fetched_utc"]).date()
    if (today - fetched).days >= REFRESH_DAYS:
        return True
    # La prossima data nota è passata: serve il risultato e la data successiva.
    future = [r for r in stored["dates"] if pd.Timestamp(r["ts"]).date() >= fetched]
    return bool(future) and pd.Timestamp(future[0]["ts"]).date() < today


def update(instruments, data_dir: Path, force: bool = False, fetcher=fetch, pause: float = PAUSE_SECONDS) -> dict:
    """Aggiorna i file delle date. Un titolo che fallisce tiene le date già salvate."""
    today = dt.datetime.now(dt.timezone.utc).date()
    report = {"updated": [], "kept": [], "failed": []}
    first = True
    for inst in instruments:
        if inst.asset_class != "stock" or not inst.active:
            continue
        p = path_for(data_dir, inst.id)
        stored = read_json(p)
        if not needs_refresh(stored, today, force):
            report["kept"].append(inst.id)
            continue
        symbol = inst.symbol_for("yahoo")
        if not symbol:
            continue
        if not first:
            time.sleep(pause)
        first = False
        try:
            dates = fetcher(symbol)
        except Exception as e:  # yfinance solleva tipi eterogenei
            log.warning("%s: trimestrali non aggiornate (%s: %s)", inst.id, type(e).__name__, e)
            report["failed"].append(inst.id)
            continue
        write_json({"id": inst.id, "symbol": symbol, "source": "yahoo",
                    "fetched_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                    "dates": dates}, p)
        report["updated"].append(inst.id)
    return report


def load(data_dir: Path, inst_id: str) -> list[pd.Timestamp]:
    """Orari di pubblicazione (fuso di New York), ordinati; lista vuota se non disponibili."""
    stored = read_json(path_for(data_dir, inst_id))
    out = []
    for r in stored.get("dates", []):
        ts = pd.Timestamp(r["ts"])
        out.append(ts.tz_localize(NY) if ts.tzinfo is None else ts.tz_convert(NY))
    return sorted(set(out))


# ---------------------------------------------------------------------------
# Finestre di reazione
# ---------------------------------------------------------------------------

def timing(ts: pd.Timestamp) -> str:
    local = ts.tz_convert(NY)
    if local.hour == 0 and local.minute == 0:
        return "n.d."
    return "prima dell'apertura" if local.time() < OPEN_TIME else "dopo la chiusura"


def _next_weekday(d: dt.date, strictly_after: bool) -> dt.date:
    d = d + dt.timedelta(days=1) if strictly_after else d
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


def reaction_windows(announcements: list[pd.Timestamp], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Una riga per annuncio: prima e ultima seduta della reazione (NaT se ancora nel futuro).

    Per le date future la seduta di reazione è stimata sui giorni feriali (festività ignorate).
    """
    rows = []
    days = index.normalize()
    for ts in announcements:
        local = ts.tz_convert(NY)
        d = local.date()
        kind = timing(ts)
        after = kind == "dopo la chiusura"
        span = 2 if kind == "n.d." else 1
        pos = int(days.searchsorted(pd.Timestamp(d), side="right" if after else "left"))
        if pos + span - 1 < len(days):
            first, last = days[pos], days[pos + span - 1]
            expected = first.date()
        else:
            first = last = pd.NaT
            expected = _next_weekday(d, after)
        rows.append({"announced": local, "timing": kind, "first": first, "last": last,
                     "expected_session": expected})
    return pd.DataFrame(rows, columns=["announced", "timing", "first", "last", "expected_session"])


def reactions(windows: pd.DataFrame, close: pd.Series, bench_close: pd.Series | None = None) -> pd.DataFrame:
    """Reazione (log-rendimento) per le finestre già concluse, anche rispetto al benchmark."""
    done = windows.dropna(subset=["first", "last"]).copy()
    if done.empty:
        return done.assign(reaction=[], reaction_vs_bench=[])
    idx = close.index.normalize()
    lc = np.log(close.to_numpy())
    pf = idx.get_indexer(pd.DatetimeIndex(done["first"]))
    pl = idx.get_indexer(pd.DatetimeIndex(done["last"]))
    ok = pf > 0
    done = done[ok]
    pf, pl = pf[ok], pl[ok]
    done["reaction"] = lc[pl] - lc[pf - 1]
    if bench_close is not None:
        lb = np.log(bench_close.reindex(close.index).to_numpy())
        done["reaction_vs_bench"] = done["reaction"] - (lb[pl] - lb[pf - 1])
    else:
        done["reaction_vs_bench"] = np.nan
    return done.reset_index(drop=True)


def window_dates(windows: pd.DataFrame, index: pd.DatetimeIndex) -> set[pd.Timestamp]:
    """Tutte le sedute che cadono in una finestra di reazione (per le anomalie "spiegate")."""
    out = set()
    days = index.normalize()
    for r in windows.dropna(subset=["first", "last"]).itertuples():
        a, b = days.searchsorted(r.first), days.searchsorted(r.last)
        out.update(index[a:b + 1])
    return out


# ---------------------------------------------------------------------------
# Riassunto per lo strumento
# ---------------------------------------------------------------------------

def sessions_between(a: dt.date, b: dt.date) -> int:
    """Sedute (giorni feriali, festività ignorate) dopo `a` fino a `b` compreso."""
    return int(np.busday_count(a + dt.timedelta(days=1), b + dt.timedelta(days=1))) if b > a else 0


def summary(announcements: list[pd.Timestamp], close: pd.Series, bench_close: pd.Series | None = None) -> dict:
    """Prossima trimestrale, ultima reazione e reazione tipica (tutto in termini relativi)."""
    if not announcements or close is None or close.empty:
        return {"available": False}
    win = reaction_windows(announcements, close.index)
    rx = reactions(win, close, bench_close)
    last_day = close.index[-1].date()
    upcoming = win[win["first"].isna()]
    nxt = None
    if len(upcoming):
        u = upcoming.iloc[0]
        nxt = {"date": u["announced"].date().isoformat(), "timing": u["timing"],
               "reaction_session": u["expected_session"].isoformat(),
               "sessions_until": sessions_between(last_day, u["expected_session"])}
    ret = np.log(close).diff().dropna()
    out = {"available": True, "next": nxt, "n_reactions": int(len(rx))}
    if len(rx):
        abs_rx = rx["reaction"].abs()
        typical_day = float(ret.abs().median()) if len(ret) else None
        out.update({
            "typical_move": float(abs_rx.median()),
            "typical_day_move": typical_day,
            "move_ratio": float(abs_rx.median() / typical_day) if typical_day else None,
            "share_up": float((rx["reaction"] > 0).mean()),
            "last": {"date": rx["announced"].iloc[-1].date().isoformat(),
                     "session": rx["last"].iloc[-1].date().isoformat(),
                     "reaction": float(rx["reaction"].iloc[-1]),
                     "reaction_vs_bench": _f(rx["reaction_vs_bench"].iloc[-1]),
                     "sessions_ago": int(len(close.index) - 1 - close.index.normalize().get_loc(rx["last"].iloc[-1]))},
            "history": [{"date": r.announced.date().isoformat(), "timing": r.timing,
                         "session": r.last.date().isoformat(), "reaction": float(r.reaction),
                         "reaction_vs_bench": _f(r.reaction_vs_bench)}
                        for r in rx.tail(HISTORY_ROWS).itertuples()],
        })
    return out


def _f(v):
    return float(v) if v is not None and math.isfinite(float(v)) else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pipeline.earnings", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", help="id separati da virgola (default: tutte le azioni dell'universo)")
    ap.add_argument("--force", action="store_true", help="riscarica anche se i dati sono recenti")
    ap.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    try:
        instruments = load_universe(args.universe)
    except ConfigError as e:
        log.error("configurazione non valida: %s", e)
        return 2
    if args.tickers:
        wanted = {t.strip() for t in args.tickers.split(",") if t.strip()}
        instruments = [i for i in instruments if i.id in wanted]
    report = update(instruments, args.data_dir, args.force)
    log.info("trimestrali: aggiornate %s, già recenti %s, fallite %s",
             report["updated"] or "-", report["kept"] or "-", report["failed"] or "-")
    write_json({"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **report},
               args.data_dir / SUBDIR / "_report.json")
    # Fallire solo se nessun titolo ha date utilizzabili (né nuove né già salvate).
    stocks = [i for i in instruments if i.asset_class == "stock" and i.active]
    usable = [i for i in stocks if load(args.data_dir, i.id)]
    return 0 if usable or not stocks else 1


if __name__ == "__main__":
    sys.exit(main())
