"""Test offline di pipeline.ingest: nessuna chiamata di rete, sorgenti finte."""

from __future__ import annotations

import datetime as dt
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline import ingest
from pipeline.ingest import (
    ConfigError,
    Ingestor,
    Instrument,
    NoDataError,
    Source,
    SourceDisabled,
    StooqSource,
    TransientError,
    last_complete_session,
    load_universe,
    normalize_frame,
    validate,
)

ROOT = Path(__file__).resolve().parents[1]
# Lunedì 26/10/2026, 22:00 UTC = 18:00 a New York: la seduta del 26 è chiusa.
AFTER_CLOSE = dt.datetime(2026, 10, 26, 22, 0, tzinfo=dt.timezone.utc)


# ---------------------------------------------------------------------------
# Utilità
# ---------------------------------------------------------------------------

def make_ohlcv(start: str, end: str, base: float = 100.0, factor: float = 1.0) -> pd.DataFrame:
    """OHLCV sintetico sui giorni feriali; `factor` simula un nuovo aggiustamento."""
    idx = pd.bdate_range(start, end, name="date")
    close = base * np.exp(np.cumsum(np.full(len(idx), 0.001))) * factor
    return pd.DataFrame(
        {"open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1_000.0},
        index=idx,
    )


class FakeSource(Source):
    def __init__(self, name: str, frames: dict[str, pd.DataFrame] | None = None,
                 errors: dict[str, list[Exception]] | None = None):
        self.name = name
        self.frames = frames or {}
        self.errors = errors or {}   # simbolo -> eccezioni da sollevare, in ordine
        self.calls: list[tuple[str, dt.date, dt.date]] = []

    def fetch(self, symbol, start, end):
        self.calls.append((symbol, start, end))
        queue = self.errors.get(symbol)
        if queue:
            raise queue.pop(0)
        if symbol not in self.frames:
            raise NoDataError("nessun dato")
        df = self.frames[symbol]
        return df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]


def inst(id_: str, sources=("yahoo",), start=dt.date(2026, 1, 1), **symbols) -> Instrument:
    return Instrument(
        id=id_, name=id_.upper(), asset_class="stock",
        symbols=symbols or {"yahoo": id_.upper()}, sources=tuple(sources),
        start=start, calendar="XNYS", currency="USD", region="US",
        benchmark=None, active=True,
    )


def make_ingestor(tmp_path, instruments, sources, now=AFTER_CLOSE, **kw) -> Ingestor:
    kw.setdefault("pause", 0)
    return Ingestor(instruments, sources, tmp_path, sleep=lambda s: None, now=lambda: now, **kw)


# ---------------------------------------------------------------------------
# Universo
# ---------------------------------------------------------------------------

def test_universe_repo_valido():
    instruments = load_universe(ROOT / "config" / "universe.yml")
    ids = [i.id for i in instruments]
    assert "ndx" in ids and "nvda" in ids and "amd" in ids
    assert "spcx" not in ids
    nvda = next(i for i in instruments if i.id == "nvda")
    assert nvda.start == dt.date(2010, 1, 1)
    assert nvda.sources[0] == "yahoo"
    assert nvda.benchmark == "ndx"
    assert nvda.symbol_for("yahoo") == "NVDA" and nvda.symbol_for("stooq") == "nvda.us"


def _write_universe(tmp_path, body: str) -> Path:
    p = tmp_path / "u.yml"
    p.write_text(textwrap.dedent(body))
    return p


def test_universe_id_duplicati(tmp_path):
    p = _write_universe(tmp_path, """
        defaults: {start: 2010-01-01, sources: [yahoo], calendar: XNYS}
        instruments:
          - {id: a, name: A, asset_class: stock, yahoo: A}
          - {id: a, name: A2, asset_class: stock, yahoo: A2}
    """)
    with pytest.raises(ConfigError, match="duplicati"):
        load_universe(p)


def test_universe_benchmark_inesistente(tmp_path):
    p = _write_universe(tmp_path, """
        defaults: {start: 2010-01-01, sources: [yahoo], calendar: XNYS, benchmark: zzz}
        instruments:
          - {id: a, name: A, asset_class: stock, yahoo: A}
    """)
    with pytest.raises(ConfigError, match="benchmark"):
        load_universe(p)


def test_universe_senza_simbolo_per_le_sorgenti(tmp_path):
    p = _write_universe(tmp_path, """
        defaults: {start: 2010-01-01, sources: [yahoo], calendar: XNYS}
        instruments:
          - {id: a, name: A, asset_class: stock, stooq: a.us}
    """)
    with pytest.raises(ConfigError, match="nessun simbolo"):
        load_universe(p)


# ---------------------------------------------------------------------------
# Sessione e validazione
# ---------------------------------------------------------------------------

def test_last_complete_session():
    # 13:32 UTC = 09:32 New York: la seduta del 26 è appena iniziata.
    during = dt.datetime(2026, 10, 26, 13, 32, tzinfo=dt.timezone.utc)
    assert last_complete_session("XNYS", during) == dt.date(2026, 10, 25)
    assert last_complete_session("XNYS", AFTER_CLOSE) == dt.date(2026, 10, 26)


def test_validate_pulisce_e_segnala():
    df = make_ohlcv("2026-10-01", "2026-10-27")
    df = pd.concat([df, df.iloc[[2]]])                          # data duplicata
    df.iloc[3, df.columns.get_loc("close")] = -1.0               # prezzo negativo
    df.iloc[4, df.columns.get_loc("high")] = df.iloc[4]["close"] * 0.9   # barra impossibile
    df.iloc[5, df.columns.get_loc("high")] = df.iloc[5]["close"] * 0.9999  # arrotondamento
    df.iloc[6, df.columns.get_loc("volume")] = -5
    clean, warnings = validate(df, dt.date(2026, 10, 26))

    assert clean.index.is_unique and clean.index.is_monotonic_increasing
    assert clean.index[-1] == pd.Timestamp("2026-10-26")        # 27 scartato
    assert df.index[3] not in clean.index and df.index[4] not in clean.index
    assert (clean["high"] >= clean[["open", "close"]].max(axis=1)).all()
    assert np.isnan(clean.loc[df.index[6], "volume"])
    text = " ".join(warnings)
    assert "seduta non conclusa" in text and "<= 0" in text and "incoerenti" in text


def test_validate_segnala_salti_senza_correggerli():
    df = make_ohlcv("2026-10-01", "2026-10-20")
    df.iloc[5:, :4] *= 1.5
    clean, warnings = validate(df, dt.date(2026, 10, 20))
    assert len(clean) == len(df)
    assert any("salti" in w for w in warnings)


def test_normalize_frame_yahoo_tz_aware():
    idx = pd.DatetimeIndex(["2026-10-22 00:00", "2026-10-23 00:00"]).tz_localize("America/New_York")
    raw = pd.DataFrame({"Open": [1, 2], "High": [1, 2], "Low": [1, 2], "Close": [1, 2],
                        "Volume": [10, 20]}, index=idx)
    df = normalize_frame(raw)
    assert list(df.columns) == ingest.COLUMNS
    assert df.index.tz is None
    assert list(df.index) == [pd.Timestamp("2026-10-22"), pd.Timestamp("2026-10-23")]


# ---------------------------------------------------------------------------
# Stooq (parsing, senza rete)
# ---------------------------------------------------------------------------

def test_stooq_parse_csv():
    df = StooqSource.parse("Date,Open,High,Low,Close,Volume\n2026-10-22,1,2,0.5,1.5,100\n")
    assert df.loc["2026-10-22", "close"] == 1.5


def test_stooq_parse_errori():
    with pytest.raises(SourceDisabled, match="quota"):
        StooqSource.parse("Exceeded the daily hits limit")
    with pytest.raises(NoDataError):
        StooqSource.parse("No data")
    with pytest.raises(SourceDisabled, match="anti-bot"):
        StooqSource.parse("<!DOCTYPE html><html><body>This site requires JavaScript</body></html>")
    with pytest.raises(TransientError):
        StooqSource.parse("   ")


def test_stooq_senza_chiave_non_disponibile():
    ok, reason = StooqSource(api_key=None).available()
    assert not ok and "STOOQ_API_KEY" in reason


def test_stooq_non_espone_la_chiave_negli_errori():
    class BoomSession:
        def get(self, url, params, **kw):
            raise ConnectionError(f"failed {url}?apikey={params['apikey']}")

    src = StooqSource(api_key="SEGRETO123", session=BoomSession())
    with pytest.raises(TransientError) as exc:
        src.fetch("nvda.us", dt.date(2026, 1, 1), dt.date(2026, 1, 2))
    assert "SEGRETO123" not in str(exc.value)


# ---------------------------------------------------------------------------
# Ingest end-to-end con sorgenti finte
# ---------------------------------------------------------------------------

def test_backfill_poi_incrementale(tmp_path):
    full = make_ohlcv("2026-01-01", "2026-10-26")
    yahoo = FakeSource("yahoo", {"AAA": full[full.index <= "2026-10-20"]})
    ing = make_ingestor(tmp_path, [inst("aaa")], {"yahoo": yahoo},
                        now=dt.datetime(2026, 10, 20, 22, tzinfo=dt.timezone.utc))
    report = ing.run("daily")
    assert report["instruments"]["aaa"]["status"] == "backfilled"
    assert (tmp_path / "aaa.parquet").exists()

    # Run successivo: arrivano nuove sedute, la sovrapposizione combacia.
    yahoo.frames["AAA"] = full
    ing2 = make_ingestor(tmp_path, [inst("aaa")], {"yahoo": yahoo})
    report = ing2.run("daily")
    r = report["instruments"]["aaa"]
    assert r["status"] == "updated" and r["new_rows"] == 4
    assert r["last_date"] == "2026-10-26"
    saved = pd.read_parquet(tmp_path / "aaa.parquet").set_index("date")
    assert len(saved) == len(full) and saved.index.is_unique
    # L'incrementale chiede solo una finestra recente, non tutto lo storico.
    assert yahoo.calls[-1][1] > dt.date(2026, 9, 1)

    # Terzo run nello stesso giorno: niente da fare, nessuna richiesta.
    n_calls = len(yahoo.calls)
    report = make_ingestor(tmp_path, [inst("aaa")], {"yahoo": yahoo}).run("daily")
    assert report["instruments"]["aaa"]["status"] == "up_to_date"
    assert len(yahoo.calls) == n_calls


def test_nuovo_aggiustamento_riscarica_tutto(tmp_path):
    old = make_ohlcv("2026-01-01", "2026-10-20")
    yahoo = FakeSource("yahoo", {"AAA": old})
    make_ingestor(tmp_path, [inst("aaa")], {"yahoo": yahoo},
                  now=dt.datetime(2026, 10, 20, 22, tzinfo=dt.timezone.utc)).run("daily")

    # Dividendo: tutti i prezzi passati vengono riscalati dello 0,5%.
    yahoo.frames["AAA"] = make_ohlcv("2026-01-01", "2026-10-26", factor=0.995)
    report = make_ingestor(tmp_path, [inst("aaa")], {"yahoo": yahoo}).run("daily")
    r = report["instruments"]["aaa"]
    assert r["status"] == "refreshed"
    saved = pd.read_parquet(tmp_path / "aaa.parquet").set_index("date")
    pd.testing.assert_series_equal(saved["close"], yahoo.frames["AAA"]["close"], check_freq=False)


def test_fallback_alla_seconda_sorgente_nel_backfill(tmp_path):
    data = make_ohlcv("2026-01-01", "2026-10-26")
    yahoo = FakeSource("yahoo", errors={"AAA": [TransientError("bloccato")] * 3})
    stooq = FakeSource("stooq", {"aaa.us": data})
    i = inst("aaa", sources=("yahoo", "stooq"), yahoo="AAA", stooq="aaa.us")
    report = make_ingestor(tmp_path, [i], {"yahoo": yahoo, "stooq": stooq}).run("daily")
    r = report["instruments"]["aaa"]
    assert r["status"] == "backfilled" and r["source"] == "stooq"
    assert len(yahoo.calls) == 3   # tre tentativi prima di passare alla riserva


def test_sorgente_appiccicosa_e_stale_se_fallisce(tmp_path):
    data = make_ohlcv("2026-01-01", "2026-10-20")
    yahoo = FakeSource("yahoo", {"AAA": data, "BBB": data})
    stooq = FakeSource("stooq", {"aaa.us": make_ohlcv("2026-01-01", "2026-10-26")})
    i = inst("aaa", sources=("yahoo", "stooq"), yahoo="AAA", stooq="aaa.us")
    j = inst("bbb")
    make_ingestor(tmp_path, [i, j], {"yahoo": yahoo, "stooq": stooq},
                  now=dt.datetime(2026, 10, 20, 22, tzinfo=dt.timezone.utc)).run("daily")

    yahoo.errors["AAA"] = [TransientError("bloccato")] * 3
    yahoo.frames["BBB"] = make_ohlcv("2026-01-01", "2026-10-26")
    report = make_ingestor(tmp_path, [i, j], {"yahoo": yahoo, "stooq": stooq}).run("daily")
    assert report["instruments"]["aaa"]["status"] == "stale"      # niente mix con Stooq
    assert report["instruments"]["bbb"]["status"] == "updated"    # il run continua
    assert report["stale"] == ["aaa"]
    assert stooq.calls == []
    saved = pd.read_parquet(tmp_path / "aaa.parquet")
    assert len(saved) == len(data)                                 # storico intatto


def test_ticker_senza_dati_non_blocca_il_run(tmp_path):
    yahoo = FakeSource("yahoo", {"BBB": make_ohlcv("2026-01-01", "2026-10-26")},
                       errors={"AAA": [NoDataError("simbolo inesistente")]})
    report = make_ingestor(tmp_path, [inst("aaa"), inst("bbb")], {"yahoo": yahoo}).run("daily")
    assert report["missing"] == ["aaa"]
    assert report["instruments"]["bbb"]["status"] == "backfilled"
    assert ingest.read_json(tmp_path / "_report.json")["missing"] == ["aaa"]


def test_budget_rimanda_e_il_run_successivo_riprende(tmp_path):
    data = make_ohlcv("2026-01-01", "2026-10-26")
    yahoo = FakeSource("yahoo", {s: data for s in ("AAA", "BBB", "CCC")})
    instruments = [inst("aaa"), inst("bbb"), inst("ccc")]
    report = make_ingestor(tmp_path, instruments, {"yahoo": yahoo}, max_requests=2).run("backfill")
    assert report["summary"] == {"backfilled": 2, "deferred": 1}
    assert report["deferred"] == ["ccc"]

    report = make_ingestor(tmp_path, instruments, {"yahoo": yahoo}, max_requests=2).run("backfill")
    assert report["instruments"]["ccc"]["status"] == "backfilled"
    assert report["instruments"]["aaa"]["status"] == "skipped"


def test_sorgente_disattivata_dopo_fallimenti_consecutivi(tmp_path):
    yahoo = FakeSource("yahoo", errors={s: [TransientError("bloccato")] * 3
                                        for s in ("AAA", "BBB", "CCC", "DDD")})
    instruments = [inst(x) for x in ("aaa", "bbb", "ccc", "ddd")]
    report = make_ingestor(tmp_path, instruments, {"yahoo": yahoo}).run("daily")
    assert "yahoo" in report["disabled_sources"]
    assert len(yahoo.calls) == 9           # 3 ticker x 3 tentativi, il quarto non chiama
    assert report["missing"] == ["aaa", "bbb", "ccc", "ddd"]


def test_quota_disattiva_la_sorgente_per_il_run(tmp_path):
    stooq = FakeSource("stooq", errors={"aaa.us": [SourceDisabled("quota esaurita")]},
                       frames={"bbb.us": make_ohlcv("2026-01-01", "2026-10-26")})
    instruments = [inst("aaa", sources=("stooq",), stooq="aaa.us"),
                   inst("bbb", sources=("stooq",), stooq="bbb.us")]
    report = make_ingestor(tmp_path, instruments, {"stooq": stooq}).run("daily")
    assert report["disabled_sources"]["stooq"] == "quota esaurita"
    assert report["missing"] == ["aaa", "bbb"]
    assert len(stooq.calls) == 1


def test_stato_salvato_dopo_ogni_ticker(tmp_path):
    data = make_ohlcv("2026-01-01", "2026-10-26")

    class CrashSource(FakeSource):
        def fetch(self, symbol, start, end):
            if symbol == "BBB":
                raise KeyboardInterrupt  # run interrotto a metà
            return super().fetch(symbol, start, end)

    src = CrashSource("yahoo", {"AAA": data, "BBB": data})
    with pytest.raises(KeyboardInterrupt):
        make_ingestor(tmp_path, [inst("aaa"), inst("bbb")], {"yahoo": src}).run("backfill")
    state = ingest.read_json(tmp_path / "_state.json")
    assert state["instruments"]["aaa"]["status"] == "backfilled"
    assert "bbb" not in state["instruments"]


def test_cambio_simbolo_forza_backfill(tmp_path):
    data = make_ohlcv("2026-01-01", "2026-10-26")
    yahoo = FakeSource("yahoo", {"AAA": data, "AAA2": data})
    make_ingestor(tmp_path, [inst("aaa")], {"yahoo": yahoo}).run("daily")
    report = make_ingestor(tmp_path, [inst("aaa", yahoo="AAA2")], {"yahoo": yahoo}).run("daily")
    assert report["instruments"]["aaa"]["status"] == "backfilled"
    assert yahoo.calls[-1][0] == "AAA2"


def test_ticker_inesistente_in_cli(tmp_path):
    rc = ingest.main(["daily", "--tickers", "zzz", "--data-dir", str(tmp_path)])
    assert rc == 2
