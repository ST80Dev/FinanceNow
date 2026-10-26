"""Test offline delle trimestrali: finestre di reazione, reazioni, riassunto, aggiornamento."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from pipeline import earnings
from pipeline.ingest import load_universe, read_json

NY = earnings.NY


def ts(s):
    return pd.Timestamp(s).tz_localize(NY)


IDX = pd.bdate_range("2024-01-01", "2024-03-29")


def test_timing():
    assert earnings.timing(ts("2024-01-10 06:00")) == "prima dell'apertura"
    assert earnings.timing(ts("2024-01-10 16:00")) == "dopo la chiusura"
    assert earnings.timing(ts("2024-01-10 00:00")) == "n.d."


def test_finestre_di_reazione():
    w = earnings.reaction_windows([ts("2024-01-10 06:00"), ts("2024-01-10 16:00"),
                                   ts("2024-01-12 00:00"), ts("2024-05-02 16:00")], IDX)
    # prima dell'apertura -> stesso giorno; dopo la chiusura -> seduta successiva
    assert w["first"][0] == pd.Timestamp("2024-01-10") and w["last"][0] == pd.Timestamp("2024-01-10")
    assert w["first"][1] == pd.Timestamp("2024-01-11")
    # orario sconosciuto (venerdì) -> venerdì e lunedì, datata al lunedì
    assert w["first"][2] == pd.Timestamp("2024-01-12") and w["last"][2] == pd.Timestamp("2024-01-15")
    # futura: nessuna seduta, giorno di reazione stimato (venerdì 3 maggio)
    assert pd.isna(w["first"][3]) and w["expected_session"][3] == dt.date(2024, 5, 3)


def test_reazione_comprende_il_salto_e_vs_benchmark():
    close = pd.Series(100.0, index=IDX)
    close[pd.Timestamp("2024-01-11"):] = 110.0          # salto il giorno dopo i conti
    bench = pd.Series(100.0, index=IDX)
    bench[pd.Timestamp("2024-01-11"):] = 102.0
    w = earnings.reaction_windows([ts("2024-01-10 16:00")], IDX)
    rx = earnings.reactions(w, close, bench)
    assert rx["reaction"][0] == pytest.approx(math.log(1.10))
    assert rx["reaction_vs_bench"][0] == pytest.approx(math.log(1.10) - math.log(1.02))
    assert earnings.window_dates(w, IDX) == {pd.Timestamp("2024-01-11")}


def test_riassunto():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", "2024-03-29")
    r = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    ann = []
    for q in pd.date_range("2020-02-01", "2024-02-01", freq="QS-FEB"):
        day = idx[idx.searchsorted(q)]
        r[idx[idx.searchsorted(day) + 1]] += 0.08 * (1 if day.month % 2 else -1)   # reazione forte
        ann.append(ts(day.strftime("%Y-%m-%d") + " 16:00"))
    ann.append(ts("2024-04-25 16:00"))                   # prossima
    close = 100 * np.exp(r.cumsum())
    s = earnings.summary(ann, close)
    assert s["available"] and s["n_reactions"] == len(ann) - 1
    assert s["next"]["date"] == "2024-04-25" and s["next"]["reaction_session"] == "2024-04-26"
    assert s["next"]["sessions_until"] == earnings.sessions_between(dt.date(2024, 3, 29), dt.date(2024, 4, 26))
    assert s["move_ratio"] > 4                           # i giorni dei conti si muovono molto di più
    assert len(s["history"]) == earnings.HISTORY_ROWS
    assert earnings.summary([], close) == {"available": False}


def test_needs_refresh():
    today = dt.date(2024, 3, 10)
    fresh = {"fetched_utc": "2024-03-08T10:00:00+00:00", "dates": [{"ts": "2024-04-25T16:00:00-04:00"}]}
    assert not earnings.needs_refresh(fresh, today)
    assert earnings.needs_refresh(fresh, dt.date(2024, 3, 20))                  # più vecchio di 7 giorni
    passed = {"fetched_utc": "2024-03-08T10:00:00+00:00", "dates": [{"ts": "2024-03-09T16:00:00-05:00"}]}
    assert earnings.needs_refresh(passed, today)                                # la prossima data è passata
    assert earnings.needs_refresh({}, today) and earnings.needs_refresh(fresh, today, force=True)


def test_update_con_fonte_finta(tmp_path):
    u = tmp_path / "u.yml"
    u.write_text("defaults: {start: 2010-01-01, sources: [yahoo], calendar: XNYS, benchmark: null}\n"
                 "instruments:\n"
                 "  - {id: idx, name: Indice, asset_class: index, yahoo: IDX}\n"
                 "  - {id: aaa, name: AAA, asset_class: stock, yahoo: AAA}\n"
                 "  - {id: bbb, name: BBB, asset_class: stock, yahoo: BBB}\n")
    calls = []

    def fake(symbol):
        calls.append(symbol)
        if symbol == "BBB":
            raise RuntimeError("bloccato")
        return [{"ts": "2024-01-10T16:00:00-05:00", "eps_estimate": 1.0, "eps_reported": 1.1, "surprise_pct": 10.0}]

    rep = earnings.update(load_universe(u), tmp_path, fetcher=fake, pause=0)
    assert rep == {"updated": ["aaa"], "kept": [], "failed": ["bbb"]}
    assert calls == ["AAA", "BBB"]                                              # gli indici non hanno trimestrali
    assert earnings.load(tmp_path, "aaa") == [ts("2024-01-10 16:00")]
    assert read_json(earnings.path_for(tmp_path, "aaa"))["symbol"] == "AAA"
    assert earnings.load(tmp_path, "bbb") == []
