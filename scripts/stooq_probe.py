"""Sonda Stooq: chiama l'endpoint CSV storico senza API key e mostra cosa risponde.

Solo libreria standard, nessuna dipendenza.

Uso:
    python scripts/stooq_probe.py                 # prova spy.us
    python scripts/stooq_probe.py spy.us ^spx xlk.us
    python scripts/stooq_probe.py --raw spy.us    # stampa anche il corpo completo
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE_URL = "https://stooq.com/q/d/l/"
USER_AGENT = "Mozilla/5.0 (FinanceNow stooq_probe)"
EXPECTED_HEADER = ["Date", "Open", "High", "Low", "Close", "Volume"]


def fetch(symbol: str, timeout: float = 30.0) -> tuple[int | None, dict[str, str], bytes, str]:
    """Restituisce (status, headers, body, url_finale). status=None se la richiesta non parte."""
    url = f"{BASE_URL}?{urllib.parse.urlencode({'s': symbol, 'i': 'd'})}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read(), resp.geturl()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e.read() or b"", url
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return None, {}, str(e).encode(), url


def diagnose(body: str) -> str:
    """Classifica la risposta in base al contenuto."""
    text = body.strip()
    low = text.lower()
    if not text:
        return "VUOTA: nessun contenuto"
    if "exceeded the daily hits limit" in low:
        return "QUOTA: limite giornaliero di download superato"
    if low.startswith("no data") or low == "brak danych":
        return "NO DATA: simbolo inesistente o senza storico"
    if "apikey" in low or "api key" in low or "api_key" in low:
        return "API KEY: la risposta menziona una chiave API (probabilmente obbligatoria)"
    if low.startswith("<!doctype") or low.startswith("<html") or "<html" in low[:500]:
        return "HTML: pagina web invece del CSV (captcha, consenso cookie, login o blocco)"
    first_line = text.splitlines()[0].split(",")
    if first_line[:5] == EXPECTED_HEADER[:5]:
        return "CSV OK"
    return "SCONOSCIUTA: formato non riconosciuto"


def summarize_csv(body: str) -> None:
    rows = list(csv.reader(io.StringIO(body.strip())))
    header, data = rows[0], rows[1:]
    print(f"  colonne     : {header}")
    print(f"  righe       : {len(data)}")
    if not data:
        return
    print(f"  prima data  : {data[0][0]}")
    print(f"  ultima data : {data[-1][0]}")
    print("  prime 3 righe:")
    for r in data[:3]:
        print("    " + ",".join(r))
    print("  ultime 3 righe:")
    for r in data[-3:]:
        print("    " + ",".join(r))


def probe(symbol: str, raw: bool) -> str:
    print("=" * 72)
    print(f"Simbolo: {symbol}")
    t0 = time.monotonic()
    status, headers, body_bytes, url = fetch(symbol)
    elapsed = time.monotonic() - t0
    print(f"URL         : {url}")
    print(f"Tempo       : {elapsed:.2f} s")

    if status is None:
        print(f"ERRORE DI RETE: {body_bytes.decode(errors='replace')}")
        return "RETE"

    print(f"HTTP status : {status}")
    for h in ("Content-Type", "Content-Disposition", "Content-Length", "Set-Cookie"):
        if h in headers:
            print(f"{h:<12}: {headers[h]}")
    body = body_bytes.decode("utf-8", errors="replace")
    print(f"Byte ricevuti: {len(body_bytes)}")

    verdict = diagnose(body)
    print(f"Diagnosi    : {verdict}")

    if verdict == "CSV OK":
        summarize_csv(body)
    if raw or verdict != "CSV OK":
        limit = None if raw else 1500
        print("--- corpo della risposta" + ("" if raw else " (primi 1500 caratteri)") + " ---")
        print(body[:limit])
        print("--- fine ---")
    return verdict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbols", nargs="*", default=["spy.us"], help="simboli Stooq (default: spy.us)")
    ap.add_argument("--raw", action="store_true", help="stampa il corpo completo della risposta")
    ap.add_argument("--pause", type=float, default=3.0, help="secondi di pausa tra un simbolo e l'altro")
    args = ap.parse_args()

    results = {}
    for i, sym in enumerate(args.symbols):
        if i:
            time.sleep(args.pause)
        results[sym] = probe(sym, args.raw)

    print("=" * 72)
    print("Riepilogo:")
    for sym, verdict in results.items():
        print(f"  {sym:<12} {verdict}")
    return 0 if all(v == "CSV OK" for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
