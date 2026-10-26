"""Riepilogo Markdown di un run (per il "job summary" di GitHub Actions).

Uso:
    python -m pipeline.summary >> "$GITHUB_STEP_SUMMARY"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.analyze import DEFAULT_RESULTS_DIR


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _fmt(x, pattern="{:.2f}"):
    return pattern.format(x) if isinstance(x, (int, float)) else "–"


def render(results_dir: Path) -> str:
    lines = ["## FinanceNow — riepilogo del run", ""]

    ingest = _load(results_dir / "status" / "ingest.json")
    if ingest:
        lines += [f"**Download** ({ingest.get('mode')}): {ingest.get('summary')} — "
                  f"richieste {ingest.get('requests')}/{ingest.get('max_requests')}"]
        for key, label in (("missing", "Senza dati"), ("stale", "Non aggiornati"), ("deferred", "Rimandati")):
            if ingest.get(key):
                lines.append(f"- ⚠️ {label}: {', '.join(ingest[key])}")
        if ingest.get("disabled_sources"):
            lines.append(f"- Sorgenti non attive: {ingest['disabled_sources']}")
        lines.append("")

    rows = []
    for p in sorted((results_dir / "analysis").glob("*.json")):
        a = _load(p)
        reg, feats = a.get("regime", {}), a.get("features_latest", {})
        rows.append(f"| {a.get('id')} | {a.get('as_of')} | {reg.get('current', '–')} | "
                    f"{reg.get('days_in_state', '–')} | {_fmt(feats.get('rv_21'))} | "
                    f"{_fmt(feats.get('drawdown'), '{:.1%}')} | {a.get('anomalies', {}).get('count_last_365_days', '–')} |")
    if rows:
        lines += ["| Strumento | Ultima seduta | Regime | Giorni | Vol. 21g | Drawdown | Anomalie 12m |",
                  "|---|---|---|---|---|---|---|", *rows, ""]

    for name, title in (("live/volatility_scores.json", "Live (previsioni registrate)"),
                        ("backtest/volatility.json", "Backtest (walk-forward storico)")):
        data = _load(results_dir / name)
        agg = data.get("aggregate", {}).get("h1")
        if data:
            extra = f", {data.get('n_forecasts_logged')} previsioni nel log" if "n_forecasts_logged" in data else ""
            lines.append(f"**Volatilità — {title}**{extra}")
        if agg:
            lines += ["", "| Modello (h=1) | Strumenti | Battono rw | Diff. media QLIKE vs rw |", "|---|---|---|---|"]
            for m, a in agg.items():
                lines.append(f"| {m} | {a['n_instruments']} | {a['wins']} | {_fmt(a['mean_qlike_diff_vs_rw'], '{:+.3f}')} |")
        elif data:
            lines.append("- nessuna previsione ancora valutabile")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pipeline.summary")
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = ap.parse_args(argv)
    sys.stdout.write(render(args.results_dir) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
