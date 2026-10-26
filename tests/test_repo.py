"""Controlli sul repository (licenze: i dati grezzi non devono mai poter finire nel repo)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _ignored(path: str) -> bool:
    return subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0


@pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="serve git")
def test_gitignore_dati_grezzi_esclusi_dashboard_inclusa():
    assert _ignored("data/raw/nvda.parquet")
    assert _ignored("data/raw/_state.json")
    assert _ignored("data/raw/_earnings/nvda.json")
    assert not _ignored("docs/data/index.json")
    assert not _ignored("docs/data/tickers/nvda.json")
    assert not _ignored("results/analysis/nvda.json")
