"""
Chemins centralisés du pipeline (source unique de vérité).

Arborescence :
    data/raw/          — collecte
    data/processed/    — preprocessing
    data/sentiment/    — scores
    data/aggregated/   — série journalière
    data/signals/      — signaux BUY/SELL/HOLD
    data/prices/       — prix (yfinance)
    data/backtest/     — résultats backtest
    models/            — artefacts ML (TF-IDF, LR, cache HuggingFace)
    logs/              — journaux d'exécution
"""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent

DATA_DIR = PACKAGE_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SENTIMENT_DIR = DATA_DIR / "sentiment"
AGGREGATED_DIR = DATA_DIR / "aggregated"
SIGNALS_DIR = DATA_DIR / "signals"
PRICES_DIR = DATA_DIR / "prices"
BACKTEST_DIR = DATA_DIR / "backtest"

MODELS_DIR = PACKAGE_ROOT / "models"
FINBERT_CACHE_DIR = MODELS_DIR / "finbert"
EMBEDDINGS_CACHE_DIR = MODELS_DIR / "embeddings"

LOGS_DIR = PACKAGE_ROOT / "logs"

_DATA_SUBDIRS = (
    RAW_DIR,
    PROCESSED_DIR,
    SENTIMENT_DIR,
)


def ensure_data_dirs() -> None:
    """Crée l'arborescence data/, models/ et logs/ si nécessaire."""
    for directory in _DATA_SUBDIRS:
        directory.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


SHARED_PROCESSED_DIR = PACKAGE_ROOT.parents[2] / "scraping" / "src" / "data" / "processed"

