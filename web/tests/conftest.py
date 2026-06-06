"""
conftest.py
===========
Fixtures partagées par toute la suite de tests.

Pytest charge ce fichier automatiquement. Les fixtures définies ici sont
disponibles dans tous les fichiers test_*.py sans import explicite.

Convention : on génère des données déterministes (seeds fixés) pour
que les tests soient reproductibles à 100%.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ajoute dashboard/ au PYTHONPATH pour que `from utils...` fonctionne
DASHBOARD_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DASHBOARD_ROOT))


# ============================================================================
# FIXTURES — Séries de prix synthétiques
# ============================================================================

@pytest.fixture
def prices_uptrend():
    """100 jours en tendance haussière régulière (+0.1%/jour)."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    prices = 100 * (1.001 ** np.arange(100))
    return pd.DataFrame({"date": dates, "price": prices})


@pytest.fixture
def prices_downtrend():
    """100 jours en tendance baissière régulière (-0.1%/jour)."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    prices = 100 * (0.999 ** np.arange(100))
    return pd.DataFrame({"date": dates, "price": prices})


@pytest.fixture
def prices_flat():
    """100 jours à prix constant — volatilité nulle, cas limite."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    return pd.DataFrame({"date": dates, "price": [100.0] * 100})


@pytest.fixture
def prices_volatile():
    """252 jours (1 an) avec volatilité réaliste (~1% / jour)."""
    rng = np.random.default_rng(seed=42)
    dates = pd.date_range("2024-01-01", periods=252, freq="D")
    daily_returns = rng.normal(loc=0.0005, scale=0.01, size=252)
    prices = 100 * np.exp(np.cumsum(daily_returns))
    return pd.DataFrame({"date": dates, "price": prices})


@pytest.fixture
def prices_with_drawdown():
    """Profil pic-creux : montée vers 150, redescente vers 120."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    up = np.linspace(100, 150, 50)
    down = np.linspace(150, 120, 50)
    return pd.DataFrame({"date": dates, "price": np.concatenate([up, down])})


# ============================================================================
# FIXTURES — Signaux de trading
# ============================================================================

def _make_signals(dates: pd.Series, signal_map: dict) -> pd.DataFrame:
    """
    Helper interne pour fabriquer un DataFrame de signaux.
    signal_map : {index_jour: "BUY"|"SELL"} ; les autres jours sont "HOLD".
    """
    signals = ["HOLD"] * len(dates)
    for idx, sig in signal_map.items():
        signals[idx] = sig
    return pd.DataFrame({
        "date": dates,
        "signal": signals,
        "confidence": [0.8] * len(dates),
        "model": ["TEST"] * len(dates),
    })


@pytest.fixture
def signals_all_hold(prices_uptrend):
    """Que des HOLD → aucun trade ne doit être exécuté."""
    return _make_signals(prices_uptrend["date"], {})


@pytest.fixture
def signals_one_cycle(prices_uptrend):
    """Un BUY au jour 10, un SELL au jour 50 → exactement 1 trade fermé."""
    return _make_signals(prices_uptrend["date"], {10: "BUY", 50: "SELL"})


@pytest.fixture
def signals_three_cycles(prices_volatile):
    """Trois cycles BUY/SELL → 3 trades fermés."""
    cycles = {10: "BUY", 30: "SELL", 50: "BUY", 80: "SELL", 100: "BUY", 150: "SELL"}
    return _make_signals(prices_volatile["date"], cycles)


@pytest.fixture
def signals_perfect_timing(prices_with_drawdown):
    """Achète au début, vend pile au pic (jour 49) → trade parfait."""
    return _make_signals(prices_with_drawdown["date"], {0: "BUY", 49: "SELL"})


# ============================================================================
# FIXTURES — DataFrames de trades fermés
# ============================================================================
# Format réel produit par simulate_portfolio :
#   [date_entry, date_exit, price_entry, price_exit, pnl, return_pct]

@pytest.fixture
def trades_all_winning():
    """3 trades, tous gagnants."""
    return pd.DataFrame({
        "date_entry": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        "date_exit":  pd.to_datetime(["2024-01-15", "2024-02-15", "2024-03-15"]),
        "price_entry": [100.0, 105.0, 110.0],
        "price_exit":  [110.0, 115.0, 125.0],
        "pnl":         [100.0, 100.0, 150.0],
        "return_pct":  [0.10, 0.095, 0.136],
    })


@pytest.fixture
def trades_mixed():
    """4 trades : 2 gagnants, 2 perdants — pour tester win_rate=50%."""
    return pd.DataFrame({
        "date_entry": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]),
        "date_exit":  pd.to_datetime(["2024-01-15", "2024-02-15", "2024-03-15", "2024-04-15"]),
        "price_entry": [100.0, 110.0, 100.0, 110.0],
        "price_exit":  [110.0, 100.0, 120.0, 90.0],
        "pnl":         [50.0, -50.0, 100.0, -200.0],
        "return_pct":  [0.10, -0.09, 0.20, -0.18],
    })


@pytest.fixture
def trades_empty():
    """DataFrame vide — pour tester les cas limites."""
    return pd.DataFrame(columns=["date_entry", "date_exit", "price_entry",
                                  "price_exit", "pnl", "return_pct"])


# ============================================================================
# FIXTURES — Courbes d'équité
# ============================================================================

@pytest.fixture
def equity_simple():
    """Courbe d'équité avec un pic et un creux clairement identifiables."""
    return pd.Series([10000, 10500, 11000, 12000, 11500, 10800, 11200, 12500, 13000])


@pytest.fixture
def returns_simple(equity_simple):
    """Rendements correspondants à equity_simple."""
    return equity_simple.pct_change().fillna(0)
