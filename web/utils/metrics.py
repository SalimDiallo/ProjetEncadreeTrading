"""
metrics.py
==========
Métriques de risque et de performance pour le backtest.
Toutes les fonctions sont pures et vectorisées (numpy/pandas).

Cohérent avec les formules du notebook ai/ml/model_training.ipynb :
- Sharpe Ratio annualisé avec √252
- Max Drawdown depuis le sommet cumulé
- Win Rate basé sur la direction des trades

Hypothèses :
- 252 jours de trading par an (standard marchés)
- Taux sans risque = 0% par défaut (ajustable)
"""

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252


def total_return(equity: pd.Series) -> float:
    """Rendement total sur la période."""
    if len(equity) < 2:
        return 0.0
    return (equity.iloc[-1] / equity.iloc[0]) - 1


def cagr(equity: pd.Series) -> float:
    """Compound Annual Growth Rate (rendement annualisé)."""
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    n_years = len(equity) / PERIODS_PER_YEAR
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1


def annualized_volatility(returns: pd.Series) -> float:
    """Volatilité annualisée des returns."""
    if returns.empty:
        return 0.0
    std = returns.std()
    if pd.isna(std) or std < 1e-12:
        return 0.0
    return float(std * np.sqrt(PERIODS_PER_YEAR))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Sharpe Ratio annualisé.
    risk_free_rate : taux annuel (ex: 0.04 pour 4%)
    """
    if returns.empty or returns.std() < 1e-12:
        return 0.0
    daily_rf = risk_free_rate / PERIODS_PER_YEAR
    excess = returns - daily_rf
    return float(np.sqrt(PERIODS_PER_YEAR) * excess.mean() / excess.std())


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Sortino Ratio : comme Sharpe mais ne pénalise que la volatilité baissière.
    """
    daily_rf = risk_free_rate / PERIODS_PER_YEAR
    excess = returns - daily_rf
    downside = excess[excess < 0]
    if downside.empty or downside.std() == 0:
        return 0.0
    return float(np.sqrt(PERIODS_PER_YEAR) * excess.mean() / downside.std())


def max_drawdown(equity: pd.Series) -> float:
    """Maximum Drawdown : pire perte depuis un sommet, en pourcentage."""
    if equity.empty:
        return 0.0
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    return float(drawdown.min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Série complète du drawdown (utile pour les graphiques)."""
    cummax = equity.cummax()
    return (equity - cummax) / cummax


def calmar_ratio(equity: pd.Series) -> float:
    """Calmar Ratio = CAGR / |Max Drawdown|."""
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return cagr(equity) / mdd


def win_rate(trades: pd.DataFrame) -> float:
    """Pourcentage de trades gagnants. Nécessite une colonne 'pnl'."""
    if trades.empty or "pnl" not in trades.columns:
        return 0.0
    wins = (trades["pnl"] > 0).sum()
    return wins / len(trades)


def profit_factor(trades: pd.DataFrame) -> float:
    """Profit Factor = somme des gains / |somme des pertes|."""
    if trades.empty or "pnl" not in trades.columns:
        return 0.0
    gains = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    losses = trades.loc[trades["pnl"] < 0, "pnl"].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / abs(losses)


def compute_all_metrics(equity: pd.Series, returns: pd.Series,
                        trades: pd.DataFrame | None = None) -> dict:
    """Calcule toutes les métriques en une passe."""
    metrics = {
        "total_return": total_return(equity),
        "cagr": cagr(equity),
        "volatility": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar_ratio(equity),
    }
    if trades is not None and not trades.empty:
        metrics["win_rate"] = win_rate(trades)
        metrics["profit_factor"] = profit_factor(trades)
        metrics["n_trades"] = len(trades)
    return metrics
