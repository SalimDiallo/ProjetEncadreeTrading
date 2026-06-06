"""
mock_data.py
============
Génère des données fictives pour les modules ML et NLP qui ne sont
pas encore livrés. Permet de développer le dashboard de bout en bout
en attendant l'intégration finale.

Stratégie utilisée pour les mocks : SMA Crossover (20/50) — la même
logique baseline que dans le notebook ML, ce qui donne des résultats
réalistes et cohérents pour la démo.
"""
import numpy as np
import pandas as pd

from utils.data_loader import load_prices


def mock_signals(asset: str, seed: int = 42) -> pd.DataFrame:
    """
    Génère des signaux factices basés sur SMA crossover 20/50.
    Cohérent avec la baseline du notebook ai/ml/model_training.ipynb.
    """
    prices = load_prices(asset)
    if prices.empty:
        return pd.DataFrame(columns=["date", "signal", "confidence", "model"])

    df = prices.copy()
    df["sma_short"] = df["price"].rolling(20).mean()
    df["sma_long"] = df["price"].rolling(50).mean()

    signals = []
    prev_diff = 0.0
    for _, row in df.iterrows():
        if pd.isna(row["sma_short"]) or pd.isna(row["sma_long"]):
            signals.append("HOLD")
            continue
        diff = row["sma_short"] - row["sma_long"]
        if diff > 0 and prev_diff <= 0:
            signals.append("BUY")
        elif diff < 0 and prev_diff >= 0:
            signals.append("SELL")
        else:
            signals.append("HOLD")
        prev_diff = diff

    rng = np.random.default_rng(seed)
    spread = (df["sma_short"] - df["sma_long"]).abs()
    max_spread = spread.max() if spread.max() > 0 else 1
    confidence = (spread / max_spread * 0.4 + 0.5 + rng.normal(0, 0.05, len(df))).clip(0.5, 0.95)

    return pd.DataFrame({
        "date": df["date"],
        "signal": signals,
        "confidence": confidence,
        "model": "MOCK_SMA_Crossover"
    })


def mock_sentiment(asset: str, seed: int = 7) -> pd.DataFrame:
    """Sentiment factice (-1 à +1) lissé sur 10 jours."""
    prices = load_prices(asset)
    if prices.empty:
        return pd.DataFrame(columns=["date", "sentiment_score", "n_articles"])

    rng = np.random.default_rng(seed)
    n = len(prices)
    raw = rng.uniform(-0.5, 0.5, size=n)
    sentiment = pd.Series(raw).rolling(window=10, min_periods=1).mean().values
    n_articles = rng.integers(1, 25, size=n)

    return pd.DataFrame({
        "date": prices["date"],
        "sentiment_score": sentiment,
        "n_articles": n_articles
    })


def mock_recommendation_reasons(signal: str) -> list[str]:
    """Raisons factices pour la recommandation finale."""
    if signal == "BUY":
        return [
            "📈 RSI à 35 — sortie de zone de survente",
            "📊 Croisement haussier MACD (golden cross)",
            "🤖 Random Forest prédit hausse à J+1",
            "📰 Sentiment positif des news (+0.42)",
        ]
    elif signal == "SELL":
        return [
            "📉 RSI à 78 — zone de surachat",
            "📊 Divergence baissière sur le MACD",
            "🤖 XGBoost prédit baisse à J+1",
            "📰 Sentiment négatif des news (-0.31)",
        ]
    else:
        return [
            "⚖️ Signaux contradictoires entre modèles",
            "📊 Prix dans un range serré, pas de tendance claire",
            "🤖 Confiance ML insuffisante (<60%)",
        ]
