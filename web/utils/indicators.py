"""
indicators.py
=============
Calcule les indicateurs techniques pour l'affichage du dashboard.

Utilise les mêmes formules que le notebook ai/ml/model_training.ipynb,
ce qui garantit la cohérence entre l'analyse visuelle (dashboard) et
les features du modèle ML.
"""
import numpy as np
import pandas as pd


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les colonnes returns et log_returns."""
    df = df.copy()
    df["returns"] = df["price"].pct_change()
    df["log_returns"] = np.log(df["price"] / df["price"].shift(1))
    return df


def compute_sma(df: pd.DataFrame, windows: list[int] = [7, 14, 30, 50]) -> pd.DataFrame:
    """Moyennes Mobiles Simples sur plusieurs fenêtres."""
    df = df.copy()
    for w in windows:
        df[f"sma_{w}"] = df["price"].rolling(window=w).mean()
    return df


def compute_ema(df: pd.DataFrame) -> pd.DataFrame:
    """Moyennes Mobiles Exponentielles (pour le MACD)."""
    df = df.copy()
    df["ema_12"] = df["price"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["price"].ewm(span=26, adjust=False).mean()
    return df


def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD = EMA12 - EMA26 + ligne de signal + histogramme."""
    df = compute_ema(df)
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """RSI — Relative Strength Index."""
    df = df.copy()
    delta = df["price"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss
    df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    return df


def compute_bollinger(df: pd.DataFrame, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """Bandes de Bollinger."""
    df = df.copy()
    df["bb_mid"] = df["price"].rolling(window=window).mean()
    bb_std = df["price"].rolling(window=window).std()
    df["bb_upper"] = df["bb_mid"] + n_std * bb_std
    df["bb_lower"] = df["bb_mid"] - n_std * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_pct"] = (df["price"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ATR — Average True Range. Nécessite high, low, close."""
    if not all(c in df.columns for c in ["high", "low", "close"]):
        return df  # OHLCV non dispo
    df = df.copy()
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs()
        )
    )
    df[f"atr_{period}"] = tr.rolling(window=period).mean()
    return df


def compute_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Volatilité court terme et moyen terme."""
    df = df.copy()
    if "returns" not in df.columns:
        df["returns"] = df["price"].pct_change()
    df["volatility_7"] = df["returns"].rolling(window=7).std()
    df["volatility_21"] = df["returns"].rolling(window=21).std()
    return df


def compute_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Momentum sur plusieurs horizons."""
    df = df.copy()
    for n in [5, 10, 20]:
        df[f"momentum_{n}"] = df["price"] / df["price"].shift(n) - 1
    return df


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline complet : applique tous les indicateurs en une passe.
    À utiliser dans le dashboard pour avoir un DataFrame riche.
    """
    df = compute_returns(df)
    df = compute_sma(df)
    df = compute_macd(df)  # inclut compute_ema
    df = compute_rsi(df)
    df = compute_bollinger(df)
    df = compute_atr(df)
    df = compute_volatility(df)
    df = compute_momentum(df)
    return df
