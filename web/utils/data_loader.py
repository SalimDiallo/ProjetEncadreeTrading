"""
data_loader.py
==============
Couche d'accès aux données du projet.

Point d'entrée unique pour lire :
- Les Parquet de prix produits par scraping/src/ (WTI, Brent)
- Le CSV OHLCV WTI (3 ans avec High/Low/Volume)
- Les actualités scrapées
- Les signaux ML et NLP (à venir)
"""
from pathlib import Path
import pandas as pd
import streamlit as st


# Chemin vers les données du module scraping
# Depuis web/utils/data_loader.py → ../../scraping/src/data/processed/
DATA_DIR = Path(__file__).resolve().parents[2] / "scraping" / "src" / "data" / "processed"


# ============================================================================
# PRIX
# ============================================================================

@st.cache_data(ttl=3600, show_spinner="Chargement des prix...")
def load_prices(asset: str) -> pd.DataFrame:
    """
    Charge les prix spot quotidiens (WTI ou Brent) depuis les Parquet.
    
    Args:
        asset: "WTI" ou "BRENT"
    
    Returns:
        DataFrame [date: datetime, price: float] trié chronologiquement.
    """
    path = DATA_DIR / f"petrol_{asset.lower()}_daily.parquet"
    
    if not path.exists():
        st.error(f"❌ Fichier introuvable : {path}")
        st.info("Lance d'abord le pipeline : `cd scraping/src && python main.py`")
        return pd.DataFrame(columns=["date", "price"])
    
    df = pd.read_parquet(path)
    # Workaround : la colonne price est stockée en string (bug du processeur)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["price"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner="Chargement des données OHLCV...")
def load_ohlcv_wti() -> pd.DataFrame:
    """
    Charge le CSV OHLCV WTI (3 ans) — pour les bougies japonaises et 
    indicateurs avancés (ATR, volume, etc.).
    
    Returns:
        DataFrame [date, open, high, low, close, volume, price]
        où price = close pour compatibilité.
    """
    path = DATA_DIR / "wti_petrole_3ans.csv"
    
    if not path.exists():
        return pd.DataFrame()
    
    # Skip 2 lignes d'en-tête (Ticker, Date)
    df = pd.read_csv(path, skiprows=2)
    df.columns = ["date", "close", "high", "low", "open", "volume"]
    df["date"] = pd.to_datetime(df["date"])
    for col in ["close", "high", "low", "open", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)
    df["price"] = df["close"]  # alias pour compatibilité
    return df


# ============================================================================
# ACTUALITÉS
# ============================================================================

@st.cache_data(ttl=3600, show_spinner="Chargement des actualités...")
def load_news(source: str = "oilprice") -> pd.DataFrame:
    """Charge les actualités scrapées."""
    path = DATA_DIR / f"petrol_news_{source}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["date", "title", "content", "source"])
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


# ============================================================================
# SIGNAUX ML & SENTIMENT NLP
# ============================================================================
# Tant que les modules ML/NLP n'ont pas livré, on retombe sur des mocks.
# Quand les fichiers réels apparaîtront, ils seront utilisés automatiquement.

@st.cache_data(ttl=3600)
def load_signals(asset: str, model: str = "RandomForest") -> pd.DataFrame:
    """
    Charge les signaux ML. Fallback sur mock si pas encore disponible.
    
    Format attendu : [date, signal (BUY/SELL/HOLD), confidence (0-1), model]
    """
    path = DATA_DIR / f"signals_ml_{asset.lower()}.parquet"
    if not path.exists():
        from utils.mock_data import mock_signals
        return mock_signals(asset)
    return pd.read_parquet(path)


@st.cache_data(ttl=3600)
def load_sentiment(asset: str) -> pd.DataFrame:
    """
    Charge les scores de sentiment NLP. Fallback sur mock.
    
    Format attendu : [date, sentiment_score (-1 à 1), n_articles]
    """
    path = DATA_DIR / f"sentiment_{asset.lower()}.parquet"
    if not path.exists():
        from utils.mock_data import mock_sentiment
        return mock_sentiment(asset)
    return pd.read_parquet(path)


def generate_sma_signals(prices_df: pd.DataFrame, short_window: int = 20, long_window: int = 50) -> pd.DataFrame:
    """Génère dynamiquement des signaux basés sur le croisement de moyennes mobiles (SMA)."""
    if prices_df.empty:
        return pd.DataFrame(columns=["date", "signal", "confidence", "model"])
        
    df = prices_df.copy().sort_values("date")
    df["sma_short"] = df["price"].rolling(short_window).mean()
    df["sma_long"] = df["price"].rolling(long_window).mean()
    
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
        
    # Calcule une confiance fictive basée sur la force de l'écart
    spread = (df["sma_short"] - df["sma_long"]).abs()
    max_spread = spread.max() if spread.max() > 0 else 1
    confidence = (spread / max_spread * 0.4 + 0.5).clip(0.5, 0.95)
    
    return pd.DataFrame({
        "date": df["date"],
        "signal": signals,
        "confidence": confidence,
        "model": f"SMA_{short_window}/{long_window}"
    })


def generate_sentiment_signals(sentiment_df: pd.DataFrame, buy_threshold: float = 0.05, sell_threshold: float = -0.05, smooth_window: int = 3) -> pd.DataFrame:
    """Génère dynamiquement des signaux de trading basés sur les scores de sentiment NLP."""
    if sentiment_df.empty:
        return pd.DataFrame(columns=["date", "signal", "confidence", "model"])
        
    df = sentiment_df.copy().sort_values("date")
    df["smooth_score"] = df["sentiment_score"].rolling(window=smooth_window, min_periods=1).mean()
    
    signals = []
    for val in df["smooth_score"]:
        if val > buy_threshold:
            signals.append("BUY")
        elif val < sell_threshold:
            signals.append("SELL")
        else:
            signals.append("HOLD")
            
    df["signal"] = signals
    df["confidence"] = df["smooth_score"].abs().clip(0.5, 1.0)
    df["model"] = f"NLP_Sentiment_Smooth_{smooth_window}"
    return df[["date", "signal", "confidence", "model"]]



# ============================================================================
# UTILITAIRES
# ============================================================================

def get_available_assets() -> list[str]:
    """Retourne la liste des actifs disponibles."""
    if not DATA_DIR.exists():
        return ["WTI", "BRENT"]
    
    assets = []
    for f in DATA_DIR.glob("petrol_*_daily.parquet"):
        name = f.stem.replace("petrol_", "").replace("_daily", "").upper()
        assets.append(name)
    return assets or ["WTI", "BRENT"]


def has_ohlcv_data(asset: str) -> bool:
    """Indique si on a des données OHLCV (pour activer les bougies)."""
    if asset.upper() == "WTI":
        return (DATA_DIR / "wti_petrole_3ans.csv").exists()
    return False
