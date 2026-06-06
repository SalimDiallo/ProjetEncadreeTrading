"""
export_signals.py
=================
Génère les signaux ML consommés par le dashboard Streamlit.

Reproduit fidèlement le feature engineering et l'entraînement de
`model_training.ipynb` (Random Forest / XGBoost) puis exporte un Parquet
au format attendu par `web/utils/data_loader.load_signals` :

    [date, signal (BUY/SELL/HOLD), confidence (0-1), model]

Le sentiment NLP (sentiment_{asset}.parquet) est intégré comme **feature**
des deux modèles : score forward-fillé avec décroissance vers neutre,
moyenne mobile 7j et volume d'articles. Chaque modèle est exporté en deux
variantes (avec / sans NLP) pour mesurer l'impact réel du sentiment.

Le target est la direction du prix à J+1 (1 = hausse). Le modèle est
entraîné sur les 80 % chronologiques initiaux ; les signaux du segment
de test sont donc out-of-sample (honnêtes), ceux du segment train sont
in-sample (pour couvrir toute la plage de dates du dashboard).

Lancement :
    cd ai/ml
    ../../.venv/bin/python export_signals.py          # exporte les 4 variantes
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Destination partagée avec le dashboard (web/utils/data_loader.DATA_DIR)
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "scraping" / "src" / "data" / "processed"

# Features identiques au notebook model_training.ipynb (basé sur le prix spot
# petrol_wti_daily.parquet, 2006-2026). Pas de volume → pas d'OBV/VWAP/volume_ratio ;
# high/low sont dérivés du prix (rolling 2 jours) pour l'ATR.
FEATURE_COLS = [
    "returns", "log_returns",
    "volatility_7", "volatility_21",
    "sma_7", "sma_14", "sma_30", "sma_50",
    "ema_12", "ema_26",
    "macd", "macd_signal", "macd_hist",
    "rsi_14",
    "bb_width", "bb_pct",
    "atr_14",
    "price_to_sma30", "price_to_sma50",
    "momentum_5", "momentum_10", "momentum_20",
]

# Features dérivées du sentiment NLP (ajoutées si use_sentiment=True)
SENTIMENT_COLS = ["sentiment_score", "sentiment_ma_7", "news_volume", "news_volume_ma_7"]

# Décroissance journalière du sentiment forward-fillé (×0.85/jour → ~neutre après ~15j)
SENTIMENT_DECAY = 0.85


def load_prices(asset: str) -> pd.DataFrame:
    """Charge le prix spot quotidien (parquet, 2006-2026) — comme le notebook."""
    path = PROCESSED_DIR / f"petrol_{asset.lower()}_daily.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Parquet de prix introuvable : {path}")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True).set_index("date").dropna(subset=["price"])
    return df[["price"]]


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduit la cellule 3.x du notebook (indicateurs techniques)."""
    df = df.copy()
    df["returns"] = df["price"].pct_change()
    df["log_returns"] = np.log(df["price"] / df["price"].shift(1))
    df["volatility_7"] = df["returns"].rolling(7).std()
    df["volatility_21"] = df["returns"].rolling(21).std()

    for window in [7, 14, 30, 50]:
        df[f"sma_{window}"] = df["price"].rolling(window).mean()

    df["ema_12"] = df["price"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["price"].ewm(span=26, adjust=False).mean()

    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["rsi_14"] = compute_rsi(df["price"], 14)

    df["bb_mid"] = df["price"].rolling(20).mean()
    df["bb_std"] = df["price"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_pct"] = (df["price"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # ATR sur high/low dérivés du prix (pas de vrai OHLCV dans le parquet spot)
    df["high"] = df["price"].rolling(2).max()
    df["low"] = df["price"].rolling(2).min()
    df["tr"] = df["high"] - df["low"]
    df["atr_14"] = df["tr"].rolling(14).mean()

    df["price_to_sma30"] = df["price"] / df["sma_30"]
    df["price_to_sma50"] = df["price"] / df["sma_50"]

    df["momentum_5"] = df["price"] / df["price"].shift(5) - 1
    df["momentum_10"] = df["price"] / df["price"].shift(10) - 1
    df["momentum_20"] = df["price"] / df["price"].shift(20) - 1

    df["target"] = (df["price"].shift(-1) > df["price"]).astype(int)
    return df


def build_sentiment_features(index: pd.DatetimeIndex, asset: str) -> pd.DataFrame:
    """
    Construit les features de sentiment NLP alignées sur les dates OHLCV.

    Le sentiment réel est épars (news scrapées en direct, ~30 jours sur 754).
    On le rend exploitable comme feature ML par :
      - forward-fill avec décroissance géométrique vers 0 (le sentiment
        d'un jour s'estompe sur ~15 jours faute de news fraîches) ;
      - moyenne mobile 7j ;
      - volume d'articles du jour (0 si pas de news) + sa MA 7j.
    Les jours antérieurs à la 1re news sont neutres (0).

    Retourne un DataFrame indexé par date avec les colonnes SENTIMENT_COLS.
    """
    out = pd.DataFrame(index=index, columns=SENTIMENT_COLS, dtype=float)
    out[:] = 0.0

    path = PROCESSED_DIR / f"sentiment_{asset.lower()}.parquet"
    if not path.exists():
        return out  # pas de sentiment → features neutres

    sen = pd.read_parquet(path)
    sen["date"] = pd.to_datetime(sen["date"]).astype("datetime64[ns]")
    sen = sen.sort_values("date")

    # Une news de week-end/jour férié informe le PROCHAIN jour de trading :
    # on snappe chaque date au jour de trading suivant, puis on agrège par jour.
    trading = pd.DataFrame({"date": index.astype("datetime64[ns]"),
                            "trade_day": index.astype("datetime64[ns]")})
    snapped = pd.merge_asof(sen, trading, on="date", direction="forward")
    agg = (snapped.dropna(subset=["trade_day"])
           .groupby("trade_day")
           .agg(sentiment_score=("sentiment_score", "mean"),
                n_articles=("n_articles", "sum")))

    raw_score = agg["sentiment_score"].reindex(index)
    raw_volume = agg["n_articles"].reindex(index)

    # Forward-fill décroissant : on garde la dernière valeur connue * decay^(jours)
    decayed = []
    last_val, days_since = 0.0, None
    for val in raw_score:
        if pd.notna(val):
            last_val, days_since = float(val), 0
        elif days_since is not None:
            days_since += 1
            last_val = last_val * SENTIMENT_DECAY
        decayed.append(last_val if days_since is not None else 0.0)

    out["sentiment_score"] = decayed
    out["sentiment_ma_7"] = out["sentiment_score"].rolling(7, min_periods=1).mean()
    out["news_volume"] = raw_volume.fillna(0.0).clip(lower=0.0)
    out["news_volume_ma_7"] = out["news_volume"].rolling(7, min_periods=1).mean()
    return out


def train_model(model_name: str):
    # Hyperparamètres FORTEMENT régularisés : prédire la direction J+1 du
    # pétrole est proche du bruit. Sans régularisation, RF/XGBoost mémorisent
    # le train (accuracy ~100%) et s'effondrent en test (~48%, sous la baseline).
    # Ces réglages réduisent l'écart train/test de ~0.45 à ~0.17 (validé en CV
    # temporelle 5 folds) → résultats modestes mais honnêtes et généralisables.
    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=100, max_depth=4, min_samples_split=60,
            min_samples_leaf=30, max_features="sqrt", random_state=42,
            n_jobs=-1, class_weight="balanced",
        )
    if model_name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5,
            reg_lambda=2.0, min_child_weight=10, random_state=42,
            eval_metric="logloss", n_jobs=-1,
        )
    raise ValueError(f"Modèle inconnu : {model_name}")


# Les 4 variantes exportées : (modèle, NLP ?, suffixe fichier, libellé `model`)
VARIANTS = [
    ("RandomForest", True,  "rf",       "RandomForest+NLP"),
    ("RandomForest", False, "rf_nonlp", "RandomForest"),
    ("XGBoost",      True,  "xgb",      "XGBoost+NLP"),
    ("XGBoost",      False, "xgb_nonlp","XGBoost"),
]


def export_signals(asset: str, model_name: str, use_sentiment: bool,
                   suffix: str, label: str) -> Path:
    df = build_features(load_prices(asset))

    feature_cols = list(FEATURE_COLS)
    if use_sentiment:
        sent = build_sentiment_features(df.index, asset)
        df = df.join(sent)
        feature_cols += SENTIMENT_COLS

    df_model = df[feature_cols + ["target", "price"]].dropna()

    split_idx = int(len(df_model) * 0.8)
    X = df_model[feature_cols]
    y = df_model["target"]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X.iloc[:split_idx])
    X_all = scaler.transform(X)

    model = train_model(model_name)
    model.fit(X_train, y.iloc[:split_idx])

    proba_up = model.predict_proba(X_all)[:, 1]
    pred = (proba_up > 0.5).astype(int)
    # confidence = proba de la classe prédite (0.5–1.0)
    confidence = np.where(pred == 1, proba_up, 1 - proba_up)
    signal = np.where(pred == 1, "BUY", "SELL")

    # is_oos = True pour les signaux out-of-sample (après le split d'entraînement).
    # Permet au dashboard de distinguer performance honnête (OOS) vs in-sample.
    is_oos = [i >= split_idx for i in range(len(df_model))]

    out = pd.DataFrame({
        "date": df_model.index,
        "signal": signal,
        "confidence": confidence,
        "model": label,
        "is_oos": is_oos,
    }).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"signals_ml_{asset.lower()}_{suffix}.parquet"
    out.to_parquet(path, index=False)

    n_oos = len(out) - split_idx
    print(f"✅ {label:18s} → {path.name}")
    print(f"   {len(out)} signaux ({split_idx} in-sample + {n_oos} OOS) | "
          f"{len(feature_cols)} features | "
          f"dist {out['signal'].value_counts().to_dict()} | "
          f"conf moy {out['confidence'].mean():.3f}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export des signaux ML pour le dashboard.")
    parser.add_argument("--asset", default="WTI", help="WTI (seul OHLCV disponible)")
    args = parser.parse_args()

    paths = {}
    for model_name, use_nlp, suffix, label in VARIANTS:
        paths[label] = export_signals(args.asset, model_name, use_nlp, suffix, label)

    # Rétro-compat : signals_ml_{asset}.parquet = variante RandomForest+NLP
    import shutil
    default = PROCESSED_DIR / f"signals_ml_{args.asset.lower()}.parquet"
    shutil.copyfile(paths["RandomForest+NLP"], default)
    print(f"✅ Rétro-compat : {default.name} = RandomForest+NLP")


if __name__ == "__main__":
    main()
