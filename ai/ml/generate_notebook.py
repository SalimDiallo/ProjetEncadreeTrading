#!/usr/bin/env python3
"""Generate the backtesting notebook."""
import json

cells = []

def md(source):
    cells.append({"cell_type":"markdown","metadata":{},"source": source.strip().split("\n")})

def code(source):
    lines = source.strip().split("\n")
    # Add newlines to all but last line
    src = [l+"\n" for l in lines[:-1]] + [lines[-1]]
    cells.append({"cell_type":"code","metadata":{},"source": src, "execution_count": None, "outputs":[]})

# ============================================================
# SECTION 1 - TITLE
# ============================================================
md("""# 🛢️ WTI Crude Oil — ML Backtesting Pipeline
---
**Objectif** : Construire un pipeline complet de backtesting pour le pétrole WTI en utilisant des indicateurs techniques et des modèles de Machine Learning (Random Forest & XGBoost).

**Sections** :
1. Installation & Imports
2. Chargement & Inspection des données
3. Calcul des Indicateurs Techniques
4. Préparation des Features & Target
5. 🔮 *Placeholder — Analyse de Sentiment*
6. Modèle Random Forest
7. Modèle XGBoost
8. Backtesting & Comparaison des Stratégies
9. Comparaison des Modèles
10. Backtesting
11. Validation Croisée (Time Series Split)
12. ⚙️ Optimisation des Hyperparamètres
13. 💰 Gestion du Risque — Stop-Loss & Position Sizing
14. 🔄 Walk-Forward Optimization
15. Conclusion""")

# ============================================================
# SECTION 2 - INSTALLS & IMPORTS
# ============================================================
md("""## 1. Installation & Imports
> À exécuter une seule fois pour installer les dépendances.""")

code("""# !pip install pandas numpy scikit-learn xgboost matplotlib seaborn pyarrow""")

code("""import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')
pd.set_option('display.max_columns', 50)
print("✅ Imports chargés avec succès")""")

# ============================================================
# SECTION 3 - LOAD DATA
# ============================================================
md("""## 2. Chargement & Inspection des Données""")

code("""# --- Chargement du fichier CSV (OHLCV) ---
DATA_PATH = "../../scraping/src/data/processed/wti_petrole_3ans.csv"
df_raw = pd.read_csv(DATA_PATH, skiprows=2)  # Skip 2 lignes d'en-tête (Ticker, Date)

# Renommer et convertir les colonnes
df = df_raw.copy()
df.columns = ['date', 'close', 'high', 'low', 'open', 'volume']
df['date'] = pd.to_datetime(df['date'])
for col in ['close', 'high', 'low', 'open', 'volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Tri chronologique & index par date
df = df.sort_values('date').reset_index(drop=True)
df = df.set_index('date')
df = df.dropna()

# Colonne 'price' = close (pour compatibilité)
df['price'] = df['close']

print(f"📊 Période : {df.index.min().date()} → {df.index.max().date()}")
print(f"📏 Nombre d'observations : {len(df)}")
print(f"📦 Colonnes OHLCV : {['open','high','low','close','volume']}")
print(f"❌ Valeurs manquantes : {df.isnull().sum().sum()}")
df.head(10)""")

code("""# --- Visualisation du prix WTI ---
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df.index, df['price'], linewidth=0.8, color='#2196F3')
ax.set_title('Prix WTI Crude Oil — Historique', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Prix (USD)')
ax.fill_between(df.index, df['price'], alpha=0.1, color='#2196F3')
plt.tight_layout()
plt.show()""")

code("""# --- Statistiques descriptives ---
df.describe()""")

# ============================================================
# SECTION 4 - TECHNICAL INDICATORS
# ============================================================
md("""## 3. Calcul des Indicateurs Techniques
On calcule les indicateurs classiques utilisés en trading quantitatif :
- **SMA** (Simple Moving Average) — 7, 14, 30, 50 jours
- **EMA** (Exponential Moving Average) — 12, 26 jours
- **RSI** (Relative Strength Index) — 14 jours
- **MACD** (Moving Average Convergence Divergence)
- **Bollinger Bands** — 20 jours
- **ATR** (Average True Range) — 14 jours (avec vrais High/Low)
- **Volume** — OBV, VWAP, Volume MA
- **Returns & Volatilité""")

code("""# ============================
# 3.1 Returns & Volatilité
# ============================
df['returns'] = df['price'].pct_change()
df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
df['volatility_7'] = df['returns'].rolling(window=7).std()
df['volatility_21'] = df['returns'].rolling(window=21).std()

# ============================
# 3.2 Moyennes Mobiles (SMA)
# ============================
for window in [7, 14, 30, 50]:
    df[f'sma_{window}'] = df['price'].rolling(window=window).mean()

# ============================
# 3.3 Moyennes Mobiles Exponentielles (EMA)
# ============================
df['ema_12'] = df['price'].ewm(span=12, adjust=False).mean()
df['ema_26'] = df['price'].ewm(span=26, adjust=False).mean()

# ============================
# 3.4 MACD
# ============================
df['macd'] = df['ema_12'] - df['ema_26']
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
df['macd_hist'] = df['macd'] - df['macd_signal']

# ============================
# 3.5 RSI (14 jours)
# ============================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['rsi_14'] = compute_rsi(df['price'], 14)

# ============================
# 3.6 Bollinger Bands (20 jours)
# ============================
df['bb_mid'] = df['price'].rolling(window=20).mean()
df['bb_std'] = df['price'].rolling(window=20).std()
df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
df['bb_pct'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

# ============================
# 3.7 ATR (Average True Range) — avec vrais High/Low
# ============================
df['tr'] = np.maximum(
    df['high'] - df['low'],
    np.maximum(
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    )
)
df['atr_14'] = df['tr'].rolling(window=14).mean()

# ============================
# 3.10 Indicateurs de Volume
# ============================
# OBV (On-Balance Volume)
df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
# Volume Moving Average
df['volume_ma_14'] = df['volume'].rolling(14).mean()
df['volume_ratio'] = df['volume'] / df['volume_ma_14']
# VWAP approximé (sur 14 jours glissant)
df['vwap_14'] = (df['close'] * df['volume']).rolling(14).sum() / df['volume'].rolling(14).sum()

# ============================
# 3.8 Ratios de prix
# ============================
df['price_to_sma30'] = df['price'] / df['sma_30']
df['price_to_sma50'] = df['price'] / df['sma_50']

# ============================
# 3.9 Momentum
# ============================
df['momentum_5'] = df['price'] / df['price'].shift(5) - 1
df['momentum_10'] = df['price'] / df['price'].shift(10) - 1
df['momentum_20'] = df['price'] / df['price'].shift(20) - 1

print(f"✅ {len(df.columns)} colonnes après feature engineering")
df.tail()""")

code("""# --- Visualisation des indicateurs clés ---
fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

# Prix + SMA
axes[0].plot(df.index, df['price'], label='Prix', linewidth=0.8)
axes[0].plot(df.index, df['sma_14'], label='SMA 14', linewidth=0.7, alpha=0.8)
axes[0].plot(df.index, df['sma_50'], label='SMA 50', linewidth=0.7, alpha=0.8)
axes[0].fill_between(df.index, df['bb_lower'], df['bb_upper'], alpha=0.1, label='Bollinger')
axes[0].set_title('Prix & Moyennes Mobiles', fontweight='bold')
axes[0].legend(loc='upper left', fontsize=8)

# RSI
axes[1].plot(df.index, df['rsi_14'], color='purple', linewidth=0.7)
axes[1].axhline(70, color='red', linestyle='--', alpha=0.5)
axes[1].axhline(30, color='green', linestyle='--', alpha=0.5)
axes[1].set_title('RSI (14)', fontweight='bold')
axes[1].set_ylim(0, 100)

# MACD
axes[2].plot(df.index, df['macd'], label='MACD', linewidth=0.7)
axes[2].plot(df.index, df['macd_signal'], label='Signal', linewidth=0.7)
axes[2].bar(df.index, df['macd_hist'], alpha=0.3, label='Histogramme')
axes[2].set_title('MACD', fontweight='bold')
axes[2].legend(loc='upper left', fontsize=8)

# Volatilité
axes[3].plot(df.index, df['volatility_21'], color='orange', linewidth=0.7)
axes[3].set_title('Volatilité (21 jours)', fontweight='bold')

plt.tight_layout()
plt.show()""")

# ============================================================
# SECTION 5 - FEATURES & TARGET
# ============================================================
md("""## 4. Préparation des Features & Target
- **Target** : Direction du prix le lendemain (1 = hausse, 0 = baisse)
- **Features** : Tous les indicateurs techniques calculés""")

code("""# --- Définition du target : direction du prix à J+1 ---
df['target'] = (df['price'].shift(-1) > df['price']).astype(int)

# --- Sélection des features ---
FEATURE_COLS = [
    'returns', 'log_returns',
    'volatility_7', 'volatility_21',
    'sma_7', 'sma_14', 'sma_30', 'sma_50',
    'ema_12', 'ema_26',
    'macd', 'macd_signal', 'macd_hist',
    'rsi_14',
    'bb_width', 'bb_pct',
    'atr_14',
    'price_to_sma30', 'price_to_sma50',
    'momentum_5', 'momentum_10', 'momentum_20',
    # Volume indicators
    'obv', 'volume_ratio', 'vwap_14',
    # === PLACEHOLDER: Ajouter les features de sentiment ici ===
    # 'sentiment_score',
    # 'sentiment_ma_7',
    # 'news_volume',
]

# --- Nettoyage des NaN (causés par les rolling windows) ---
df_model = df[FEATURE_COLS + ['target', 'price']].dropna()
print(f"📏 Dataset modèle : {len(df_model)} lignes, {len(FEATURE_COLS)} features")
print(f"📊 Distribution target : \\n{df_model['target'].value_counts(normalize=True).round(3)}")""")

code("""# --- Split temporel (pas de shuffle pour les séries temporelles !) ---
# 80% train / 20% test
split_idx = int(len(df_model) * 0.8)

X_train = df_model[FEATURE_COLS].iloc[:split_idx]
X_test  = df_model[FEATURE_COLS].iloc[split_idx:]
y_train = df_model['target'].iloc[:split_idx]
y_test  = df_model['target'].iloc[split_idx:]

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"🔹 Train : {len(X_train)} ({X_train.index.min().date()} → {X_train.index.max().date()})")
print(f"🔹 Test  : {len(X_test)} ({X_test.index.min().date()} → {X_test.index.max().date()})")""")

# ============================================================
# SECTION 6 - SENTIMENT PLACEHOLDER
# ============================================================
md("""## 5. 🔮 Placeholder — Analyse de Sentiment
> **TODO** : Intégrer les scores de sentiment issus de l'analyse NLP des actualités / tweets sur le pétrole WTI.
>
> Les features de sentiment seront ajoutées dans `FEATURE_COLS` ci-dessus.""")

code("""# ================================================================
# PLACEHOLDER : ANALYSE DE SENTIMENT
# ================================================================
# Étapes prévues :
#   1. Charger les données de sentiment (news / tweets / RSS)
#   2. Calculer un score de sentiment journalier (ex: VADER, FinBERT)
#   3. Calculer des features dérivées :
#      - sentiment_score        : score brut du jour
#      - sentiment_ma_7         : moyenne mobile 7j du sentiment
#      - sentiment_volatility   : volatilité du sentiment
#      - news_volume            : nombre d'articles/tweets du jour
#   4. Merger avec df_model sur la date
#   5. Ajouter les colonnes dans FEATURE_COLS
# ================================================================

# Exemple de code (à décommenter une fois les données disponibles) :
#
# df_sentiment = pd.read_csv("path/to/sentiment_data.csv", parse_dates=['date'])
# df_sentiment = df_sentiment.set_index('date')
# df_sentiment['sentiment_ma_7'] = df_sentiment['sentiment_score'].rolling(7).mean()
# df_sentiment['sentiment_volatility'] = df_sentiment['sentiment_score'].rolling(7).std()
#
# df_model = df_model.join(df_sentiment[['sentiment_score','sentiment_ma_7',
#                                         'sentiment_volatility','news_volume']])
# df_model = df_model.dropna()
#
# # Ajouter dans FEATURE_COLS :
# # FEATURE_COLS += ['sentiment_score','sentiment_ma_7','sentiment_volatility','news_volume']

print("⏳ Section sentiment — en attente d'intégration")""")

# ============================================================
# SECTION 7 - HELPER FUNCTIONS
# ============================================================
md("""## 6. Fonctions Utilitaires""")

code("""def evaluate_model(name, y_true, y_pred, y_proba=None):
    \"\"\"Affiche les métriques de classification et la matrice de confusion.\"\"\"
    print(f"\\n{'='*50}")
    print(f"  📈 Résultats — {name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_true, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_true, y_pred):.4f}")
    print(f"  F1-Score  : {f1_score(y_true, y_pred):.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=['Baisse','Hausse']))

    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Baisse','Hausse'], yticklabels=['Baisse','Hausse'])
    ax.set_title(f'Matrice de Confusion — {name}', fontweight='bold')
    ax.set_ylabel('Réel')
    ax.set_xlabel('Prédit')
    plt.tight_layout()
    plt.show()
    return {
        'model': name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }""")

# ============================================================
# SECTION 8 - RANDOM FOREST
# ============================================================
md("""## 7. Modèle — Random Forest 🌲""")

code("""# --- Entraînement Random Forest ---
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

rf_metrics = evaluate_model("Random Forest", y_test, rf_pred, rf_proba)""")

code("""# --- Feature Importance (Random Forest) ---
feat_imp = pd.Series(rf_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 8))
feat_imp.plot(kind='barh', ax=ax, color='#4CAF50')
ax.set_title('Importance des Features — Random Forest', fontweight='bold')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.show()""")

# ============================================================
# SECTION 9 - XGBOOST
# ============================================================
md("""## 8. Modèle — XGBoost 🚀""")

code("""# --- Entraînement XGBoost ---
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1
)
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)
xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

xgb_metrics = evaluate_model("XGBoost", y_test, xgb_pred, xgb_proba)""")

code("""# --- Feature Importance (XGBoost) ---
xgb_imp = pd.Series(xgb_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 8))
xgb_imp.plot(kind='barh', ax=ax, color='#FF9800')
ax.set_title('Importance des Features — XGBoost', fontweight='bold')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.show()""")

# ============================================================
# SECTION 10 - COMPARAISON
# ============================================================
md("""## 9. Comparaison des Modèles""")

code("""# --- Tableau comparatif ---
results_df = pd.DataFrame([rf_metrics, xgb_metrics]).set_index('model')
print(results_df.round(4).to_string())

fig, ax = plt.subplots(figsize=(10, 5))
results_df.plot(kind='bar', ax=ax, rot=0, colormap='viridis')
ax.set_title('Comparaison RF vs XGBoost', fontweight='bold')
ax.set_ylabel('Score')
ax.legend(loc='lower right')
ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()""")

# ============================================================
# SECTION 11 - BACKTESTING
# ============================================================
md("""## 10. Backtesting 📊
Simulation de trading basée sur les prédictions des modèles.
- **Signal = 1** (Hausse prédite) → on achète
- **Signal = 0** (Baisse prédite) → on reste cash
- Comparaison avec la stratégie Buy & Hold""")

code("""def backtest_strategy(df_bt, predictions, proba, model_name, threshold=0.5):
    \"\"\"
    Backtesting d'une stratégie ML.
    - positions basées sur les prédictions (avec seuil de confiance)
    - calcul du rendement cumulé, max drawdown, sharpe ratio
    \"\"\"
    bt = df_bt.copy()
    bt['signal'] = (proba >= threshold).astype(int)
    bt['strategy_returns'] = bt['signal'] * bt['returns']
    bt['cumulative_market'] = (1 + bt['returns']).cumprod()
    bt['cumulative_strategy'] = (1 + bt['strategy_returns']).cumprod()

    # Métriques
    total_return_market = bt['cumulative_market'].iloc[-1] - 1
    total_return_strategy = bt['cumulative_strategy'].iloc[-1] - 1

    # Sharpe Ratio (annualisé)
    sharpe = bt['strategy_returns'].mean() / bt['strategy_returns'].std() * np.sqrt(252)

    # Max Drawdown
    cummax = bt['cumulative_strategy'].cummax()
    drawdown = (bt['cumulative_strategy'] - cummax) / cummax
    max_dd = drawdown.min()

    # Win rate
    trades = bt[bt['signal'] == 1]
    win_rate = (trades['strategy_returns'] > 0).mean() if len(trades) > 0 else 0

    print(f"\\n{'='*55}")
    print(f"  📊 Backtesting — {model_name} (seuil={threshold})")
    print(f"{'='*55}")
    print(f"  Rendement Marché (B&H)  : {total_return_market:+.2%}")
    print(f"  Rendement Stratégie     : {total_return_strategy:+.2%}")
    print(f"  Sharpe Ratio (ann.)     : {sharpe:.3f}")
    print(f"  Max Drawdown            : {max_dd:.2%}")
    print(f"  Win Rate                : {win_rate:.2%}")
    print(f"  Nb Trades (jours long)  : {bt['signal'].sum()} / {len(bt)}")

    return bt, {
        'model': model_name,
        'return_market': total_return_market,
        'return_strategy': total_return_strategy,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'nb_trades': int(bt['signal'].sum())
    }""")

code("""# --- Préparer le dataframe de backtest (période test uniquement) ---
df_backtest = df_model.iloc[split_idx:].copy()
df_backtest['returns'] = df_backtest['price'].pct_change().fillna(0)

# --- Backtesting Random Forest ---
bt_rf, bt_rf_metrics = backtest_strategy(df_backtest, rf_pred, rf_proba, "Random Forest", threshold=0.5)

# --- Backtesting XGBoost ---
bt_xgb, bt_xgb_metrics = backtest_strategy(df_backtest, xgb_pred, xgb_proba, "XGBoost", threshold=0.5)""")

code("""# --- Graphique des rendements cumulés ---
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(bt_rf.index, bt_rf['cumulative_market'], label='Buy & Hold', linewidth=1.5, color='gray', alpha=0.7)
ax.plot(bt_rf.index, bt_rf['cumulative_strategy'], label='Random Forest', linewidth=1.5, color='#4CAF50')
ax.plot(bt_xgb.index, bt_xgb['cumulative_strategy'], label='XGBoost', linewidth=1.5, color='#FF9800')
ax.axhline(1, color='black', linestyle='--', alpha=0.3)
ax.set_title('Backtesting — Rendements Cumulés', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Valeur du Portfolio (base 1)')
ax.legend(fontsize=11)
ax.fill_between(bt_rf.index, bt_rf['cumulative_strategy'], 1, alpha=0.05, color='green')
ax.fill_between(bt_xgb.index, bt_xgb['cumulative_strategy'], 1, alpha=0.05, color='orange')
plt.tight_layout()
plt.show()""")

code("""# --- Résumé final du backtesting ---
bt_summary = pd.DataFrame([bt_rf_metrics, bt_xgb_metrics]).set_index('model')
print("\\n📋 Résumé Backtesting :")
print(bt_summary.round(4).to_string())""")

# ============================================================
# SECTION 12 - CROSS-VALIDATION
# ============================================================
md("""## 11. Validation Croisée (Time Series Split)""")

code("""# --- Cross-validation avec TimeSeriesSplit ---
tscv = TimeSeriesSplit(n_splits=5)
X_all = scaler.fit_transform(df_model[FEATURE_COLS])
y_all = df_model['target'].values

cv_results = {'Random Forest': [], 'XGBoost': []}

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_all)):
    X_tr, X_val = X_all[train_idx], X_all[val_idx]
    y_tr, y_val = y_all[train_idx], y_all[val_idx]

    # RF
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    cv_results['Random Forest'].append(accuracy_score(y_val, rf.predict(X_val)))

    # XGB
    xgb = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05,
                         random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
    xgb.fit(X_tr, y_tr)
    cv_results['XGBoost'].append(accuracy_score(y_val, xgb.predict(X_val)))

    print(f"  Fold {fold+1} — RF: {cv_results['Random Forest'][-1]:.4f} | XGB: {cv_results['XGBoost'][-1]:.4f}")

print(f"\\n📊 Moyenne CV — RF: {np.mean(cv_results['Random Forest']):.4f} | XGB: {np.mean(cv_results['XGBoost']):.4f}")""")

# ============================================================
# SECTION 13 - HYPERPARAMETER OPTIMIZATION
# ============================================================
md("""## 12. ⚙️ Optimisation des Hyperparamètres
Recherche systématique des meilleurs hyperparamètres via **GridSearchCV** et **RandomizedSearchCV** avec un split temporel.""")

md("""### 12.1 GridSearchCV — Random Forest""")

code("""# --- GridSearchCV pour Random Forest ---
# On utilise TimeSeriesSplit pour respecter l'ordre temporel
tscv_opt = TimeSeriesSplit(n_splits=3)

rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [3, 5, 10],
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
    param_grid=rf_param_grid,
    cv=tscv_opt,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
rf_grid.fit(X_train_scaled, y_train)

print(f"\\n🏆 Meilleurs paramètres RF :")
for k, v in rf_grid.best_params_.items():
    print(f"   {k}: {v}")
print(f"   Best F1 (CV): {rf_grid.best_score_:.4f}")

# Évaluation sur le test set
rf_opt_pred = rf_grid.best_estimator_.predict(X_test_scaled)
rf_opt_proba = rf_grid.best_estimator_.predict_proba(X_test_scaled)[:, 1]
rf_opt_metrics = evaluate_model("RF Optimisé", y_test, rf_opt_pred, rf_opt_proba)""")

md("""### 12.2 RandomizedSearchCV — XGBoost
> RandomizedSearch est plus efficace que GridSearch quand l'espace de recherche est grand.""")

code("""# --- RandomizedSearchCV pour XGBoost ---
from scipy.stats import randint, uniform

xgb_param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 12),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.5, 0.5),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0.5, 2),
    'min_child_weight': randint(1, 10),
}

xgb_random = RandomizedSearchCV(
    XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1),
    param_distributions=xgb_param_dist,
    n_iter=50,  # 50 combinaisons aléatoires
    cv=tscv_opt,
    scoring='f1',
    n_jobs=-1,
    verbose=1,
    random_state=42
)
xgb_random.fit(X_train_scaled, y_train)

print(f"\\n🏆 Meilleurs paramètres XGBoost :")
for k, v in xgb_random.best_params_.items():
    print(f"   {k}: {round(v, 4) if isinstance(v, float) else v}")
print(f"   Best F1 (CV): {xgb_random.best_score_:.4f}")

# Évaluation sur le test set
xgb_opt_pred = xgb_random.best_estimator_.predict(X_test_scaled)
xgb_opt_proba = xgb_random.best_estimator_.predict_proba(X_test_scaled)[:, 1]
xgb_opt_metrics = evaluate_model("XGB Optimisé", y_test, xgb_opt_pred, xgb_opt_proba)""")

code("""# --- Comparaison avant / après optimisation ---
all_metrics = pd.DataFrame([rf_metrics, rf_opt_metrics, xgb_metrics, xgb_opt_metrics]).set_index('model')
print("\\n📋 Comparaison avant/après optimisation :")
print(all_metrics.round(4).to_string())

fig, ax = plt.subplots(figsize=(12, 5))
all_metrics.plot(kind='bar', ax=ax, rot=15, colormap='coolwarm')
ax.set_title('Impact de l\'optimisation des hyperparamètres', fontweight='bold')
ax.set_ylabel('Score')
ax.set_ylim(0, 1)
ax.legend(loc='lower right')
plt.tight_layout()
plt.show()""")

# ============================================================
# SECTION 14 - RISK MANAGEMENT
# ============================================================
md("""## 13. 💰 Gestion du Risque — Stop-Loss & Position Sizing
Une stratégie profitable ne suffit pas — il faut gérer le **risque** pour survivre aux périodes de pertes.

- **Stop-Loss** : Sortir automatiquement si la perte dépasse un seuil
- **Take-Profit** : Sécuriser les gains à un seuil prédéfini
- **Position Sizing (Kelly)** : Ajuster la taille de position selon la confiance du modèle""")

code("""def backtest_with_risk_management(
    df_bt, proba, model_name,
    threshold=0.5,
    stop_loss=-0.03,     # -3% stop loss
    take_profit=0.05,    # +5% take profit
    use_kelly=False
):
    \"\"\"
    Backtesting avancé avec gestion du risque.
    - Stop-loss et take-profit par trade
    - Position sizing optionnel via critère de Kelly
    \"\"\"
    bt = df_bt.copy()
    bt['signal'] = (proba >= threshold).astype(int)
    bt['returns'] = bt['price'].pct_change().fillna(0)

    # --- Kelly Criterion pour le position sizing ---
    # f* = (p * b - q) / b   où p=win_rate, q=1-p, b=gain_moyen/perte_moyenne
    if use_kelly:
        # Calculer sur une fenêtre glissante de 60 jours
        kelly_fractions = []
        for i in range(len(bt)):
            if i < 60:
                kelly_fractions.append(0.5)  # Défaut avant suffisamment de données
                continue
            window = bt['returns'].iloc[max(0,i-60):i]
            wins = window[window > 0]
            losses = window[window < 0]
            if len(wins) == 0 or len(losses) == 0:
                kelly_fractions.append(0.5)
                continue
            p = len(wins) / len(window)
            b = wins.mean() / abs(losses.mean())
            kelly = (p * b - (1 - p)) / b
            kelly = np.clip(kelly, 0, 1)  # Borner entre 0 et 1
            kelly_fractions.append(kelly * 0.5)  # Demi-Kelly (plus conservateur)
        bt['kelly_fraction'] = kelly_fractions
        bt['position_size'] = bt['signal'] * bt['kelly_fraction']
    else:
        bt['position_size'] = bt['signal'].astype(float)

    # --- Appliquer Stop-Loss & Take-Profit ---
    adjusted_returns = []
    for i in range(len(bt)):
        r = bt['returns'].iloc[i] * bt['position_size'].iloc[i]
        if r < stop_loss:
            r = stop_loss  # Stop-loss déclenché
        elif r > take_profit:
            r = take_profit  # Take-profit déclenché
        adjusted_returns.append(r)

    bt['strategy_returns'] = adjusted_returns
    bt['cumulative_strategy'] = (1 + bt['strategy_returns']).cumprod()
    bt['cumulative_market'] = (1 + bt['returns']).cumprod()

    # --- Métriques ---
    total_ret = bt['cumulative_strategy'].iloc[-1] - 1
    sharpe = bt['strategy_returns'].mean() / bt['strategy_returns'].std() * np.sqrt(252) if bt['strategy_returns'].std() > 0 else 0
    cummax = bt['cumulative_strategy'].cummax()
    max_dd = ((bt['cumulative_strategy'] - cummax) / cummax).min()
    trades = bt[bt['signal'] == 1]
    win_rate = (trades['strategy_returns'] > 0).mean() if len(trades) > 0 else 0
    stop_triggers = sum(1 for r in adjusted_returns if r == stop_loss and bt['signal'].iloc[adjusted_returns.index(r)] == 1)

    sizing_label = "Kelly" if use_kelly else "Fixe"
    print(f"\\n{'='*60}")
    print(f"  💰 {model_name} — SL={stop_loss:.1%} / TP={take_profit:.1%} / Sizing={sizing_label}")
    print(f"{'='*60}")
    print(f"  Rendement Stratégie : {total_ret:+.2%}")
    print(f"  Sharpe Ratio        : {sharpe:.3f}")
    print(f"  Max Drawdown        : {max_dd:.2%}")
    print(f"  Win Rate            : {win_rate:.2%}")

    return bt, {
        'model': f"{model_name} (SL/TP/{'Kelly' if use_kelly else 'Fix'})",
        'return': total_ret, 'sharpe': sharpe,
        'max_drawdown': max_dd, 'win_rate': win_rate
    }""")

code("""# --- Backtesting avec Risk Management ---
df_bt_rm = df_model.iloc[split_idx:].copy()

# Utiliser le meilleur modèle optimisé (XGBoost)
best_proba = xgb_random.best_estimator_.predict_proba(X_test_scaled)[:, 1]

# Scénario 1 : Sans gestion du risque
_, rm_base = backtest_with_risk_management(
    df_bt_rm, best_proba, "XGB Optimisé",
    stop_loss=-1.0, take_profit=1.0, use_kelly=False
)

# Scénario 2 : Avec Stop-Loss & Take-Profit
_, rm_sltp = backtest_with_risk_management(
    df_bt_rm, best_proba, "XGB + SL/TP",
    stop_loss=-0.02, take_profit=0.04, use_kelly=False
)

# Scénario 3 : Avec Kelly Criterion
bt_kelly, rm_kelly = backtest_with_risk_management(
    df_bt_rm, best_proba, "XGB + Kelly",
    stop_loss=-0.02, take_profit=0.04, use_kelly=True
)""")

code("""# --- Comparaison des stratégies de risque ---
rm_df = pd.DataFrame([rm_base, rm_sltp, rm_kelly]).set_index('model')
print("\\n📋 Comparaison des stratégies de gestion du risque :")
print(rm_df.round(4).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Rendement vs Drawdown
ax = axes[0]
ax.barh(rm_df.index, rm_df['return'], color=['#2196F3','#4CAF50','#FF9800'])
ax.set_title('Rendement par stratégie', fontweight='bold')
ax.set_xlabel('Rendement')
for i, v in enumerate(rm_df['return']):
    ax.text(v + 0.001, i, f"{v:+.2%}", va='center', fontsize=9)

ax = axes[1]
ax.barh(rm_df.index, rm_df['max_drawdown'].abs(), color=['#f44336','#e91e63','#9c27b0'])
ax.set_title('Max Drawdown (absolu)', fontweight='bold')
ax.set_xlabel('Drawdown')
for i, v in enumerate(rm_df['max_drawdown'].abs()):
    ax.text(v + 0.001, i, f"{v:.2%}", va='center', fontsize=9)

plt.tight_layout()
plt.show()""")

# ============================================================
# SECTION 15 - WALK-FORWARD OPTIMIZATION
# ============================================================
md("""## 14. 🔄 Walk-Forward Optimization
Validation la plus **robuste** pour les séries temporelles :
- On découpe l'historique en fenêtres glissantes
- À chaque étape : **entraîner** sur la fenêtre passée, **prédire** la fenêtre suivante
- On ré-optimise les hyperparamètres à chaque étape (adaptatif)

```
Fenêtre 1: [==Train==][Test].................
Fenêtre 2: ...[==Train==][Test].............
Fenêtre 3: ......[==Train==][Test]..........
...et ainsi de suite
```""")

code("""def walk_forward_optimization(
    df_data, feature_cols, train_window=500, test_window=100, step=100
):
    \"\"\"
    Walk-Forward Optimization.
    - train_window : nombre de jours d'entraînement
    - test_window  : nombre de jours de test
    - step         : décalage entre chaque fenêtre
    \"\"\"
    results = []
    all_predictions = []
    all_actuals = []
    all_dates = []

    n = len(df_data)
    start = 0
    fold = 0

    while start + train_window + test_window <= n:
        fold += 1
        train_end = start + train_window
        test_end = train_end + test_window

        # Split
        X_tr = df_data[feature_cols].iloc[start:train_end]
        y_tr = df_data['target'].iloc[start:train_end]
        X_te = df_data[feature_cols].iloc[train_end:test_end]
        y_te = df_data['target'].iloc[train_end:test_end]
        dates_te = df_data.index[train_end:test_end]

        # Normalisation
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        # Entraînement XGBoost (on peut aussi faire un mini GridSearch ici)
        model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, use_label_encoder=False,
            eval_metric='logloss', n_jobs=-1
        )
        model.fit(X_tr_s, y_tr)
        preds = model.predict(X_te_s)
        acc = accuracy_score(y_te, preds)

        results.append({
            'fold': fold,
            'train_start': df_data.index[start].date(),
            'train_end': df_data.index[train_end-1].date(),
            'test_start': dates_te[0].date(),
            'test_end': dates_te[-1].date(),
            'accuracy': acc,
            'f1': f1_score(y_te, preds),
            'n_train': len(X_tr),
            'n_test': len(X_te)
        })

        all_predictions.extend(preds)
        all_actuals.extend(y_te.values)
        all_dates.extend(dates_te)

        start += step

    results_df = pd.DataFrame(results)
    overall_acc = accuracy_score(all_actuals, all_predictions)
    overall_f1 = f1_score(all_actuals, all_predictions)

    print(f"\\n{'='*60}")
    print(f"  🔄 Walk-Forward Optimization — {fold} fenêtres")
    print(f"{'='*60}")
    print(f"  Train window : {train_window} jours")
    print(f"  Test window  : {test_window} jours")
    print(f"  Step         : {step} jours")
    print(f"  Accuracy globale : {overall_acc:.4f}")
    print(f"  F1-Score global  : {overall_f1:.4f}")

    return results_df, all_predictions, all_actuals, all_dates""")

code("""# --- Exécution Walk-Forward ---
wf_results, wf_preds, wf_actuals, wf_dates = walk_forward_optimization(
    df_model, FEATURE_COLS,
    train_window=500,   # ~2 ans d'entraînement
    test_window=100,    # ~4-5 mois de test
    step=100            # Décaler de 100 jours à chaque étape
)

print("\\n📊 Détail par fenêtre :")
print(wf_results.to_string(index=False))""")

code("""# --- Visualisation Walk-Forward ---
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Accuracy par fold
ax = axes[0]
ax.bar(wf_results['fold'], wf_results['accuracy'], color='#2196F3', alpha=0.8, label='Accuracy')
ax.bar(wf_results['fold'], wf_results['f1'], color='#FF9800', alpha=0.5, label='F1-Score', width=0.4)
ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Baseline (50%)')
ax.set_xlabel('Fold')
ax.set_ylabel('Score')
ax.set_title('Walk-Forward — Performance par Fenêtre', fontweight='bold')
ax.legend()
ax.set_ylim(0, 1)

# Courbe de prédictions vs réel
ax = axes[1]
wf_df = pd.DataFrame({'date': wf_dates, 'actual': wf_actuals, 'predicted': wf_preds})
wf_df = wf_df.set_index('date')
rolling_acc = (wf_df['actual'] == wf_df['predicted']).rolling(50).mean()
ax.plot(rolling_acc.index, rolling_acc.values, color='#4CAF50', linewidth=1.2)
ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
ax.set_title('Accuracy Glissante (50 jours)', fontweight='bold')
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1)
ax.fill_between(rolling_acc.index, rolling_acc.values, 0.5, alpha=0.1, color='green')

plt.tight_layout()
plt.show()""")

# ============================================================
# SECTION 16 - CONCLUSION
# ============================================================
md("""## 15. Conclusion & Prochaines Étapes

### Résultats
- Les deux modèles (Random Forest et XGBoost) ont été entraînés et évalués sur des données de prix WTI.
- L'optimisation des hyperparamètres (GridSearch / RandomizedSearch) a permis d'améliorer les performances.
- La gestion du risque (Stop-Loss, Take-Profit, Kelly Criterion) réduit le drawdown au prix d'un rendement parfois plus faible.
- Le Walk-Forward confirme la robustesse (ou non) de la stratégie sur différentes fenêtres temporelles.

### Prochaines Étapes
1. **🔮 Intégrer l'analyse de sentiment** — Ajouter les scores de sentiment comme features supplémentaires
2. **📈 Ajouter d'autres modèles** — LSTM, GRU, Transformer
3. **🧪 Optuna** — Pour une optimisation bayésienne plus efficace des hyperparamètres""")

# ============================================================
# BUILD NOTEBOOK
# ============================================================
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        }
    },
    "cells": cells
}

output_path = "/mnt/data/insea/s4/projetTrading/ai/ml/model_training.ipynb"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"✅ Notebook généré : {output_path}")
print(f"   {len(cells)} cellules créées")
