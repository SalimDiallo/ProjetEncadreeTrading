# 🛢️ Web Dashboard — Petrol Trading Platform

Interface graphique du projet, basée sur **Streamlit**.
Affiche les prix, les indicateurs techniques, la recommandation
ACHETER / CONSERVER / VENDRE et les métriques de risque.

## 🚀 Lancement rapide

Depuis la racine du projet (avec l'environnement Python actif) :

```bash
# Installer les dépendances spécifiques au web
pip install -r web/requirements.txt

# Lancer le dashboard
cd web
streamlit run app.py
```

Le dashboard s'ouvre dans le navigateur sur `http://localhost:8501`.

## 📁 Structure

```
web/
├── app.py                      # Point d'entrée Streamlit
├── requirements.txt            # Dépendances web (streamlit, plotly...)
├── pytest.ini                  # Config tests
├── .streamlit/
│   └── config.toml             # Thème
│
├── components/                 # Composants UI réutilisables
│   ├── sidebar.py              # Filtres utilisateur
│   ├── price_chart.py          # Graphique (ligne/bougies) + indicateurs
│   ├── recommendation.py       # Verdict BUY/HOLD/SELL
│   ├── metrics_panel.py        # KPIs (Sharpe, Drawdown...)
│   └── trade_log.py            # Tableau des trades
│
├── utils/                      # Logique métier
│   ├── data_loader.py          # Lecture des Parquet/CSV
│   ├── indicators.py           # SMA, RSI, MACD, Bollinger, ATR
│   ├── metrics.py              # Sharpe, MDD, CAGR, Calmar...
│   ├── backtest.py             # Moteur de simulation
│   ├── formatters.py           # Helpers d'affichage
│   └── mock_data.py            # Données factices (fallback)
│
├── tests/                      # Suite de tests pytest
│   ├── conftest.py             # Fixtures partagées
│   ├── test_metrics.py         # 44 tests
│   ├── test_backtest.py        # 23 tests
│   ├── test_formatters.py      # 26 tests
│   └── test_integration.py     # 12 tests
│
└── assets/                     # Images, logos
```

## 🔌 Sources de données consommées

Le dashboard lit depuis `../scraping/src/data/processed/` :

| Fichier | Producteur | Statut |
|---|---|---|
| `petrol_wti_daily.parquet` | scraping/ | ✅ Disponible |
| `petrol_brent_daily.parquet` | scraping/ | ✅ Disponible |
| `wti_petrole_3ans.csv` | scraping/ | ✅ Disponible (OHLCV) |
| `petrol_news_oilprice.parquet` | scraping/ | ✅ Disponible |
| `signals_ml_wti.parquet` | ai/ml/ | 🟡 À venir |
| `signals_ml_brent.parquet` | ai/ml/ | 🟡 À venir |
| `sentiment_wti.parquet` | ai/nlp/ | 🟡 À venir |

⚠️ Tant que les fichiers ML/NLP n'existent pas, le dashboard utilise des **mocks** (stratégie SMA crossover 20/50, cohérente avec le baseline du notebook).

## 🧪 Tests

```bash
cd web
pytest                  # lance les 105 tests (~2s)
pytest --cov=utils      # avec couverture
pytest -x               # stop au 1er échec
```

**Couverture** : 100% sur `metrics.py`, `backtest.py`, `formatters.py`.

## 📊 Fonctionnalités

### Onglet "Recommandation"
- Verdict final ACHETER / CONSERVER / VENDRE en gros
- Score de confiance + justifications
- Mini-graphique des 90 derniers jours

### Onglet "Analyse de prix"
- Graphique en ligne ou bougies japonaises (OHLCV WTI)
- Indicateurs configurables : SMA, Bollinger, RSI, MACD
- Signaux BUY/SELL en surimpression

### Onglet "Backtest"
- Simulation complète avec frais configurables
- Comparaison stratégie vs Buy & Hold
- Courbe d'équité + drawdown
- 10 métriques (Sharpe, Sortino, Calmar, MDD, CAGR, win rate, profit factor...)
- Journal des trades avec P&L coloré

### Onglet "Actualités"
- Articles scrapés depuis OilPrice.com

## ⚠️ Bug connu côté scraping

La colonne `price` est stockée en `string` au lieu de `float` dans les Parquet.
Le `data_loader.py` fait un cast `pd.to_numeric` à la lecture en attendant
la correction dans `scraping/src/processors/petrol_processor.py`.

---

📝 **Avertissement** : outil à vocation **éducative**. Ne constitue pas
un conseil en investissement.
