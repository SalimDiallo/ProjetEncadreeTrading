# 🛢️ Projet Trading — WTI Crude Oil

> Projet encadré — INSEA S4 | Analyse et prédiction du prix du pétrole WTI

Plateforme modulaire de **backtesting algorithmique** combinant analyse technique,
Machine Learning (Random Forest, XGBoost) et analyse de sentiment NLP, le tout
exposé via un **dashboard Streamlit interactif**.

---

## 📁 Structure du Projet

```
projetTrading/
│
├── ai/                              # 🤖 Intelligence Artificielle
│   ├── main.py                      #    Point d'entrée AI
│   ├── ml/                          #    Machine Learning
│   │   ├── model_training.ipynb     #    Notebook backtesting (RF, XGBoost)
│   │   ├── model_trainingV1.ipynb   #    Version 1 du notebook
│   │   └── indicateurs_techniques.md#    Guide des indicateurs techniques
│   └── nlp/                         #    Traitement du Langage Naturel
│       └── run.py                   #    Analyse de sentiment (à venir)
│
├── scraping/                        # 🕷️ Collecte de Données
│   ├── main.ipynb                   #    Notebook de scraping
│   ├── README.md                    #    Documentation scraping
│   └── src/
│       ├── main.py                  #    Pipeline ETL
│       ├── config.py                #    Configuration
│       ├── collectors/              #    Collecteurs (EIA, OilPrice)
│       ├── processors/              #    Traitement des données
│       └── data/
│           ├── raw/                 #    JSON bruts horodatés
│           └── processed/           #    🔌 BUS DE DONNÉES PARTAGÉ
│               ├── wti_petrole_3ans.csv          # OHLCV 3 ans
│               ├── wti_petrole_horaire_2ans.csv  # OHLCV horaire 2 ans
│               ├── petrol_wti_daily.parquet      # Prix WTI quotidien
│               ├── petrol_brent_daily.parquet    # Prix Brent quotidien
│               └── petrol_news_oilprice.parquet  # Actualités
│
├── web/                             # 🎨 Dashboard Streamlit
│   ├── app.py                       #    Point d'entrée
│   ├── components/                  #    Composants UI
│   │   ├── sidebar.py               #    Filtres & paramètres
│   │   ├── price_chart.py           #    Graphiques + indicateurs
│   │   ├── recommendation.py        #    Verdict ACHETER/VENDRE
│   │   ├── metrics_panel.py         #    KPIs (Sharpe, MDD...)
│   │   └── trade_log.py             #    Journal des trades
│   ├── utils/                       #    Logique métier
│   │   ├── data_loader.py           #    Lecture des Parquet/CSV
│   │   ├── indicators.py            #    SMA, RSI, MACD, Bollinger
│   │   ├── metrics.py               #    Sharpe, MDD, CAGR...
│   │   ├── backtest.py              #    Moteur de simulation
│   │   ├── formatters.py            #    Helpers d'affichage
│   │   └── mock_data.py             #    Données factices (fallback)
│   ├── tests/                       #    97 tests pytest
│   ├── .streamlit/config.toml       #    Thème
│   ├── requirements.txt             #    Dépendances web
│   └── README.md                    #    Doc dashboard
│
├── .gitignore
├── pyproject.toml                   # Config Python (uv)
├── uv.lock                          # Lock des dépendances
├── main.py                          # Point d'entrée principal
├── INTERFACES.md                    # 🔌 Contrats entre modules
└── README.md                        # Ce fichier
```

---

## ⚙️ Stack Technique

| Composant | Technologies |
|-----------|-------------|
| **Scraping** | Python, requests, BeautifulSoup, yfinance |
| **ML / Backtesting** | scikit-learn, XGBoost, pandas, numpy |
| **NLP** | *(à venir — VADER / FinBERT)* |
| **Visualisation Notebook** | matplotlib, seaborn |
| **Dashboard Web** | **Streamlit, Plotly** |
| **Tests** | pytest, pytest-cov |
| **Gestion projet** | uv |

---

## 🚀 Installation

### Option 1 — Avec Docker (RECOMMANDÉ — zéro pollution locale)

Voir le guide complet dans **[DOCKER_QUICKSTART.md](./DOCKER_QUICKSTART.md)**.

```bash
# Build (une seule fois, ~5 min)
docker compose build

# Lancer le dashboard
docker compose up dashboard
# → http://localhost:8501
```

Pour développer dans VS Code directement dans le conteneur :
1. Installer l'extension **Dev Containers**
2. Ouvrir le projet → cliquer sur **"Reopen in Container"**

### Option 2 — Avec `uv`

```bash
# Cloner le projet
git clone git@github.com:SalimDiallo/ProjetEncadreeTrading.git
cd ProjetEncadreeTrading

# Installer toutes les dépendances
uv sync
uv add pandas numpy scikit-learn xgboost matplotlib seaborn pyarrow streamlit plotly pytest
```

### Option 3 — Avec `pip` (pour le dashboard seulement)

```bash
python -m venv venv
source venv/bin/activate          # Linux/Mac
# OU
venv\Scripts\activate             # Windows

pip install -r web/requirements.txt
```

---

## 🎯 Utilisation

### 1️⃣ Récupérer les données

```bash
cd scraping/src
python main.py
```

Ceci crée les fichiers Parquet dans `scraping/src/data/processed/`.

### 2️⃣ Entraîner les modèles ML (optionnel)

Ouvre `ai/ml/model_training.ipynb` dans Jupyter ou VS Code.

### 3️⃣ Lancer le dashboard web

```bash
cd web
streamlit run app.py
```

Le dashboard s'ouvre sur `http://localhost:8501`.

### 4️⃣ Lancer les tests

```bash
cd web
pytest                              # 97 tests, ~2 secondes
pytest --cov=utils                  # avec couverture
```

---

## 📊 Fonctionnalités du Dashboard

| Onglet | Contenu |
|---|---|
| 🎯 **Recommandation** | Verdict ACHETER/CONSERVER/VENDRE + score de confiance |
| 📈 **Analyse de prix** | Graphique (ligne ou bougies), SMA/RSI/MACD/Bollinger |
| 💼 **Backtest** | Simulation portefeuille, 10 métriques de risque, courbe d'équité |
| 📰 **Actualités** | Articles scrapés depuis OilPrice.com |

---

## 🔌 Communication entre modules

Voir **[INTERFACES.md](./INTERFACES.md)** pour les contrats détaillés.

En résumé : tous les modules écrivent/lisent dans `scraping/src/data/processed/`,
et le dashboard détecte automatiquement les nouveaux fichiers.

---

## 👥 Équipe

| Module | Responsable |
|---|---|
| 🕷️ Scraping & ETL | [Nom] |
| 🤖 Machine Learning | [Nom] |
| 📰 NLP Sentiment | [Nom] |
| 🎨 Dashboard Web | [Toi] |

---

## ⚠️ Avertissement

Cet outil est à vocation **éducative**. Il ne constitue **pas un conseil en investissement**.
Les performances passées ne préjugent pas des performances futures.
