# 🛢️ Projet Trading — WTI Crude Oil

> Projet encadré — INSEA S4 | Analyse et prédiction du prix du pétrole WTI

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
│       ├── main.py                  #    Point d'entrée scraping
│       ├── config.py                #    Configuration
│       ├── collectors/              #    Collecteurs de données
│       │   ├── base.py              #       Classe de base
│       │   ├── price_collector.py   #       Collecteur de prix WTI/Brent
│       │   └── news_collector.py    #       Collecteur d'actualités
│       ├── processors/              #    Traitement des données
│       │   └── petrol_processor.py  #       Processeur pétrole
│       └── data/
│           ├── raw/                 #    Données brutes (JSON)
│           └── processed/           #    Données traitées
│               ├── wti_petrole_3ans.csv        # OHLCV journalier 3 ans
│               ├── wti_petrole_horaire_2ans.csv# OHLCV horaire 2 ans
│               ├── petrol_wti_daily.parquet    # Prix WTI quotidien
│               ├── petrol_brent_daily.parquet  # Prix Brent quotidien
│               └── petrol_news_oilprice.parquet# Actualités pétrole
│
├── web/                             # 🌐 Interface Web (Dashboard)
│   └── notes.md                     #    Notes de développement
│
├── .gitignore                       #    Fichiers ignorés par Git
├── pyproject.toml                   #    Configuration Python (uv)
├── uv.lock                          #    Lock des dépendances
├── main.py                          #    Point d'entrée principal
└── README.md                        #    Ce fichier
```

---

## ⚙️ Stack Technique

| Composant | Technologies |
|-----------|-------------|
| **Scraping** | Python, yfinance, BeautifulSoup |
| **ML / Backtesting** | scikit-learn, XGBoost, pandas, numpy |
| **NLP** | *(à venir — VADER / FinBERT)* |
| **Visualisation** | matplotlib, seaborn |
| **Web** | *(à venir)* |

## 🚀 Installation

```bash
# Cloner le projet
git clone git@github.com:SalimDiallo/ProjetEncadreeTrading.git
cd ProjetEncadreeTrading

# Installer les dépendances
uv sync
uv add pandas numpy scikit-learn xgboost matplotlib seaborn pyarrow
```
