# 🛢️ Oil Sentiment NLP Trading Pipeline

> **Pipeline end-to-end d'analyse de sentiment sur le marché pétrolier, de la collecte de données textuelles au backtesting de stratégies de trading.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-blueviolet)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 📋 Table des matières

- [Aperçu du projet](#-aperçu-du-projet)
- [Architecture du pipeline](#-architecture-du-pipeline)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Étapes du pipeline](#-étapes-du-pipeline)
  - [1 — Data Ingestion](#étape-1--data-ingestion)
  - [2 — Preprocessing](#étape-2--preprocessing)
  - [3 — Feature Engineering](#étape-3--feature-engineering)
  - [4 — Modeling (Analyse de Sentiment)](#étape-4--modeling-analyse-de-sentiment)
  - [5 — Agrégation](#étape-5--agrégation)
  - [6 — Signal & Backtesting](#étape-6--signal--backtesting)
- [Lancement du pipeline complet](#-lancement-du-pipeline-complet)
- [Flux de données](#-flux-de-données)
- [Organisation en équipe](#-organisation-en-équipe)
- [Tests](#-tests)
- [Structure du projet](#-structure-du-projet)
- [Licence](#-licence)

---

## 🎯 Aperçu du projet

Ce projet construit un **pipeline NLP complet** qui :

1. **Collecte** des textes financiers (actualités, Reddit, Twitter/X, SEC EDGAR)
2. **Nettoie et normalise** les textes pour le domaine pétrolier
3. **Extrait des features** (TF-IDF, embeddings FinBERT)
4. **Analyse le sentiment** avec 3 niveaux de modèles (lexical → LR → FinBERT)
5. **Agrège** les scores en série temporelle
6. **Génère des signaux** de trading (BUY / SELL / HOLD)
7. **Backteste** la stratégie sur les prix historiques du pétrole (WTI)

### Modèles de sentiment disponibles

| Modèle | Type | Dépendances | Précision |
|--------|------|-------------|-----------|
| **Lexical** | Dictionnaire (~60 termes bullish/bearish) | Aucune | Baseline |
| **Logistic Regression** | Supervisé (TF-IDF) | scikit-learn | Intermédiaire |
| **FinBERT** | Transformer pré-entraîné (ProsusAI/finbert) | torch, transformers | Meilleure |

### Stratégies de trading

| Stratégie | Description |
|-----------|-------------|
| `threshold` | Seuils fixes sur le score de sentiment |
| `momentum` | Croisement de moyennes mobiles (MA3 / MA7) |
| `zscore` | Signal sur dépassement du z-score (±0.5, fenêtre 14j) |
| `regime` | Détection de régimes par quantiles rolling (21j) |

---

## 🏗️ Architecture du pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Data Ingestion │────▶│  Preprocessing   │────▶│ Feature Engineering │
│  (4 sources)    │     │  (nettoyage NLP) │     │  (TF-IDF + embeds)  │
└─────────────────┘     └──────────────────┘     └──────────┬──────────┘
                                                            │
                                                            ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   Backtesting   │◀────│  Signal Trading  │◀────│  Sentiment Modeling │
│  (métriques)    │     │  (BUY/SELL/HOLD) │     │ (FinBERT / LR / Lex)│
└─────────────────┘     └──────────────────┘     └─────────────────────┘
                                ▲
                                │
                        ┌───────┴────────┐
                        │  Agrégation    │
                        │ (série tempo.) │
                        └────────────────┘
```

---

## 🚀 Installation

### Prérequis

- **Python 3.10+**
- **pip** (gestionnaire de paquets)

### Installation rapide

```powershell
# Cloner le projet
git clone <url-du-repo>
cd NLP

# Créer et activer l'environnement virtuel
python -m venv .venv
.venv\Scripts\Activate.ps1

# Installer toutes les dépendances (core + ML + dev)
pip install -r requirements.txt

# Ou via pip install (mode éditable, recommandé pour le dev)
pip install -e ".[ml,dev]"
```

> **Note :** Si vous n'avez pas besoin de FinBERT (torch ~2GB), les dépendances `transformers` et `torch` sont optionnelles. Le pipeline basculera automatiquement sur le modèle lexical.

### Variables d'environnement (optionnel)

```powershell
Copy-Item .env.example .env
# Éditer .env pour renseigner vos clés API
```

| Variable | Usage | Requis ? |
|----------|-------|----------|
| `REDDIT_CLIENT_ID` | API Reddit (PRAW) | Non — mode mock sans clé |
| `REDDIT_CLIENT_SECRET` | API Reddit (PRAW) | Non — mode mock sans clé |
| `TWITTER_BEARER_TOKEN` | API Twitter/X v2 | Non — mode mock sans clé |
| `SEC_USER_AGENT` | API EDGAR (SEC) | Recommandé |
| `HF_HOME` | Cache local HuggingFace | Non |

> Sans clés API, le pipeline fonctionne en **mode mock** avec des données simulées réalistes.

---

## ⚙️ Configuration

La configuration centrale se trouve dans `config/pipeline.yaml` :

```yaml
pipeline:
  sources: [news, reddit, twitter, edgar]   # Sources de données
  model: auto                                # auto | finbert | logistic_regression | lexical
  strategy: threshold                        # threshold | momentum | zscore | regime
  freq: "1D"                                 # Granularité : 1min, 15min, 1H, 1D, 1W
  start_date: "2023-01-01"                   # Début période backtest
  end_date: "2024-12-31"                     # Fin période backtest
  ticker: "CL=F"                             # Ticker pétrole WTI
  buy_threshold: 0.05                        # Seuil BUY
  sell_threshold: -0.05                      # Seuil SELL
  smooth_window: 3                           # Lissage sentiment (nb périodes)

aggregation:
  weight_by_confidence: true                 # Pondérer par confiance du modèle
  weight_by_oil_density: true                # Pondérer par densité termes pétrole
  weight_by_source: true                     # Pondérer par fiabilité source
  source_weights:                            # Poids personnalisés par source
    edgar: 1.4
    yahoo_finance: 1.2
    reuters: 1.15
    reddit: 0.85
    twitter: 0.75
```

Tous les paramètres sont **surchargeables en CLI** (voir [Options CLI](#options-cli-complètes)).

---

## 📦 Étapes du pipeline

### Étape 1 — Data Ingestion

Collecte de données textuelles depuis 4 sources. Chaque record produit :

```json
{ "text": "OPEC cuts production...", "date": "2024-03-15T08:00:00Z", "source": "reuters" }
```

| Source | Méthode | Commande |
|--------|---------|----------|
| **News** | Yahoo Finance API + 5 flux RSS | `python -m oil_sentiment_pipeline.data_ingestion.news_scraper` |
| **Reddit** | PRAW (r/investing, r/oil, r/energy) | `python -m oil_sentiment_pipeline.data_ingestion.reddit_collector` |
| **Twitter/X** | API v2 par mots-clés | `python -m oil_sentiment_pipeline.data_ingestion.twitter_scraper` |
| **SEC EDGAR** | Filings 10-K, 10-Q, 8-K | `python -m oil_sentiment_pipeline.data_ingestion.edgar_parser` |

**Collecteur consolidé** (toutes sources → un seul CSV) :

```powershell
python -m oil_sentiment_pipeline.data_ingestion.collector
# → data/raw/all_sources_YYYYMMDD_HHMMSS.csv
```

---

### Étape 2 — Preprocessing

Nettoyage et normalisation des textes bruts pour le domaine pétrolier.

| Opération | Exemple |
|-----------|---------|
| Suppression URLs | `http://...` → *(supprimé)* |
| Suppression mentions/cashtags | `@Reuters`, `$XOM` → *(supprimé)* |
| Lowercasing | `OPEC` → `opec` |
| Suppression stopwords | `the, is, at` → *(supprimés)* |
| Lemmatisation | `drops` → `drop`, `raises` → `raise` |
| Normalisation vocabulaire pétrole | `wti crude oil` → `crude` |
| Gestion des négations | `not bullish` → `not_bullish` |
| Score de densité pétrole | Ratio de termes pétrole dans le texte |

```powershell
# Pipeline complet preprocessing
python -m oil_sentiment_pipeline.preprocessing.pipeline
# → data/processed/processed_YYYYMMDD_HHMMSS.csv
```

---

### Étape 3 — Feature Engineering

Transformation des textes nettoyés en représentations numériques.

- **TF-IDF** — Matrice sparse (unigrammes + bigrammes), optimisée vocabulaire pétrolier
- **Embeddings** — Vecteurs denses 768D via FinBERT (ou mock si torch absent)

```powershell
python -m oil_sentiment_pipeline.feature_engineering.pipeline
# → models/tfidf_*.pkl + models/embeddings_*.npy
```

---

### Étape 4 — Modeling (Analyse de Sentiment)

Scoring de chaque record avec un label et un score de sentiment.

**Chaque record en sortie est enrichi avec :**

```python
{
    **record_original,                          # Tous les champs d'entrée conservés
    "sentiment_label":      str,               # "positive" | "neutral" | "negative"
    "sentiment_score":      float,             # [-1.0, +1.0]
    "sentiment_confidence": float,             # [0.0, 1.0]
    "model":                str,               # "finbert" | "logistic_regression" | "lexical"
}
```

**Sélection automatique** (`model=auto`) : FinBERT si `torch` disponible, sinon lexical.

```powershell
# Pipeline modeling complet
python -m oil_sentiment_pipeline.modeling.pipeline
# → data/sentiment/sentiment_YYYYMMDD_HHMMSS.csv
```

---

### Étape 5 — Agrégation

Consolidation des scores individuels en **série temporelle** par période (minute, heure, jour, semaine).

**Colonnes produites :**

| Colonne | Description |
|---------|-------------|
| `date` | Timestamp aligné sur la granularité |
| `sentiment_mean` | Moyenne brute des scores |
| `sentiment_weighted` | Moyenne pondérée (confiance × source × densité pétrole) |
| `sentiment_std` | Écart-type (divergence d'opinions) |
| `volume` | Nombre de textes sur la période |
| `positive_ratio` / `negative_ratio` | Proportion de textes positifs / négatifs |
| `net_ratio` | `(positifs - négatifs) / total` |

```powershell
python -m oil_sentiment_pipeline.aggregation.aggregator
# → data/aggregated/sentiment_aggregated_YYYYMMDD_HHMMSS.csv
```

---

### Étape 6 — Signal & Backtesting

**Génération de signaux** : Transforme la série de sentiment en positions discrètes : **BUY (+1)** / **HOLD (0)** / **SELL (-1)**.

**Backtesting** : Simule la stratégie sur les prix historiques du pétrole (WTI `CL=F` via yfinance).

| Métrique | Description |
|----------|-------------|
| Total Return | Performance cumulée |
| CAGR | Taux de croissance annuel composé |
| Sharpe Ratio | Rendement ajusté du risque |
| Max Drawdown | Perte maximale depuis un pic |
| Win Rate | % de jours rentables |
| Alpha / Beta | Surperformance vs Buy & Hold |
| Calmar Ratio | CAGR / Max Drawdown |

```powershell
python -m oil_sentiment_pipeline.signal.signal_generator
# → data/signals/signals_YYYYMMDD_HHMMSS.csv

python -m oil_sentiment_pipeline.backtest.backtester
# → data/backtest/backtest_daily_YYYYMMDD.csv
# → data/backtest/backtest_metrics_YYYYMMDD.json
# → data/backtest/backtest_chart_YYYYMMDD.png
```

---

## 🎮 Lancement du pipeline complet

```powershell
# Lancement standard (modèle auto, stratégie seuils)
python -m oil_sentiment_pipeline.main

# Avec FinBERT + stratégie momentum
python -m oil_sentiment_pipeline.main --model finbert --strategy momentum

# Test rapide sans graphique
python -m oil_sentiment_pipeline.main --model lexical --strategy threshold --no-plot --no-save

# Long-only avec période personnalisée
python -m oil_sentiment_pipeline.main --no-short --start-date 2022-01-01 --end-date 2024-12-31

# Agrégation horaire + walk-forward validation (3 folds)
python -m oil_sentiment_pipeline.main --freq 1H --walk-forward 3
```

### Options CLI complètes

| Option | Défaut | Description |
|--------|--------|-------------|
| `--model` | `auto` | `auto` / `finbert` / `logistic_regression` / `lexical` |
| `--strategy` | `threshold` | `threshold` / `momentum` / `zscore` / `regime` |
| `--sources` | `news reddit twitter edgar` | Sources à collecter |
| `--freq` | `1D` | Granularité d'agrégation (`1min`, `15min`, `1H`, `1D`, `1W`) |
| `--start-date` | `2023-01-01` | Début de la période de backtest |
| `--end-date` | `2024-12-31` | Fin de la période de backtest |
| `--ticker` | `CL=F` | Ticker du pétrole (WTI) |
| `--buy-threshold` | `0.05` | Seuil de déclenchement signal BUY |
| `--sell-threshold` | `-0.05` | Seuil de déclenchement signal SELL |
| `--smooth-window` | `3` | Fenêtre de lissage du sentiment (en périodes) |
| `--cost` | `0.001` | Coût de transaction (fraction, 0.1%) |
| `--signal-lag` | `1` | Jours entre signal et prise de position (anti look-ahead) |
| `--walk-forward` | `0` | Nombre de folds walk-forward (0 = désactivé) |
| `--max-per-source` | `30` | Nombre max de records par source |
| `--no-short` | — | Mode long-only (pas de positions short) |
| `--no-plot` | — | Désactive les graphiques |
| `--no-save` | — | Désactive la sauvegarde des fichiers |
| `--log-level` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

---

## 📊 Flux de données

```
Sources texte                    Répertoire de sortie          Format
───────────────────────────────────────────────────────────────────────
Yahoo Finance / RSS
Reddit / Twitter / EDGAR   ───▶ data/raw/                     CSV
        │
        ▼
   Preprocessing            ───▶ data/processed/               CSV
   (nettoyage, tokens)
        │
        ▼
  Feature Engineering       ───▶ models/                       .pkl, .npy
  (TF-IDF, embeddings)
        │
        ▼
   Sentiment Modeling       ───▶ data/sentiment/               CSV
   (lexical / LR / FinBERT)
        │
        ▼
   Agrégation temporelle    ───▶ data/aggregated/              CSV
        │
        ▼
   Signaux de trading       ───▶ data/signals/                 CSV
        │
        ▼
   Backtesting              ───▶ data/backtest/                CSV + JSON + PNG
```

> Les dossiers `data/` et `models/` contiennent des **artefacts générés** : ils ne sont pas versionnés (voir `.gitignore`).

---

## 👥 Organisation en équipe

Le pipeline est conçu pour être **modulaire et indépendant** entre les étapes. Voici un découpage possible pour une équipe de 4 personnes :

| Rôle | Modules | Entrée | Sortie |
|------|---------|--------|--------|
| **Data Engineer** | `data_ingestion/`, `preprocessing/` | Sources brutes | `data/processed/*.csv` |
| **NLP / Sentiment Analyst** | `modeling/` | Records prétraités | `data/sentiment/*.csv` avec `sentiment_label`, `sentiment_score`, `sentiment_confidence`, `model` |
| **ML Engineer** | `feature_engineering/`, entraînement modèles | Records prétraités | `models/*.pkl`, modèles entraînés |
| **Quant / Backtester** | `aggregation/`, `signal/`, `backtest/` | Records sentimentés | Signaux, métriques, graphiques |

### Contrat d'interface entre les modules

Chaque étape consomme la sortie de l'étape précédente via un format standardisé `List[Dict]` :

```
Ingestion → {"text", "date", "source"}
     ↓
Preprocessing → + {"text_clean", "tokens", "oil_density", "text_normalized"}
     ↓
Modeling → + {"sentiment_label", "sentiment_score", "sentiment_confidence", "model"}
     ↓
Agrégation → pd.DataFrame avec {"date", "sentiment_mean", "volume", ...}
```

---

## 🧪 Tests

```powershell
# Lancer tous les tests
pytest

# Tests avec verbosité
pytest -v

# Exclure les tests lourds (FinBERT / torch)
pytest -m "not slow"
```

---

## 🗂️ Structure du projet

```
NLP/
├── config/
│   └── pipeline.yaml              # Configuration centrale (surchargeable en CLI)
├── oil_sentiment_pipeline/
│   ├── data_ingestion/            # Collecte multi-sources (news, Reddit, Twitter, EDGAR)
│   ├── preprocessing/             # Nettoyage et normalisation des textes
│   ├── feature_engineering/       # TF-IDF et embeddings HuggingFace
│   ├── modeling/                  # Sentiment : lexical, Logistic Regression, FinBERT
│   ├── aggregation/               # Série temporelle de sentiment (multi-fréquence)
│   ├── evaluation/                # Évaluation des modèles (gold standard)
│   ├── signal/                    # Génération de signaux de trading
│   ├── backtest/                  # Simulation, métriques, graphiques
│   ├── data/                      # Artefacts générés (non versionnés)
│   │   ├── raw/                   #   Données brutes collectées
│   │   ├── processed/             #   Données après preprocessing
│   │   ├── sentiment/             #   Records scorés
│   │   ├── aggregated/            #   Agrégation temporelle
│   │   ├── signals/               #   Signaux de trading
│   │   ├── prices/                #   Prix pétrole (yfinance)
│   │   └── backtest/              #   Résultats + graphiques
│   ├── models/                    # Modèles sauvegardés (.pkl, .npy)
│   ├── logs/                      # Logs d'exécution
│   ├── main.py                    # Orchestrateur CLI complet
│   ├── paths.py                   # Résolution des chemins
│   └── settings.py                # Chargement de la configuration YAML
├── tests/                         # Suite de tests (pytest)
├── .env.example                   # Template des variables d'environnement
├── .gitignore
├── pyproject.toml                 # Metadata projet + dépendances
├── requirements.txt               # Dépendances (core + ML + dev)
└── README.md
```

---

## 📄 Licence

Ce projet est distribué sous licence **MIT**. Voir le fichier `LICENSE` pour plus de détails.
