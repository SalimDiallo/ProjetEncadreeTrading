# 🛢️ Rapport de Pipeline - Trading Algorithmique WTI Crude Oil

Ce document décrit le fonctionnement global et l'architecture technique du pipeline de trading algorithmique mis en place pour ce TP.

---

## 📊 Représentation Graphique du Pipeline

```mermaid
graph TD
    %% Styling
    classDef step1 fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#0D47A1;
    classDef step2 fill:#FFF3E0,stroke:#EF6C00,stroke-width:2px,color:#E65100;
    classDef step3a fill:#F3E5F5,stroke:#6A1B9A,stroke-width:2px,color:#4A148C;
    classDef step3b fill:#E0F2F1,stroke:#00695C,stroke-width:2px,color:#004D40;
    classDef step4 fill:#FFFDE7,stroke:#F57F17,stroke-width:2px,color:#F57F17;
    classDef step5 fill:#FFEBEE,stroke:#C62828,stroke-width:2px,color:#B71C1C;
    classDef step6 fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20;
    classDef database fill:#ECEFF1,stroke:#37474F,stroke-width:2px,color:#263238;

    subgraph Step1["1. COLLECTE DES DONNÉES (SCRAPING)"]
        News["📰 News (Web Scraping)<br/>Reuters, Bloomberg, OilPrice.com"]
        APIs["📈 API Financières<br/>Yahoo Finance, EIA, Alpha Vantage"]
        Raw["🗄️ Stockage Brut<br/>(Raw Data json/csv)"]:::database
        News --> Raw
        APIs --> Raw
    end
    class News,APIs step1;

    subgraph Step2["2. ORCHESTRATION & AUTOMATISATION"]
        Airflow["⚙️ Apache Airflow / Script Python<br/>(Scheduling, DAGs, Monitoring, Logging)"]:::step2
    end
    Raw --> Airflow

    subgraph Step3A["3A. TRAITEMENT DES NEWS (NLP)"]
        NLP_Pre["🐍 Prétraitement du texte<br/>(Nettoyage, tokenisation, ponctuation)"]
        NLP_Sent["🧠 Analyse de Sentiment<br/>Score de sentiment (-1 à 1) (FinBERT, VADER)"]
        NLP_Pre --> NLP_Sent
    end
    class NLP_Pre,NLP_Sent step3a;

    subgraph Step3B["3B. TRAITEMENT DES DONNÉES DE MARCHÉ"]
        Mkt_Pre["📉 Prétraitement des données<br/>(OHLCV, Nettoyage, Alignement)"]
        Mkt_Ind["📊 Indicateurs Techniques<br/>(RSI, MACD, Moyennes mobiles, Bollinger)"]
        Mkt_Pre --> Mkt_Ind
    end
    class Mkt_Pre,Mkt_Ind step3b;

    Airflow --> NLP_Pre
    Airflow --> Mkt_Pre

    subgraph Step4["4. FUSION SENTIMENTS + INDICATEURS"]
        Model["🤖 Modèle d'apprentissage supervisé<br/>RANDOM FOREST / XGBOOST"]
        Signal["🎯 Sortie : Signal de trading<br/>(BUY, HOLD, SELL)"]
        Model --> Signal
    end
    class Model,Signal step4;

    NLP_Sent --> Model
    Mkt_Ind --> Model

    subgraph Step5["5. BACKTESTING"]
        Backtest["📈 Simulation de stratégie<br/>Calcul de performance historique"]
        Metrics["⭐ Évaluation de la stratégie<br/>(Sharpe, Drawdown, Win rate)"]
        Backtest --> Metrics
    end
    class Backtest,Metrics step5;

    Signal --> Backtest

    subgraph Step6["6. VISUALISATION & INTERFACE"]
        Streamlit["👑 Streamlit Dashboard<br/>Graphiques, Indicateurs, Journal, Signaux"]:::step6
    end

    Metrics --> Streamlit
```

---

## 📝 Description Étape par Étape

### 1. Collecte des Données (Scraping)
* **Actualités (NLP) :** Récupération automatique d'articles et de titres à partir de sites d'information financière clés (comme *OilPrice.com*, *Reuters*, *Bloomberg*). Ces données sont stockées au format brut pour alimenter le pipeline de traitement de langage naturel.
* **API Financières :** Téléchargement des données de prix historiques (données quotidiennes et horaires OHLCV - Open, High, Low, Close, Volume) à partir de Yahoo Finance, de l'EIA (Energy Information Administration) et d'autres APIs financières.
* **Fichiers intermédiaires :** Fichiers enregistrés dans `scraping/src/data/raw/`.

### 2. Orchestration & Automatisation
* Orchestration complète du flux à l'aide d'**Apache Airflow** (ou via des scripts d'automatisation Python comme `scraping/src/main.py` et `ai/nlp/run.py`).
* Automatisation de l'exécution récurrente des tâches (Scheduling), création de DAGs de dépendance, surveillance en temps réel (Monitoring) et journalisation centralisée (Logging).

### 3A. Traitement des Nouvelles (NLP)
* **Prétraitement :** Nettoyage des textes, suppression des caractères spéciaux, tokenisation et classification des articles selon la pertinence par rapport au pétrole (`oil_density`).
* **Classification par actif :** Identification automatique de l'impact sur le **WTI** ou le **Brent** par reconnaissance de mots-clés.
* **Analyse de Sentiment :** Utilisation de modèles de pointe comme **FinBERT** (réseau de neurones spécialisé dans la finance) et **VADER** (analyse lexicale) pour obtenir un score de sentiment continu (de `-1` très négatif à `+1` très positif).
* **Agrégation journalière :** Compilation des sentiments quotidiens pondérés par la pertinence et le niveau de confiance du modèle.

### 3B. Traitement des Données de Marché
* **Prétraitement :** Nettoyage des données historiques de prix, gestion des valeurs manquantes et synchronisation temporelle.
* **Indicateurs Techniques :** Calcul d'indicateurs classiques pour le trading quantitatif :
  * **Moyennes Mobiles (SMA/EMA) :** Suivi de tendance à court/moyen terme.
  * **RSI (Relative Strength Index) :** Identification des zones de surachat/survente.
  * **MACD (Moving Average Convergence Divergence) :** Dynamique de tendance.
  * **Bandes de Bollinger :** Volatilité des prix.

### 4. Fusion Sentiments + Indicateurs
* Regroupement des indicateurs techniques de prix et des signaux de sentiment NLP agrégés.
* Entraînement d'un modèle d'apprentissage supervisé : **Random Forest** (Forêt d'arbres décisionnels) ou **XGBoost**.
* **Sortie du modèle :** Génération d'un signal de trading quotidien : `BUY` (Achat), `HOLD` (Conserver), ou `SELL` (Vente).

### 5. Backtesting
* Simulation historique de la stratégie de trading sur les données passées pour évaluer sa rentabilité.
* Calcul précis des métriques de performance et de risque clés :
  * **Ratio de Sharpe :** Performance ajustée au risque.
  * **Maximum Drawdown (MDD) :** Perte maximale historique subie depuis un sommet.
  * **Win Rate :** Taux de réussite des transactions.
  * **CAGR :** Taux de croissance annuel composé.

### 6. Visualisation & Interface
* Interface utilisateur interactive développée sous **Streamlit** accessible localement (port `8501`).
* **Fonctionnalités :**
  * **Recommandation :** Verdict clair en temps réel avec score de confiance.
  * **Analyse de prix :** Graphiques interactifs Plotly avec options d'affichage des indicateurs techniques.
  * **Backtest :** Courbes d'équité et analyse comparée par rapport à une stratégie *Buy & Hold*.
  * **Journal des transactions :** Suivi et historique des signaux générés.
