# 🛢️ Modèle de Rapport de Projet Fin d'Études / TP

> **Projet Encadré — INSEA S4 | Analyse et prédiction du prix du pétrole WTI**
> **Sujet :** Plateforme modulaire de backtesting algorithmique combinant analyse technique, Machine Learning (Random Forest, XGBoost) et analyse de sentiment NLP.

---

## 📌 Page de Garde [À Personnaliser]
* **Titre :** Plateforme Modulaire de Trading Algorithmique sur le WTI Crude Oil
* **Sous-titre :** Intégration d'indicateurs quantitatifs, de modèles de Machine Learning et d'analyse sentimentale NLP
* **Membres du groupe :** [Prénom Nom (Scraping)], [Prénom Nom (Machine Learning)], [Prénom Nom (NLP)], [Prénom Nom (Web & Intégration)]
* **Encadrant :** [Nom du Professeur]
* **Date :** Juin 2026

---

## 📖 Sommaire
1. [Introduction & Contexte](#1-introduction--contexte)
2. [Module 1 : Collecte de Données (Scraping & Ingestion)](#2-module-1--collecte-de-données-scraping--ingestion)
3. [Module 2 : Analyse Technique & Traitement du Signal](#3-module-2--analyse-technique--traitement-du-signal)
4. [Module 3 : Pipeline NLP & Analyse de Sentiment](#4-module-3--pipeline-nlp--analyse-de-sentiment)
5. [Module 4 : Modélisation Machine Learning](#5-module-4--modélisation-machine-learning)
6. [Module 5 : Moteur de Backtesting & Métriques de Performance](#6-module-5--moteur-de-backtesting--métriques-de-performance)
7. [Module 6 : Dashboard Streamlit & Visualisation](#7-module-6--dashboard-streamlit--visualisation)
8. [Conclusion & Perspectives](#8-conclusion--perspectives)

---

## 1. Introduction & Contexte
Ce projet s'inscrit dans le cadre des projets encadrés de la quatrième année (S4) à l'INSEA. Il a pour but de concevoir et implémenter une plateforme complète et modulaire de trading algorithmique dédiée au pétrole brut de référence **WTI (West Texas Intermediate)**.

Le marché pétrolier est caractérisé par sa forte volatilité et sa sensibilité à la fois aux facteurs économiques (données d'inventaires de l'EIA, décisions de l'OPEP) et géopolitiques (actualités internationales). Pour capturer cette complexité, la plateforme intègre trois approches complémentaires :
1. **L'analyse quantitative** basée sur des indicateurs techniques classiques.
2. **Le Machine Learning (ML)** supervisé pour la prédiction de la direction du prix.
3. **Le Traitement du Langage Naturel (NLP)** pour extraire le sentiment du marché à partir de l'actualité et des réseaux sociaux.

---

## 2. Module 1 : Collecte de Données (Scraping & Ingestion)
Le module de données (situé dans le répertoire `scraping/`) sert de point d'entrée pour toute la plateforme. Il applique le principe d'**acquisition immuable** : les données brutes sont archivées avant d'être nettoyées.

```
📁 scraping/src/
├── collectors/           # Collecteurs EIA & News
└── data/
    ├── raw/              # Fichiers bruts (JSON/HTML)
    └── processed/        # Fichiers d'échange Parquet (BUS DE DONNÉES)
```

### A. Collecte des Prix (API EIA & Yahoo Finance)
* **API EIA :** Requêtes automatisées pour récupérer les données officielles de prix hebdomadaires et quotidiens.
* **Yahoo Finance :** Extraction des données de prix quotidiennes et horaires au format **OHLCV** (Open, High, Low, Close, Volume) pour le ticker `CL=F` (WTI Crude Oil Future).

### B. Collecte des Actualités et Médias Sociaux
* **News Financières :** Web scraping automatisé via `BeautifulSoup` et flux RSS d'actualités financières spécialisées (ex: *OilPrice.com*, *Reuters*).
* **Médias Sociaux (Optionnel/Simulé) :** Collecteurs Reddit (via l'API PRAW sur les subreddits `/r/oil`, `/r/investing`) et Twitter/X.

### C. Pipeline ETL & Format de Sortie
Toutes les données nettoyées sont converties en format **Parquet** pour optimiser les performances de lecture/écriture et préserver le typage des colonnes (dates, réels) :
* `petrol_wti_daily.parquet` : Séries temporelles quotidiennes du WTI.
* `petrol_news_oilprice.parquet` : Base de données des articles récoltés.

---

## 3. Module 2 : Analyse Technique & Traitement du Signal
Ce module transforme les séries temporelles brutes de prix en features d'apprentissage. Les indicateurs calculés sont décrits ci-dessous :

### A. Rendements (Returns)
* **Rendement Simple (`returns`) :** 
  $$\text{returns}_t = \frac{\text{Price}_t - \text{Price}_{t-1}}{\text{Price}_{t-1}}$$
* **Rendement Logarithmique (`log_returns`) :**
  $$\text{log\_returns}_t = \ln\left(\frac{\text{Price}_t}{\text{Price}_{t-1}}\right)$$
  *Intérêt :* Les rendements log sont additifs temporellement et stables numériquement.

### B. Moyennes Mobiles (Moving Averages)
* **Moyenne Mobile Simple (SMA) :** Calculée sur des fenêtres de 7, 14, 30, et 50 jours pour capturer la tendance globale.
* **Moyenne Mobile Exponentielle (EMA 12 et 26) :** Accorde plus d'importance aux prix récents pour réduire le retard de signal.

### C. MACD (Moving Average Convergence Divergence)
* $$\text{MACD} = \text{EMA}_{12} - \text{EMA}_{26}$$
* $$\text{Signal} = \text{EMA}_9(\text{MACD})$$
* $$\text{Histogramme} = \text{MACD} - \text{Signal}$$

### D. RSI (Relative Strength Index)
Oscillateur de momentum borné entre 0 et 100 servant à identifier les zones de surachat ($> 70$) et de survente ($< 30$).
$$\text{RSI} = 100 - \frac{100}{1 + \text{RS}} \quad \text{où} \quad \text{RS} = \frac{\text{Moyenne des gains sur 14j}}{\text{Moyenne des pertes sur 14j}}$$

### E. Volatilité & Écart à la Moyenne
* **Bandes de Bollinger :** Encadrent le prix avec une moyenne mobile sur 20 jours à plus ou moins 2 écarts-types.
  * *Features extraites :* Largeur de bande (`bb_width`) et position relative (`bb_pct`).
* **ATR (Average True Range) :** Mesure la volatilité intrinsèque de l'actif sur 14 jours.
* **Momentum :** Élan des prix sur 5, 10 et 20 jours.
* **Ratios de Prix :** Écart relatif du prix actuel par rapport à sa SMA (`price_to_sma30`).

---

## 4. Module 3 : Pipeline NLP & Analyse de Sentiment
Le pipeline NLP extrait l'orientation psychologique du marché à partir des flux de nouvelles récoltées.

```
Texte Brut ──▶ Preprocessing ──▶ TF-IDF / Embeddings ──▶ Modélisation Sentiment ──▶ Agrégation Journalière ──▶ Export Bus
```

### A. Preprocessing NLP
1. Nettoyage : Suppression des URLs, des tags HTML et des caractères non alphanumériques.
2. Normalisation linguistique : Passage en minuscules, suppression des *stopwords* et lemmatisation.
3. Calcul de la **densité pétrolière** (`oil_density`) : Indice vérifiant le ratio de mots-clés liés au secteur pétrolier (ex : *barrel, OPEC, production, drilling*).

### B. Modèles d'Analyse de Sentiment
La plateforme implémente trois niveaux d'analyse configurables :
1. **Modèle Lexical :** Approche par dictionnaire optimisé pour le pétrole (bullish/bearish).
2. **Logistic Regression :** Entraîné sur des vecteurs de features TF-IDF.
3. **FinBERT (ProsusAI) :** Modèle de type Transformer pré-entraîné sur des corpus financiers, offrant la meilleure précision pour la détection de nuances contextuelles.

chaque article reçoit un score continu compris entre **-1.0 (très baissier)** et **+1.0 (très haussier)**.

### C. Agrégation Temporelle & Distribution
Les sentiments individuels sont regroupés quotidiennement :
* **Pondération :** Chaque sentiment est pondéré par la confiance du modèle, la pertinence (`oil_density`) de l'article, et la fiabilité de la source.
* **Classification par Actif :** Les articles sont classés automatiquement dans la catégorie **WTI** ou **Brent** selon les entités nommées détectées.
* **Exportation :** Génération des fichiers `sentiment_wti.parquet` et `sentiment_brent.parquet`.

---

## 5. Module 4 : Modélisation Machine Learning
L'objectif est d'utiliser l'ensemble des indicateurs techniques combinés aux sentiments agrégés pour anticiper la direction du marché à $T+1$.

### A. Formulation du Problème
On définit une target de classification binaire :
$$Y_{t+1} = \begin{cases} 1 & \text{si } \text{Price}_{t+1} > \text{Price}_t \\ 0 & \text{sinon} \end{cases}$$

### B. Algorithmes Implémentés
1. **Random Forest (Forêt d'arbres décisionnels) :**
   * *Avantage :* Très robuste aux valeurs aberrantes, limite le risque d'overfitting.
   * *Paramètres :* `n_estimators=200`, `max_depth=10`, `class_weight='balanced'`.
2. **XGBoost (Extreme Gradient Boosting) :**
   * *Avantage :* Apprentissage séquentiel optimisé, souvent supérieur en précision.
   * *Paramètres :* `n_estimators=300`, `learning_rate=0.05`, `max_depth=6`.

### C. Validation Croisée Temporelle
Pour éviter le *data leakage* (le fait d'utiliser des données futures pour prédire le passé), nous utilisons un schéma de **TimeSeriesSplit** :

```
Fold 1: [Train: Janv-Juin] ──▶ [Val: Juillet]
Fold 2: [Train: Janv-Juillet] ──▶ [Val: Août]
Fold 3: [Train: Janv-Août] ──▶ [Val: Septembre]
```

---

## 6. Module 5 : Moteur de Backtesting & Métriques de Performance
Le backtester simule les gains et les pertes générés par la stratégie sur la base des signaux émis par les modèles (`BUY = +1`, `HOLD = 0`, `SELL = -1`).

### A. Paramètres de Simulation
* **Frais de transaction :** Configurables (par défaut : $0.1\%$ par transaction).
* **Lag de signal :** Retard de 1 jour (`signal_lag=1`) pour simuler le délai de passage d'ordre en situation réelle.

### B. Métriques Clés d'Évaluation
* **Rendement Cumulé :** Évolution globale du portefeuille.
* **Ratio de Sharpe (Annualisé) :**
  $$\text{Sharpe} = \frac{\mathbb{E}[R_p - R_f]}{\sigma_p} \times \sqrt{252}$$
* **Maximum Drawdown (MDD) :** La plus forte baisse enregistrée depuis un sommet historique.
* **Win Rate :** Ratio d'opérations rentables sur le nombre total d'opérations.
* **Ratio de Calmar :** Ratio de rendement annuelisé sur le Maximum Drawdown (mesure d'efficacité face au risque extrême).

---

## 7. Module 6 : Dashboard Streamlit & Visualisation
L'interface utilisateur permet d'explorer les résultats de manière dynamique sans coder.

```
📁 web/
├── app.py                      # Point d'entrée Streamlit
├── components/                 # Composants visuels (Price Chart, Recommendation...)
└── utils/                      # Logique d'affichage et de calcul
```

### A. Fonctionnalités de l'Interface
1. **Onglet Recommandation :** Affiche le verdict en direct (ACHETER, CONSERVER, VENDRE) accompagné d'un indice de confiance et des explications du modèle.
2. **Onglet Analyse Technique :** Graphique Plotly interactif des prix (courbes ou bougies) sur lequel l'utilisateur peut superposer la SMA, le RSI, le MACD et les Bandes de Bollinger.
3. **Onglet Backtesting :** Comparaison graphique des rendements cumulés de notre stratégie face à une stratégie passive d'achat-conservation (*Buy & Hold*).
4. **Actualités :** Affichage des articles scrapés et de leur score individuel de sentiment.

---

## 8. Conclusion & Perspectives
Le projet démontre la valeur ajoutée de l'hybridation des données quantitatives et textuelles (sentiments) dans la prédiction des marchés de matières premières hautement spéculatifs comme le pétrole.

### Pistes d'Amélioration :
* **Intégration d'autres sources NLP :** Analyse des rapports de politique monétaire de la Fed et des communiqués officiels de l'OPEP.
* **Modèles Deep Learning :** Utilisation de réseaux de neurones récurrents (LSTM) ou de modèles de type Transformer temporels (TFT) pour capturer les dépendances à long terme.
* **Gestion du risque avancée :** Ajout de stop-loss et de take-profit dynamiques basés sur l'ATR (Average True Range).
