# 📚 Guide des Indicateurs Techniques & Concepts ML

> Explication détaillée de tous les indicateurs et concepts utilisés dans le notebook de backtesting WTI.

---

## 1. Returns (Rendements)

### `returns` — Rendement Simple
```
returns = (Prix_t - Prix_{t-1}) / Prix_{t-1}
```
- **Quoi ?** : Le pourcentage de variation du prix d'un jour à l'autre.
- **Pourquoi ?** : C'est la mesure la plus fondamentale en finance. Un return de +0.02 signifie que le prix a monté de **2%**.
- **Exemple** : Prix passe de 100$ à 103$ → return = **+3%**

### `log_returns` — Rendement Logarithmique
```
log_returns = ln(Prix_t / Prix_{t-1})
```
- **Quoi ?** : Le logarithme du ratio de prix.
- **Pourquoi ?** : Les log-returns sont **additifs** (on peut les sommer sur plusieurs jours), suivent mieux une distribution normale, et sont plus stables numériquement.
- **Différence avec returns** : Pour de petites variations, les deux sont quasi identiques. Pour de grandes variations, le log-return est plus conservateur.

---

## 2. Moyennes Mobiles

### `sma_N` — Simple Moving Average (Moyenne Mobile Simple)
```
SMA_N = (Prix_t + Prix_{t-1} + ... + Prix_{t-N+1}) / N
```
- **Quoi ?** : La moyenne arithmétique des N derniers prix.
- **Fenêtres utilisées** : 7, 14, 30, 50 jours
- **Pourquoi ?** :
  - **SMA courte (7, 14)** : Réagit vite aux changements → **tendance court terme**
  - **SMA longue (30, 50)** : Lisse le bruit → **tendance long terme**
- **Signal classique** : Quand la SMA courte **croise au-dessus** de la SMA longue → signal d'achat (**Golden Cross**). L'inverse → signal de vente (**Death Cross**).

### `ema_12`, `ema_26` — Exponential Moving Average (Moyenne Mobile Exponentielle)
```
EMA_t = α × Prix_t + (1 - α) × EMA_{t-1}     où α = 2/(N+1)
```
- **Quoi ?** : Comme la SMA, mais donne **plus de poids aux prix récents**.
- **Pourquoi ?** : L'EMA réagit plus rapidement aux changements de prix que la SMA. C'est la base du calcul du **MACD**.
- **EMA 12** : Réactive (court terme)
- **EMA 26** : Plus lente (moyen terme)

> [!TIP]
> La différence clé entre SMA et EMA : l'EMA « oublie » moins vite les prix récents, ce qui la rend plus sensible aux mouvements actuels du marché.

---

## 3. MACD (Moving Average Convergence Divergence)

```
MACD        = EMA_12 - EMA_26
Signal      = EMA_9(MACD)
Histogramme = MACD - Signal
```

| Composante | Rôle |
|---|---|
| `macd` | Mesure la **convergence/divergence** entre tendance courte et moyenne |
| `macd_signal` | Lisse le MACD → sert de **déclencheur** de signal |
| `macd_hist` | Visualise la **force** de la tendance |

- **Signal d'achat** : MACD croise au-dessus du Signal (histogramme passe positif)
- **Signal de vente** : MACD croise en dessous du Signal (histogramme passe négatif)
- **Interprétation** : Un histogramme qui grandit = tendance qui **s'accélère**. Un histogramme qui rétrécit = tendance qui **s'essouffle**.

---

## 4. RSI (Relative Strength Index)

```
RSI = 100 - (100 / (1 + RS))
RS  = Moyenne des gains (14j) / Moyenne des pertes (14j)
```

- **Quoi ?** : Oscillateur borné entre **0 et 100** qui mesure la vitesse et l'ampleur des variations de prix.
- **Interprétation** :

| Zone | Valeur RSI | Signification |
|------|-----------|---------------|
| 🔴 Surachat | RSI > 70 | Le prix a trop monté, correction probable |
| 🟢 Survente | RSI < 30 | Le prix a trop baissé, rebond probable |
| ⚪ Neutre | 30 < RSI < 70 | Pas de signal extrême |

- **Pourquoi 14 jours ?** : C'est la période standard définie par Wilder (créateur du RSI). Assez pour lisser le bruit, assez court pour rester réactif.

> [!IMPORTANT]
> Le RSI ne prédit pas la direction, il indique quand un mouvement est **excessif**. Un RSI > 70 ne signifie pas forcément une baisse immédiate — en tendance forte, le RSI peut rester en zone de surachat longtemps.

---

## 5. Bandes de Bollinger

```
Bande Haute   = SMA_20 + 2 × σ_20
Bande Moyenne = SMA_20
Bande Basse   = SMA_20 - 2 × σ_20
```

*(σ_20 = écart-type du prix sur 20 jours)*

### Features dérivées :
- **`bb_width`** = (Haute - Basse) / Moyenne → mesure la **volatilité**. Bandes larges = marché volatile.
- **`bb_pct`** = (Prix - Basse) / (Haute - Basse) → **position relative** du prix dans les bandes (0 = sur la bande basse, 1 = sur la haute).

### Signaux :
- Prix touche la **bande haute** → potentiel surachat
- Prix touche la **bande basse** → potentiel survente
- **Squeeze** (bandes se resserrent) → explosion de volatilité imminente

---

## 6. ATR (Average True Range)

```
True Range = Max(High - Low, |High - Close_prev|, |Low - Close_prev|)
ATR_14     = Moyenne(True Range) sur 14 jours
```

- **Quoi ?** : Mesure de la **volatilité réelle** du marché, indépendamment de la direction.
- **Pourquoi ?** : Contrairement à la volatilité basée sur les returns, l'ATR capture les **gaps** et les **mèches** intraday.
- **Usage** :
  - ATR élevé → marché très volatile → augmenter les stop-loss
  - ATR faible → marché calme → resserrer les stop-loss

> [!NOTE]
> Dans notre dataset, on n'a que le prix de clôture (pas de High/Low réels), donc on approxime avec un rolling max/min sur 2 jours. C'est une approximation acceptable pour un prix journalier unique.

---

## 7. Volatilité

```
volatility_N = écart-type des returns sur N jours
```

- **`volatility_7`** : Volatilité court terme (1 semaine)
- **`volatility_21`** : Volatilité moyen terme (~1 mois)
- **Pourquoi ?** : La volatilité tend à être **persistante** (volatility clustering) — les périodes de forte volatilité sont suivies de forte volatilité. C'est une feature prédictive puissante.

---

## 8. Ratios de Prix

### `price_to_sma30`, `price_to_sma50`
```
ratio = Prix / SMA_N
```

- **Ratio > 1** : Le prix est **au-dessus** de sa moyenne → tendance haussière
- **Ratio < 1** : Le prix est **en dessous** de sa moyenne → tendance baissière
- **Pourquoi un ratio ?** : Normalise l'information. Que le pétrole soit à 60$ ou 120$, un ratio de 1.05 signifie toujours « 5% au-dessus de la moyenne ».

---

## 9. Momentum

```
momentum_N = (Prix_t / Prix_{t-N}) - 1
```

- **`momentum_5`** : Élan sur 5 jours (1 semaine)
- **`momentum_10`** : Élan sur 10 jours (2 semaines)
- **`momentum_20`** : Élan sur 20 jours (1 mois)
- **Interprétation** : Un momentum de +0.08 = le prix a monté de 8% en N jours.
- **Concept clé** : Les tendances ont tendance à **persister** à court terme (effet momentum) mais à **s'inverser** à long terme (mean reversion).

---

## 10. Concepts ML

### Target — Direction du Prix
```python
target = 1 si Prix_{t+1} > Prix_t   (Hausse)
target = 0 si Prix_{t+1} ≤ Prix_t   (Baisse)
```
On transforme le problème de **régression** (prédire le prix) en problème de **classification binaire** (prédire la direction). C'est plus adapté au trading car on n'a besoin que de la direction pour décider d'acheter ou non.

### Random Forest 🌲
```
Ensemble de N arbres de décision entraînés sur des sous-échantillons aléatoires
Prédiction finale = vote majoritaire des arbres
```

| Hyperparamètre | Valeur | Rôle |
|---|---|---|
| `n_estimators=200` | 200 arbres | Plus d'arbres = plus stable mais plus lent |
| `max_depth=10` | Profondeur max 10 | Limite la complexité → évite l'overfitting |
| `min_samples_split=10` | Min 10 samples pour split | Évite les splits sur du bruit |
| `class_weight='balanced'` | Poids équilibrés | Compense si une classe est plus fréquente |

**Avantages** : Robuste au bruit, peu de risque d'overfitting, feature importance native.
**Limites** : Moins performant sur les patterns séquentiels (séries temporelles).

### XGBoost 🚀
```
Boosting = on construit les arbres SÉQUENTIELLEMENT
Chaque nouvel arbre corrige les erreurs du précédent
```

| Hyperparamètre | Valeur | Rôle |
|---|---|---|
| `n_estimators=300` | 300 arbres | Plus d'itérations de boosting |
| `learning_rate=0.05` | Taux d'apprentissage | Petit = apprentissage lent mais plus précis |
| `max_depth=6` | Profondeur max 6 | Arbres moins profonds qu'en RF (le boosting compense) |
| `subsample=0.8` | 80% des données | Régularisation par sous-échantillonnage |
| `reg_alpha=0.1` | Régularisation L1 | Pousse les poids faibles vers 0 (feature selection) |
| `reg_lambda=1.0` | Régularisation L2 | Empêche les poids de devenir trop grands |

**Avantages** : Souvent le meilleur en compétition, gère bien les features hétérogènes.
**Limites** : Plus sensible aux hyperparamètres, risque d'overfitting si mal réglé.

> [!TIP]
> **RF vs XGBoost** : Random Forest construit les arbres **en parallèle** (bagging), XGBoost les construit **en série** (boosting). Le boosting tend à être plus puissant mais plus fragile.

### TimeSeriesSplit — Validation Croisée Temporelle
```
Fold 1: [Train: ████░░░░░░] [Val: ██░░░░░░░░]
Fold 2: [Train: ██████░░░░] [Val: ██░░░░░░░░]
Fold 3: [Train: ████████░░] [Val: ██░░░░░░░░]
```
- **Pourquoi pas un K-Fold classique ?** : En séries temporelles, on ne peut **jamais** entraîner sur le futur pour prédire le passé (data leakage). Le TimeSeriesSplit respecte l'ordre chronologique.

---

## 11. Métriques de Backtesting

| Métrique | Formule | Interprétation |
|---|---|---|
| **Rendement** | `(Portfolio_final / Portfolio_initial) - 1` | Gain ou perte total |
| **Sharpe Ratio** | `mean(returns) / std(returns) × √252` | Rendement ajusté au risque. > 1 = bon, > 2 = excellent |
| **Max Drawdown** | `max((Peak - Trough) / Peak)` | Pire chute depuis un sommet. -20% = on a perdu 20% au pire moment |
| **Win Rate** | `Trades gagnants / Total trades` | % de jours où la prédiction était correcte |

> [!WARNING]
> Un bon modèle en classification (accuracy 55%) ne garantit pas une stratégie profitable. Le **Sharpe Ratio** et le **Max Drawdown** sont plus importants que l'accuracy pour évaluer une stratégie de trading.
