# 🔌 Contrats d'Interface entre Modules

> Document de référence partagé entre les équipes Scraping, ML, NLP et Web.
> **Toute modification doit passer par une Pull Request** validée par tous.

## Vue d'ensemble

Tous les modules communiquent via le dossier **`scraping/src/data/processed/`**,
qui sert de bus de données partagé.

```
┌──────────────┐
│  scraping/   │  ──► écrit  petrol_*.parquet, wti_*.csv
└──────────────┘            │
                            ▼
                  ┌────────────────────┐
        ┌─────────┤  data/processed/   ├──────────┐
        │         └────────────────────┘          │
        ▼                                          ▼
  ┌──────────┐                              ┌──────────┐
  │  ai/ml/  │  ──► écrit signals_ml_*    │  web/    │  ──► lit tout
  └──────────┘                              └──────────┘
        ▲
        │
  ┌──────────┐
  │ ai/nlp/  │  ──► écrit sentiment_*
  └──────────┘
```

---

## 📦 Module SCRAPING → tous les autres

### Fichiers produits (déjà fonctionnels)

#### `petrol_wti_daily.parquet`
```python
{
  "date": str,      # ⚠️ format ISO, à convertir en datetime
  "price": str,     # ⚠️ stocké en string — BUG à corriger
}
```

#### `petrol_brent_daily.parquet`
Idem WTI.

#### `wti_petrole_3ans.csv` (OHLCV)
```python
{
  "date": datetime,
  "open": float,
  "high": float,
  "low": float,
  "close": float,
  "volume": int,
}
```
⚠️ Le CSV a 2 lignes d'en-tête à skipper (`pd.read_csv(path, skiprows=2)`).

#### `petrol_news_oilprice.parquet`
```python
{
  "date": datetime,    # ⚠️ actuellement = date de scrape, pas de publication
  "title": str,
  "content": str,
  "source": str,
}
```

### 🐛 Bugs à corriger côté scraping

1. **Cast `price` en float** dans `processors/petrol_processor.py` :
   ```python
   df["price"] = pd.to_numeric(df["price"], errors="coerce")
   ```
2. **Extraire la vraie date de publication** des articles (pas la date du scrape)

---

## 🤖 Module ML (ai/ml/) → web/

### Fichier à produire

#### `scraping/src/data/processed/signals_ml_{asset}.parquet`
où `{asset}` = `wti` ou `brent`.

```python
{
  "date": datetime,           # date du signal
  "signal": str,              # "BUY" | "SELL" | "HOLD"
  "confidence": float,        # score entre 0 et 1
  "predicted_return": float,  # rendement prédit à J+1 (optionnel)
  "model": str,               # "RandomForest" | "XGBoost"
}
```

### Notes
- ✅ Le seuil de décision (`threshold`) du notebook `model_training.ipynb` produit déjà un signal binaire — il faut juste le convertir en `BUY/HOLD` et exporter le DataFrame en Parquet.
- ⚠️ **Pas de look-ahead bias** : les features au temps `t` ne doivent utiliser que des données disponibles à `t`.

### Exemple de génération à ajouter à la fin du notebook

```python
# Conversion des prédictions en signaux + export
signals_df = pd.DataFrame({
    "date": df_bt.index,
    "signal": np.where(predictions_rf == 1, "BUY", "HOLD"),
    "confidence": proba_rf,
    "predicted_return": rf.predict_proba(X_test)[:, 1] - 0.5,
    "model": "RandomForest",
})
signals_df.to_parquet(
    "../../scraping/src/data/processed/signals_ml_wti.parquet",
    index=False
)
```

---

## 📰 Module NLP (ai/nlp/) → web/

### Fichier à produire

#### `scraping/src/data/processed/sentiment_{asset}.parquet`

```python
{
  "date": datetime,           # date d'agrégation (1 ligne par jour)
  "sentiment_score": float,   # score entre -1 (négatif) et +1 (positif)
  "n_articles": int,          # nombre d'articles agrégés ce jour
}
```

### Notes
- Granularité journalière (pas horaire)
- 1 fichier par actif (`wti`, `brent`)
- Agrégation : moyenne pondérée par fiabilité de la source, ou moyenne simple

---

## 🎨 Module WEB → utilisateur

Pas de fichier produit en sortie, seulement l'affichage.

Le dashboard est conçu pour :
- **Détecter automatiquement** la présence de chaque fichier optionnel
- **Tomber en mocks** si un module n'a pas encore livré
- **Basculer automatiquement** vers les vrais signaux dès qu'ils sont disponibles

→ Aucune coordination temporelle n'est nécessaire. Chaque équipe livre quand elle veut.

---

## ✅ Checklist de validation

Pour qu'un module soit considéré "livré" :

- [ ] Le fichier Parquet est créé au bon emplacement
- [ ] Les colonnes ont les bons noms et types
- [ ] Aucune valeur NaN dans les colonnes critiques (signal, confidence)
- [ ] La date couvre la même période que `petrol_wti_daily.parquet`
- [ ] Le dashboard affiche correctement les nouvelles données (test visuel)
- [ ] Les tests `pytest web/tests/` passent toujours

---

📅 **Dernière mise à jour** : à compléter
✍️ **Validé par** : équipes Scraping, ML, NLP, Web
