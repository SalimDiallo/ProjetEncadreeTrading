# 🧪 Tests Unitaires — Dashboard

Suite de tests `pytest` qui valide les fonctions critiques du dashboard.
Tous les tests s'exécutent en moins de 2 secondes et ne nécessitent
**aucune donnée externe** (données synthétiques générées à la volée).

## 🚀 Lancement

Depuis le dossier `dashboard/` (avec ton venv activé) :

```bash
# Installer pytest si pas déjà fait
pip install pytest pytest-cov

# Lancer toute la suite
pytest

# Avec rapport de couverture
pytest --cov=utils --cov-report=term-missing

# Un seul fichier
pytest tests/test_metrics.py

# Une seule classe
pytest tests/test_metrics.py::TestSharpeRatio

# Un seul test
pytest tests/test_metrics.py::TestSharpeRatio::test_zero_volatility_returns_zero

# S'arrêter au premier échec (utile pour debug)
pytest -x

# Avec plus de verbosité
pytest -v
```

## 📋 Structure des tests

```
tests/
├── conftest.py            # Fixtures partagées (prix, signaux, trades synthétiques)
├── test_metrics.py        # 🔴 Sharpe, Sortino, MDD, CAGR, Calmar, Win Rate, Profit Factor
├── test_backtest.py       # 🔴 Moteur de simulation : trades, fees, P&L, buy_and_hold
├── test_formatters.py     # 🟡 fmt_currency, fmt_percent, fmt_ratio, fmt_signal
├── test_integration.py    # 🔴 Pipeline end-to-end Prix → Backtest → Métriques
└── README.md              # Ce fichier
```

## 🎯 Couverture des tests

| Module | Cible | Importance |
|---|---|---|
| `utils/metrics.py` | Calculs financiers | 🔴 Critique |
| `utils/backtest.py` | Moteur de simulation | 🔴 Critique |
| `utils/formatters.py` | Helpers d'affichage | 🟡 Moyenne |
| Pipeline complet | Prix → Métriques | 🔴 Critique |

**Objectif de couverture : > 80% sur `utils/`.**

## 💡 Bonnes pratiques

### Avant chaque commit Git
```bash
pytest -x  # tous les tests doivent passer
```

### Si tu corriges un bug
1. **Écris d'abord un test** qui reproduit le bug (il doit échouer)
2. **Corrige le bug** dans le code
3. **Vérifie** que le test passe maintenant
4. **Commit** le bug fix + le test ensemble

C'est la garantie que le bug ne reviendra jamais.

### Si tu ajoutes une fonction
Tu ajoutes au minimum :
- Un test du cas nominal
- Un test du cas limite (input vide, valeurs extrêmes)
- Un test du cas d'erreur (input invalide)

## 🐛 Bugs détectés par cette suite

Cette suite de tests a déjà permis de détecter et corriger :

1. **`sharpe_ratio` retournait `7e+16`** quand la volatilité était nulle
   (bug de précision flottante avec `pandas.std() == 0`)
2. **`annualized_volatility` retournait `NaN`** sur une série vide
3. **`simulate_portfolio` ne reconnaissait pas le paramètre `fee_pct`**
   utilisé par `app.py` (alias manquant)

Sans ces tests, ces bugs seraient en production maintenant.

## 🛡️ Fixtures disponibles dans `conftest.py`

### Prix
- `prices_uptrend` — 100 jours en hausse régulière
- `prices_downtrend` — 100 jours en baisse régulière
- `prices_flat` — 100 jours constants (volatilité nulle)
- `prices_volatile` — 252 jours avec vol réaliste de ~1%/jour
- `prices_with_drawdown` — Montée puis chute (pour tester MDD)

### Signaux
- `signals_all_hold` — Que des HOLD (aucun trade)
- `signals_one_cycle` — 1 BUY + 1 SELL
- `signals_three_cycles` — 3 cycles d'achat/vente
- `signals_perfect_timing` — BUY au plus bas, SELL au pic

### Trades & equity
- `trades_all_winning` — 3 trades, tous gagnants
- `trades_mixed` — 4 trades, 2 gains + 2 pertes
- `trades_empty` — DataFrame vide pour cas limites
- `equity_simple` — Courbe d'équité simple avec pic et creux
- `returns_simple` — Rendements correspondants

---

## ⚙️ Intégration Continue (optionnel)

Pour lancer les tests automatiquement à chaque push GitHub :

```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r dashboard/requirements.txt pytest pytest-cov
      - run: cd dashboard && pytest --cov=utils
```

Tu obtiens un badge ✅ "tests passing" sur ton README — toujours appréciable
pour un projet académique.
