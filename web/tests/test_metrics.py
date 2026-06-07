"""
test_metrics.py
===============
Tests des fonctions de calcul financier dans utils/metrics.py.

Importance : 🔴 CRITIQUE
Une erreur ici fausse les recommandations affichées à l'utilisateur.
"""
import numpy as np
import pandas as pd
import pytest

from utils.metrics import (
    total_return,
    cagr,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    drawdown_series,
    calmar_ratio,
    win_rate,
    profit_factor,
    compute_all_metrics,
    PERIODS_PER_YEAR,
)


# ============================================================================
# total_return
# ============================================================================

class TestTotalReturn:
    """Vérifie le calcul du rendement total."""

    def test_doubling_returns_100_percent(self):
        equity = pd.Series([100, 150, 200])
        assert total_return(equity) == pytest.approx(1.0)

    def test_halving_returns_negative_50_percent(self):
        equity = pd.Series([100, 75, 50])
        assert total_return(equity) == pytest.approx(-0.5)

    def test_no_change_returns_zero(self):
        equity = pd.Series([100, 100, 100])
        assert total_return(equity) == 0.0

    def test_single_value_returns_zero(self):
        """Cas limite : une seule valeur → impossible de calculer un rendement."""
        equity = pd.Series([100])
        assert total_return(equity) == 0.0

    def test_empty_series_returns_zero(self):
        """Cas limite : série vide → 0 par convention, pas d'erreur."""
        equity = pd.Series([], dtype=float)
        assert total_return(equity) == 0.0


# ============================================================================
# cagr
# ============================================================================

class TestCAGR:
    """Vérifie le taux de croissance annualisé."""

    def test_one_year_doubling(self):
        """Sur 252 jours, doubler donne un CAGR de 100%."""
        equity = pd.Series(np.linspace(100, 200, PERIODS_PER_YEAR))
        assert cagr(equity) == pytest.approx(1.0, abs=0.05)

    def test_no_growth_returns_zero(self):
        equity = pd.Series([100.0] * PERIODS_PER_YEAR)
        assert cagr(equity) == 0.0

    def test_zero_initial_capital_returns_zero(self):
        """Protection contre division par zéro."""
        equity = pd.Series([0.0, 50.0, 100.0])
        assert cagr(equity) == 0.0

    def test_short_series_returns_zero(self):
        equity = pd.Series([100.0])
        assert cagr(equity) == 0.0


# ============================================================================
# annualized_volatility
# ============================================================================

class TestAnnualizedVolatility:
    """Vérifie le calcul de la volatilité annualisée."""

    def test_constant_returns_zero_volatility(self):
        """Tous les rendements identiques → vol = 0 (bug réel détecté)."""
        returns = pd.Series([0.001] * 100)
        assert annualized_volatility(returns) == 0.0

    def test_high_dispersion_high_volatility(self):
        """Vol journalière de 5% → vol annualisée ≈ 5% × √252 ≈ 79%."""
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0, 0.05, PERIODS_PER_YEAR))
        result = annualized_volatility(returns)
        assert 0.7 < result < 0.9

    def test_empty_returns_zero(self):
        """Cas limite : série vide → 0 (pas de NaN qui casse l'affichage)."""
        returns = pd.Series([], dtype=float)
        assert annualized_volatility(returns) == 0.0

    def test_result_is_python_float(self):
        """Le résultat doit être un float natif, pas un numpy.float64."""
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0, 0.01, 100))
        assert isinstance(annualized_volatility(returns), float)


# ============================================================================
# sharpe_ratio
# ============================================================================

class TestSharpeRatio:
    """
    Vérifie le calcul du Sharpe Ratio.
    C'est LA métrique la plus regardée par les investisseurs.
    """

    def test_positive_excess_returns_positive_sharpe(self):
        """Des rendements positifs avec faible vol → Sharpe > 1."""
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.001, 0.005, PERIODS_PER_YEAR))
        assert sharpe_ratio(returns) > 1.0

    def test_negative_returns_give_negative_sharpe(self):
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(-0.001, 0.005, PERIODS_PER_YEAR))
        assert sharpe_ratio(returns) < 0

    def test_zero_volatility_returns_zero(self):
        """
        ⚠️ Test régression : un bug précédent retournait 7e+16 dans ce cas
        à cause de la précision flottante de pandas .std().
        """
        returns = pd.Series([0.001] * 100)
        assert sharpe_ratio(returns) == 0.0

    def test_empty_returns_zero(self):
        returns = pd.Series([], dtype=float)
        assert sharpe_ratio(returns) == 0.0

    def test_higher_risk_free_rate_lowers_sharpe(self):
        """Augmenter le taux sans risque doit faire baisser le Sharpe."""
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.001, 0.005, PERIODS_PER_YEAR))
        sharpe_no_rf = sharpe_ratio(returns, risk_free_rate=0.0)
        sharpe_with_rf = sharpe_ratio(returns, risk_free_rate=0.05)
        assert sharpe_no_rf > sharpe_with_rf


# ============================================================================
# sortino_ratio
# ============================================================================

class TestSortinoRatio:
    """
    Vérifie le Sortino Ratio.
    Contrairement au Sharpe, il ne pénalise que la volatilité baissière.
    """

    def test_only_negative_returns_with_variance(self):
        """
        Avec des pertes VARIÉES (downside std > 0), Sortino doit être calculé.
        Cas avec uniquement des rendements négatifs → Sortino négatif.
        """
        returns = pd.Series([-0.01, -0.02, -0.005, -0.015, -0.025])
        result = sortino_ratio(returns)
        assert result < 0

    def test_no_negative_returns_returns_zero(self):
        """
        Aucune perte → pas de downside std → 0 (évite division par zéro).
        """
        returns = pd.Series([0.01, 0.02, 0.005, 0.015])
        assert sortino_ratio(returns) == 0.0

    def test_sortino_higher_than_sharpe_when_asymmetric(self):
        """
        Avec des gains forts et des pertes variées mais modérées,
        Sortino doit être plus élevé que Sharpe.
        """
        returns = pd.Series([0.05, 0.04, -0.001, 0.03, -0.005, 0.02, -0.002, 0.01, -0.003])
        assert sortino_ratio(returns) > sharpe_ratio(returns)


# ============================================================================
# max_drawdown & drawdown_series
# ============================================================================

class TestMaxDrawdown:
    """Vérifie le calcul de la perte maximale depuis un sommet."""

    def test_simple_peak_trough(self):
        """Pic à 150, creux à 100 → drawdown = (100-150)/150 = -33.33%."""
        equity = pd.Series([100, 120, 150, 130, 100, 110])
        assert max_drawdown(equity) == pytest.approx(-1/3, abs=0.01)

    def test_monotonic_growth_zero_drawdown(self):
        """Croissance pure → drawdown = 0."""
        equity = pd.Series([100, 110, 120, 130, 140])
        assert max_drawdown(equity) == 0.0

    def test_drawdown_is_never_positive(self):
        """Propriété mathématique : MDD ∈ [-1, 0]."""
        rng = np.random.default_rng(0)
        equity = pd.Series(100 + rng.normal(0, 5, 200).cumsum())
        assert max_drawdown(equity) <= 0

    def test_empty_series_returns_zero(self):
        assert max_drawdown(pd.Series([], dtype=float)) == 0.0


class TestDrawdownSeries:
    """Vérifie la série complète de drawdowns (pour les graphiques)."""

    def test_series_length_matches_input(self, equity_simple):
        result = drawdown_series(equity_simple)
        assert len(result) == len(equity_simple)

    def test_first_drawdown_is_zero(self, equity_simple):
        """Au jour 0, le drawdown est toujours 0 (pas d'historique)."""
        result = drawdown_series(equity_simple)
        assert result.iloc[0] == 0.0

    def test_all_values_non_positive(self, equity_simple):
        result = drawdown_series(equity_simple)
        assert (result <= 0).all()


# ============================================================================
# calmar_ratio
# ============================================================================

class TestCalmarRatio:
    """Vérifie le Calmar Ratio = CAGR / |Max Drawdown|."""

    def test_zero_drawdown_returns_zero(self):
        """Sans drawdown, division impossible → on retourne 0."""
        equity = pd.Series([100, 110, 120, 130])
        assert calmar_ratio(equity) == 0.0

    def test_positive_with_growth_and_drawdown(self, prices_with_drawdown):
        """Avec croissance + drawdown, Calmar doit être un nombre fini."""
        equity = pd.Series(prices_with_drawdown["price"].values)
        result = calmar_ratio(equity)
        assert np.isfinite(result)


# ============================================================================
# win_rate
# ============================================================================

class TestWinRate:
    """Vérifie le taux de réussite des trades."""

    def test_all_winning(self, trades_all_winning):
        assert win_rate(trades_all_winning) == 1.0

    def test_half_winning(self, trades_mixed):
        """trades_mixed : 2 gains / 4 trades → 50%."""
        assert win_rate(trades_mixed) == 0.5

    def test_empty_trades(self, trades_empty):
        assert win_rate(trades_empty) == 0.0

    def test_dataframe_without_pnl_column(self):
        """Si la colonne 'pnl' n'existe pas, retourne 0 (pas de crash)."""
        df = pd.DataFrame({"other": [1, 2, 3]})
        assert win_rate(df) == 0.0


# ============================================================================
# profit_factor
# ============================================================================

class TestProfitFactor:
    """Vérifie le Profit Factor = somme gains / |somme pertes|."""

    def test_only_winning_trades_returns_infinity(self, trades_all_winning):
        """Pas de pertes → infini."""
        assert profit_factor(trades_all_winning) == float("inf")

    def test_balanced_gains_and_losses(self):
        """Gains = pertes → PF = 1."""
        trades = pd.DataFrame({"pnl": [100, -100, 50, -50]})
        assert profit_factor(trades) == 1.0

    def test_mixed_trades(self, trades_mixed):
        """trades_mixed : gains=150, pertes=250 → PF = 0.6."""
        assert profit_factor(trades_mixed) == pytest.approx(150 / 250)

    def test_empty_trades(self, trades_empty):
        assert profit_factor(trades_empty) == 0.0


# ============================================================================
# compute_all_metrics — fonction d'orchestration
# ============================================================================

class TestComputeAllMetrics:
    """Vérifie le calcul groupé de toutes les métriques."""

    def test_returns_required_keys(self, equity_simple, returns_simple):
        """Toutes les clés essentielles doivent être présentes."""
        result = compute_all_metrics(equity_simple, returns_simple)
        required = {"total_return", "cagr", "volatility", "sharpe",
                    "sortino", "max_drawdown", "calmar"}
        assert required.issubset(result.keys())

    def test_includes_trade_metrics_when_provided(self, equity_simple,
                                                   returns_simple, trades_mixed):
        """Avec trades, les clés win_rate / profit_factor / n_trades apparaissent."""
        result = compute_all_metrics(equity_simple, returns_simple, trades_mixed)
        assert result["n_trades"] == 4
        assert "win_rate" in result
        assert "profit_factor" in result

    def test_no_trade_keys_when_trades_is_none(self, equity_simple, returns_simple):
        result = compute_all_metrics(equity_simple, returns_simple, trades=None)
        assert "win_rate" not in result
        assert "n_trades" not in result

    def test_all_values_are_numeric(self, equity_simple, returns_simple):
        """Aucune valeur ne doit être NaN (sinon le dashboard casse)."""
        result = compute_all_metrics(equity_simple, returns_simple)
        for key, value in result.items():
            assert isinstance(value, (int, float))
            assert not pd.isna(value), f"{key} est NaN !"
