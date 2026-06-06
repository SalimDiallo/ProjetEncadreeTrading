"""
test_integration.py
===================
Tests d'intégration end-to-end.

Importance : 🔴 CRITIQUE
Si ces tests passent, le dashboard fonctionnera. Si un seul échoue,
c'est qu'un maillon de la chaîne est cassé.

Pipeline testé :
    Prix synthétiques → Signaux → simulate_portfolio → compute_all_metrics
"""
import pandas as pd
import pytest

from utils.backtest import simulate_portfolio, buy_and_hold
from utils.metrics import compute_all_metrics


# ============================================================================
# Pipeline complet
# ============================================================================

class TestFullPipeline:
    """Vérifie la chaîne complète sans dépendre des fichiers Parquet réels."""

    def test_strategy_pipeline_runs_without_error(self, prices_volatile,
                                                    signals_three_cycles):
        """Le pipeline complet doit s'exécuter sans lever d'exception."""
        equity_df, trades_df = simulate_portfolio(
            prices_volatile, signals_three_cycles,
            initial_capital=10_000, fee_rate=0.001
        )
        metrics = compute_all_metrics(equity_df["equity"],
                                       equity_df["returns"], trades_df)

        # Sanity checks de base
        assert len(equity_df) == len(prices_volatile)
        assert len(trades_df) == 3
        assert "sharpe" in metrics

    def test_benchmark_and_strategy_have_same_metric_keys(self, prices_uptrend,
                                                           signals_one_cycle):
        """Stratégie et benchmark doivent produire la même structure."""
        strat, _ = simulate_portfolio(prices_uptrend, signals_one_cycle, fee_rate=0.0)
        bh = buy_and_hold(prices_uptrend, initial_capital=10_000)

        strat_metrics = compute_all_metrics(strat["equity"], strat["returns"])
        bh_metrics = compute_all_metrics(bh["equity"], bh["returns"])

        assert set(strat_metrics.keys()) == set(bh_metrics.keys())

    def test_all_hold_produces_flat_metrics(self, prices_volatile, signals_all_hold):
        """
        Si on ne trade jamais, le capital reste constant donc :
        - total_return = 0
        - sharpe = 0
        - max_drawdown = 0
        """
        # On adapte signals_all_hold (fixture basée sur prices_uptrend) à prices_volatile
        flat_signals = pd.DataFrame({
            "date": prices_volatile["date"],
            "signal": ["HOLD"] * len(prices_volatile),
            "confidence": [0.5] * len(prices_volatile),
            "model": ["TEST"] * len(prices_volatile),
        })

        equity_df, trades_df = simulate_portfolio(prices_volatile, flat_signals)
        metrics = compute_all_metrics(equity_df["equity"],
                                       equity_df["returns"], trades_df)

        assert metrics["total_return"] == 0
        assert metrics["sharpe"] == 0
        assert metrics["max_drawdown"] == 0

    def test_no_metric_is_nan(self, prices_volatile, signals_three_cycles):
        """
        ⚠️ Test de robustesse : aucune métrique ne doit être NaN,
        sinon ça casse l'affichage du dashboard.
        """
        equity_df, trades_df = simulate_portfolio(prices_volatile,
                                                   signals_three_cycles)
        metrics = compute_all_metrics(equity_df["equity"],
                                       equity_df["returns"], trades_df)

        for key, value in metrics.items():
            assert not pd.isna(value), f"{key} = NaN !"


# ============================================================================
# Composant de recommandation (logique d'agrégation)
# ============================================================================

class TestRecommendationAggregation:
    """Vérifie l'agrégation des signaux pour la recommandation finale."""

    def test_returns_valid_signal(self, signals_three_cycles):
        """Le signal final doit être l'un des trois autorisés."""
        from components.recommendation import aggregate_signals
        signal, confidence = aggregate_signals(signals_three_cycles)
        assert signal in {"BUY", "SELL", "HOLD"}

    def test_confidence_in_valid_range(self, signals_three_cycles):
        """La confiance doit être un score entre 0 et 1."""
        from components.recommendation import aggregate_signals
        _, confidence = aggregate_signals(signals_three_cycles)
        assert 0.0 <= confidence <= 1.0

    def test_empty_signals_fallback_to_hold(self):
        """Aucun signal → fallback HOLD (pas de crash)."""
        from components.recommendation import aggregate_signals
        empty = pd.DataFrame(columns=["date", "signal", "confidence", "model"])
        signal, confidence = aggregate_signals(empty)
        assert signal == "HOLD"
        assert 0.0 <= confidence <= 1.0


# ============================================================================
# Scénarios métier réalistes
# ============================================================================

class TestRealisticScenarios:
    """Scénarios qu'on s'attend à voir en production."""

    def test_short_period_doesnt_crash(self, prices_uptrend):
        """Un backtest sur très peu de jours doit fonctionner."""
        short_prices = prices_uptrend.head(5).copy()
        signals = pd.DataFrame({
            "date": short_prices["date"],
            "signal": ["BUY", "HOLD", "HOLD", "HOLD", "SELL"],
            "confidence": [0.8] * 5,
            "model": ["TEST"] * 5,
        })
        equity_df, trades = simulate_portfolio(short_prices, signals)
        assert len(equity_df) == 5
        assert len(trades) == 1

    def test_large_capital(self, prices_volatile, signals_three_cycles):
        """Un capital initial très élevé doit fonctionner (pas d'overflow)."""
        equity_df, _ = simulate_portfolio(prices_volatile, signals_three_cycles,
                                           initial_capital=1_000_000)
        assert equity_df["equity"].iloc[0] == 1_000_000
        assert equity_df["equity"].iloc[-1] > 0

    def test_consistent_equity_continuity(self, prices_uptrend, signals_one_cycle):
        """L'equity ne doit jamais avoir de saut brutal (continuité mark-to-market)."""
        equity_df, _ = simulate_portfolio(prices_uptrend, signals_one_cycle, fee_rate=0.0)
        # Rendements journaliers raisonnables (< 50% par jour en valeur absolue)
        assert equity_df["returns"].abs().max() < 0.5
