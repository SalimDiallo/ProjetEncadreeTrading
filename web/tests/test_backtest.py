"""
test_backtest.py
================
Tests du moteur de simulation de portefeuille dans utils/backtest.py.

Importance : 🔴 CRITIQUE
Un bug ici fausserait toutes les métriques et la recommandation finale.

Format réel des trades retournés par simulate_portfolio :
    [date_entry, date_exit, price_entry, price_exit, pnl, return_pct]
Chaque ligne = un trade COMPLET (achat + revente).
"""
import pandas as pd
import pytest

from utils.backtest import simulate_portfolio, buy_and_hold


# ============================================================================
# simulate_portfolio — comportement des trades
# ============================================================================

class TestSimulatePortfolioTrades:
    """Vérifie que les trades sont exécutés correctement."""

    def test_all_hold_produces_no_trades(self, prices_uptrend, signals_all_hold):
        """Aucun signal d'action → aucun trade ne doit être enregistré."""
        _, trades = simulate_portfolio(prices_uptrend, signals_all_hold)
        assert len(trades) == 0

    def test_one_cycle_produces_one_trade(self, prices_uptrend, signals_one_cycle):
        """Un BUY suivi d'un SELL → 1 trade fermé."""
        _, trades = simulate_portfolio(prices_uptrend, signals_one_cycle, fee_rate=0.0)
        assert len(trades) == 1

    def test_three_cycles_produce_three_trades(self, prices_volatile, signals_three_cycles):
        _, trades = simulate_portfolio(prices_volatile, signals_three_cycles, fee_rate=0.0)
        assert len(trades) == 3

    def test_sell_without_position_is_ignored(self, prices_uptrend):
        """Un SELL sans position en cours doit être ignoré silencieusement."""
        dates = prices_uptrend["date"]
        signals = pd.DataFrame({
            "date": dates,
            "signal": ["HOLD"] * len(dates),
            "confidence": [0.8] * len(dates),
            "model": ["TEST"] * len(dates),
        })
        signals.loc[10, "signal"] = "SELL"  # SELL alors qu'on n'a rien acheté

        _, trades = simulate_portfolio(prices_uptrend, signals, fee_rate=0.0)
        assert len(trades) == 0

    def test_consecutive_buys_dont_open_multiple_positions(self, prices_uptrend):
        """Un 2e BUY pendant qu'on est déjà long doit être ignoré."""
        dates = prices_uptrend["date"]
        signals = pd.DataFrame({
            "date": dates,
            "signal": ["HOLD"] * len(dates),
            "confidence": [0.8] * len(dates),
            "model": ["TEST"] * len(dates),
        })
        signals.loc[10, "signal"] = "BUY"
        signals.loc[20, "signal"] = "BUY"  # ignoré
        signals.loc[50, "signal"] = "SELL"

        _, trades = simulate_portfolio(prices_uptrend, signals, fee_rate=0.0)
        assert len(trades) == 1


# ============================================================================
# simulate_portfolio — format de sortie
# ============================================================================

class TestSimulatePortfolioOutputFormat:
    """Vérifie la structure exacte des DataFrames retournés."""

    def test_equity_df_has_required_columns(self, prices_uptrend, signals_all_hold):
        equity_df, _ = simulate_portfolio(prices_uptrend, signals_all_hold)
        required = {"date", "price", "signal", "position", "cash", "equity", "returns"}
        assert required.issubset(equity_df.columns)

    def test_equity_df_length_matches_prices(self, prices_uptrend, signals_all_hold):
        """Une ligne par jour de prix, peu importe les signaux."""
        equity_df, _ = simulate_portfolio(prices_uptrend, signals_all_hold)
        assert len(equity_df) == len(prices_uptrend)

    def test_first_return_is_zero(self, prices_uptrend, signals_all_hold):
        """Le premier rendement est forcément 0 (pas de jour précédent)."""
        equity_df, _ = simulate_portfolio(prices_uptrend, signals_all_hold)
        assert equity_df["returns"].iloc[0] == 0.0

    def test_trades_df_has_required_columns(self, prices_uptrend, signals_one_cycle):
        """Format spécifique des trades fermés."""
        _, trades = simulate_portfolio(prices_uptrend, signals_one_cycle, fee_rate=0.0)
        required = {"date_entry", "date_exit", "price_entry", "price_exit",
                    "pnl", "return_pct"}
        assert required.issubset(trades.columns)


# ============================================================================
# simulate_portfolio — logique financière
# ============================================================================

class TestSimulatePortfolioFinance:
    """Vérifie que la logique de P&L est cohérente."""

    def test_uptrend_long_is_profitable(self, prices_uptrend, signals_one_cycle):
        """Achat bas + vente haut sur tendance haussière → profit."""
        equity_df, trades = simulate_portfolio(prices_uptrend, signals_one_cycle, fee_rate=0.0)
        assert equity_df["equity"].iloc[-1] > 10_000
        assert trades["pnl"].iloc[0] > 0

    def test_downtrend_long_loses_money(self, prices_downtrend):
        """Acheter sur une tendance baissière → perte."""
        dates = prices_downtrend["date"]
        signals = pd.DataFrame({
            "date": dates,
            "signal": ["BUY"] + ["HOLD"] * (len(dates) - 2) + ["SELL"],
            "confidence": [0.8] * len(dates),
            "model": ["TEST"] * len(dates),
        })
        equity_df, trades = simulate_portfolio(prices_downtrend, signals, fee_rate=0.0)
        assert equity_df["equity"].iloc[-1] < 10_000
        assert trades["pnl"].iloc[0] < 0

    def test_pnl_sign_matches_price_direction(self, prices_uptrend, signals_one_cycle):
        """Si price_exit > price_entry, pnl doit être > 0."""
        _, trades = simulate_portfolio(prices_uptrend, signals_one_cycle, fee_rate=0.0)
        t = trades.iloc[0]
        if t["price_exit"] > t["price_entry"]:
            assert t["pnl"] > 0
        else:
            assert t["pnl"] <= 0

    def test_return_pct_calculation(self, prices_uptrend, signals_one_cycle):
        """return_pct doit être (price_exit / price_entry) - 1."""
        _, trades = simulate_portfolio(prices_uptrend, signals_one_cycle, fee_rate=0.0)
        t = trades.iloc[0]
        expected = (t["price_exit"] / t["price_entry"]) - 1
        assert t["return_pct"] == pytest.approx(expected)


# ============================================================================
# simulate_portfolio — frais & paramètres
# ============================================================================

class TestSimulatePortfolioFees:
    """Vérifie la gestion des frais de transaction."""

    def test_higher_fees_reduce_equity(self, prices_uptrend, signals_one_cycle):
        """Plus de frais → moins d'argent à l'arrivée."""
        eq_no_fee, _ = simulate_portfolio(prices_uptrend, signals_one_cycle, fee_rate=0.0)
        eq_high_fee, _ = simulate_portfolio(prices_uptrend, signals_one_cycle, fee_rate=0.01)
        assert eq_high_fee["equity"].iloc[-1] < eq_no_fee["equity"].iloc[-1]

    def test_fee_pct_alias_works(self, prices_uptrend, signals_one_cycle):
        """
        ⚠️ Test régression : app.py utilise fee_pct=, et le code doit accepter
        cet alias en plus de fee_rate=.
        """
        eq1, _ = simulate_portfolio(prices_uptrend, signals_one_cycle, fee_rate=0.01)
        eq2, _ = simulate_portfolio(prices_uptrend, signals_one_cycle, fee_pct=0.01)
        assert eq1["equity"].iloc[-1] == pytest.approx(eq2["equity"].iloc[-1])

    def test_custom_initial_capital(self, prices_uptrend, signals_all_hold):
        """Le capital initial doit être respecté."""
        equity_df, _ = simulate_portfolio(prices_uptrend, signals_all_hold,
                                           initial_capital=50_000)
        assert equity_df["equity"].iloc[0] == 50_000


# ============================================================================
# buy_and_hold — stratégie de référence
# ============================================================================

class TestBuyAndHold:
    """Vérifie la stratégie passive de référence."""

    def test_doubling_prices_doubles_equity(self):
        prices = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "price": [100, 110, 120, 130, 140, 150, 160, 170, 180, 200],
        })
        result = buy_and_hold(prices, initial_capital=10_000)
        assert result["equity"].iloc[-1] == pytest.approx(20_000)

    def test_flat_prices_keep_equity_constant(self):
        prices = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "price": [100.0] * 10,
        })
        result = buy_and_hold(prices, initial_capital=10_000)
        assert (result["equity"] == 10_000).all()

    def test_halving_prices_halves_equity(self):
        prices = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "price": [100, 90, 80, 70, 50],
        })
        result = buy_and_hold(prices, initial_capital=10_000)
        assert result["equity"].iloc[-1] == pytest.approx(5_000)

    def test_initial_equity_matches_capital(self, prices_volatile):
        """Au jour 0, equity = capital initial exactement."""
        result = buy_and_hold(prices_volatile, initial_capital=10_000)
        assert result["equity"].iloc[0] == pytest.approx(10_000)


# ============================================================================
# Cohérence stratégie vs benchmark
# ============================================================================

class TestStrategyBeatsBenchmark:
    """Vérifie qu'une stratégie 'parfaite' bat bien le buy & hold."""

    def test_perfect_timing_beats_buy_and_hold(self, prices_with_drawdown,
                                                signals_perfect_timing):
        """
        Acheter au plus bas et vendre au pic = mieux que buy & hold
        qui subit le drawdown de la deuxième moitié.
        """
        strat, _ = simulate_portfolio(prices_with_drawdown,
                                       signals_perfect_timing, fee_rate=0.0)
        bh = buy_and_hold(prices_with_drawdown, initial_capital=10_000)
        assert strat["equity"].iloc[-1] > bh["equity"].iloc[-1]
