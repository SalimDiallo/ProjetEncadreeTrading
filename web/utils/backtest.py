"""
backtest.py
===========
Moteur de backtest simple pour simuler un portefeuille à partir
de signaux d'achat/vente.

Hypothèses :
- Long-only (pas de short)
- All-in / all-out (pas de position sizing fractionnel)
- Frais de transaction configurable (défaut : 0.1%)
- Pas de slippage modélisé
- Exécution au prix de clôture du jour du signal
"""
import pandas as pd


def simulate_portfolio(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    initial_capital: float = 10_000,
    fee_rate: float = 0.001,
    fee_pct: float = None,  # alias pour fee_rate (compatibilité)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simule un portefeuille long-only à partir de signaux.

    Returns:
        equity_df: [date, price, signal, position, cash, equity, returns]
        trades_df: [date_entry, date_exit, price_entry, price_exit, pnl, return_pct]
    """
    if fee_pct is not None:
        fee_rate = fee_pct

    df = prices.merge(signals[["date", "signal"]], on="date", how="left")
    df["signal"] = df["signal"].fillna("HOLD")

    cash = initial_capital
    position = 0.0
    equity_history = []
    position_history = []
    cash_history = []

    trades = []
    open_trade = None

    for _, row in df.iterrows():
        price = row["price"]
        signal = row["signal"]

        if signal == "BUY" and position == 0:
            position = (cash * (1 - fee_rate)) / price
            cash = 0
            open_trade = {
                "date_entry": row["date"],
                "price_entry": price,
            }

        elif signal == "SELL" and position > 0:
            cash = position * price * (1 - fee_rate)
            if open_trade is not None:
                open_trade["date_exit"] = row["date"]
                open_trade["price_exit"] = price
                open_trade["pnl"] = (price - open_trade["price_entry"]) * position
                open_trade["return_pct"] = (price / open_trade["price_entry"]) - 1
                trades.append(open_trade)
                open_trade = None
            position = 0.0

        equity = cash + position * price
        equity_history.append(equity)
        position_history.append(position)
        cash_history.append(cash)

    df["position"] = position_history
    df["cash"] = cash_history
    df["equity"] = equity_history
    df["returns"] = df["equity"].pct_change().fillna(0)

    trades_df = pd.DataFrame(trades)
    return df, trades_df


def buy_and_hold(prices: pd.DataFrame, initial_capital: float = 10_000) -> pd.DataFrame:
    """Stratégie passive de référence."""
    df = prices.copy()
    units = initial_capital / df["price"].iloc[0]
    df["equity"] = units * df["price"]
    df["returns"] = df["equity"].pct_change().fillna(0)
    return df
