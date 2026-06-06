"""
backtest.py
===========
Moteur de backtest pour simuler un portefeuille à partir de signaux.

Modèle « position cible » :
- À chaque jour, le signal + la confiance déterminent une position cible :
  +1 (long), -1 (short, si allow_short), ou 0 (neutre / cash).
- Si la confiance < confidence_threshold, on reste neutre (0).
- Quand la position cible change, on clôture l'ancienne (trade réalisé,
  frais payés) et on ouvre la nouvelle.

Hypothèses :
- Position « tout ou rien » sur le capital courant (pas de sizing fractionnel)
- Frais de transaction configurables (défaut 0.1%), payés à chaque changement
- Pas de slippage ; exécution au prix de clôture du jour du signal
"""
import pandas as pd


def simulate_portfolio(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    initial_capital: float = 10_000,
    fee_rate: float = 0.001,
    fee_pct: float = None,            # alias pour fee_rate (compatibilité)
    allow_short: bool = False,        # True → SELL ouvre une position courte
    confidence_threshold: float = 0.0,  # ne prend position que si conf > seuil
    stop_loss: float = None,          # ex: 0.05 → clôture si perte ≥ 5%
    take_profit: float = None,        # ex: 0.10 → clôture si gain ≥ 10%
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simule un portefeuille à partir de signaux BUY/SELL/HOLD.

    Args:
        allow_short: si True, un signal SELL ouvre une position COURTE
            (on profite des baisses) au lieu de simplement sortir.
        confidence_threshold: si la confiance du signal est sous ce seuil,
            la position cible est neutre (0). Permet de ne trader que les
            signaux « francs ».
        stop_loss: perte max tolérée par trade (fraction positive, ex 0.05).
            Dès que le rendement de la position ≤ -stop_loss, on clôture.
        take_profit: gain cible par trade (fraction positive, ex 0.10).
            Dès que le rendement de la position ≥ take_profit, on clôture.
            Après un SL/TP, on reste neutre jusqu'à un nouveau signal.

    Returns:
        equity_df: [date, price, signal, position, cash, equity, returns]
        trades_df: [date_entry, date_exit, side, price_entry, price_exit, pnl, return_pct, exit_reason]
    """
    if fee_pct is not None:
        fee_rate = fee_pct

    cols = ["date", "signal"]
    if "confidence" in signals.columns:
        cols.append("confidence")
    df = prices.merge(signals[cols], on="date", how="left")
    df["signal"] = df["signal"].fillna("HOLD")
    if "confidence" not in df.columns:
        df["confidence"] = 1.0
    df["confidence"] = df["confidence"].fillna(0.0)

    # --- Position cible quotidienne (+1 / -1 / 0) ---
    def target_position(signal, conf):
        if conf < confidence_threshold:
            return 0
        if signal == "BUY":
            return 1
        if signal == "SELL":
            return -1 if allow_short else 0
        return 0  # HOLD → conserver la position précédente (géré ci-dessous)

    # État du portefeuille (variables mutées via la closure ci-dessous)
    state = {"cash": initial_capital, "units": 0.0, "side": 0,
             "entry_price": None, "entry_date": None}
    equity_hist, position_hist, cash_hist = [], [], []
    trades = []
    prev_target = 0
    locked_target = None  # après un SL/TP, on attend que le signal change

    def position_return(price):
        """Rendement courant de la position ouverte (long ou short)."""
        ep = state["entry_price"]
        if state["side"] == 1:
            return price / ep - 1
        if state["side"] == -1:
            return ep / price - 1
        return 0.0

    def close_position(price, date, reason):
        if state["side"] == 1:
            state["cash"] = state["units"] * price * (1 - fee_rate)
            pnl = (price - state["entry_price"]) * state["units"]
            ret = price / state["entry_price"] - 1
        else:  # short : rachat
            pnl = (state["entry_price"] - price) * abs(state["units"])
            state["cash"] = state["cash"] + pnl - abs(state["units"]) * price * fee_rate
            ret = state["entry_price"] / price - 1
        trades.append({
            "date_entry": state["entry_date"], "date_exit": date,
            "side": "LONG" if state["side"] == 1 else "SHORT",
            "price_entry": state["entry_price"], "price_exit": price,
            "pnl": pnl, "return_pct": ret, "exit_reason": reason,
        })
        state.update(units=0.0, side=0, entry_price=None, entry_date=None)

    def open_position(target, price, date):
        if target == 1:
            state["units"] = (state["cash"] * (1 - fee_rate)) / price
            state["cash"] = 0.0
            state.update(side=1, entry_price=price, entry_date=date)
        elif target == -1:
            notional = state["cash"]
            state["units"] = -notional / price
            state["cash"] = state["cash"] - notional * fee_rate
            state.update(side=-1, entry_price=price, entry_date=date)

    for _, row in df.iterrows():
        price, date = row["price"], row["date"]
        raw_target = target_position(row["signal"], row["confidence"])
        target = prev_target if row["signal"] == "HOLD" else raw_target
        prev_target = target

        # 1) Stop-loss / take-profit sur la position ouverte
        if state["side"] != 0:
            r = position_return(price)
            if stop_loss is not None and r <= -abs(stop_loss):
                close_position(price, date, "Stop-loss")
                locked_target = target   # on ne ré-ouvre pas avant un nouveau signal
            elif take_profit is not None and r >= abs(take_profit):
                close_position(price, date, "Take-profit")
                locked_target = target

        # 2) Déverrouillage : dès que le signal cible change, on redevient libre
        if locked_target is not None and target != locked_target:
            locked_target = None

        # 3) Changement de position selon le signal (si non verrouillé)
        if locked_target is None and target != state["side"]:
            if state["side"] != 0:
                close_position(price, date, "Signal")
            open_position(target, price, date)

        # --- Valorisation ---
        if state["side"] == 1:
            equity = state["cash"] + state["units"] * price
        elif state["side"] == -1:
            equity = state["cash"] + (state["entry_price"] - price) * abs(state["units"])
        else:
            equity = state["cash"]

        equity_hist.append(equity)
        position_hist.append(state["units"])
        cash_hist.append(state["cash"])

    df["position"] = position_hist
    df["cash"] = cash_hist
    df["equity"] = equity_hist
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
