"""
model_comparison.py
===================
Comparateur de stratégies : backteste RF, XGBoost (avec/sans NLP), NLP seul,
SMA et Buy & Hold sur la MÊME période, puis affiche un tableau de métriques,
l'impact réel du sentiment NLP et les courbes d'équité superposées.

C'est l'écran qui matérialise l'intégration ML × NLP demandée :
le sentiment est une feature des deux modèles, et on mesure son effet.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.backtest import simulate_portfolio, buy_and_hold
from utils.metrics import compute_all_metrics
from utils.data_loader import (
    load_signals, load_sentiment, generate_sma_signals, generate_sentiment_signals,
    get_oos_start_date,
)
from utils.formatters import fmt_percent, fmt_ratio


# Stratégies comparées : libellé → (type, paramètre)
# type "ml" → clé de modèle pour load_signals ; "sma"/"nlp"/"bh" → calcul dynamique
COMPARISON_STRATEGIES = [
    ("Random Forest +NLP", "ml", "RandomForest+NLP"),
    ("Random Forest", "ml", "RandomForest"),
    ("XGBoost +NLP", "ml", "XGBoost+NLP"),
    ("XGBoost", "ml", "XGBoost"),
    ("Sentiment NLP seul", "nlp", None),
    ("Technique (SMA 20/50)", "sma", None),
    ("Buy & Hold", "bh", None),
]


def _signals_for(kind: str, param, prices: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Retourne les signaux d'une stratégie, filtrés sur la période des prix."""
    if kind == "ml":
        sig = load_signals(asset, param)
    elif kind == "sma":
        sig = generate_sma_signals(prices, 20, 50)
    elif kind == "nlp":
        sig = generate_sentiment_signals(load_sentiment(asset))
    else:  # buy & hold : pas de signaux discrets
        return pd.DataFrame(columns=["date", "signal", "confidence", "model"])

    if sig.empty:
        return sig
    start, end = prices["date"].min(), prices["date"].max()
    return sig[(sig["date"] >= start) & (sig["date"] <= end)].reset_index(drop=True)


def _backtest_metrics(prices, signals, capital, fee_rate, is_bh=False):
    if is_bh:
        eq = buy_and_hold(prices, capital)
        return compute_all_metrics(eq["equity"], eq["returns"]), eq
    eq, trades = simulate_portfolio(prices, signals, capital, fee_rate)
    return compute_all_metrics(eq["equity"], eq["returns"], trades), eq


def render_model_comparison(prices: pd.DataFrame, asset: str,
                            capital: float, fee_rate: float):
    """Affiche le comparateur complet des stratégies."""
    st.subheader("Comparaison des modèles ML & NLP")
    st.caption(
        "Stratégies backtestées sur la même période et le même capital. "
        "Les variantes +NLP intègrent le sentiment comme feature du modèle."
    )

    # --- Choix honnête de la période d'évaluation ---
    oos_start = get_oos_start_date(asset)
    eval_mode = st.radio(
        "Période d'évaluation",
        ["Out-of-sample uniquement (honnête)", "Tout l'historique (avec in-sample)"],
        horizontal=True,
        help="Les modèles ML sont entraînés sur les 80% initiaux. Évaluer sur "
             "l'in-sample gonfle artificiellement les rendements (le modèle 'connaît' "
             "déjà ces mouvements). L'out-of-sample reflète la vraie capacité prédictive.",
    )
    if eval_mode.startswith("Out-of-sample") and oos_start is not None:
        prices = prices[prices["date"] >= oos_start].reset_index(drop=True)
        st.caption(f"Période out-of-sample : {oos_start.date()} → "
                   f"{prices['date'].max().date()} ({len(prices)} jours)")
        if len(prices) < 20:
            st.warning("Période OOS trop courte sur la plage sélectionnée — "
                       "élargis les dates dans la sidebar.")
    else:
        st.warning("Mode in-sample inclus : les rendements des modèles ML sont "
                   "optimistes (données vues à l'entraînement). À interpréter avec prudence.")

    rows, equity_curves = [], {}
    for label, kind, param in COMPARISON_STRATEGIES:
        signals = _signals_for(kind, param, prices, asset)
        metrics, eq = _backtest_metrics(prices, signals, capital, fee_rate,
                                        is_bh=(kind == "bh"))
        equity_curves[label] = eq
        rows.append({
            "Stratégie": label,
            "Rendement": fmt_percent(metrics.get("total_return", 0)),
            "CAGR": fmt_percent(metrics.get("cagr", 0)),
            "Sharpe": fmt_ratio(metrics.get("sharpe", 0)),
            "Max DD": fmt_percent(metrics.get("max_drawdown", 0)),
            "Win rate": fmt_percent(metrics["win_rate"]) if "win_rate" in metrics else "—",
            "Trades": str(metrics.get("n_trades", "—")),
            "_ret": metrics.get("total_return", 0),  # tri/colonne cachée
        })

    table = pd.DataFrame(rows)

    def _highlight(row):
        if "+NLP" in row["Stratégie"]:
            return ["background-color: rgba(16,185,129,0.10)"] * len(row)
        if row["Stratégie"] == "Buy & Hold":
            return ["background-color: rgba(107,114,128,0.10)"] * len(row)
        return [""] * len(row)

    styled = (table.drop(columns=["_ret"])
              .style.apply(_highlight, axis=1))
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # --- Impact réel du NLP (Δ rendement avec − sans) ---
    ret = {r["Stratégie"]: r["_ret"] for r in rows}
    st.markdown("#### Impact du sentiment NLP")
    c1, c2 = st.columns(2)
    with c1:
        d_rf = ret["Random Forest +NLP"] - ret["Random Forest"]
        st.metric("Random Forest : +NLP vs sans",
                  fmt_percent(ret["Random Forest +NLP"]),
                  delta=fmt_percent(d_rf),
                  help="Différence de rendement total apportée par le sentiment")
    with c2:
        d_xgb = ret["XGBoost +NLP"] - ret["XGBoost"]
        st.metric("XGBoost : +NLP vs sans",
                  fmt_percent(ret["XGBoost +NLP"]),
                  delta=fmt_percent(d_xgb))

    # --- Note de transparence sur la couverture du sentiment ---
    _render_coverage_note(prices, asset)

    # --- Courbes d'équité superposées ---
    st.markdown("#### Courbes d'équité comparées")
    default_sel = ["Random Forest +NLP", "XGBoost +NLP", "Buy & Hold"]
    selected = st.multiselect(
        "Stratégies à afficher",
        list(equity_curves.keys()),
        default=default_sel,
    )
    _render_equity_overlay(equity_curves, selected)


def _render_coverage_note(prices: pd.DataFrame, asset: str):
    """Affiche honnêtement la couverture du sentiment réel."""
    sen = load_sentiment(asset)
    if sen.empty:
        return
    start, end = prices["date"].min(), prices["date"].max()
    sen_period = sen[(sen["date"] >= start) & (sen["date"] <= end)]
    n_days_sen = sen_period["date"].nunique()
    n_days_total = prices["date"].nunique()
    st.info(
        f"Transparence NLP : le sentiment réel ne couvre que {n_days_sen} jours "
        f"sur {n_days_total} (news scrapées en direct, sans archives). Son impact "
        f"se concentre sur la période récente ; l'écart avec/sans NLP reste modeste."
    )


def _render_equity_overlay(equity_curves: dict, selected: list):
    if not selected:
        st.caption("Sélectionne au moins une stratégie.")
        return
    palette = ["#FF6B35", "#10b981", "#3b82f6", "#8b5cf6", "#ef4444",
               "#f59e0b", "#6b7280"]
    fig = go.Figure()
    for i, label in enumerate(selected):
        eq = equity_curves.get(label)
        if eq is None or eq.empty:
            continue
        fig.add_trace(go.Scatter(
            x=eq["date"], y=eq["equity"], mode="lines", name=label,
            line=dict(width=2, color=palette[i % len(palette)]),
        ))
    fig.update_layout(
        template="plotly_white", height=420,
        xaxis_title="Date", yaxis_title="Capital ($)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)
