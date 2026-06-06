"""
metrics_panel.py
================
Grille de cartes st.metric affichant les KPIs de performance & risque.
"""
import streamlit as st
from utils.formatters import fmt_percent, fmt_ratio


def render_metrics_panel(metrics: dict, benchmark_metrics: dict = None):
    """Affiche le panneau des métriques en colonnes."""
    st.subheader("Métriques de performance")

    # Ligne 1 : performance globale
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        delta = None
        if benchmark_metrics:
            delta = fmt_percent(metrics.get("total_return", 0)
                                - benchmark_metrics.get("total_return", 0))
        st.metric("Rendement total",
                  fmt_percent(metrics.get("total_return", 0)),
                  delta=delta,
                  help="Performance totale sur la période")
    with col2:
        delta = None
        if benchmark_metrics:
            delta = fmt_percent(metrics.get("cagr", 0)
                                - benchmark_metrics.get("cagr", 0))
        st.metric("CAGR",
                  fmt_percent(metrics.get("cagr", 0)),
                  delta=delta,
                  help="Taux de croissance annualisé composé")
    with col3:
        delta = None
        if benchmark_metrics:
            delta = fmt_ratio(metrics.get("sharpe", 0)
                              - benchmark_metrics.get("sharpe", 0))
        st.metric("Sharpe Ratio",
                  fmt_ratio(metrics.get("sharpe", 0)),
                  delta=delta,
                  help="Rendement ajusté au risque. >1 bon, >2 très bon")
    with col4:
        delta = None
        if benchmark_metrics:
            delta = fmt_percent(metrics.get("max_drawdown", 0)
                                - benchmark_metrics.get("max_drawdown", 0))
        st.metric("Max Drawdown",
                  fmt_percent(metrics.get("max_drawdown", 0)),
                  delta=delta, delta_color="inverse",
                  help="Perte maximale depuis un pic")

    # Ligne 2 : métriques secondaires
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Sortino Ratio",
                  fmt_ratio(metrics.get("sortino", 0)),
                  help="Comme Sharpe mais ne pénalise que la volatilité baissière")
    with col6:
        st.metric("Calmar Ratio",
                  fmt_ratio(metrics.get("calmar", 0)),
                  help="CAGR / |Max Drawdown|")
    with col7:
        st.metric("Volatilité",
                  fmt_percent(metrics.get("volatility", 0)),
                  help="Volatilité annualisée")
    with col8:
        if "win_rate" in metrics:
            st.metric("Taux de réussite",
                      fmt_percent(metrics["win_rate"]),
                      help="% de trades gagnants")
        else:
            st.metric("Nb trades", metrics.get("n_trades", 0))

    if "profit_factor" in metrics:
        col9, col10, _, _ = st.columns(4)
        with col9:
            st.metric("Profit Factor",
                      fmt_ratio(metrics["profit_factor"]),
                      help="Somme gains / Somme pertes — > 1 = profitable")
        with col10:
            st.metric("Nb trades", metrics.get("n_trades", 0))
