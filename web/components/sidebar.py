"""
sidebar.py
==========
Barre latérale avec tous les filtres et paramètres utilisateur.
"""
import streamlit as st
from datetime import date
from utils.data_loader import get_available_assets, has_ohlcv_data


def render_sidebar() -> dict:
    """Affiche la sidebar et retourne les paramètres choisis."""

    st.sidebar.title("⚙️ Paramètres d'analyse")

    # --- Actif ---
    st.sidebar.subheader("📊 Actif")
    available = get_available_assets()
    asset = st.sidebar.selectbox(
        "Actif pétrolier",
        available,
        help="WTI = référence US · Brent = référence Europe/Asie"
    )

    # Type de graphique selon la dispo OHLCV
    chart_options = ["Ligne (prix spot)"]
    if has_ohlcv_data(asset):
        chart_options.append("Bougies japonaises (OHLCV)")
    chart_type = st.sidebar.radio(
        "Type de graphique",
        chart_options,
        help="Les bougies sont disponibles uniquement pour les actifs avec données OHLCV"
    )

    # --- Période ---
    st.sidebar.subheader("📅 Période")
    date_range = st.sidebar.date_input(
        "Plage de dates",
        value=(date(2024, 1, 1), date(2026, 4, 13)),
        min_value=date(2020, 1, 1),
        max_value=date(2026, 12, 31),
    )

    # --- Stratégie ---
    st.sidebar.subheader("🎯 Stratégie")
    strategy = st.sidebar.selectbox(
        "Stratégie à backtester",
        [
            "Technique (SMA crossover)",
            "ML — Random Forest",
            "ML — XGBoost",
            "Sentiment NLP",
            "Hybride (vote majoritaire)",
        ],
    )

    sma_short = 20
    sma_long = 50
    buy_threshold = 0.05
    sell_threshold = -0.05
    smooth_window = 3

    if strategy == "Technique (SMA crossover)":
        sma_short = st.sidebar.slider("SMA court (jours)", min_value=2, max_value=50, value=20, step=1)
        sma_long = st.sidebar.slider("SMA long (jours)", min_value=10, max_value=150, value=50, step=1)
    elif strategy == "Sentiment NLP":
        buy_threshold = st.sidebar.slider("Seuil achat (BUY)", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
        sell_threshold = st.sidebar.slider("Seuil vente (SELL)", min_value=-0.5, max_value=0.0, value=-0.05, step=0.01)
        smooth_window = st.sidebar.slider("Lissage sentiment (jours)", min_value=1, max_value=10, value=3, step=1)

    # --- Paramètres avancés ---
    with st.sidebar.expander("⚙️ Paramètres avancés"):
        capital = st.number_input(
            "Capital initial ($)",
            min_value=1_000, max_value=10_000_000,
            value=10_000, step=1_000
        )
        fee_pct = st.slider(
            "Frais par trade (%)",
            min_value=0.0, max_value=1.0,
            value=0.1, step=0.05
        ) / 100
        show_benchmark = st.checkbox("Comparer au Buy & Hold", value=True)

    # --- Indicateurs ---
    with st.sidebar.expander("📈 Indicateurs techniques"):
        show_sma = st.checkbox("SMA 20 & 50", value=True)
        show_bollinger = st.checkbox("Bandes de Bollinger", value=False)
        show_rsi = st.checkbox("RSI (14)", value=True)
        show_macd = st.checkbox("MACD", value=False)

    # --- Bouton ---
    st.sidebar.markdown("---")
    run = st.sidebar.button(
        "🚀 Lancer l'analyse",
        type="primary",
        use_container_width=True
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "🛢️ **Petrol Trading Platform**\n\n"
        "Projet encadré INSEA S4\n"
        "Outil éducatif"
    )

    return {
        "asset": asset,
        "chart_type": chart_type,
        "date_range": date_range,
        "strategy": strategy,
        "capital": capital,
        "fee_pct": fee_pct,
        "show_benchmark": show_benchmark,
        "show_sma": show_sma,
        "show_bollinger": show_bollinger,
        "show_rsi": show_rsi,
        "show_macd": show_macd,
        "run": run,
        "sma_short": sma_short,
        "sma_long": sma_long,
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "smooth_window": smooth_window,
    }
