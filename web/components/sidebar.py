"""
sidebar.py
==========
Barre latérale avec tous les filtres et paramètres utilisateur.
"""
import streamlit as st
from datetime import date
from utils.data_loader import get_available_assets, has_ohlcv_data


# Stratégies proposées dans le dashboard. Pour les stratégies ML, la valeur
# associée est la clé de modèle attendue par load_signals (ML_MODEL_FILES).
STRATEGIES = {
    "Buy & Hold (référence)": None,
    "Technique (SMA crossover)": None,
    "ML — Random Forest (+NLP)": "RandomForest+NLP",
    "ML — XGBoost (+NLP)": "XGBoost+NLP",
    "Sentiment NLP seul": None,
    "Hybride (vote ML+NLP)": None,
}


def render_sidebar() -> dict:
    """Affiche la sidebar et retourne les paramètres choisis."""

    st.sidebar.title("Paramètres")

    # --- Actif ---
    st.sidebar.subheader("Actif")
    available = get_available_assets()
    asset = st.sidebar.selectbox(
        "Actif pétrolier",
        available,
        help="WTI — référence US (West Texas Intermediate)"
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
    st.sidebar.subheader("Période")
    date_range = st.sidebar.date_input(
        "Plage de dates",
        value=(date(2024, 1, 1), date(2026, 4, 13)),
        min_value=date(2020, 1, 1),
        max_value=date(2026, 12, 31),
    )

    # --- Stratégie ---
    st.sidebar.subheader("Stratégie à backtester")
    strategy = st.sidebar.selectbox(
        "Stratégie",
        list(STRATEGIES.keys()),
        index=2,  # ML — Random Forest (+NLP) par défaut
        label_visibility="collapsed",
    )
    model_key = STRATEGIES.get(strategy)

    sma_short = 20
    sma_long = 50
    buy_threshold = 0.05
    sell_threshold = -0.05
    smooth_window = 3

    if strategy == "Technique (SMA crossover)":
        sma_short = st.sidebar.slider("SMA court (jours)", min_value=2, max_value=50, value=20, step=1)
        sma_long = st.sidebar.slider("SMA long (jours)", min_value=10, max_value=150, value=50, step=1)
    elif strategy == "Sentiment NLP seul":
        buy_threshold = st.sidebar.slider("Seuil achat (BUY)", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
        sell_threshold = st.sidebar.slider("Seuil vente (SELL)", min_value=-0.5, max_value=0.0, value=-0.05, step=0.01)
        smooth_window = st.sidebar.slider("Lissage sentiment (jours)", min_value=1, max_value=10, value=3, step=1)

    # --- Paramètres avancés ---
    with st.sidebar.expander("Paramètres avancés"):
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

        position_mode = st.radio(
            "Mode de position",
            ["Long-only", "Long / Short"],
            key="position_mode",
            help="Long-only : on achète ou on reste cash. "
                 "Long/Short : un signal SELL ouvre une position courte "
                 "(on parie sur la baisse).",
        )
        allow_short = position_mode == "Long / Short"

        confidence_threshold = st.slider(
            "Seuil de confiance",
            min_value=0.50, max_value=0.70, value=0.50, step=0.01,
            key="confidence_threshold",
            help="On ne prend position que si la confiance du modèle dépasse "
                 "ce seuil ; sinon on reste neutre. Filtre les signaux faibles.",
        )

        # Gestion du risque : stop-loss / take-profit par trade
        use_risk = st.checkbox(
            "Stop-loss / Take-profit", value=False, key="use_risk",
            help="Clôture automatiquement une position dès qu'elle perd "
                 "(stop-loss) ou gagne (take-profit) le seuil choisi. "
                 "Réduit les gros drawdowns.",
        )
        stop_loss = None
        take_profit = None
        if use_risk:
            stop_loss = st.slider("Stop-loss (%)", min_value=1.0, max_value=15.0,
                                  value=5.0, step=0.5, key="stop_loss_pct") / 100
            take_profit = st.slider("Take-profit (%)", min_value=1.0, max_value=30.0,
                                    value=10.0, step=0.5, key="take_profit_pct") / 100

        show_benchmark = st.checkbox("Comparer au Buy & Hold", value=True)

    # --- Indicateurs (tous activés par défaut) ---
    with st.sidebar.expander("Indicateurs techniques", expanded=True):
        show_sma = st.checkbox("SMA 20 & 50", value=True)
        show_bollinger = st.checkbox("Bandes de Bollinger", value=True)
        show_rsi = st.checkbox("RSI (14)", value=True)
        show_macd = st.checkbox("MACD", value=True)

    st.sidebar.markdown("---")
    st.sidebar.caption("Outil éducatif — pas un conseil en investissement.")

    return {
        "asset": asset,
        "chart_type": chart_type,
        "date_range": date_range,
        "strategy": strategy,
        "model_key": model_key,
        "capital": capital,
        "fee_pct": fee_pct,
        "allow_short": allow_short,
        "confidence_threshold": confidence_threshold,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "show_benchmark": show_benchmark,
        "show_sma": show_sma,
        "show_bollinger": show_bollinger,
        "show_rsi": show_rsi,
        "show_macd": show_macd,
        "sma_short": sma_short,
        "sma_long": sma_long,
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "smooth_window": smooth_window,
    }
