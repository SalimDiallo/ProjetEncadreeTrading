"""
=============================================================
🛢️  PETROL TRADING — PLATEFORME DE BACKTESTING
=============================================================
Backtesting de stratégies sur le pétrole (WTI) :
- Stratégie technique (SMA crossover)
- Machine Learning : Random Forest & XGBoost, avec le sentiment NLP
  intégré comme feature
- Sentiment NLP seul
- Hybride (vote majoritaire ML + NLP)

Modules consommés :
- scraping/src/  → prix WTI + actualités
- ai/ml/         → signaux ML (signals_ml_*.parquet)
- ai/nlp/        → sentiment (sentiment_*.parquet)

Lancement :
    cd web && streamlit run app.py
=============================================================
"""
import streamlit as st
import pandas as pd

from components.sidebar import render_sidebar
from components.price_chart import (
    render_price_chart, render_equity_curve, render_drawdown_chart
)
from components.recommendation import (
    render_recommendation, aggregate_signals, build_real_reasons
)
from components.metrics_panel import render_metrics_panel
from components.trade_log import render_trade_log

from utils.data_loader import (
    load_prices, load_ohlcv_wti, load_signals, load_sentiment,
    has_ohlcv_data, generate_sma_signals, generate_sentiment_signals,
    generate_hybrid_signals, DATA_DIR,
)
from utils.backtest import simulate_portfolio, buy_and_hold
from utils.metrics import compute_all_metrics


# ============================================================================
# Configuration
# ============================================================================
st.set_page_config(
    page_title="Petrol Backtesting",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Backtesting — Trading Pétrolier")

params = render_sidebar()


# ============================================================================
# Chargement des prix
# ============================================================================
if params["chart_type"].startswith("Bougies") and has_ohlcv_data(params["asset"]):
    prices = load_ohlcv_wti()
else:
    prices = load_prices(params["asset"])

if prices.empty:
    st.error("Aucune donnée de prix disponible. "
             "Lancer le pipeline : `cd scraping/src && python main.py`")
    st.stop()

# Filtrage par période
if isinstance(params["date_range"], tuple) and len(params["date_range"]) == 2:
    start, end = params["date_range"]
    mask = (prices["date"] >= pd.Timestamp(start)) & (prices["date"] <= pd.Timestamp(end))
    prices_filtered = prices.loc[mask].reset_index(drop=True)
else:
    prices_filtered = prices
    start, end = prices["date"].min(), prices["date"].max()

if prices_filtered.empty:
    st.warning("Aucune donnée sur la plage sélectionnée. Élargis la période.")
    st.stop()


# ============================================================================
# Génération des signaux de la stratégie sélectionnée
# ============================================================================
strategy = params["strategy"]

if strategy == "Buy & Hold (référence)":
    signals = pd.DataFrame(columns=["date", "signal", "confidence", "model"])
elif strategy == "Technique (SMA crossover)":
    signals = generate_sma_signals(prices, params["sma_short"], params["sma_long"])
elif strategy == "Sentiment NLP seul":
    signals = generate_sentiment_signals(
        load_sentiment(params["asset"]),
        buy_threshold=params["buy_threshold"],
        sell_threshold=params["sell_threshold"],
        smooth_window=params["smooth_window"],
    )
elif strategy == "Hybride (vote ML+NLP)":
    signals = generate_hybrid_signals(params["asset"], prices)
else:  # stratégies ML (RandomForest+NLP / XGBoost+NLP)
    signals = load_signals(params["asset"], params["model_key"])

signals_filtered = signals[
    (signals["date"] >= pd.Timestamp(start)) &
    (signals["date"] <= pd.Timestamp(end))
].reset_index(drop=True) if not signals.empty else signals


# ============================================================================
# Onglets — le BACKTEST est l'écran principal
# ============================================================================
tab_bt, tab_tech = st.tabs([
    "Backtest",
    "Analyse technique",
])

# ----------------------------------------------------------------------------
# TAB 1 — BACKTEST (écran central)
# ----------------------------------------------------------------------------
with tab_bt:
    st.markdown(f"### Stratégie : **{strategy}**")

    is_bh = strategy == "Buy & Hold (référence)"
    is_ml = strategy in ("ML — Random Forest (+NLP)", "ML — XGBoost (+NLP)")

    # Garde-fou pour les modèles ML : par défaut, on n'évalue que la période
    # out-of-sample (jamais vue à l'entraînement). Évaluer en in-sample gonfle
    # artificiellement les rendements (le modèle « connaît » déjà ces mouvements).
    bt_prices = prices_filtered
    if is_ml:
        from utils.data_loader import get_oos_start_date
        oos_start = get_oos_start_date(params["asset"])
        eval_mode = st.radio(
            "Période d'évaluation",
            ["Out-of-sample uniquement (réaliste)", "Tout l'historique (in-sample inclus)"],
            horizontal=True,
            key="eval_mode",
            help="Le modèle est entraîné sur les 80% initiaux. L'évaluer sur ces "
                 "données gonfle les rendements. L'out-of-sample mesure la vraie "
                 "capacité prédictive.",
        )
        if eval_mode.startswith("Out-of-sample") and oos_start is not None:
            bt_prices = prices_filtered[prices_filtered["date"] >= oos_start].reset_index(drop=True)
            st.caption(f"Période out-of-sample : {oos_start.date()} → "
                       f"{bt_prices['date'].max().date()} ({len(bt_prices)} jours)")
            if len(bt_prices) < 20:
                st.warning("Période out-of-sample trop courte sur la plage choisie — "
                           "élargis les dates dans la sidebar.")
                st.stop()
        else:
            st.warning("Mode in-sample inclus : les rendements affichés sont "
                       "optimistes (données vues à l'entraînement) et ne reflètent "
                       "pas la performance réelle. À interpréter avec prudence.")

    if is_bh:
        equity_df = buy_and_hold(bt_prices, params["capital"])
        trades_df = pd.DataFrame()
        metrics = compute_all_metrics(equity_df["equity"], equity_df["returns"])
        benchmark_df = None
    else:
        equity_df, trades_df = simulate_portfolio(
            bt_prices, signals_filtered,
            initial_capital=params["capital"], fee_rate=params["fee_pct"],
            allow_short=params["allow_short"],
            confidence_threshold=params["confidence_threshold"],
            stop_loss=params["stop_loss"],
            take_profit=params["take_profit"],
        )
        metrics = compute_all_metrics(equity_df["equity"], equity_df["returns"], trades_df)
        benchmark_df = buy_and_hold(bt_prices, params["capital"]) \
            if params["show_benchmark"] else None

        # Récapitulatif du mode de trading actif
        mode = "Long/Short" if params["allow_short"] else "Long-only"
        thr = params["confidence_threshold"]
        risk = ""
        if params["stop_loss"] is not None:
            risk = (f" · SL {params['stop_loss']*100:.1f}% / "
                    f"TP {params['take_profit']*100:.1f}%")
        st.caption(f"Mode : **{mode}** · seuil de confiance : **{thr:.2f}**{risk} · "
                   f"{len(trades_df)} trade(s) déclenché(s)")

    benchmark_metrics = None
    if benchmark_df is not None:
        benchmark_metrics = compute_all_metrics(benchmark_df["equity"], benchmark_df["returns"])

    # Métriques de performance
    render_metrics_panel(metrics, benchmark_metrics)

    st.markdown("---")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        render_equity_curve(equity_df, benchmark_df)
    with col_b:
        render_drawdown_chart(equity_df)

    st.markdown("---")
    render_trade_log(trades_df)

    # Recommandation actuelle (raisons réelles) — sauf Buy & Hold
    if not is_bh:
        st.markdown("---")
        st.markdown("### Recommandation actuelle")
        signal, confidence = aggregate_signals(signals_filtered)
        reasons = build_real_reasons(
            prices_filtered, signals_filtered,
            sentiment=load_sentiment(params["asset"]), strategy=strategy,
        )
        render_recommendation(
            signal=signal, confidence=confidence, reasons=reasons,
            asset=params["asset"],
            current_price=float(prices_filtered["price"].iloc[-1]),
        )

# ----------------------------------------------------------------------------
# TAB 2 — ANALYSE TECHNIQUE
# ----------------------------------------------------------------------------
with tab_tech:
    col1, col2, col3, col4 = st.columns(4)
    last_price = prices_filtered["price"].iloc[-1]
    first_price = prices_filtered["price"].iloc[0]
    perf = (last_price / first_price - 1) * 100
    col1.metric("Prix actuel", f"${last_price:.2f}")
    col2.metric("Prix moyen", f"${prices_filtered['price'].mean():.2f}")
    col3.metric("Performance période", f"{perf:+.2f}%")
    col4.metric("Jours analysés", len(prices_filtered))

    st.markdown("---")
    render_price_chart(
        prices_filtered,
        asset=params["asset"],
        chart_type=params["chart_type"],
        signals=signals_filtered,
        show_sma=params["show_sma"],
        show_bollinger=params["show_bollinger"],
        show_rsi=params["show_rsi"],
        show_macd=params["show_macd"],
    )

# ============================================================================
# Footer
# ============================================================================
st.markdown("---")
st.caption("Petrol Trading Backtesting · INSEA S4 · Outil éducatif")
