"""
=============================================================
🛢️  PETROL TRADING DASHBOARD
=============================================================
Interface graphique du projet "Trading Encadré WTI Crude Oil".

Modules consommés :
- scraping/src/  → données de prix WTI/Brent + actualités
- ai/ml/         → signaux ML (Random Forest, XGBoost)
- ai/nlp/        → scores de sentiment

Lancement :
    cd web
    streamlit run app.py

Auteur : [Ton nom]
Projet encadré INSEA S4
=============================================================
"""
import streamlit as st
import pandas as pd

# Composants UI
from components.sidebar import render_sidebar
from components.price_chart import (
    render_price_chart, render_equity_curve, render_drawdown_chart
)
from components.recommendation import render_recommendation, aggregate_signals
from components.metrics_panel import render_metrics_panel
from components.trade_log import render_trade_log

# Logique métier
from utils.data_loader import (
    load_prices, load_ohlcv_wti, load_signals, load_news, has_ohlcv_data
)
from utils.backtest import simulate_portfolio, buy_and_hold
from utils.metrics import compute_all_metrics
from utils.mock_data import mock_recommendation_reasons


# ============================================================================
# Configuration de la page
# ============================================================================
st.set_page_config(
    page_title="Petrol Trading Dashboard",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛢️ Plateforme de Backtesting — Trading Pétrolier")
st.caption(
    "Analyse technique · Machine Learning (Random Forest / XGBoost) · Sentiment NLP · "
    "WTI & Brent Spot Prices"
)


# ============================================================================
# Sidebar
# ============================================================================
params = render_sidebar()


# ============================================================================
# Chargement des données
# ============================================================================
# Si l'utilisateur veut des bougies et qu'on a l'OHLCV, on charge le CSV
# Sinon on charge les prix spot (Parquet)
if (params["chart_type"].startswith("Bougies")
        and has_ohlcv_data(params["asset"])):
    prices = load_ohlcv_wti()
else:
    prices = load_prices(params["asset"])

if prices.empty:
    st.error("⚠️ Aucune donnée disponible.")
    st.info("Lance d'abord le pipeline : `cd scraping/src && python main.py`")
    st.stop()

# Filtrage par plage de dates
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

# Chargement ou génération dynamique des signaux selon la stratégie sélectionnée
if params["strategy"] == "Technique (SMA crossover)":
    from utils.data_loader import generate_sma_signals
    signals = generate_sma_signals(prices, params["sma_short"], params["sma_long"])
elif params["strategy"] == "Sentiment NLP":
    from utils.data_loader import load_sentiment, generate_sentiment_signals
    sentiment_df = load_sentiment(params["asset"])
    signals = generate_sentiment_signals(
        sentiment_df,
        buy_threshold=params["buy_threshold"],
        sell_threshold=params["sell_threshold"],
        smooth_window=params["smooth_window"]
    )
else:
    signals = load_signals(params["asset"])

signals_filtered = signals[
    (signals["date"] >= pd.Timestamp(start)) &
    (signals["date"] <= pd.Timestamp(end))
].reset_index(drop=True) if not signals.empty else signals


# ============================================================================
# Layout en onglets
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Recommandation",
    "📈 Analyse de prix",
    "💼 Backtest",
    "📰 Actualités"
])

# ----------------------------------------------------------------------------
# TAB 1 — Recommandation
# ----------------------------------------------------------------------------
with tab1:
    signal, confidence = aggregate_signals(signals_filtered)
    current_price = float(prices_filtered["price"].iloc[-1])
    reasons = mock_recommendation_reasons(signal)

    render_recommendation(
        signal=signal, confidence=confidence,
        reasons=reasons, asset=params["asset"],
        current_price=current_price,
    )

    st.markdown("### Aperçu de la tendance récente")
    recent = prices_filtered.tail(90)
    recent_signals = signals_filtered.tail(90) if not signals_filtered.empty else None
    render_price_chart(
        recent,
        asset=params["asset"],
        chart_type="Ligne (prix spot)",
        signals=recent_signals,
        show_sma=True, show_bollinger=False, show_rsi=False, show_macd=False,
    )

# ----------------------------------------------------------------------------
# TAB 2 — Analyse de prix
# ----------------------------------------------------------------------------
with tab2:
    col1, col2, col3, col4 = st.columns(4)
    last_price = prices_filtered["price"].iloc[-1]
    first_price = prices_filtered["price"].iloc[0]
    perf = (last_price / first_price - 1) * 100

    col1.metric("💵 Prix actuel", f"${last_price:.2f}")
    col2.metric("📊 Prix moyen", f"${prices_filtered['price'].mean():.2f}")
    col3.metric("📈 Performance période", f"{perf:+.2f}%")
    col4.metric("📅 Jours analysés", len(prices_filtered))

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

# ----------------------------------------------------------------------------
# TAB 3 — Backtest complet
# ----------------------------------------------------------------------------
with tab3:
    st.markdown(f"### Stratégie testée : **{params['strategy']}**")
    # Détecte si la stratégie utilise des données réelles ou simulées (mock)
    using_mock = True
    if params["strategy"] == "Sentiment NLP":
        from utils.data_loader import DATA_DIR
        using_mock = not (DATA_DIR / f"sentiment_{params['asset'].lower()}.parquet").exists()
    elif params["strategy"] in ["ML — Random Forest", "ML — XGBoost"]:
        from utils.data_loader import DATA_DIR
        using_mock = not (DATA_DIR / f"signals_ml_{params['asset'].lower()}.parquet").exists()
    elif params["strategy"] == "Technique (SMA crossover)":
        using_mock = False  # Calculé dynamiquement sur les prix réels

    if using_mock:
        st.info(
            f"ℹ️ La stratégie **{params['strategy']}** utilise actuellement des signaux "
            "factices (mock). Le dashboard remplacera automatiquement "
            "ces mocks par les vrais signaux dès que les modules ML/NLP les livreront."
        )
    else:
        st.success(
            f"✅ La stratégie **{params['strategy']}** est active et utilise les signaux "
            "réels générés par le module correspondant."
        )

    # Simulation
    equity_df, trades_df = simulate_portfolio(
        prices_filtered, signals_filtered,
        initial_capital=params["capital"],
        fee_rate=params["fee_pct"],
    )

    benchmark_df = buy_and_hold(prices_filtered, params["capital"]) if params["show_benchmark"] else None

    metrics = compute_all_metrics(equity_df["equity"], equity_df["returns"], trades_df)
    benchmark_metrics = None
    if benchmark_df is not None:
        benchmark_metrics = compute_all_metrics(benchmark_df["equity"], benchmark_df["returns"])

    render_metrics_panel(metrics, benchmark_metrics)

    st.markdown("---")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        render_equity_curve(equity_df, benchmark_df)
    with col_b:
        render_drawdown_chart(equity_df)

    st.markdown("---")
    render_trade_log(trades_df)

# ----------------------------------------------------------------------------
# TAB 4 — Actualités
# ----------------------------------------------------------------------------
with tab4:
    st.subheader("📰 Actualités pétrolières — source : OilPrice.com")
    news = load_news("oilprice")

    if news.empty:
        st.info(
            "Aucune actualité chargée. Lance le scraping :\n\n"
            "`cd scraping/src && python main.py`"
        )
    else:
        st.caption(f"{len(news)} articles disponibles")
        for _, row in news.head(20).iterrows():
            with st.container():
                st.markdown(f"**{row.get('title', 'Sans titre')}**")
                if row.get("content"):
                    content = str(row["content"])[:200]
                    st.caption(content + ("..." if len(str(row["content"])) > 200 else ""))
                st.caption(f"📅 {row.get('date', '?')} · 🌐 {row.get('source', 'N/A')}")
                st.markdown("---")


# ============================================================================
# Footer
# ============================================================================
st.markdown("---")
st.caption(
    "🛢️ **Petrol Trading Backtesting Platform** · Projet encadré INSEA S4 · "
    "Modules : Scraping · ML (RF/XGBoost) · NLP · Dashboard Streamlit"
)
