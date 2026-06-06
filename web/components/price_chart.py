"""
price_chart.py
==============
Graphiques de prix (ligne ou bougies) avec indicateurs techniques
en surimpression et sous-graphiques (RSI, MACD).
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.indicators import (
    compute_sma, compute_bollinger, compute_rsi, compute_macd
)


# ============================================================================
# Graphique principal de prix
# ============================================================================

def render_price_chart(
    df: pd.DataFrame,
    asset: str = "WTI",
    chart_type: str = "Ligne (prix spot)",
    signals: pd.DataFrame = None,
    show_sma: bool = True,
    show_bollinger: bool = False,
    show_rsi: bool = False,
    show_macd: bool = False,
):
    """
    Affiche le graphique principal avec indicateurs choisis.
    Si show_rsi ou show_macd → graphique multi-panneaux.
    """
    if df.empty:
        st.warning("Aucune donnée à afficher.")
        return

    # Préparation des indicateurs
    df_ind = df.copy()
    if show_sma:
        df_ind = compute_sma(df_ind, windows=[20, 50])
    if show_bollinger:
        df_ind = compute_bollinger(df_ind)
    if show_rsi:
        df_ind = compute_rsi(df_ind)
    if show_macd:
        df_ind = compute_macd(df_ind)

    # Structure du subplot
    n_panels = 1 + (1 if show_rsi else 0) + (1 if show_macd else 0)
    heights = [0.6 if n_panels > 1 else 1.0]
    if show_rsi:
        heights.append(0.2)
    if show_macd:
        heights.append(0.2)

    fig = make_subplots(
        rows=n_panels, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=heights,
    )

    # --- Panneau 1 : Prix ---
    is_candles = (chart_type.startswith("Bougies") and 
                  all(c in df_ind.columns for c in ["open", "high", "low", "close"]))

    if is_candles:
        fig.add_trace(go.Candlestick(
            x=df_ind["date"],
            open=df_ind["open"], high=df_ind["high"],
            low=df_ind["low"], close=df_ind["close"],
            name=asset,
            increasing_line_color="#0d9488",
            decreasing_line_color="#dc2626",
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=df_ind["date"], y=df_ind["price"],
            mode="lines", name=f"Prix {asset}",
            line=dict(color="#FF6B35", width=2.5),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Prix: $%{y:.2f}<extra></extra>"
        ), row=1, col=1)

    # Bollinger
    if show_bollinger and "bb_upper" in df_ind.columns:
        fig.add_trace(go.Scatter(
            x=df_ind["date"], y=df_ind["bb_upper"],
            mode="lines", line=dict(color="rgba(150,150,150,0.3)", width=1),
            showlegend=False, name="BB haute"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_ind["date"], y=df_ind["bb_lower"],
            mode="lines", line=dict(color="rgba(150,150,150,0.3)", width=1),
            fill="tonexty", fillcolor="rgba(150,150,150,0.1)",
            name="Bandes de Bollinger"
        ), row=1, col=1)

    # SMA
    if show_sma:
        if "sma_20" in df_ind.columns:
            fig.add_trace(go.Scatter(
                x=df_ind["date"], y=df_ind["sma_20"],
                mode="lines", name="SMA 20",
                line=dict(color="#3b82f6", width=1.5, dash="dot")
            ), row=1, col=1)
        if "sma_50" in df_ind.columns:
            fig.add_trace(go.Scatter(
                x=df_ind["date"], y=df_ind["sma_50"],
                mode="lines", name="SMA 50",
                line=dict(color="#8b5cf6", width=1.5, dash="dash")
            ), row=1, col=1)

    # Signaux BUY/SELL
    if signals is not None and not signals.empty:
        merged = df_ind.merge(signals[["date", "signal"]], on="date", how="left")
        buys = merged[merged["signal"] == "BUY"]
        sells = merged[merged["signal"] == "SELL"]
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys["date"], y=buys["price"],
                mode="markers", name="Signal ACHAT",
                marker=dict(symbol="triangle-up", size=12, color="#10b981",
                          line=dict(color="white", width=1))
            ), row=1, col=1)
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells["date"], y=sells["price"],
                mode="markers", name="Signal VENTE",
                marker=dict(symbol="triangle-down", size=12, color="#ef4444",
                          line=dict(color="white", width=1))
            ), row=1, col=1)

    # --- Panneau RSI ---
    current_row = 2
    if show_rsi and "rsi_14" in df_ind.columns:
        fig.add_trace(go.Scatter(
            x=df_ind["date"], y=df_ind["rsi_14"],
            mode="lines", name="RSI 14",
            line=dict(color="#8b5cf6", width=1.5)
        ), row=current_row, col=1)
        # Zones surachat/survente
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=current_row, col=1)
        fig.update_yaxes(title_text="RSI", row=current_row, col=1, range=[0, 100])
        current_row += 1

    # --- Panneau MACD ---
    if show_macd and "macd" in df_ind.columns:
        fig.add_trace(go.Scatter(
            x=df_ind["date"], y=df_ind["macd"],
            mode="lines", name="MACD",
            line=dict(color="#3b82f6", width=1.5)
        ), row=current_row, col=1)
        fig.add_trace(go.Scatter(
            x=df_ind["date"], y=df_ind["macd_signal"],
            mode="lines", name="Signal",
            line=dict(color="#ef4444", width=1.5, dash="dash")
        ), row=current_row, col=1)
        # Histogramme
        colors = ["#10b981" if v > 0 else "#ef4444" for v in df_ind["macd_hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=df_ind["date"], y=df_ind["macd_hist"],
            name="Histogramme", marker_color=colors, opacity=0.5
        ), row=current_row, col=1)
        fig.update_yaxes(title_text="MACD", row=current_row, col=1)

    fig.update_layout(
        title=f"Évolution du prix — {asset}",
        template="plotly_white",
        height=600 if n_panels == 1 else 700 + 100 * (n_panels - 1),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(title_text="Prix ($/baril)", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=n_panels, col=1)

    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Courbes d'équité & drawdown
# ============================================================================

def render_equity_curve(equity_df: pd.DataFrame, benchmark_df: pd.DataFrame = None):
    """Courbe d'équité du portefeuille vs benchmark."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df["date"], y=equity_df["equity"],
        mode="lines", name="Stratégie",
        line=dict(color="#FF6B35", width=2.5),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Capital: $%{y:,.2f}<extra></extra>"
    ))
    if benchmark_df is not None and not benchmark_df.empty:
        fig.add_trace(go.Scatter(
            x=benchmark_df["date"], y=benchmark_df["equity"],
            mode="lines", name="Buy & Hold",
            line=dict(color="#6b7280", width=2, dash="dash")
        ))
    fig.update_layout(
        title="📈 Courbe d'équité du portefeuille",
        xaxis_title="Date", yaxis_title="Capital ($)",
        template="plotly_white", height=400,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)


def render_drawdown_chart(equity_df: pd.DataFrame):
    """Courbe de drawdown."""
    from utils.metrics import drawdown_series
    dd = drawdown_series(equity_df["equity"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df["date"], y=dd * 100,
        mode="lines", line=dict(color="#ef4444", width=1.5),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.2)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Drawdown: %{y:.2f}%<extra></extra>"
    ))
    fig.update_layout(
        title="📉 Drawdown",
        xaxis_title="Date", yaxis_title="Drawdown (%)",
        template="plotly_white", height=300, showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
