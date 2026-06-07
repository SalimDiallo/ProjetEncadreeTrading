"""
trade_log.py
============
Journal interactif des trades exécutés pendant le backtest.
"""
import streamlit as st
import pandas as pd


def render_trade_log(trades: pd.DataFrame):
    """Affiche le tableau des trades fermés."""
    st.subheader("📋 Journal des trades")

    if trades.empty:
        st.info("Aucun trade exécuté sur la période sélectionnée.")
        return

    # Formatage colonnes
    df_display = trades.copy()
    if "date_entry" in df_display.columns:
        df_display["date_entry"] = pd.to_datetime(df_display["date_entry"]).dt.strftime("%Y-%m-%d")
    if "date_exit" in df_display.columns:
        df_display["date_exit"] = pd.to_datetime(df_display["date_exit"]).dt.strftime("%Y-%m-%d")

    def color_pnl(val):
        if pd.isna(val):
            return ""
        color = "#10b981" if val > 0 else "#ef4444"
        return f"color: {color}; font-weight: 600;"

    styled = df_display.style.format({
        "price_entry": "${:.2f}",
        "price_exit": "${:.2f}",
        "pnl": lambda x: f"${x:.2f}" if pd.notna(x) else "—",
        "return_pct": lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "—",
    })
    if "pnl" in df_display.columns:
        styled = styled.map(color_pnl, subset=["pnl"])
    if "return_pct" in df_display.columns:
        styled = styled.map(color_pnl, subset=["return_pct"])

    st.dataframe(styled, use_container_width=True, height=350, hide_index=True)

    # Récap
    if "pnl" in trades.columns:
        total_pnl = trades["pnl"].dropna().sum()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("P&L total réalisé", f"${total_pnl:,.2f}")
        with col2:
            st.metric("Nombre total de trades", len(trades))
        with col3:
            winning = (trades["pnl"] > 0).sum() if "pnl" in trades.columns else 0
            st.metric("Trades gagnants", f"{winning} / {len(trades)}")
