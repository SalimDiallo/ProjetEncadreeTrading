"""
recommendation.py
=================
Composant central : la carte de recommandation finale
ACHETER / CONSERVER / VENDRE en gros et coloré.
"""
import streamlit as st
import pandas as pd


SIGNAL_CONFIG = {
    "BUY": {
        "color": "#10b981", "bg": "rgba(16,185,129,0.1)",
        "emoji": "🟢", "label": "ACHETER",
        "subtitle": "Conditions favorables identifiées",
    },
    "HOLD": {
        "color": "#f59e0b", "bg": "rgba(245,158,11,0.1)",
        "emoji": "🟡", "label": "CONSERVER",
        "subtitle": "Pas de signal franc — attendre",
    },
    "SELL": {
        "color": "#ef4444", "bg": "rgba(239,68,68,0.1)",
        "emoji": "🔴", "label": "VENDRE",
        "subtitle": "Risque baissier détecté",
    },
}


def render_recommendation(signal: str, confidence: float,
                          reasons: list[str] = None,
                          asset: str = "WTI",
                          current_price: float = None):
    cfg = SIGNAL_CONFIG.get(signal, SIGNAL_CONFIG["HOLD"])

    price_html = ""
    if current_price is not None:
        price_html = f'<div style="font-size: 14px; color: #6b7280; margin-top: 8px;">Prix actuel {asset} : <strong>${current_price:.2f}</strong>/baril</div>'

    st.markdown(f"""
<div style="
    background: {cfg['bg']};
    border-left: 8px solid {cfg['color']};
    padding: 28px;
    border-radius: 12px;
    margin: 16px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
">
    <div style="font-size: 13px; color: #6b7280; text-transform: uppercase; letter-spacing: 1px;">
        🎯 Recommandation
    </div>
    <div style="font-size: 48px; font-weight: 800; color: {cfg['color']}; margin: 12px 0 4px 0; line-height: 1;">
        {cfg['emoji']} {cfg['label']}
    </div>
    <div style="font-size: 15px; color: #6b7280; margin-bottom: 16px;">
        {cfg['subtitle']}
    </div>
    <div>
        <span style="
            background: {cfg['color']};
            color: white;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
        ">
            Confiance : {confidence*100:.1f}%
        </span>
    </div>
    {price_html}
</div>
    """, unsafe_allow_html=True)

    if reasons:
        with st.expander("📋 Pourquoi cette recommandation ?", expanded=True):
            for reason in reasons:
                st.markdown(f"- {reason}")

    st.caption(
        "⚠️ **Avertissement** : cet outil est à vocation éducative et ne constitue "
        "pas un conseil en investissement. Les performances passées ne préjugent "
        "pas des performances futures."
    )


def aggregate_signals(signals_df: pd.DataFrame) -> tuple[str, float]:
    """Calcule la recommandation à partir du dernier signal disponible."""
    if signals_df.empty:
        return "HOLD", 0.5
    last = signals_df.sort_values("date").iloc[-1]
    return last["signal"], float(last["confidence"])
