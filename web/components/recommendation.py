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
        "color": "#16a34a", "bg": "rgba(22,163,74,0.08)",
        "label": "ACHETER",
        "subtitle": "Conditions favorables identifiées",
    },
    "HOLD": {
        "color": "#d97706", "bg": "rgba(217,119,6,0.08)",
        "label": "CONSERVER",
        "subtitle": "Pas de signal franc — attendre",
    },
    "SELL": {
        "color": "#dc2626", "bg": "rgba(220,38,38,0.08)",
        "label": "VENDRE",
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
        Recommandation
    </div>
    <div style="font-size: 48px; font-weight: 800; color: {cfg['color']}; margin: 12px 0 4px 0; line-height: 1;">
        {cfg['label']}
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
        with st.expander("Pourquoi cette recommandation ?", expanded=True):
            for reason in reasons:
                st.markdown(f"- {reason}")

    st.caption(
        "Avertissement : outil éducatif, pas un conseil en investissement. "
        "Les performances passées ne préjugent pas des performances futures."
    )


def aggregate_signals(signals_df: pd.DataFrame) -> tuple[str, float]:
    """Calcule la recommandation à partir du dernier signal disponible."""
    if signals_df.empty:
        return "HOLD", 0.5
    last = signals_df.sort_values("date").iloc[-1]
    return last["signal"], float(last["confidence"])


def build_real_reasons(prices: pd.DataFrame, signals: pd.DataFrame,
                       sentiment: pd.DataFrame = None,
                       strategy: str = "") -> list[str]:
    """
    Construit les raisons RÉELLES de la recommandation à partir des données :
    RSI et MACD calculés sur les prix, dernier signal du modèle, sentiment NLP
    du jour. Remplace l'ancien texte factice (mock_recommendation_reasons).
    """
    from utils.indicators import compute_rsi, compute_macd

    reasons = []

    # --- RSI réel ---
    if not prices.empty and len(prices) > 14:
        rsi = compute_rsi(prices)["rsi_14"].iloc[-1]
        if pd.notna(rsi):
            if rsi >= 70:
                reasons.append(f"RSI à {rsi:.0f} — zone de surachat (signal baissier)")
            elif rsi <= 30:
                reasons.append(f"RSI à {rsi:.0f} — zone de survente (signal haussier)")
            else:
                reasons.append(f"RSI à {rsi:.0f} — zone neutre")

    # --- MACD réel ---
    if not prices.empty and len(prices) > 26:
        macd_df = compute_macd(prices)
        macd = macd_df["macd"].iloc[-1]
        macd_sig = macd_df["macd_signal"].iloc[-1]
        if pd.notna(macd) and pd.notna(macd_sig):
            if macd > macd_sig:
                reasons.append("MACD au-dessus de sa ligne de signal — momentum haussier")
            else:
                reasons.append("MACD sous sa ligne de signal — momentum baissier")

    # --- Signal du modèle ML/stratégie ---
    if not signals.empty:
        last = signals.sort_values("date").iloc[-1]
        model_name = last.get("model", strategy or "modèle")
        verb = {"BUY": "une hausse", "SELL": "une baisse"}.get(last["signal"], "un statu quo")
        reasons.append(
            f"{model_name} prédit {verb} à J+1 "
            f"(confiance {float(last['confidence'])*100:.0f}%)"
        )

    # --- Sentiment NLP réel du jour ---
    if sentiment is not None and not sentiment.empty:
        last_s = sentiment.sort_values("date").iloc[-1]
        score = float(last_s["sentiment_score"])
        n = int(last_s.get("n_articles", 0))
        tone = "positif" if score > 0.05 else "négatif" if score < -0.05 else "neutre"
        reasons.append(
            f"Sentiment NLP {tone} ({score:+.2f}) sur {n} article(s) récent(s)"
        )

    return reasons or ["Données insuffisantes pour détailler la recommandation."]
