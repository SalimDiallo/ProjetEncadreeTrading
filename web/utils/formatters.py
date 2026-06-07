"""
formatters.py
=============
Helpers de formatage pour l'affichage dans le dashboard.
"""


def fmt_currency(value: float, currency: str = "$") -> str:
    """Formate une valeur monétaire : 12345.6 -> '$12,345.60'."""
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    return f"{currency}{value:,.2f}"


def fmt_percent(value: float, decimals: int = 2) -> str:
    """Formate un pourcentage : 0.1234 -> '12.34%'."""
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    return f"{value * 100:.{decimals}f}%"


def fmt_ratio(value: float, decimals: int = 2) -> str:
    """Formate un ratio : 1.456 -> '1.46'."""
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    if value == float("inf"):
        return "∞"
    return f"{value:.{decimals}f}"


def fmt_signal(signal: str) -> tuple[str, str, str]:
    """Retourne (emoji, label_fr, couleur_hex) pour un signal."""
    mapping = {
        "BUY": ("▲", "ACHETER", "#0d9488"),
        "SELL": ("▼", "VENDRE", "#dc2626"),
        "HOLD": ("●", "CONSERVER", "#d97706"),
    }
    return mapping.get(signal.upper(), ("○", "INDÉTERMINÉ", "#6b7280"))
