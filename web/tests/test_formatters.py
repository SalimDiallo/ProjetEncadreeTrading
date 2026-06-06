"""
test_formatters.py
==================
Tests des helpers de formatage dans utils/formatters.py.

Importance : 🟡 MOYENNE
Pas de logique métier critique, mais important pour l'affichage cohérent
dans le dashboard.
"""
import pytest

from utils.formatters import fmt_currency, fmt_percent, fmt_ratio, fmt_signal


# ============================================================================
# fmt_currency
# ============================================================================

class TestFmtCurrency:
    """Vérifie le formatage monétaire : 12345.6 → '$12,345.60'."""

    def test_basic_formatting(self):
        assert fmt_currency(1234.5) == "$1,234.50"

    def test_zero(self):
        assert fmt_currency(0) == "$0.00"

    def test_large_value_has_thousands_separator(self):
        result = fmt_currency(1_000_000)
        assert "," in result

    def test_custom_currency_symbol(self):
        assert fmt_currency(100, currency="€") == "€100.00"

    def test_none_returns_dash(self):
        """None → '—' (pas de crash)."""
        assert fmt_currency(None) == "—"

    def test_nan_returns_dash(self):
        """NaN → '—' (l'affichage reste propre)."""
        assert fmt_currency(float("nan")) == "—"


# ============================================================================
# fmt_percent
# ============================================================================

class TestFmtPercent:
    """Vérifie le formatage en pourcentage : 0.1234 → '12.34%'."""

    def test_basic_positive(self):
        assert fmt_percent(0.1234) == "12.34%"

    def test_negative_value(self):
        result = fmt_percent(-0.05)
        assert "-5.00%" in result or "-5%" in result

    def test_zero(self):
        assert fmt_percent(0.0) == "0.00%"

    def test_custom_decimals(self):
        assert fmt_percent(0.12345, decimals=4) == "12.3450%"
        assert fmt_percent(0.12345, decimals=0) == "12%"

    def test_none_returns_dash(self):
        assert fmt_percent(None) == "—"

    def test_nan_returns_dash(self):
        assert fmt_percent(float("nan")) == "—"


# ============================================================================
# fmt_ratio
# ============================================================================

class TestFmtRatio:
    """Vérifie le formatage des ratios : 1.456 → '1.46'."""

    def test_basic(self):
        assert fmt_ratio(1.456) == "1.46"

    def test_infinity_returns_symbol(self):
        """∞ pour les profit factors sans pertes."""
        assert fmt_ratio(float("inf")) == "∞"

    def test_zero(self):
        assert fmt_ratio(0.0) == "0.00"

    def test_negative(self):
        assert fmt_ratio(-1.5) == "-1.50"

    def test_none_returns_dash(self):
        assert fmt_ratio(None) == "—"

    def test_nan_returns_dash(self):
        assert fmt_ratio(float("nan")) == "—"


# ============================================================================
# fmt_signal
# ============================================================================

class TestFmtSignal:
    """Vérifie le mapping signal → (emoji, label FR, couleur)."""

    def test_buy_signal(self):
        emoji, label, color = fmt_signal("BUY")
        assert label == "ACHETER"
        assert color.startswith("#")

    def test_sell_signal(self):
        emoji, label, color = fmt_signal("SELL")
        assert label == "VENDRE"

    def test_hold_signal(self):
        emoji, label, color = fmt_signal("HOLD")
        assert label == "CONSERVER"

    def test_unknown_signal_returns_default(self):
        """Signal inconnu → fallback INDÉTERMINÉ (pas de crash)."""
        emoji, label, color = fmt_signal("UNKNOWN")
        assert label == "INDÉTERMINÉ"

    def test_case_insensitive(self):
        """'buy' minuscule doit fonctionner aussi."""
        _, label, _ = fmt_signal("buy")
        assert label == "ACHETER"

    def test_returns_three_elements(self):
        """La fonction retourne toujours un tuple à 3 éléments."""
        result = fmt_signal("BUY")
        assert len(result) == 3
