"""
normalizer.py
-------------
Gestion du vocabulaire financier spécialisé pétrole.

Fonctionnalités :
  - Dictionnaire de synonymes / abréviations → terme canonique
  - Expansion des acronymes financiers oil
  - Détection de négations (modifie le token suivant)
  - Score de densité oil (ratio mots-clés / total tokens)
"""

import logging
import re
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dictionnaire de normalisation — abréviations → forme canonique
# ---------------------------------------------------------------------------

SYNONYM_MAP: Dict[str, str] = {
    # Acronymes pétrole
    "wti":          "west_texas_intermediate",
    "wtio":         "west_texas_intermediate",
    "cl":           "crude_oil_futures",
    "bno":          "brent_oil",
    "uso":          "us_oil_fund",
    "lng":          "liquefied_natural_gas",
    "lpg":          "liquefied_petroleum_gas",
    "ngl":          "natural_gas_liquids",
    "boe":          "barrel_oil_equivalent",
    "bbl":          "barrel",
    "bbls":         "barrel",
    "mmbbl":        "million_barrels",
    "mbpd":         "thousand_barrels_per_day",
    "mmbpd":        "million_barrels_per_day",
    "capex":        "capital_expenditure",
    "opex":         "operational_expenditure",

    # Organisations
    "opec":         "opec_organization",
    "opecplus":     "opec_plus",
    "iea":          "international_energy_agency",
    "eia":          "energy_information_administration",
    "doe":          "department_of_energy",

    # Termes de marché
    "ath":          "all_time_high",
    "atl":          "all_time_low",
    "mo":          "month",
    "yoy":          "year_over_year",
    "qoq":          "quarter_over_quarter",
    "ytd":          "year_to_date",
    "eod":          "end_of_day",
    "eow":          "end_of_week",

    # Sentiment implicite
    "dd":           "drawdown",
    "rip":          "decline",
    "mooning":      "rising_sharply",
    "dumping":      "falling_sharply",
    "ripping":      "rising_sharply",
    "tanking":      "falling_sharply",
    "crashing":     "sharp_decline",
    "surging":      "sharp_increase",
    "pumping":      "rising",
    "spiking":      "sharp_increase",
    "slumping":     "decline",
    "plunging":     "sharp_decline",
    "soaring":      "sharp_increase",

    # Abréviations communes Twitter/Reddit
    "imo":          "in_my_opinion",
    "imho":         "in_my_humble_opinion",
    "afaik":        "as_far_as_i_know",
    "tbh":          "to_be_honest",
    "fwiw":         "for_what_its_worth",
    "fyi":          "for_your_information",
}

# ---------------------------------------------------------------------------
# Patterns de négation — "not bullish" → "not_bullish"
# ---------------------------------------------------------------------------

NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "without", "hardly", "barely", "scarcely"}

SENTIMENT_WORDS = {
    "bullish", "bearish", "positive", "negative", "good", "bad",
    "strong", "weak", "rising", "falling", "up", "down",
    "increase", "decrease", "growth", "decline", "rally", "selloff",
    "optimistic", "pessimistic", "confident", "worried", "concerned",
}

# ---------------------------------------------------------------------------
# Fonctions de normalisation
# ---------------------------------------------------------------------------

def expand_synonyms(tokens: List[str]) -> List[str]:
    """Remplace chaque token par sa forme canonique si présent dans SYNONYM_MAP."""
    return [SYNONYM_MAP.get(t, t) for t in tokens]


def handle_negations(tokens: List[str]) -> List[str]:
    """
    Détecte les mots de négation et les fusionne avec le mot de sentiment suivant.
    Ex: ["not", "bullish"] → ["not_bullish"]
    """
    result = []
    skip_next = False

    for i, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue

        if token in NEGATION_WORDS and i + 1 < len(tokens):
            next_token = tokens[i + 1]
            if next_token in SENTIMENT_WORDS:
                result.append(f"not_{next_token}")
                skip_next = True
                continue

        result.append(token)

    return result


def compute_oil_density(tokens: List[str]) -> float:
    """
    Calcule le ratio de tokens oil-related / total tokens.
    Utile pour filtrer les textes hors-sujet.

    Returns
    -------
    float : entre 0.0 (aucun mot oil) et 1.0 (tout oil)
    """
    if not tokens:
        return 0.0

    oil_terms = {
        "oil", "crude", "petroleum", "brent", "west_texas_intermediate",
        "opec", "opec_organization", "opec_plus", "barrel", "boe",
        "energy", "refinery", "drilling", "offshore", "pipeline",
        "lng", "lpg", "shale", "permian", "upstream", "downstream",
        "midstream", "gasoline", "diesel", "naphtha", "kerosene",
        "crack_spread", "rig", "wellbore", "fracking", "natural_gas",
    }

    count = sum(1 for t in tokens if any(oil_kw in t for oil_kw in oil_terms))
    return count / len(tokens)


def normalize_tokens(
    tokens: List[str],
    do_synonym_expansion: bool = True,
    do_negation_handling: bool = True,
) -> List[str]:
    """
    Applique la chaîne complète de normalisation financière.

    Parameters
    ----------
    tokens : List[str]
        Tokens déjà nettoyés (depuis text_cleaner.clean_text).
    do_synonym_expansion : bool
        Expansion des abréviations/synonymes.
    do_negation_handling : bool
        Fusion des négations avec les mots de sentiment.

    Returns
    -------
    List[str] : tokens normalisés
    """
    if do_synonym_expansion:
        tokens = expand_synonyms(tokens)

    if do_negation_handling:
        tokens = handle_negations(tokens)

    return tokens


def normalize_text(
    text: str,
    do_synonym_expansion: bool = True,
    do_negation_handling: bool = True,
) -> Tuple[str, float]:
    """
    Normalise un texte déjà nettoyé par text_cleaner.

    Returns
    -------
    Tuple[str, float] : (texte normalisé, score de densité oil)
    """
    tokens = text.split()
    tokens = normalize_tokens(tokens, do_synonym_expansion, do_negation_handling)
    density = compute_oil_density(tokens)
    return " ".join(tokens), density


def normalize_batch(
    texts: List[str],
    min_oil_density: float = 0.0,
) -> List[Dict]:
    """
    Normalise une liste de textes.
    Retourne une liste de dicts avec le texte normalisé et le score de densité.

    Parameters
    ----------
    texts : List[str]
        Textes pré-nettoyés.
    min_oil_density : float
        Filtre optionnel : exclut les textes avec densité oil < seuil.
    """
    results = []
    for text in texts:
        try:
            normalized, density = normalize_text(text)
            if density >= min_oil_density:
                results.append({"text_normalized": normalized, "oil_density": density})
            else:
                results.append({"text_normalized": normalized, "oil_density": density})
        except Exception as exc:
            logger.warning("Erreur normalisation : %s", exc)
            results.append({"text_normalized": text, "oil_density": 0.0})
    return results


# ---------------------------------------------------------------------------
# CLI rapide pour test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    samples = [
        "opec cut production wti brent not bullish outlook",
        "eia report draw bbls inventory not bearish surprise",
        "soaring crude oil prices mooning shale permian ripping",
        "opecplus discipline holds lng exports surging demand strong",
        "imho wti tanking recession fears bearish sentiment growing",
    ]

    print(f"{'INPUT':<55} | {'NORMALIZED':<55} | DENSITY")
    print("-" * 125)
    for text in samples:
        norm, density = normalize_text(text)
        print(f"{text[:53]:<55} | {norm[:53]:<55} | {density:.3f}")
