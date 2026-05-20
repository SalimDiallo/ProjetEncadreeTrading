"""
text_cleaner.py
---------------
Nettoyage et normalisation du texte brut collecté depuis toutes les sources.

Opérations appliquées (dans l'ordre) :
  1. Lowercasing
  2. Suppression URLs
  3. Suppression mentions (@user) et cashtags ($OIL, $XOM...)
  4. Suppression hashtags (conserve le mot : #OOTT → oott)
  5. Suppression caractères spéciaux / ponctuation excessive
  6. Suppression des nombres isolés
  7. Suppression des stopwords anglais (NLTK)
  8. Lemmatisation (WordNetLemmatizer)
  9. Suppression tokens trop courts (< 2 chars)
 10. Normalisation des espaces
"""

import logging
import re
import string
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Initialisation NLTK (téléchargement des ressources si absentes)
# ---------------------------------------------------------------------------

def _init_nltk() -> tuple:
    """Charge les ressources NLTK nécessaires. Retourne (stopwords_set, lemmatizer)."""
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        for resource in ["stopwords", "wordnet", "omw-1.4", "punkt"]:
            try:
                nltk.data.find(f"corpora/{resource}" if resource != "punkt" else f"tokenizers/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)

        sw = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        return sw, lemmatizer

    except ImportError:
        logger.warning("NLTK non disponible — stopwords et lemmatisation désactivés.")
        return set(), None


_STOPWORDS, _LEMMATIZER = _init_nltk()

# ---------------------------------------------------------------------------
# Patterns regex compilés
# ---------------------------------------------------------------------------

_URL_PATTERN        = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_PATTERN    = re.compile(r"@\w+")
_CASHTAG_PATTERN    = re.compile(r"\$[A-Z]{1,6}\b")
_HASHTAG_PATTERN    = re.compile(r"#(\w+)")          # garde le mot sans #
_HTML_ENTITY_PATTERN = re.compile(r"&[a-z]+;|&#\d+;", re.IGNORECASE)
_SPECIAL_CHARS      = re.compile(r"[^a-z0-9\s\'\-]") # garde lettres, chiffres, apostrophes, tirets
_MULTI_SPACE        = re.compile(r"\s+")
_ISOLATED_NUMBERS   = re.compile(r"\b\d+(\.\d+)?\b")
_REPEATED_CHARS     = re.compile(r"(.)\1{3,}")        # "ooooil" → "oil"

# ---------------------------------------------------------------------------
# Vocabulaire financier oil — conservation (ne pas supprimer comme stopwords)
# ---------------------------------------------------------------------------

OIL_FINANCE_VOCAB = {
    # Termes techniques pétrole
    "wti", "brent", "crude", "opec", "barrel", "bbls", "boe",
    "upstream", "downstream", "midstream", "refinery", "drilling",
    "offshore", "onshore", "pipeline", "lng", "lpg", "ngl",
    "shale", "permian", "fracking", "rig", "wellbore",
    # Termes financiers
    "bullish", "bearish", "long", "short", "rally", "selloff",
    "breakout", "support", "resistance", "momentum", "volatility",
    "hedge", "futures", "options", "spread", "crack", "swap",
    "earnings", "revenue", "capex", "guidance", "outlook",
    # Acteurs marché
    "saudi", "russia", "iran", "iraq", "uae", "venezuela",
    "aramco", "exxon", "chevron", "bp", "shell", "total",
    "halliburton", "schlumberger", "conoco", "occidental",
    # Indicateurs macro
    "gdp", "inflation", "recession", "fed", "fomc", "eia",
    "iea", "opecplus", "inventory", "stockpile", "drawdown",
    "production", "supply", "demand", "export", "import",
}

# Stopwords à conserver car pertinents pour le sentiment en contexte financier
KEEP_FROM_STOPWORDS = {
    "not", "no", "nor", "never", "neither", "without",
    "above", "below", "up", "down", "over", "under",
    "more", "less", "than", "too", "very", "most", "least",
}


# ---------------------------------------------------------------------------
# Fonctions de nettoyage atomiques
# ---------------------------------------------------------------------------

def remove_urls(text: str) -> str:
    return _URL_PATTERN.sub(" ", text)


def remove_mentions(text: str) -> str:
    return _MENTION_PATTERN.sub(" ", text)


def remove_cashtags(text: str) -> str:
    """Supprime les cashtags ($XOM, $CL=F...). Mono-actif pétrole → on retire tout."""
    return _CASHTAG_PATTERN.sub(" ", text)


def normalize_hashtags(text: str) -> str:
    """#OOTT → oott (conserve le terme sans le #)."""
    return _HASHTAG_PATTERN.sub(lambda m: m.group(1).lower(), text)


def remove_html_entities(text: str) -> str:
    return _HTML_ENTITY_PATTERN.sub(" ", text)


def remove_special_chars(text: str) -> str:
    return _SPECIAL_CHARS.sub(" ", text)


def remove_isolated_numbers(text: str) -> str:
    return _ISOLATED_NUMBERS.sub(" ", text)


def fix_repeated_chars(text: str) -> str:
    """'sooooo bullish' → 'so bullish'."""
    return _REPEATED_CHARS.sub(r"\1\1", text)


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Retire les stopwords NLTK sauf ceux utiles au sentiment financier."""
    if not _STOPWORDS:
        return tokens
    keep = _STOPWORDS - KEEP_FROM_STOPWORDS
    return [t for t in tokens if t not in keep or t in OIL_FINANCE_VOCAB]


def lemmatize(tokens: List[str]) -> List[str]:
    """Lemmatise chaque token (nécessite WordNet)."""
    if _LEMMATIZER is None:
        return tokens
    return [_LEMMATIZER.lemmatize(t) for t in tokens]


def tokenize(text: str) -> List[str]:
    """Tokenisation simple par espaces après nettoyage."""
    return [t for t in text.split() if len(t) >= 2]


# ---------------------------------------------------------------------------
# Pipeline de nettoyage principal
# ---------------------------------------------------------------------------

def clean_text(
    text: str,
    remove_stops: bool = True,
    do_lemmatize: bool = True,
    remove_numbers: bool = True,
    return_tokens: bool = False,
) -> str | List[str]:
    """
    Nettoie un texte brut selon le pipeline complet.

    Parameters
    ----------
    text : str
        Texte brut à nettoyer.
    remove_stops : bool
        Appliquer la suppression des stopwords.
    do_lemmatize : bool
        Appliquer la lemmatisation.
    remove_numbers : bool
        Supprimer les nombres isolés.
    return_tokens : bool
        Si True, retourne une liste de tokens plutôt qu'une chaîne.

    Returns
    -------
    str ou List[str]
    """
    if not text or not isinstance(text, str):
        return [] if return_tokens else ""

    # 1. Lowercase
    text = text.lower()

    # 2. Entités HTML
    text = remove_html_entities(text)

    # 3. URLs
    text = remove_urls(text)

    # 4. Mentions
    text = remove_mentions(text)

    # 5. Cashtags
    text = remove_cashtags(text)

    # 6. Hashtags (normalisation)
    text = normalize_hashtags(text)

    # 7. Répétitions de caractères
    text = fix_repeated_chars(text)

    # 8. Caractères spéciaux
    text = remove_special_chars(text)

    # 9. Nombres isolés
    if remove_numbers:
        text = remove_isolated_numbers(text)

    # 10. Normalisation espaces
    text = _MULTI_SPACE.sub(" ", text).strip()

    # 11. Tokenisation
    tokens = tokenize(text)

    # 12. Stopwords
    if remove_stops:
        tokens = remove_stopwords(tokens)

    # 13. Lemmatisation
    if do_lemmatize:
        tokens = lemmatize(tokens)

    # 14. Filtre tokens trop courts
    tokens = [t for t in tokens if len(t) >= 2]

    if return_tokens:
        return tokens

    return " ".join(tokens)


def clean_batch(
    texts: List[str],
    remove_stops: bool = True,
    do_lemmatize: bool = True,
    remove_numbers: bool = True,
) -> List[str]:
    """Nettoie une liste de textes. Retourne une liste de textes nettoyés."""
    cleaned = []
    for text in texts:
        try:
            result = clean_text(
                text,
                remove_stops=remove_stops,
                do_lemmatize=do_lemmatize,
                remove_numbers=remove_numbers,
                return_tokens=False,
            )
            cleaned.append(result)
        except Exception as exc:
            logger.warning("Erreur nettoyage texte : %s — texte remplacé par chaîne vide.", exc)
            cleaned.append("")
    return cleaned


# ---------------------------------------------------------------------------
# CLI rapide pour test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    samples = [
        "BREAKING: OPEC+ agrees to cut production by 1M bbl/day! #OOTT #oil @Reuters https://t.co/xyz123",
        "WTI crude drops below $75 on weak US jobs data... Energy sector selloff $XOM $CVX",
        "Brent crude up 2.5%!!! Red Sea shipping disruptions worsen 🚢⬆️ supply risk premium returns",
        "   oil oil oil  123.45  @mention #crudeoil www.oilprice.com &amp; check this out!!!   ",
        "Goldman Sachs raises Brent forecast to $95/barrel by Q3 2024 — bullish on energy sector",
    ]

    print(f"{'RAW':<80} | {'CLEANED'}")
    print("-" * 160)
    for raw in samples:
        cleaned = clean_text(raw)
        print(f"{raw[:78]:<80} | {cleaned[:78]}")
