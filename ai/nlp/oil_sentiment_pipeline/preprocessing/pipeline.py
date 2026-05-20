"""
pipeline.py
-----------
Orchestrateur du module preprocessing.

Prend en entrée une liste de records bruts (format standard data_ingestion) :
    [{"text": str, "date": str, "source": str}, ...]

Retourne une liste enrichie :
    [{
        "text":             str  (original),
        "text_clean":       str  (nettoyé),
        "text_normalized":  str  (normalisé),
        "tokens":           List[str],
        "oil_density":      float,
        "date":             str,
        "source":           str,
    }, ...]

Peut aussi charger/sauvegarder depuis/vers CSV.
"""

import csv
import logging
import os
from typing import List, Dict, Optional

from oil_sentiment_pipeline.preprocessing.text_cleaner import clean_text, clean_batch
from oil_sentiment_pipeline.preprocessing.normalizer import normalize_text, compute_oil_density

logger = logging.getLogger(__name__)

from oil_sentiment_pipeline.paths import PROCESSED_DIR, RAW_DIR

DEFAULT_OUTPUT_DIR = str(PROCESSED_DIR)

# ---------------------------------------------------------------------------
# Traitement d'un record unique
# ---------------------------------------------------------------------------

def preprocess_record(
    record: Dict,
    remove_stops: bool = True,
    do_lemmatize: bool = True,
    do_normalize: bool = True,
    min_tokens: int = 3,
) -> Optional[Dict]:
    """
    Applique le pipeline complet à un record data_ingestion.
   
    Retourne None si le texte nettoyé est trop court (< min_tokens tokens).
    """
    raw_text = record.get("text", "")

    if not raw_text or not raw_text.strip():
        return None

    # 1. Nettoyage
    text_clean = clean_text(
        raw_text,
        remove_stops=remove_stops,
        do_lemmatize=do_lemmatize,
        remove_numbers=True,
        return_tokens=False,
    )

    tokens = text_clean.split() if text_clean else []

    if len(tokens) < min_tokens:
        logger.debug("Texte trop court après nettoyage (%d tokens) — ignoré.", len(tokens))
        return None

    # 2. Normalisation financière
    text_normalized = text_clean
    oil_density = 0.0

    if do_normalize:
        text_normalized, oil_density = normalize_text(text_clean)
    else:
        oil_density = compute_oil_density(tokens)

    return {
        "text":             raw_text,
        "text_clean":       text_clean,
        "text_normalized":  text_normalized,
        "tokens":           tokens,
        "oil_density":      round(oil_density, 4),
        "date":             record.get("date", ""),
        "source":           record.get("source", ""),
    }


# ---------------------------------------------------------------------------
# Traitement d'un batch
# ---------------------------------------------------------------------------

def preprocess_batch(
    records: List[Dict],
    remove_stops: bool = True,
    do_lemmatize: bool = True,
    do_normalize: bool = True,
    min_tokens: int = 3,
    min_oil_density: float = 0.0,
) -> List[Dict]:
    """
    Applique le pipeline complet à une liste de records.

    Parameters
    ----------
    records : List[Dict]
        Records au format standard data_ingestion.
    remove_stops : bool
        Activer la suppression des stopwords.
    do_lemmatize : bool
        Activer la lemmatisation.
    do_normalize : bool
        Activer la normalisation financière (synonymes, négations).
    min_tokens : int
        Nombre minimum de tokens après nettoyage (filtre les textes vides).
    min_oil_density : float
        Score minimum de densité oil pour conserver un record (0.0 = tout garder).

    Returns
    -------
    List[Dict] : records prétraités et filtrés.
    """
    processed = []
    skipped_empty = 0
    skipped_density = 0

    for record in records:
        try:
            result = preprocess_record(
                record,
                remove_stops=remove_stops,
                do_lemmatize=do_lemmatize,
                do_normalize=do_normalize,
                min_tokens=min_tokens,
            )

            if result is None:
                skipped_empty += 1
                continue

            if result["oil_density"] < min_oil_density:
                skipped_density += 1
                continue

            processed.append(result)

        except Exception as exc:
            logger.warning("Erreur preprocessing record : %s", exc)

    logger.info(
        "Preprocessing : %d/%d records conservés (ignorés: %d vides, %d hors-sujet).",
        len(processed),
        len(records),
        skipped_empty,
        skipped_density,
    )
    return processed


# ---------------------------------------------------------------------------
# I/O CSV
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_raw_csv(filepath: str) -> List[Dict]:
    """Charge un CSV brut (colonnes: text, date, source)."""
    records = []
    try:
        with open(filepath, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    "text":   row.get("text", ""),
                    "date":   row.get("date", ""),
                    "source": row.get("source", ""),
                })
        logger.info("CSV chargé : %s (%d records)", filepath, len(records))
    except FileNotFoundError:
        logger.error("Fichier non trouvé : %s", filepath)
    except Exception as exc:
        logger.error("Erreur chargement CSV %s : %s", filepath, exc)
    return records


def save_processed_csv(records: List[Dict], filepath: str) -> None:
    """Sauvegarde les records prétraités en CSV."""
    _ensure_dir(os.path.dirname(filepath))
    fieldnames = ["date", "source", "oil_density", "text_clean", "text_normalized", "text"]

    try:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        logger.info("CSV prétraité sauvegardé : %s (%d lignes)", filepath, len(records))
    except Exception as exc:
        logger.error("Erreur sauvegarde CSV %s : %s", filepath, exc)


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def run_preprocessing(
    records: List[Dict] = None,
    input_csv: str = None,
    save_csv: bool = True,
    output_path: str = None,
    remove_stops: bool = True,
    do_lemmatize: bool = True,
    do_normalize: bool = True,
    min_tokens: int = 3,
    min_oil_density: float = 0.0,
) -> List[Dict]:
    """
    Pipeline de preprocessing complet.

    Accepte soit une liste de records Python, soit un chemin vers un CSV brut.

    Returns
    -------
    List[Dict] : records prétraités.
    """
    if records is None and input_csv:
        records = load_raw_csv(input_csv)

    if not records:
        logger.warning("Aucun record à prétraiter.")
        return []

    logger.info("Démarrage preprocessing sur %d records...", len(records))

    processed = preprocess_batch(
        records,
        remove_stops=remove_stops,
        do_lemmatize=do_lemmatize,
        do_normalize=do_normalize,
        min_tokens=min_tokens,
        min_oil_density=min_oil_density,
    )

    if save_csv and processed:
        if output_path is None:
            from datetime import datetime, timezone
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(DEFAULT_OUTPUT_DIR, f"processed_{ts}.csv")
        save_processed_csv(processed, output_path)

    return processed


# ---------------------------------------------------------------------------
# CLI rapide pour test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import glob
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Cherche le dernier all_sources_*.csv dans data/raw/
    csv_files = sorted(glob.glob(str(RAW_DIR / "all_sources_*.csv")))

    if csv_files:
        input_csv = csv_files[-1]  # le plus récent
        logger.info("Fichier source : %s", input_csv)
        results = run_preprocessing(input_csv=input_csv, save_csv=True)
    else:
        logger.warning("Aucun fichier all_sources_*.csv trouvé dans data/raw/ — utilisation des données mock.")
        mock_records = [
            {"text": "BREAKING: OPEC+ agrees to cut production by 1M bbl/day! #OOTT #oil @Reuters https://t.co/xyz", "date": "2024-03-15T08:00:00Z", "source": "twitter_mock"},
            {"text": "WTI crude drops below $75 on weak US jobs data... $XOM $CVX bearish energy sector", "date": "2024-03-14T14:30:00Z", "source": "reddit_mock"},
            {"text": "Goldman Sachs raises Brent forecast to $95/barrel by Q3 2024 bullish energy", "date": "2024-03-13T10:00:00Z", "source": "yahoo_finance"},
            {"text": "OPEC discipline holds. Saudi Arabia confirms no production increase. Bullish crude.", "date": "2024-03-11T11:45:00Z", "source": "reuters"},
        ]
        results = run_preprocessing(records=mock_records, save_csv=True)

    print(f"\n{'='*70}")
    print(f"RÉSULTATS PREPROCESSING ({len(results)} records)")
    print(f"{'='*70}")
    for r in results[:10]:  # affiche les 10 premiers
        print(f"\n[{r['source']}] density={r['oil_density']:.3f}")
        print(f"  RAW    : {r['text'][:80]}")
        print(f"  CLEAN  : {r['text_clean'][:80]}")
        print(f"  NORM   : {r['text_normalized'][:80]}")
        print(f"  TOKENS : {r['tokens'][:10]}")
    if len(results) > 10:
        print(f"\n... ({len(results) - 10} records supplémentaires sauvegardés dans data/processed/)")
