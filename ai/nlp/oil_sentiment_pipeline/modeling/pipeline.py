"""
modeling/pipeline.py
--------------------
Orchestrateur du module modeling.

Stratégie d'inférence :
  1. FinBERT (si torch disponible) — modèle principal
  2. Logistic Regression (si données TF-IDF disponibles) — modèle secondaire
  3. Lexical baseline — toujours disponible, zéro dépendance

Chaque record en sortie est enrichi avec :
    {
        "sentiment_label":       str   ("positive" / "neutral" / "negative")
        "sentiment_score":       float  ([-1.0, +1.0])
        "sentiment_confidence":  float  ([0.0, 1.0])
        "model":                 str    (nom du modèle utilisé)
    }
"""

import logging
import os
from typing import List, Dict, Optional, Literal

logger = logging.getLogger(__name__)

ModelType = Literal["finbert", "logistic_regression", "lexical", "auto"]

from oil_sentiment_pipeline.paths import MODELS_DIR, PROCESSED_DIR, SENTIMENT_DIR

DEFAULT_MODEL_DIR = str(MODELS_DIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _torch_available() -> bool:
    try:
        import torch
        import transformers
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_sentiment_analysis(
    records: List[Dict],
    model: ModelType = "auto",
    text_field: str = "text_clean",
    tfidf_matrix=None,
    lr_model=None,
    finbert_batch_size: int = 8,
    save_results: bool = False,
    output_path: str = None,
) -> List[Dict]:
    """
    Applique l'analyse de sentiment sur une liste de records prétraités.

    Parameters
    ----------
    records : List[Dict]
        Records issus de preprocessing.pipeline.run_preprocessing().
    model : str
        "auto"               → FinBERT si dispo, sinon lexical
        "finbert"            → FinBERT uniquement (fallback mock si torch absent)
        "logistic_regression"→ LR sur TF-IDF (requiert tfidf_matrix + lr_model)
        "lexical"            → Dictionnaire de polarité (toujours dispo)
    text_field : str
        Champ texte à utiliser ("text", "text_clean", "text_normalized").
    tfidf_matrix : scipy.sparse, optionnel
        Matrice TF-IDF pré-calculée (pour le modèle LR).
    lr_model : LogisticRegressionSentiment, optionnel
        Modèle LR pré-entraîné.
    finbert_batch_size : int
        Taille des batches pour FinBERT.
    save_results : bool
        Sauvegarde les résultats en CSV.
    output_path : str
        Chemin CSV de sortie.

    Returns
    -------
    List[Dict] : records enrichis avec le sentiment.
    """
    if not records:
        logger.warning("Aucun record reçu pour l'analyse de sentiment.")
        return []

    logger.info("Analyse sentiment sur %d records (modèle: %s)...", len(records), model)

    # ------------------------------------------------------------------
    # Sélection du modèle
    # ------------------------------------------------------------------
    effective_model = _resolve_model(model, tfidf_matrix, lr_model)
    logger.info("Modèle effectif : %s", effective_model)

    results: List[Dict] = []

    # ------------------------------------------------------------------
    # FinBERT
    # ------------------------------------------------------------------
    if effective_model in ("finbert", "finbert_mock"):
        from oil_sentiment_pipeline.modeling.finbert_classifier import get_finbert_classifier
        clf = get_finbert_classifier(fallback_mock=True)
        results = clf.predict_records(records, text_field=text_field, batch_size=finbert_batch_size)

    # ------------------------------------------------------------------
    # Logistic Regression
    # ------------------------------------------------------------------
    elif effective_model == "logistic_regression":
        if tfidf_matrix is None or lr_model is None:
            logger.warning("LR requiert tfidf_matrix + lr_model — fallback lexical.")
            results = _run_lexical(records)
        else:
            try:
                preds = lr_model.predict_with_score(tfidf_matrix)
                results = []
                for rec, (label, score) in zip(records, preds):
                    results.append({
                        **rec,
                        "sentiment_label":      label,
                        "sentiment_score":      score,
                        "sentiment_confidence": abs(score),
                        "model":                "logistic_regression",
                    })
            except Exception as exc:
                logger.error("Erreur LR : %s — fallback lexical.", exc)
                results = _run_lexical(records)

    # ------------------------------------------------------------------
    # Lexical baseline
    # ------------------------------------------------------------------
    else:
        results = _run_lexical(records)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    _log_distribution(results)

    if save_results and results:
        _save_sentiment_csv(results, output_path)

    return results


def _resolve_model(
    model: str,
    tfidf_matrix,
    lr_model,
) -> str:
    """Détermine le modèle effectif à utiliser."""
    if model == "auto":
        if _torch_available():
            return "finbert"
        return "lexical"

    if model == "logistic_regression":
        if tfidf_matrix is not None and lr_model is not None:
            return "logistic_regression"
        logger.warning("LR demandé mais tfidf_matrix/lr_model absents — fallback lexical.")
        return "lexical"

    return model  # "finbert", "lexical"


def _run_lexical(records: List[Dict]) -> List[Dict]:
    from oil_sentiment_pipeline.modeling.baseline import predict_lexical
    return predict_lexical(records)


def _log_distribution(results: List[Dict]) -> None:
    labels = [r.get("sentiment_label", "unknown") for r in results]
    dist = {l: labels.count(l) for l in ["positive", "neutral", "negative"]}
    total = len(labels)
    logger.info(
        "Distribution sentiment : positive=%d (%.0f%%) | neutral=%d (%.0f%%) | negative=%d (%.0f%%)",
        dist.get("positive", 0), dist.get("positive", 0) / max(total, 1) * 100,
        dist.get("neutral",  0), dist.get("neutral",  0) / max(total, 1) * 100,
        dist.get("negative", 0), dist.get("negative", 0) / max(total, 1) * 100,
    )


# ---------------------------------------------------------------------------
# Sauvegarde CSV
# ---------------------------------------------------------------------------

def _save_sentiment_csv(results: List[Dict], path: str = None) -> None:
    import csv
    from datetime import datetime, timezone

    if path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        os.makedirs(SENTIMENT_DIR, exist_ok=True)
        path = str(SENTIMENT_DIR / f"sentiment_{ts}.csv")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    fieldnames = ["date", "source", "sentiment_label", "sentiment_score",
                  "sentiment_confidence", "model", "text_clean"]

    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        logger.info("Résultats sentiment sauvegardés : %s (%d lignes)", path, len(results))
    except Exception as exc:
        logger.error("Erreur sauvegarde sentiment CSV : %s", exc)


# ---------------------------------------------------------------------------
# Entraînement supervisé du modèle LR (bootstrap complet)
# ---------------------------------------------------------------------------

def train_lr_model(
    records: List[Dict],
    tfidf_matrix=None,
    auto_label: bool = True,
    save_model: bool = True,
) -> tuple:
    """
    Entraîne la Logistic Regression sur des records prétraités.

    Si auto_label=True, génère les labels via le dictionnaire lexical (bootstrap).
    Sinon, les records doivent contenir un champ 'sentiment_label'.

    Returns
    -------
    Tuple[LogisticRegressionSentiment, List[str]] : (modèle entraîné, labels)
    """
    from oil_sentiment_pipeline.modeling.baseline import (
        LogisticRegressionSentiment, auto_label_records
    )
    from oil_sentiment_pipeline.feature_engineering.tfidf_vectorizer import build_tfidf_matrix

    if auto_label:
        labeled_records, labels = auto_label_records(records)
    else:
        labeled_records = records
        labels = [r.get("sentiment_label", "neutral") for r in records]

    if tfidf_matrix is None:
        texts = [r.get("text_normalized") or r.get("text_clean", "") for r in labeled_records]
        tfidf_matrix, vectorizer = build_tfidf_matrix(texts)
    else:
        vectorizer = None

    clf = LogisticRegressionSentiment()
    clf.fit(tfidf_matrix, labels)

    if save_model:
        clf.save()

    return clf, labels, vectorizer


# ---------------------------------------------------------------------------
# CLI rapide pour test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import glob, csv as _csv
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    csv_files = sorted(glob.glob(str(PROCESSED_DIR / "processed_*.csv")))

    if csv_files:
        with open(csv_files[-1], encoding="utf-8") as f:
            records = [dict(r) for r in _csv.DictReader(f) if r.get("text_clean")]
        logger.info("Records chargés : %d depuis %s", len(records), csv_files[-1])
    else:
        logger.warning("Aucun fichier processed trouvé — utilisation des records mock.")
        records = [
            {"text_clean": "opec cut production barrel crude bullish rally",     "tokens": ["opec", "cut", "production", "barrel", "crude", "bullish", "rally"],       "date": "2024-03-15T08:00:00Z", "source": "twitter"},
            {"text_clean": "wti crude drop weak demand bearish inventory build", "tokens": ["wti", "crude", "drop", "weak", "demand", "bearish", "inventory", "build"], "date": "2024-03-14T14:00:00Z", "source": "reddit"},
            {"text_clean": "brent crude stable market balanced supply demand",   "tokens": ["brent", "crude", "stable", "market", "balanced", "supply", "demand"],      "date": "2024-03-13T10:00:00Z", "source": "reuters"},
            {"text_clean": "goldman raises brent forecast bullish energy sector","tokens": ["goldman", "raise", "brent", "forecast", "bullish", "energy", "sector"],    "date": "2024-03-12T09:00:00Z", "source": "bloomberg"},
            {"text_clean": "china demand recovery slow crude bearish outlook",   "tokens": ["china", "demand", "recovery", "slow", "crude", "bearish", "outlook"],      "date": "2024-03-11T11:00:00Z", "source": "yahoo"},
        ]

    model_type = "finbert"
    print(f"\n{'='*70}")
    print(f"MODÈLE : {model_type.upper()} — {len(records)} records")
    print(f"{'='*70}")
    results = run_sentiment_analysis(records, model=model_type)
    _save_sentiment_csv(results)
    for r in results[:10]:
        print(
            f"[{r.get('sentiment_label','?'):8s}] "
            f"score={r.get('sentiment_score', 0):+.4f} | "
            f"conf={r.get('sentiment_confidence', 0):.4f} | "
            f"{r.get('text_clean','')[:60]}"
        )
    if len(results) > 10:
        print(f"\n... ({len(results) - 10} records supplémentaires sauvegardés dans data/sentiment/)")
