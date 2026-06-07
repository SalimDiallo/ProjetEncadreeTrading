"""
main.py
-------
Pipeline complet d'analyse de sentiment pétrole + backtesting.

Flux d'exécution :
  1. Collecte des données (news, Reddit, Twitter, EDGAR)
  2. Preprocessing (nettoyage + normalisation)
  3. Feature engineering (TF-IDF + embeddings)
  4. Modélisation sentiment (FinBERT ou Logistic Regression)
  5. Agrégation journalière
  6. Génération des signaux de trading
  7. Backtesting + métriques + graphiques
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from typing import List, Optional

from oil_sentiment_pipeline.paths import LOGS_DIR, ensure_data_dirs

# ---------------------------------------------------------------------------
# Configuration du logging
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    ensure_data_dirs()
    log_file = LOGS_DIR / f"pipeline_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s",
        handlers=handlers,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_pipeline(
    sources:          List[str]  = None,
    model:            str        = "auto",
    max_per_source:   int        = 30,
    start_date:       str        = "2023-01-01",
    end_date:         str        = "2024-12-31",
    save_data:        bool       = True,
    allow_mock:       bool       = True,
) -> dict:
    """
    Exécute le pipeline complet du début à la fin.

    Returns
    -------
    dict avec toutes les données intermédiaires et résultats.
    """
    if sources is None:
        sources = ["news", "reddit", "twitter", "edgar"]

    ensure_data_dirs()
    results = {}

    logger.info("=" * 70)
    logger.info("OIL SENTIMENT PIPELINE — DÉMARRAGE")
    logger.info(
        "Sources: %s | Modèle: %s",
        sources,
        model,
    )
    logger.info("Période backtest: %s → %s", start_date, end_date)
    logger.info("=" * 70)

    # ──────────────────────────────────────────────────────────────────────
    # ÉTAPE 1 — Collecte des données
    # ──────────────────────────────────────────────────────────────────────
    logger.info("\n[1/6] COLLECTE DES DONNÉES...")
    try:
        from oil_sentiment_pipeline.data_ingestion.collector import collect_all
        raw_records = collect_all(
            max_per_source=max_per_source,
            save_csv=save_data,
            sources=sources,
            start_date=start_date,
            end_date=end_date,
            allow_mock=allow_mock,
        )
        results["raw_records"] = raw_records
        logger.info("→ %d records bruts collectés.", len(raw_records))
    except Exception as exc:
        logger.error("Erreur collecte : %s", exc)
        return {"error": str(exc)}

    if not raw_records:
        logger.error("Aucun record collecté — arrêt du pipeline.")
        return {"error": "Aucune donnée collectée"}

    # ──────────────────────────────────────────────────────────────────────
    # ÉTAPE 2 — Preprocessing
    # ──────────────────────────────────────────────────────────────────────
    logger.info("\n[2/6] PREPROCESSING...")
    try:
        from oil_sentiment_pipeline.preprocessing.pipeline import run_preprocessing
        processed_records = run_preprocessing(
            records=raw_records,
            save_csv=save_data,
            min_tokens=3,
            min_oil_density=0.0,
        )
        results["processed_records"] = processed_records
        logger.info("→ %d records après preprocessing.", len(processed_records))
    except Exception as exc:
        logger.error("Erreur preprocessing : %s", exc)
        processed_records = raw_records  # fallback sur brut

    # ──────────────────────────────────────────────────────────────────────
    # ÉTAPE 3 — Feature Engineering
    # ──────────────────────────────────────────────────────────────────────
    logger.info("\n[3/6] FEATURE ENGINEERING...")
    features = {}
    tfidf_matrix = None
    vectorizer   = None

    try:
        from oil_sentiment_pipeline.feature_engineering.pipeline import build_features
        features = build_features(
            processed_records,
            use_tfidf=True,
            use_embeddings=(model in ("finbert", "auto")),
            embedding_model="finbert",
            save_artifacts=save_data,
        )
        tfidf_matrix = features.get("tfidf_matrix")
        vectorizer   = features.get("vectorizer")
        results["features"] = {
            "tfidf_shape":     tfidf_matrix.shape if tfidf_matrix is not None else None,
            "embeddings_shape": features["embeddings"].shape if features.get("embeddings") is not None else None,
        }
        logger.info("→ TF-IDF: %s | Embeddings: %s",
                    features["tfidf_matrix"].shape if tfidf_matrix is not None else None,
                    features["embeddings"].shape if features.get("embeddings") is not None else None)
    except Exception as exc:
        logger.error("Erreur feature engineering : %s", exc)

    # ──────────────────────────────────────────────────────────────────────
    # ÉTAPE 4 — Modélisation Sentiment
    # ──────────────────────────────────────────────────────────────────────
    logger.info("\n[4/6] ANALYSE DE SENTIMENT (modèle: %s)...", model)
    lr_model = None

    try:
        # Entraînement LR si demandé et TF-IDF disponible
        if model == "logistic_regression" and tfidf_matrix is not None:
            from oil_sentiment_pipeline.modeling.pipeline import train_lr_model
            lr_model, _, _ = train_lr_model(
                processed_records,
                tfidf_matrix=tfidf_matrix,
                auto_label=True,
                save_model=save_data,
            )

        from oil_sentiment_pipeline.modeling.pipeline import run_sentiment_analysis
        sentiment_records = run_sentiment_analysis(
            processed_records,
            model=model,
            text_field="text",  # <-- Changé: FinBERT a besoin du texte brut (avec grammaire/ponctuation), pas du texte nettoyé !
            tfidf_matrix=tfidf_matrix,
            lr_model=lr_model,
            save_results=save_data,
        )
        results["sentiment_records"] = sentiment_records
        labels = [r.get("sentiment_label", "?") for r in sentiment_records]
        dist   = {l: labels.count(l) for l in ["positive", "neutral", "negative"]}
        logger.info("→ %d records scorés | Distribution: %s", len(sentiment_records), dist)

    except Exception as exc:
        logger.error("Erreur modeling : %s", exc)
        sentiment_records = processed_records

    # ──────────────────────────────────────────────────────────────────────
    # ÉTAPE 5 — Agrégation & Export Dashboard
    # ──────────────────────────────────────────────────────────────────────
    logger.info("\n[5/6] AGRÉGATION JOURNALIÈRE & EXPORT PARQUET...")
    try:
        from oil_sentiment_pipeline.aggregation.aggregator import aggregate_daily_sentiment
        from oil_sentiment_pipeline.paths import SHARED_PROCESSED_DIR
        from oil_sentiment_pipeline.settings import PipelineSettings

        cfg = PipelineSettings.from_yaml()
        aggregated_data = aggregate_daily_sentiment(sentiment_records, cfg)
        
        for asset, df_asset in aggregated_data.items():
            out_path = SHARED_PROCESSED_DIR / f"sentiment_{asset}.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df_asset.to_parquet(out_path, index=False)
            logger.info("→ Exporté : %s (%d jours)", out_path, len(df_asset))
            
    except Exception as exc:
        logger.error("Erreur d'agrégation / export dashboard : %s", exc)

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE TERMINÉ — Données exportées pour le Dashboard.")
    logger.info("Fichiers de sentiment disponibles dans : %s", SHARED_PROCESSED_DIR)
    logger.info("=" * 70)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    from pathlib import Path

    from oil_sentiment_pipeline.settings import PipelineSettings

    cfg = PipelineSettings.from_yaml()

    parser = argparse.ArgumentParser(
        description="Oil Sentiment NLP Trading Pipeline (Sentiment Extraction Only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Chemin vers config/pipeline.yaml (défaut : config/pipeline.yaml).",
    )
    parser.add_argument("--sources",     nargs="+", default=cfg.sources,
                        choices=["news", "reddit", "twitter", "edgar"],
                        help="Sources de données à collecter.")
    parser.add_argument("--model",       default=cfg.model,
                        choices=["auto", "finbert", "logistic_regression", "lexical"],
                        help="Modèle de sentiment.")
    parser.add_argument("--max-per-source", type=int, default=cfg.max_per_source)
    parser.add_argument("--start-date",  default=cfg.start_date)
    parser.add_argument("--end-date",    default=cfg.end_date)
    parser.add_argument("--no-save",     action="store_true",
                        help="Désactive la sauvegarde des données.")
    parser.add_argument("--log-level",   default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    """Point d'entrée CLI (python -m oil_sentiment_pipeline.main ou oil-sentiment)."""
    args = _parse_args()
    setup_logging(args.log_level)

    from oil_sentiment_pipeline.settings import PipelineSettings

    cfg = PipelineSettings.from_yaml(args.config) if args.config else PipelineSettings.from_yaml()

    run_pipeline(
        sources=args.sources,
        model=args.model,
        max_per_source=args.max_per_source,
        start_date=args.start_date,
        end_date=args.end_date,
        save_data=not args.no_save,
        allow_mock=cfg.allow_mock,
    )


if __name__ == "__main__":
    main()
