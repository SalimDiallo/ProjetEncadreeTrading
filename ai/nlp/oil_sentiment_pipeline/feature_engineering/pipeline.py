"""
pipeline.py
-----------
Orchestrateur du module feature_engineering.

Prend en entrée les records prétraités (depuis preprocessing.pipeline) et
produit deux types de features :
  - Matrice TF-IDF sparse (sklearn-compatible)
  - Embeddings denses HuggingFace (numpy array)

Format d'entrée attendu :
    [{
        "text_clean": str,
        "text_normalized": str,
        "tokens": List[str],
        "oil_density": float,
        "date": str,
        "source": str,
    }, ...]

Format de sortie :
    {
        "tfidf_matrix":   scipy.sparse.csr_matrix  (n, vocab_size),
        "embeddings":     np.ndarray               (n, 768),
        "feature_names":  List[str],               (vocab TF-IDF)
        "texts":          List[str],               (textes utilisés)
        "metadata":       List[Dict],              (date, source, oil_density)
        "vectorizer":     OilTfidfVectorizer,
        "encoder":        EmbeddingEncoder | MockEmbeddingEncoder,
    }
"""

import logging
import os
from typing import List, Dict, Optional

import numpy as np

from oil_sentiment_pipeline.feature_engineering.tfidf_vectorizer import (
    OilTfidfVectorizer,
    build_tfidf_matrix,
)
from oil_sentiment_pipeline.feature_engineering.embeddings import (
    get_encoder,
    save_embeddings,
)

logger = logging.getLogger(__name__)

from oil_sentiment_pipeline.paths import MODELS_DIR, PROCESSED_DIR

DEFAULT_ARTIFACTS_DIR = str(MODELS_DIR)


# ---------------------------------------------------------------------------
# Sélection du texte à vectoriser
# ---------------------------------------------------------------------------

def _select_text_field(record: Dict, prefer_normalized: bool = True) -> str:
    """
    Choisit le champ texte à utiliser pour la vectorisation.
    Priorité : text_normalized > text_clean > text
    """
    if prefer_normalized and record.get("text_normalized", "").strip():
        return record["text_normalized"]
    if record.get("text_clean", "").strip():
        return record["text_clean"]
    return record.get("text", "")


# ---------------------------------------------------------------------------
# Extraction des métadonnées
# ---------------------------------------------------------------------------

def _extract_metadata(records: List[Dict]) -> List[Dict]:
    return [
        {
            "date":        r.get("date", ""),
            "source":      r.get("source", ""),
            "oil_density": r.get("oil_density", 0.0),
        }
        for r in records
    ]


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def build_features(
    records: List[Dict],
    use_tfidf: bool = True,
    use_embeddings: bool = True,
    prefer_normalized: bool = True,
    tfidf_params: Dict = None,
    embedding_model: str = "finbert",
    embedding_pooling: str = "cls",
    embedding_batch_size: int = 16,
    save_artifacts: bool = False,
    artifacts_dir: str = None,
) -> Dict:
    """
    Construit toutes les features à partir des records prétraités.

    Parameters
    ----------
    records : List[Dict]
        Records issus de preprocessing.pipeline.run_preprocessing().
    use_tfidf : bool
        Construire la matrice TF-IDF.
    use_embeddings : bool
        Générer les embeddings HuggingFace.
    prefer_normalized : bool
        Utilise text_normalized plutôt que text_clean.
    tfidf_params : Dict
        Paramètres personnalisés du TF-IDF (surcharge les défauts).
    embedding_model : str
        Modèle HuggingFace : "finbert", "distilbert", "minilm", "mpnet".
    embedding_pooling : str
        Stratégie de pooling : "cls" ou "mean".
    embedding_batch_size : int
        Taille des batches pour l'inférence.
    save_artifacts : bool
        Sauvegarde le TF-IDF (pkl) et les embeddings (.npy).
    artifacts_dir : str
        Répertoire de sauvegarde des artefacts.

    Returns
    -------
    Dict avec clés : tfidf_matrix, embeddings, feature_names, texts,
                     metadata, vectorizer, encoder
    """
    if not records:
        logger.warning("Aucun record reçu — features vides.")
        return {}

    if artifacts_dir is None:
        artifacts_dir = DEFAULT_ARTIFACTS_DIR

    # Extraction des textes
    texts = [_select_text_field(r, prefer_normalized) for r in records]
    texts = [t if t.strip() else "unknown" for t in texts]

    metadata = _extract_metadata(records)

    result = {
        "texts":    texts,
        "metadata": metadata,
    }

    # ------------------------------------------------------------------
    # TF-IDF
    # ------------------------------------------------------------------
    if use_tfidf:
        logger.info("Construction matrice TF-IDF sur %d textes...", len(texts))
        try:
            tfidf_path = os.path.join(artifacts_dir, "tfidf_vectorizer.pkl") if save_artifacts else None
            matrix, vectorizer = build_tfidf_matrix(
                texts,
                params=tfidf_params,
                save_model=save_artifacts,
                model_path=tfidf_path,
            )
            result["tfidf_matrix"]  = matrix
            result["feature_names"] = vectorizer.get_feature_names()
            result["vectorizer"]    = vectorizer
            logger.info("TF-IDF OK : shape=%s, vocab=%d.", matrix.shape, vectorizer.vocab_size)
        except Exception as exc:
            logger.error("Erreur TF-IDF : %s", exc)
            result["tfidf_matrix"]  = None
            result["feature_names"] = []
            result["vectorizer"]    = None

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    if use_embeddings:
        logger.info("Génération embeddings [%s] sur %d textes...", embedding_model, len(texts))
        try:
            encoder = get_encoder(
                model_name=embedding_model,
                pooling=embedding_pooling,
                fallback_mock=True,
            )
            embeddings = encoder.encode(
                texts,
                batch_size=embedding_batch_size,
                normalize=True,
            )
            result["embeddings"] = embeddings
            result["encoder"]    = encoder

            if save_artifacts:
                emb_path = os.path.join(artifacts_dir, "embeddings.npy")
                save_embeddings(embeddings, emb_path)

            logger.info("Embeddings OK : shape=%s.", embeddings.shape)

        except Exception as exc:
            logger.error("Erreur embeddings : %s", exc)
            result["embeddings"] = None
            result["encoder"]    = None

    logger.info("Feature engineering terminé. %d records traités.", len(texts))
    return result


# ---------------------------------------------------------------------------
# Inférence sur de nouveaux textes (après entraînement)
# ---------------------------------------------------------------------------

def transform_new_texts(
    texts: List[str],
    vectorizer: OilTfidfVectorizer,
    encoder=None,
    use_tfidf: bool = True,
    use_embeddings: bool = True,
) -> Dict:
    """
    Applique les transformations sur de nouveaux textes
    avec un vectoriseur et encodeur déjà entraînés.
    """
    result = {"texts": texts}

    if use_tfidf and vectorizer is not None:
        try:
            result["tfidf_matrix"] = vectorizer.transform(texts)
        except Exception as exc:
            logger.error("Erreur transform TF-IDF : %s", exc)
            result["tfidf_matrix"] = None

    if use_embeddings and encoder is not None:
        try:
            result["embeddings"] = encoder.encode(texts, normalize=True)
        except Exception as exc:
            logger.error("Erreur transform embeddings : %s", exc)
            result["embeddings"] = None

    return result


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
        print(f"Records chargés : {len(records)} depuis {csv_files[-1]}")
    else:
        print("Aucun fichier processed trouvé — utilisation des records mock.")
        records = [
            {"text_clean": "opec cut production barrel crude bullish", "text_normalized": "opec_organization cut production barrel crude bullish", "oil_density": 0.33, "date": "2024-03-15T08:00:00Z", "source": "twitter_mock"},
            {"text_clean": "wti crude drops weak demand bearish inventory", "text_normalized": "west_texas_intermediate crude drops weak demand bearish inventory", "oil_density": 0.28, "date": "2024-03-14T14:30:00Z", "source": "reddit_mock"},
            {"text_clean": "brent crude rallies opec discipline supply tight", "text_normalized": "brent crude rallies opec_organization discipline supply tight", "oil_density": 0.40, "date": "2024-03-13T10:00:00Z", "source": "reuters"},
            {"text_clean": "goldman sachs raises brent forecast bullish energy", "text_normalized": "goldman sachs raises brent forecast bullish energy", "oil_density": 0.22, "date": "2024-03-12T09:00:00Z", "source": "yahoo_finance"},
            {"text_clean": "china demand recovery slow crude bearish", "text_normalized": "china demand recovery slow crude bearish", "oil_density": 0.25, "date": "2024-03-11T11:00:00Z", "source": "bloomberg"},
        ]

    features = build_features(
        records,
        use_tfidf=True,
        use_embeddings=True,
        embedding_model="finbert",
        save_artifacts=True,
        artifacts_dir=str(MODELS_DIR),
    )

    print(f"\n{'='*60}")
    print("FEATURE ENGINEERING — RÉSULTATS")
    print(f"{'='*60}")
    print(f"Textes traités    : {len(features['texts'])}")

    if features.get("tfidf_matrix") is not None:
        m = features["tfidf_matrix"]
        print(f"TF-IDF shape      : {m.shape}")
        print(f"Vocab size        : {features['vectorizer'].vocab_size}")
        top = features["vectorizer"].top_features(m, n=8)
        print(f"Top 8 termes      : {[t for t, _ in top]}")

    if features.get("embeddings") is not None:
        emb = features["embeddings"]
        print(f"Embeddings shape  : {emb.shape}")
        print(f"Norme vecteur[0]  : {np.linalg.norm(emb[0]):.4f}")
