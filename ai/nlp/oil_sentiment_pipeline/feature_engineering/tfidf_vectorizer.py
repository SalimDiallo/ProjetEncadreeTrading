"""
tfidf_vectorizer.py
-------------------
Vectorisation TF-IDF des textes prétraités.

Fonctionnalités :
  - Entraînement / fit du vectoriseur sur un corpus
  - Transform d'un batch de textes en matrice sparse
  - Sauvegarde / chargement du modèle (pickle)
  - Extraction des features les plus importantes
  - Matrice dense optionnelle pour compatibilité modèles sklearn
"""

import logging
import os
import pickle
from typing import List, Dict, Optional, Tuple

import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

from oil_sentiment_pipeline.paths import MODELS_DIR, PROCESSED_DIR

DEFAULT_MODEL_DIR = str(MODELS_DIR)

# ---------------------------------------------------------------------------
# Configuration par défaut du TF-IDF
# ---------------------------------------------------------------------------

DEFAULT_TFIDF_PARAMS = {
    "max_features":   10_000,   # vocab limité
    "ngram_range":    (1, 2),   # unigrams + bigrams
    "min_df":         2,        # ignore les termes < 2 docs
    "max_df":         0.95,     # ignore les termes > 95% des docs
    "sublinear_tf":   True,     # log(tf) — réduit l'effet des répétitions
    "strip_accents":  "unicode",
    "analyzer":       "word",
    "token_pattern":  r"(?u)\b\w[\w\-_]+\b",  # accepte les tokens normalisés (underscores)
    "use_idf":        True,
    "smooth_idf":     True,
}


# ---------------------------------------------------------------------------
# Classe OilTfidfVectorizer
# ---------------------------------------------------------------------------

class OilTfidfVectorizer:
    """
    Wrapper autour de sklearn TfidfVectorizer adapté au corpus pétrole.

    Usage typique :
        vec = OilTfidfVectorizer()
        X_train = vec.fit_transform(train_texts)
        X_test  = vec.transform(test_texts)
        vec.save()
    """

    def __init__(self, params: Dict = None):
        config = {**DEFAULT_TFIDF_PARAMS, **(params or {})}
        self.vectorizer = TfidfVectorizer(**config)
        self._is_fitted = False
        logger.info("OilTfidfVectorizer initialisé (max_features=%d, ngrams=%s).",
                    config["max_features"], config["ngram_range"])

    # ------------------------------------------------------------------
    # Fit / Transform
    # ------------------------------------------------------------------

    def fit(self, texts: List[str]) -> "OilTfidfVectorizer":
        """Entraîne le vectoriseur sur le corpus."""
        if not texts:
            raise ValueError("Corpus vide — impossible d'entraîner le TF-IDF.")
        logger.info("Fit TF-IDF sur %d textes...", len(texts))
        self.vectorizer.fit(texts)
        self._is_fitted = True
        logger.info("Vocabulaire TF-IDF : %d termes.", len(self.vectorizer.vocabulary_))
        return self

    def transform(self, texts: List[str]) -> spmatrix:
        """Transforme une liste de textes en matrice TF-IDF sparse."""
        self._check_fitted()
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: List[str]) -> spmatrix:
        """Fit + Transform en une seule passe."""
        if not texts:
            raise ValueError("Corpus vide.")
        logger.info("Fit+Transform TF-IDF sur %d textes...", len(texts))
        matrix = self.vectorizer.fit_transform(texts)
        self._is_fitted = True
        logger.info("Vocabulaire TF-IDF : %d termes. Matrice: %s.", 
                    len(self.vectorizer.vocabulary_), matrix.shape)
        return matrix

    def to_dense(self, matrix: spmatrix) -> np.ndarray:
        """Convertit la matrice sparse en dense numpy array."""
        return matrix.toarray()

    # ------------------------------------------------------------------
    # Analyse des features
    # ------------------------------------------------------------------

    def get_feature_names(self) -> List[str]:
        """Retourne les noms des features (termes du vocabulaire)."""
        self._check_fitted()
        return self.vectorizer.get_feature_names_out().tolist()

    def top_features(self, matrix: spmatrix, n: int = 20) -> List[Tuple[str, float]]:
        """
        Retourne les N termes TF-IDF les plus importants du corpus entier
        (somme des scores TF-IDF par terme).
        """
        self._check_fitted()
        scores = np.asarray(matrix.sum(axis=0)).flatten()
        feature_names = self.get_feature_names()
        top_indices = scores.argsort()[::-1][:n]
        return [(feature_names[i], float(scores[i])) for i in top_indices]

    def top_features_for_doc(
        self, text: str, n: int = 10
    ) -> List[Tuple[str, float]]:
        """Retourne les N termes les plus importants pour un document donné."""
        self._check_fitted()
        vec = self.transform([text])
        scores = np.asarray(vec.todense()).flatten()
        feature_names = self.get_feature_names()
        top_indices = scores.argsort()[::-1][:n]
        return [(feature_names[i], float(scores[i])) for i in top_indices if scores[i] > 0]

    # ------------------------------------------------------------------
    # Persistance
    # ------------------------------------------------------------------

    def save(self, path: str = None) -> str:
        """Sauvegarde le vectoriseur en pickle."""
        self._check_fitted()
        if path is None:
            os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)
            path = os.path.join(DEFAULT_MODEL_DIR, "tfidf_vectorizer.pkl")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        logger.info("TF-IDF sauvegardé : %s", path)
        return path

    @classmethod
    def load(cls, path: str = None) -> "OilTfidfVectorizer":
        """Charge un vectoriseur depuis un fichier pickle."""
        if path is None:
            path = os.path.join(DEFAULT_MODEL_DIR, "tfidf_vectorizer.pkl")
        instance = cls.__new__(cls)
        with open(path, "rb") as f:
            instance.vectorizer = pickle.load(f)
        instance._is_fitted = True
        logger.info("TF-IDF chargé depuis : %s (vocab=%d).", 
                    path, len(instance.vectorizer.vocabulary_))
        return instance

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Le vectoriseur TF-IDF n'a pas encore été entraîné. Appelez fit() d'abord.")

    @property
    def vocab_size(self) -> int:
        self._check_fitted()
        return len(self.vectorizer.vocabulary_)


# ---------------------------------------------------------------------------
# Fonctions utilitaires standalone
# ---------------------------------------------------------------------------

def build_tfidf_matrix(
    texts: List[str],
    params: Dict = None,
    save_model: bool = False,
    model_path: str = None,
) -> Tuple[spmatrix, OilTfidfVectorizer]:
    """
    Construit la matrice TF-IDF complète sur un corpus.

    Returns
    -------
    Tuple[spmatrix, OilTfidfVectorizer]
    """
    vec = OilTfidfVectorizer(params=params)
    matrix = vec.fit_transform(texts)
    if save_model:
        vec.save(model_path)
    return matrix, vec


def texts_to_tfidf(
    texts: List[str],
    vectorizer: OilTfidfVectorizer,
    dense: bool = False,
) -> np.ndarray | spmatrix:
    """Transforme des textes avec un vectoriseur déjà entraîné."""
    matrix = vectorizer.transform(texts)
    return vectorizer.to_dense(matrix) if dense else matrix


# ---------------------------------------------------------------------------
# CLI rapide pour test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import glob, csv as _csv
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Lit le dernier fichier processed
    csv_files = sorted(glob.glob(str(PROCESSED_DIR / "processed_*.csv")))

    if csv_files:
        with open(csv_files[-1], encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        corpus = [r.get("text_normalized") or r.get("text_clean", "") for r in rows if r.get("text_clean")]
        print(f"Corpus chargé : {len(corpus)} textes depuis {csv_files[-1]}")
    else:
        print("Aucun fichier processed trouvé — utilisation du corpus mock.")
        corpus = [
            "opec cut production barrel crude bullish",
            "wti crude drops weak demand bearish",
            "wti crude rallies opec discipline supply tight",
            "eia report draw inventory bullish crude oil",
            "shale production permian growth bearish supply",
            "goldman sachs raises wti forecast bullish energy",
            "china demand recovery slow crude oil bearish",
            "saudi arabia confirms no production increase bullish",
            "us crude inventory build bearish wti",
            "opec plus meeting production decision crude oil",
        ]

    os.makedirs(MODELS_DIR, exist_ok=True)
    matrix, vec = build_tfidf_matrix(
        corpus,
        save_model=True,
        model_path=str(MODELS_DIR / "tfidf.pkl"),
    )
    print(f"Matrice TF-IDF : {matrix.shape}")
    print(f"Vocabulaire    : {vec.vocab_size} termes\n")

    print("Top 15 termes globaux :")
    for term, score in vec.top_features(matrix, n=15):
        print(f"  {term:<35} {score:.4f}")

    print("\nTop features pour le doc 0 :")
    for term, score in vec.top_features_for_doc(corpus[0]):
        print(f"  {term:<35} {score:.4f}")
