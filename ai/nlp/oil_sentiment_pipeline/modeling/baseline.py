"""
baseline.py
-----------
Modèle baseline de classification de sentiment : Logistic Regression sur TF-IDF.

Labels :
  - "positive"  (+1)
  - "neutral"   ( 0)
  - "negative"  (-1)

Fonctionnalités :
  - Entraînement supervisé (si données labellisées disponibles)
  - Mode lexical non-supervisé (dictionnaire de polarité) — opérationnel sans labels
  - Prédiction : label + score de confiance
  - Sauvegarde / chargement pickle
  - Rapport de classification sklearn
"""

import logging
import os
import pickle
from typing import List, Dict, Optional, Tuple

import numpy as np
from scipy.sparse import spmatrix

logger = logging.getLogger(__name__)

from oil_sentiment_pipeline.paths import MODELS_DIR

DEFAULT_MODEL_DIR = str(MODELS_DIR)

# ---------------------------------------------------------------------------
# Constantes labels
# ---------------------------------------------------------------------------

LABEL_POSITIVE = "positive"
LABEL_NEUTRAL  = "neutral"
LABEL_NEGATIVE = "negative"
LABEL_TO_INT   = {LABEL_POSITIVE: 1, LABEL_NEUTRAL: 0, LABEL_NEGATIVE: -1}
INT_TO_LABEL   = {v: k for k, v in LABEL_TO_INT.items()}

# ---------------------------------------------------------------------------
# Dictionnaire de polarité lexicale (mode non-supervisé)
# ---------------------------------------------------------------------------

POSITIVE_LEXICON = {
    # Sentiment haussier pétrole
    "bullish", "rally", "surge", "soar", "jump", "rise", "gain", "climb",
    "outperform", "strong", "recovery", "rebound", "positive", "upside",
    "beat", "exceed", "growth", "expand", "increase", "improve",
    "optimistic", "confident", "support", "breakout", "momentum",
    # Termes spécifiques oil bullish
    "cut", "draw", "drawdown", "deficit", "tight", "squeeze",
    "opec_discipline", "production_cut", "supply_shortage",
    "not_bearish", "not_negative", "not_weak",
}

NEGATIVE_LEXICON = {
    # Sentiment baissier pétrole
    "bearish", "selloff", "drop", "fall", "decline", "crash", "plunge",
    "slump", "weak", "underperform", "negative", "downside", "miss",
    "disappoint", "contraction", "reduce", "decrease", "worsen",
    "pessimistic", "concerned", "worried", "breakdown", "resistance",
    # Termes spécifiques oil bearish
    "build", "glut", "oversupply", "surplus", "flood", "dump",
    "demand_destruction", "recession", "slowdown", "inventory_build",
    "not_bullish", "not_positive", "not_strong",
}

INTENSIFIERS = {
    "very": 1.5, "extremely": 2.0, "highly": 1.5, "significantly": 1.5,
    "sharply": 1.5, "strongly": 1.5, "massively": 2.0, "slightly": 0.5,
    "marginally": 0.5, "somewhat": 0.7, "major": 1.5, "minor": 0.5,
}


# ---------------------------------------------------------------------------
# Scorer lexical
# ---------------------------------------------------------------------------

def lexical_sentiment_score(tokens: List[str]) -> float:
    """
    Calcule un score de sentiment brut basé sur le dictionnaire de polarité.

    Returns
    -------
    float : score normalisé entre -1.0 (très négatif) et +1.0 (très positif).
            0.0 si neutral ou corpus vide.
    """
    if not tokens:
        return 0.0

    score = 0.0
    multiplier = 1.0

    for i, token in enumerate(tokens):
        # Intensifieur sur le prochain token
        if token in INTENSIFIERS:
            multiplier = INTENSIFIERS[token]
            continue

        if token in POSITIVE_LEXICON:
            score += 1.0 * multiplier
        elif token in NEGATIVE_LEXICON:
            score -= 1.0 * multiplier
        else:
            multiplier = 1.0  # reset si pas de sentiment

        multiplier = 1.0  # reset après application

    # Normalisation [-1, 1]
    normalizer = max(len(tokens), 1)
    return float(np.clip(score / normalizer * 5, -1.0, 1.0))


def score_to_label(score: float, pos_threshold: float = 0.05, neg_threshold: float = -0.05) -> str:
    """Convertit un score flottant en label catégoriel."""
    if score > pos_threshold:
        return LABEL_POSITIVE
    if score < neg_threshold:
        return LABEL_NEGATIVE
    return LABEL_NEUTRAL


def predict_lexical(
    records: List[Dict],
    pos_threshold: float = 0.05,
    neg_threshold: float = -0.05,
) -> List[Dict]:
    """
    Prédit le sentiment de façon non-supervisée via le dictionnaire lexical.
    Ne nécessite pas d'entraînement.

    Parameters
    ----------
    records : List[Dict]
        Records prétraités (doit contenir 'tokens' ou 'text_clean').

    Returns
    -------
    List[Dict] : records enrichis avec 'sentiment_label' et 'sentiment_score'.
    """
    results = []
    for rec in records:
        tokens = rec.get("tokens") or rec.get("text_clean", "").split()
        score = lexical_sentiment_score(tokens)
        label = score_to_label(score, pos_threshold, neg_threshold)
        results.append({
            **rec,
            "sentiment_label": label,
            "sentiment_score": round(score, 4),
            "model":           "lexical_baseline",
        })
    return results


# ---------------------------------------------------------------------------
# Modèle Logistic Regression (supervisé)
# ---------------------------------------------------------------------------

class LogisticRegressionSentiment:
    """
    Classificateur de sentiment basé sur Logistic Regression + TF-IDF.

    Nécessite des données labellisées pour l'entraînement.
    En l'absence de labels, utilise predict_lexical() pour auto-labelliser.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000, class_weight: str = "balanced"):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder

        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=42,
        )
        self.label_encoder = LabelEncoder()
        self._is_fitted = False
        self.classes_ = [LABEL_NEGATIVE, LABEL_NEUTRAL, LABEL_POSITIVE]
        logger.info("LogisticRegressionSentiment initialisé (C=%.2f).", C)

    # ------------------------------------------------------------------

    def fit(
        self,
        X: spmatrix | np.ndarray,
        y: List[str],
    ) -> "LogisticRegressionSentiment":
        """
        Entraîne le classificateur.

        Parameters
        ----------
        X : matrice TF-IDF (n, vocab) ou embeddings (n, 768)
        y : labels string ["positive", "neutral", "negative"]
        """
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        y_enc = self.label_encoder.fit_transform(y)
        logger.info("Entraînement LR sur %d exemples (%s)...", len(y), dict(zip(*np.unique(y, return_counts=True))))
        self.model.fit(X, y_enc)
        self._is_fitted = True

        # Cross-validation rapide
        try:
            cv = StratifiedKFold(n_splits=min(5, len(set(y))), shuffle=True, random_state=42)
            scores = cross_val_score(self.model, X, y_enc, cv=cv, scoring="f1_macro")
            logger.info("CV F1-macro : %.4f ± %.4f", scores.mean(), scores.std())
        except Exception as exc:
            logger.warning("CV échouée : %s", exc)

        return self

    def predict(self, X: spmatrix | np.ndarray) -> List[str]:
        """Prédit les labels pour une matrice de features."""
        self._check_fitted()
        y_enc = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_enc).tolist()

    def predict_proba(self, X: spmatrix | np.ndarray) -> np.ndarray:
        """Retourne les probabilités par classe (negative, neutral, positive)."""
        self._check_fitted()
        return self.model.predict_proba(X)

    def predict_with_score(
        self, X: spmatrix | np.ndarray
    ) -> List[Tuple[str, float]]:
        """
        Retourne les prédictions avec un score de confiance.

        Score = P(positive) - P(negative), normalisé en [-1, 1].

        Returns
        -------
        List of (label, score)
        """
        self._check_fitted()
        probas = self.predict_proba(X)
        labels = self.predict(X)
        classes = self.label_encoder.classes_.tolist()

        pos_idx = classes.index(LABEL_POSITIVE) if LABEL_POSITIVE in classes else -1
        neg_idx = classes.index(LABEL_NEGATIVE) if LABEL_NEGATIVE in classes else -1

        results = []
        for i, label in enumerate(labels):
            if pos_idx >= 0 and neg_idx >= 0:
                score = float(probas[i, pos_idx] - probas[i, neg_idx])
            else:
                score = 0.0
            results.append((label, round(score, 4)))

        return results

    def evaluate(
        self,
        X: spmatrix | np.ndarray,
        y_true: List[str],
    ) -> Dict:
        """Évalue le modèle et retourne un rapport de classification."""
        from sklearn.metrics import classification_report, accuracy_score, f1_score

        self._check_fitted()
        y_pred = self.predict(X)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

        logger.info("Évaluation LR — Accuracy: %.4f | F1-macro: %.4f", accuracy, f1)
        return {
            "accuracy":  accuracy,
            "f1_macro":  f1,
            "report":    report,
        }

    # ------------------------------------------------------------------
    # Persistance
    # ------------------------------------------------------------------

    def save(self, path: str = None) -> str:
        self._check_fitted()
        if path is None:
            os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)
            path = os.path.join(DEFAULT_MODEL_DIR, "logistic_regression_sentiment.pkl")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "label_encoder": self.label_encoder}, f)
        logger.info("Modèle LR sauvegardé : %s", path)
        return path

    @classmethod
    def load(cls, path: str = None) -> "LogisticRegressionSentiment":
        if path is None:
            path = os.path.join(DEFAULT_MODEL_DIR, "logistic_regression_sentiment.pkl")
        instance = cls.__new__(cls)
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance.model         = data["model"]
        instance.label_encoder = data["label_encoder"]
        instance._is_fitted    = True
        logger.info("Modèle LR chargé depuis : %s", path)
        return instance

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Le modèle n'est pas encore entraîné. Appelez fit() d'abord.")


# ---------------------------------------------------------------------------
# Auto-labellisation via lexique (bootstrap sans données annotées)
# ---------------------------------------------------------------------------

def auto_label_records(
    records: List[Dict],
    pos_threshold: float = 0.08,
    neg_threshold: float = -0.08,
) -> Tuple[List[Dict], List[str]]:
    """
    Génère automatiquement des labels via le dictionnaire lexical.
    Utilisé comme bootstrap pour entraîner la Logistic Regression.

    Returns
    -------
    Tuple[List[Dict], List[str]] : (records avec labels, liste de labels)
    """
    labeled = predict_lexical(records, pos_threshold, neg_threshold)
    labels = [r["sentiment_label"] for r in labeled]
    dist = {l: labels.count(l) for l in [LABEL_POSITIVE, LABEL_NEUTRAL, LABEL_NEGATIVE]}
    logger.info("Auto-labellisation : %s", dist)
    return labeled, labels


# ---------------------------------------------------------------------------
# CLI rapide pour test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    mock_records = [
        {"tokens": ["opec", "cut", "production", "barrel", "crude", "bullish", "rally"], "text_clean": "opec cut production barrel crude bullish rally", "date": "2024-03-15", "source": "mock"},
        {"tokens": ["wti", "crude", "drop", "weak", "demand", "bearish", "selloff"],     "text_clean": "wti crude drop weak demand bearish selloff",     "date": "2024-03-14", "source": "mock"},
        {"tokens": ["brent", "stable", "market", "neutral", "wait"],                     "text_clean": "brent stable market neutral wait",               "date": "2024-03-13", "source": "mock"},
        {"tokens": ["opec", "discipline", "supply", "tight", "bullish", "surge"],        "text_clean": "opec discipline supply tight bullish surge",     "date": "2024-03-12", "source": "mock"},
        {"tokens": ["inventory", "build", "crude", "bearish", "glut", "decline"],        "text_clean": "inventory build crude bearish glut decline",     "date": "2024-03-11", "source": "mock"},
        {"tokens": ["not", "bullish", "outlook", "very", "weak", "demand"],              "text_clean": "not bullish outlook very weak demand",           "date": "2024-03-10", "source": "mock"},
    ]

    print("=== Mode lexical (non-supervisé) ===")
    results = predict_lexical(mock_records)
    for r in results:
        print(f"  [{r['sentiment_label']:8s}] score={r['sentiment_score']:+.4f} | {r['text_clean']}")

    print("\n=== Mode supervisé (auto-labellisation + LR) ===")
    from oil_sentiment_pipeline.feature_engineering.tfidf_vectorizer import build_tfidf_matrix

    texts = [r["text_clean"] for r in mock_records]
    X, vec = build_tfidf_matrix(texts)
    labeled, labels = auto_label_records(mock_records)

    clf = LogisticRegressionSentiment(C=1.0)
    clf.fit(X, labels)

    preds = clf.predict_with_score(X)
    for (label, score), orig in zip(preds, mock_records):
        print(f"  [{label:8s}] score={score:+.4f} | {orig['text_clean']}")
