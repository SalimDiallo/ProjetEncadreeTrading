"""
finbert_classifier.py
---------------------
Classificateur de sentiment avancé basé sur FinBERT (ProsusAI/finbert).

FinBERT est un modèle BERT pré-entraîné sur des corpus financiers,
fine-tuné pour la classification de sentiment en 3 classes :
    - positive  (+1)
    - neutral   ( 0)
    - negative  (-1)

Avantages vs baseline lexical :
  - Comprend le contexte (negations, ironie partielle)
  - Spécialisé finance (Reuters, Bloomberg, SEC filings)
  - Pas d'entraînement supplémentaire requis (zero-shot)

Bascule automatique sur le scorer lexical si torch absent.
"""

import logging
import os
from typing import List, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from oil_sentiment_pipeline.paths import FINBERT_CACHE_DIR

DEFAULT_MODEL_DIR = str(FINBERT_CACHE_DIR)

FINBERT_MODEL_ID  = "ProsusAI/finbert"
MAX_LENGTH        = 512
BATCH_SIZE        = 8

LABEL_MAP = {
    "positive": "positive",
    "neutral":  "neutral",
    "negative": "negative",
}


# ---------------------------------------------------------------------------
# Vérification disponibilité torch / transformers
# ---------------------------------------------------------------------------

def _torch_available() -> bool:
    try:
        import torch
        import transformers
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Classe FinBERTClassifier
# ---------------------------------------------------------------------------

class FinBERTClassifier:
    """
    Classificateur de sentiment zéro-shot basé sur ProsusAI/finbert.

    Usage :
        clf = FinBERTClassifier()
        results = clf.predict(texts)
        # [{"label": "positive", "score": 0.92, "probas": {...}}, ...]
    """

    def __init__(
        self,
        model_id: str = FINBERT_MODEL_ID,
        device: str = None,
        cache_dir: str = None,
    ):
        if not _torch_available():
            raise ImportError("torch et transformers requis. pip install torch transformers")

        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.model_id  = model_id
        self.cache_dir = cache_dir or DEFAULT_MODEL_DIR

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info("Chargement FinBERT [%s] sur %s...", self.model_id, self.device)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, cache_dir=self.cache_dir
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id, cache_dir=self.cache_dir
            )
            self.model.to(self.device)
            self.model.eval()

            # Labels du modèle (ordre selon config)
            self.id2label = self.model.config.id2label
            logger.info("FinBERT chargé. Labels : %s", self.id2label)

        except Exception as exc:
            raise RuntimeError(f"Impossible de charger FinBERT : {exc}") from exc

    # ------------------------------------------------------------------
    # Prédiction
    # ------------------------------------------------------------------

    def predict(
        self,
        texts: List[str],
        batch_size: int = BATCH_SIZE,
        return_all_scores: bool = False,
    ) -> List[Dict]:
        """
        Classe une liste de textes en positive / neutral / negative.

        Parameters
        ----------
        texts : List[str]
            Textes bruts ou nettoyés (FinBERT gère la tokenisation).
        batch_size : int
        return_all_scores : bool
            Si True, inclut les probabilités pour chaque classe.

        Returns
        -------
        List[Dict] :
            [{"label": str, "score": float, "probas": {label: float}}, ...]
        """
        import torch
        import torch.nn.functional as F

        results = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start: start + batch_size]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)
                logits  = outputs.logits
                probas  = F.softmax(logits, dim=-1).cpu().numpy()

            for i in range(len(batch)):
                pred_idx   = int(np.argmax(probas[i]))
                raw_label  = self.id2label[pred_idx].lower()
                label      = LABEL_MAP.get(raw_label, "neutral")
                conf_score = float(probas[i, pred_idx])

                # Score directionnel : P(positive) - P(negative)
                pos_idx = self._label_to_idx("positive")
                neg_idx = self._label_to_idx("negative")
                directional = float(probas[i, pos_idx] - probas[i, neg_idx])

                entry = {
                    "label":            label,
                    "score":            round(directional, 4),
                    "confidence":       round(conf_score, 4),
                }

                if return_all_scores:
                    entry["probas"] = {
                        self.id2label[j].lower(): round(float(probas[i, j]), 4)
                        for j in range(len(self.id2label))
                    }

                results.append(entry)

        logger.info("FinBERT : %d textes classifiés.", len(results))
        return results

    def predict_records(
        self,
        records: List[Dict],
        text_field: str = "text",
        batch_size: int = BATCH_SIZE,
    ) -> List[Dict]:
        """
        Applique la prédiction sur une liste de records et enrichit chaque record.

        Parameters
        ----------
        records : List[Dict]
        text_field : str
            Champ texte à utiliser ("text", "text_clean", "text_normalized").

        Returns
        -------
        List[Dict] : records enrichis avec sentiment_label, sentiment_score,
                     sentiment_confidence, model.
        """
        texts = [r.get(text_field, "") or "" for r in records]
        preds = self.predict(texts, batch_size=batch_size, return_all_scores=True)

        enriched = []
        for rec, pred in zip(records, preds):
            enriched.append({
                **rec,
                "sentiment_label":      pred["label"],
                "sentiment_score":      pred["score"],
                "sentiment_confidence": pred["confidence"],
                "sentiment_probas":     pred.get("probas", {}),
                "model":                "finbert",
            })

        return enriched

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _label_to_idx(self, label: str) -> int:
        """Retourne l'index d'un label dans la config du modèle."""
        for idx, lbl in self.id2label.items():
            if lbl.lower() == label.lower():
                return int(idx)
        return 0


# ---------------------------------------------------------------------------
# Pipeline HuggingFace (alternative légère via pipeline API)
# ---------------------------------------------------------------------------

class FinBERTPipeline:
    """
    Wrapper autour de la pipeline API HuggingFace — plus simple, moins configurable.
    Idéal pour les tests rapides.
    """

    def __init__(self, model_id: str = FINBERT_MODEL_ID, device: int = -1):
        if not _torch_available():
            raise ImportError("torch et transformers requis.")
        from transformers import pipeline as hf_pipeline
        logger.info("Initialisation pipeline HuggingFace [%s]...", model_id)
        self._pipe = hf_pipeline(
            "text-classification",
            model=model_id,
            tokenizer=model_id,
            device=device,
            truncation=True,
            max_length=MAX_LENGTH,
            top_k=None,
        )
        logger.info("Pipeline HuggingFace prête.")

    def predict(self, texts: List[str], batch_size: int = BATCH_SIZE) -> List[Dict]:
        raw = self._pipe(texts, batch_size=batch_size)
        results = []
        for scores_list in raw:
            probas = {item["label"].lower(): item["score"] for item in scores_list}
            pos = probas.get("positive", 0.0)
            neg = probas.get("negative", 0.0)
            label = max(probas, key=probas.get)
            results.append({
                "label":      LABEL_MAP.get(label, "neutral"),
                "score":      round(pos - neg, 4),
                "confidence": round(probas[label], 4),
                "probas":     {k: round(v, 4) for k, v in probas.items()},
            })
        return results

    def predict_records(
        self, records: List[Dict], text_field: str = "text", batch_size: int = BATCH_SIZE
    ) -> List[Dict]:
        texts = [r.get(text_field, "") or "" for r in records]
        preds = self.predict(texts, batch_size=batch_size)
        return [
            {**rec, "sentiment_label": p["label"], "sentiment_score": p["score"],
             "sentiment_confidence": p["confidence"], "sentiment_probas": p.get("probas", {}),
             "model": "finbert_pipeline"}
            for rec, p in zip(records, preds)
        ]


# ---------------------------------------------------------------------------
# Mock FinBERT (sans torch) — scores déterministes basés sur lexique
# ---------------------------------------------------------------------------

class MockFinBERTClassifier:
    """
    Simulateur FinBERT pour les environnements sans GPU/torch.
    Utilise le scorer lexical de baseline.py avec une légère variation aléatoire.
    """

    def __init__(self):
        logger.warning("MockFinBERTClassifier actif — scores simulés (sans torch).")

    def predict(self, texts: List[str], **kwargs) -> List[Dict]:
        from oil_sentiment_pipeline.modeling.baseline import (
            lexical_sentiment_score, score_to_label
        )
        results = []
        for text in texts:
            tokens = text.split()
            score  = lexical_sentiment_score(tokens)
            # Légère variation déterministe
            seed = hash(text[:30]) % (2**16)
            rng  = np.random.default_rng(seed)
            noise = rng.uniform(-0.05, 0.05)
            score = float(np.clip(score + noise, -1.0, 1.0))
            label = score_to_label(score)
            conf  = abs(score) * 0.4 + 0.5  # [0.5, 0.9]
            results.append({
                "label":      label,
                "score":      round(score, 4),
                "confidence": round(min(conf, 0.99), 4),
                "probas":     _score_to_probas(score),
            })
        return results

    def predict_records(
        self, records: List[Dict], text_field: str = "text", **kwargs
    ) -> List[Dict]:
        texts = [r.get(text_field, "") or "" for r in records]
        preds = self.predict(texts)
        return [
            {**rec, "sentiment_label": p["label"], "sentiment_score": p["score"],
             "sentiment_confidence": p["confidence"], "sentiment_probas": p.get("probas", {}),
             "model": "mock_finbert"}
            for rec, p in zip(records, preds)
        ]


def _score_to_probas(score: float) -> Dict[str, float]:
    """Convertit un score [-1, 1] en distribution de probabilités approximative."""
    pos = float(np.clip((score + 1) / 2 * 0.8, 0.01, 0.95))
    neg = float(np.clip((1 - score) / 2 * 0.8, 0.01, 0.95))
    neu = max(0.01, round(1.0 - pos - neg, 4))
    total = pos + neg + neu
    return {
        "positive": round(pos / total, 4),
        "neutral":  round(neu / total, 4),
        "negative": round(neg / total, 4),
    }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_finbert_classifier(
    use_pipeline_api: bool = False,
    fallback_mock: bool = True,
) -> FinBERTClassifier | FinBERTPipeline | MockFinBERTClassifier:
    """
    Retourne le classificateur FinBERT adapté à l'environnement.

    Priority : FinBERTClassifier > FinBERTPipeline > MockFinBERTClassifier
    """
    if not _torch_available():
        if fallback_mock:
            return MockFinBERTClassifier()
        raise ImportError("torch/transformers requis.")

    try:
        if use_pipeline_api:
            return FinBERTPipeline()
        return FinBERTClassifier()
    except Exception as exc:
        logger.error("Erreur init FinBERT : %s", exc)
        if fallback_mock:
            logger.warning("Basculement sur MockFinBERTClassifier.")
            return MockFinBERTClassifier()
        raise


# ---------------------------------------------------------------------------
# CLI rapide pour test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    texts = [
        "OPEC+ agrees to cut production by 1M bbl/day. Crude oil market expected to tighten significantly.",
        "WTI crude drops below $75 on weak US jobs data and rising inventory builds.",
        "WTI crude oil prices remain stable as supply and demand remain balanced.",
        "Goldman Sachs raises WTI forecast to $95/barrel, bullish on energy sector.",
        "China demand recovery slower than expected — bearish signal for crude oil markets.",
        "Not bullish on oil prices despite OPEC cuts due to weak global growth outlook.",
    ]

    clf = get_finbert_classifier(fallback_mock=True)
    results = clf.predict(texts, return_all_scores=True) if hasattr(clf, 'predict') else clf.predict(texts)

    print(f"\n{'TEXT':<65} | {'LABEL':8} | {'SCORE':7} | CONF")
    print("-" * 100)
    for text, res in zip(texts, results):
        print(f"{text[:63]:<65} | {res['label']:8} | {res['score']:+.4f} | {res['confidence']:.4f}")
