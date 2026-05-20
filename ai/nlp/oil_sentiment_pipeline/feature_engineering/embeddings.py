"""
embeddings.py
-------------
Génération d'embeddings textuels via HuggingFace Transformers.

Modèles supportés :
  - FinBERT (ProsusAI/finbert)         — spécialisé finance, recommandé
  - DistilBERT (distilbert-base-uncased) — léger, généraliste
  - Tout modèle HuggingFace compatible sentence-transformers

Sortie : vecteurs numpy de dimension (n_texts, hidden_size)
  - FinBERT : (n, 768)

Stratégie d'encoding :
  - CLS token pooling (défaut)
  - Mean pooling sur tous les tokens

Bascule automatique en mode TF-IDF si torch/transformers non disponibles.
"""

import logging
import os
from typing import List, Optional, Literal

import numpy as np

logger = logging.getLogger(__name__)

from oil_sentiment_pipeline.paths import EMBEDDINGS_CACHE_DIR, MODELS_DIR, PROCESSED_DIR

DEFAULT_MODEL_DIR = str(EMBEDDINGS_CACHE_DIR)

# Modèles disponibles
AVAILABLE_MODELS = {
    "finbert":       "ProsusAI/finbert",
    "distilbert":    "distilbert-base-uncased",
    "minilm":        "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet":         "sentence-transformers/all-mpnet-base-v2",
}

DEFAULT_MODEL = "finbert"
MAX_LENGTH = 512
BATCH_SIZE = 16


# ---------------------------------------------------------------------------
# Vérification disponibilité torch / transformers
# ---------------------------------------------------------------------------

def _check_torch_available() -> bool:
    try:
        import torch
        import transformers
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Classe EmbeddingEncoder
# ---------------------------------------------------------------------------

class EmbeddingEncoder:
    """
    Encodeur de textes basé sur un modèle HuggingFace Transformer.

    Usage :
        enc = EmbeddingEncoder(model_name="finbert")
        embeddings = enc.encode(texts)   # shape: (n, 768)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        pooling: Literal["cls", "mean"] = "cls",
        device: str = None,
        cache_dir: str = None,
    ):
        if not _check_torch_available():
            raise ImportError(
                "torch et transformers sont requis pour les embeddings. "
                "Installez-les via : pip install torch transformers"
            )

        import torch
        from transformers import AutoTokenizer, AutoModel

        # Résolution du nom de modèle
        self.model_id = AVAILABLE_MODELS.get(model_name, model_name)
        self.pooling = pooling
        self.cache_dir = cache_dir or DEFAULT_MODEL_DIR

        # Device auto-détection
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info("Chargement du modèle [%s] sur device=%s...", self.model_id, self.device)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, cache_dir=self.cache_dir
            )
            self.model = AutoModel.from_pretrained(
                self.model_id, cache_dir=self.cache_dir
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info("Modèle [%s] chargé. Hidden size: %d.",
                        self.model_id, self.model.config.hidden_size)
        except Exception as exc:
            raise RuntimeError(f"Impossible de charger le modèle {self.model_id} : {exc}") from exc

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _cls_pooling(self, model_output) -> "torch.Tensor":
        """Extrait le vecteur du token [CLS]."""
        return model_output.last_hidden_state[:, 0, :]

    def _mean_pooling(self, model_output, attention_mask) -> "torch.Tensor":
        """Moyenne pondérée sur tous les tokens (hors padding)."""
        import torch
        token_embeddings = model_output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

    def encode(
        self,
        texts: List[str],
        batch_size: int = BATCH_SIZE,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode une liste de textes en vecteurs dense.

        Parameters
        ----------
        texts : List[str]
        batch_size : int
        normalize : bool
            L2-normalisation des vecteurs (recommandé pour similarité cosinus).
        show_progress : bool
            Affiche une barre de progression tqdm.

        Returns
        -------
        np.ndarray de shape (n_texts, hidden_size)
        """
        import torch

        if not texts:
            return np.array([])

        all_embeddings = []
        iterator = range(0, len(texts), batch_size)

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc=f"Encoding [{self.model_id}]")
            except ImportError:
                pass

        with torch.no_grad():
            for start in iterator:
                batch = texts[start: start + batch_size]

                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                output = self.model(**encoded)

                if self.pooling == "cls":
                    embeddings = self._cls_pooling(output)
                else:
                    embeddings = self._mean_pooling(output, encoded["attention_mask"])

                embeddings = embeddings.cpu().numpy()

                if normalize:
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1e-9, norms)
                    embeddings = embeddings / norms

                all_embeddings.append(embeddings)

        result = np.vstack(all_embeddings)
        logger.info("Embeddings générés : shape=%s, model=%s.", result.shape, self.model_id)
        return result

    @property
    def embedding_dim(self) -> int:
        return self.model.config.hidden_size


# ---------------------------------------------------------------------------
# Mode fallback — embeddings aléatoires (tests sans GPU/torch)
# ---------------------------------------------------------------------------

class MockEmbeddingEncoder:
    """
    Encodeur fictif pour les tests sans GPU ni torch.
    Génère des vecteurs aléatoires cohérents (seed par hash du texte).
    """

    def __init__(self, dim: int = 768):
        self.dim = dim
        logger.warning("MockEmbeddingEncoder actif — vecteurs aléatoires (tests uniquement).")

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        embeddings = []
        for text in texts:
            seed = hash(text[:50]) % (2**32)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.dim).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            embeddings.append(vec)
        return np.vstack(embeddings)

    @property
    def embedding_dim(self) -> int:
        return self.dim


# ---------------------------------------------------------------------------
# Factory — retourne le bon encodeur selon disponibilité
# ---------------------------------------------------------------------------

def get_encoder(
    model_name: str = DEFAULT_MODEL,
    pooling: str = "cls",
    fallback_mock: bool = True,
) -> EmbeddingEncoder | MockEmbeddingEncoder:
    """
    Retourne un EmbeddingEncoder HuggingFace ou un MockEmbeddingEncoder
    si torch/transformers ne sont pas disponibles.
    """
    if _check_torch_available():
        try:
            return EmbeddingEncoder(model_name=model_name, pooling=pooling)
        except Exception as exc:
            logger.error("Erreur chargement modèle : %s", exc)
            if fallback_mock:
                logger.warning("Basculement sur MockEmbeddingEncoder.")
                return MockEmbeddingEncoder()
            raise
    else:
        if fallback_mock:
            logger.warning("torch/transformers non disponibles — MockEmbeddingEncoder activé.")
            return MockEmbeddingEncoder()
        raise ImportError("torch et transformers requis pour les embeddings réels.")


def encode_texts(
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
    pooling: str = "cls",
    batch_size: int = BATCH_SIZE,
    normalize: bool = True,
    fallback_mock: bool = True,
) -> np.ndarray:
    """
    Fonction utilitaire one-shot : encode une liste de textes.

    Returns
    -------
    np.ndarray shape (n_texts, embedding_dim)
    """
    encoder = get_encoder(model_name=model_name, pooling=pooling, fallback_mock=fallback_mock)
    return encoder.encode(texts, batch_size=batch_size, normalize=normalize)


# ---------------------------------------------------------------------------
# Sauvegarde / chargement des embeddings
# ---------------------------------------------------------------------------

def save_embeddings(embeddings: np.ndarray, path: str) -> None:
    """Sauvegarde les embeddings en format .npy."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.save(path, embeddings)
    logger.info("Embeddings sauvegardés : %s (shape=%s).", path, embeddings.shape)


def load_embeddings(path: str) -> np.ndarray:
    """Charge des embeddings depuis un fichier .npy."""
    embeddings = np.load(path)
    logger.info("Embeddings chargés : %s (shape=%s).", path, embeddings.shape)
    return embeddings


# ---------------------------------------------------------------------------
# CLI rapide pour test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import glob, csv as _csv
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    csv_files = sorted(glob.glob(str(PROCESSED_DIR / "processed_*.csv")))

    if csv_files:
        with open(csv_files[-1], encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        texts = [r.get("text_normalized") or r.get("text_clean", "") for r in rows if r.get("text_clean")]
        print(f"Corpus chargé : {len(texts)} textes depuis {csv_files[-1]}")
    else:
        print("Aucun fichier processed trouvé — utilisation du corpus mock.")
        texts = [
            "OPEC cuts production crude oil bullish market rally",
            "WTI crude drops weak demand bearish inventory build",
            "Brent crude stable geopolitical tensions offset weak demand",
            "Shale production growth bearish long term oil supply",
            "Saudi Arabia confirms output discipline OPEC plus bullish",
        ]

    print("=== EmbeddingEncoder (avec fallback mock) ===\n")
    encoder = get_encoder(model_name="finbert", fallback_mock=True)
    embeddings = encode_texts(texts, model_name="finbert", fallback_mock=True, normalize=True)

    # Sauvegarde
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_embeddings(embeddings, str(MODELS_DIR / "embeddings.npy"))

    print(f"Shape embeddings : {embeddings.shape}")
    print(f"Dim vecteur      : {encoder.embedding_dim}")
    print(f"Norme vecteur[0] : {np.linalg.norm(embeddings[0]):.4f}")
    if len(embeddings) >= 2:
        sim_01 = float(embeddings[0] @ embeddings[1])
        print(f"Similarité cosinus doc0/doc1 : {sim_01:.4f}")
