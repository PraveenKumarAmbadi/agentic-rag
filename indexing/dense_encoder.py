"""Dense vector encoder using sentence-transformers bi-encoder.

All tunables (model name, batch size, max seq length) are read from
config.yaml's ``dense:`` section. The model is loaded lazily on the first
encode call to avoid heavy import-time side effects.
"""

import logging
from typing import Iterable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from configs.loader import load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Read defaults from config (fail fast if config is malformed)
# ---------------------------------------------------------------------------
_cfg = load_config()
_dense_cfg: dict = _cfg.get("dense", {})

_DEFAULT_MODEL: str = _dense_cfg.get(
    "model_name", "sentence-transformers/all-MiniLM-L6-v2"
)
_DEFAULT_BATCH_SIZE: int = _dense_cfg.get("batch_size", 64)
_DEFAULT_MAX_SEQ_LENGTH: int = _dense_cfg.get("max_seq_length", 512)


class DenseEncoder:
    """Stateful dense vector encoder. Loads model lazily on first use.

    Instantiate with explicit arguments to override config defaults,
    or use the module-level convenience functions ``encode_texts()`` /
    ``encode_query()`` which use a singleton backed by config.
    """

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        max_seq_length: int | None = None,
    ) -> None:
        self.model_name = model_name or _DEFAULT_MODEL
        self.batch_size = batch_size or _DEFAULT_BATCH_SIZE
        self.max_seq_length = max_seq_length or _DEFAULT_MAX_SEQ_LENGTH

        self._model: SentenceTransformer | None = None
        self._device: torch.device | None = None

    # -----------------------------------------------------------------------
    # Lazy model loading
    # -----------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        """Load model on first call. Idempotent."""
        if self._model is not None:
            return

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(
            "Dense encoder loading model '%s' on %s ...",
            self.model_name,
            self._device,
        )

        self._model = SentenceTransformer(
            self.model_name,
            device=str(self._device),
        )
        # sentence-transformers handles its own eval mode and gradient freezing

        embedding_dim = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Dense encoder ready: %s (%d dims)",
            self.model_name,
            embedding_dim,
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def encode_texts(
        self,
        texts: Iterable[str],
        batch_size: int | None = None,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode multiple texts into dense vectors.

        Args:
            texts: Iterable of raw text strings.
            batch_size: Override default batch size (tune for GPU memory).
            show_progress_bar: If True, show a tqdm progress bar.

        Returns:
            [N, D] numpy array of float32 vectors, where D is the model's
            embedding dimension. Vectors are L2-normalized by the model.
        """
        self._ensure_loaded()

        texts_list = list(texts)
        if not texts_list:
            return np.array([])

        bs = batch_size or self.batch_size

        with torch.no_grad():
            vectors = self._model.encode(
                texts_list,
                batch_size=bs,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalize -> cosine similarity via dot product
            )

        return vectors  # type: ignore[return-value]

    def encode_query(
        self,
        text: str,
    ) -> np.ndarray:
        """Encode a single query string.

        Args:
            text: Raw query text.

        Returns:
            [D] numpy array of float32.
        """
        result = self.encode_texts([text], batch_size=1)
        return result[0]


# ---------------------------------------------------------------------------
# Module-level singleton for convenience
# ---------------------------------------------------------------------------
_default_encoder: DenseEncoder | None = None


def _get_default() -> DenseEncoder:
    global _default_encoder
    if _default_encoder is None:
        _default_encoder = DenseEncoder()
    return _default_encoder


def encode_texts(
    texts: Iterable[str],
    batch_size: int | None = None,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """Convenience function using the default singleton encoder."""
    return _get_default().encode_texts(texts, batch_size, show_progress_bar)


def encode_query(text: str) -> np.ndarray:
    """Convenience function using the default singleton encoder."""
    return _get_default().encode_query(text)


# ---------------------------------------------------------------------------
# Smoke test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    vec = encode_query("neural networks for protein folding")
    print(f"\nVector shape: {vec.shape}")
    print(f"Vector norm (should be ~1.0): {np.linalg.norm(vec):.4f}")
    print(f"First 5 dims: {vec[:5]}")
