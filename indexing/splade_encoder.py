"""SPLADE sparse vector encoder.

All tunables (model name, top-k, batch size, max seq length) are read from
config.yaml's ``splade:`` section. The model is loaded lazily on the first
encode call to avoid heavy import-time side effects.

The encoder is model-agnostic — it works with any HuggingFace
``AutoModelForMaskedLM`` because the pooling logic (max over tokens of
log(1 + ReLU(MLM logits))) is the standard SPLADE formula.
"""

import logging
from typing import Iterable

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from configs.loader import load_config

logger = logging.getLogger(__name__)

def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging: your code talks, libraries stay quiet."""
    # 1. Root handler — controls the output format
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 2. Quiet down chatty third-party loggers
    noisy_loggers = [
        "elastic_transport",   # ES HTTP request logs
        "elasticsearch",       # Older ES client logs
        "urllib3",             # HTTP connection pool logs
        "requests",            # If you use requests elsewhere
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Read defaults from config (fail fast if config is malformed)
# ---------------------------------------------------------------------------
_cfg = load_config()
_splade_cfg: dict = _cfg.get("splade", {})

_DEFAULT_MODEL: str = _splade_cfg.get(
    "model_name", "naver/splade-cocondenser-ensembledistil"
)
_DEFAULT_TOP_K: int = _splade_cfg.get("top_k", 256)
_DEFAULT_BATCH_SIZE: int = _splade_cfg.get("batch_size", 32)
_DEFAULT_MAX_SEQ_LENGTH: int = _splade_cfg.get("max_seq_length", 512)


class SpladeEncoder:
    """Stateful SPLADE encoder. Loads model lazily on first use.

    Instantiate with explicit arguments to override config defaults,
    or use the module-level convenience functions ``encode_texts()`` /
    ``encode_query()`` which use a singleton backed by config.
    """

    def __init__(
        self,
        model_name: str | None = None,
        top_k: int | None = None,
        batch_size: int | None = None,
        max_seq_length: int | None = None,
    ) -> None:
        self.model_name = model_name or _DEFAULT_MODEL
        self.top_k = top_k or _DEFAULT_TOP_K
        self.batch_size = batch_size or _DEFAULT_BATCH_SIZE
        self.max_seq_length = max_seq_length or _DEFAULT_MAX_SEQ_LENGTH

        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModelForMaskedLM | None = None
        self._device: torch.device | None = None

    # -----------------------------------------------------------------------
    # Lazy model loading
    # -----------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        """Load tokenizer and model on first call. Idempotent."""
        if self._model is not None:
            return

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(
            "SPLADE loading model '%s' on %s ...",
            self.model_name,
            self._device,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self._model.to(self._device)
        self._model.eval()

        # Freeze parameters — inference only
        for param in self._model.parameters():
            param.requires_grad = False

        num_params = sum(p.numel() for p in self._model.parameters())
        num_layers = getattr(
            self._model.config, "num_hidden_layers", "unknown"
        )
        logger.info(
            "SPLADE ready: %s (%s layers, %d params)",
            self.model_name,
            num_layers,
            num_params,
        )

    # -----------------------------------------------------------------------
    # Core SPLADE pooling
    # -----------------------------------------------------------------------
    @staticmethod
    def _pool(
        logits: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """SPLADE max-pooling over token positions.

        Formula:
            vec = max over tokens [ log(1 + ReLU(logits)) * mask ]

        Args:
            logits: [batch, seq_len, vocab_size] from MLM head.
            attention_mask: [batch, seq_len] — 1 for real tokens, 0 for pad.

        Returns:
            [batch, vocab_size] pooled vectors.
        """
        relu = torch.relu(logits)
        compressed = torch.log1p(relu)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (compressed * mask).max(dim=1).values
        return pooled

    @staticmethod
    def _sparsify(
        vectors: torch.Tensor,
        top_k: int,
    ) -> list[dict[int, float]]:
        """Keep only the top-k highest weights per document.

        Args:
            vectors: [batch, vocab_size] dense pooled vectors.
            top_k: Number of highest entries to retain.

        Returns:
            List of {token_id: weight} dicts. Only strictly positive
            weights are included; zeros are dropped.
        """
        topk_vals, topk_idx = torch.topk(vectors, k=top_k, dim=1)

        sparse: list[dict[int, float]] = []
        for vals, idxs in zip(topk_vals, topk_idx):
            doc: dict[int, float] = {}
            for idx, val in zip(idxs.tolist(), vals.tolist()):
                if val > 0.0:
                    doc[idx] = round(val, 6)
            sparse.append(doc)

        return sparse

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def encode_texts(
        self,
        texts: Iterable[str],
        batch_size: int | None = None,
        top_k: int | None = None,
    ) -> list[dict[int, float]]:
        """Encode multiple texts into SPLADE sparse vectors.

        Args:
            texts: Iterable of raw text strings.
            batch_size: Override default batch size (tune for GPU memory).
            top_k: Override default top-k (e.g., smaller for queries).

        Returns:
            List of {token_id: weight} dicts, one per input text, in order.
        """
        self._ensure_loaded()

        texts_list = list(texts)
        if not texts_list:
            return []

        bs = batch_size or self.batch_size
        k = top_k or self.top_k
        results: list[dict[int, float]] = []

        with torch.no_grad():
            for i in range(0, len(texts_list), bs):
                batch = texts_list[i : i + bs]

                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(self._device)
                attn_mask = encoded["attention_mask"].to(self._device)

                outputs = self._model(
                    input_ids=input_ids, attention_mask=attn_mask
                )
                logits = outputs.logits

                pooled = self._pool(logits, attn_mask)
                sparse_batch = self._sparsify(pooled, top_k=k)
                results.extend(sparse_batch)

        return results

    def encode_query(
        self,
        text: str,
        top_k: int | None = None,
    ) -> dict[int, float]:
        """Encode a single query string.

        Args:
            text: Raw query text.
            top_k: Override default top-k.

        Returns:
            {token_id: weight} dict for the query.
        """
        result = self.encode_texts([text], batch_size=1, top_k=top_k)
        return result[0]


# ---------------------------------------------------------------------------
# Module-level singleton for convenience
# ---------------------------------------------------------------------------
_default_encoder: SpladeEncoder | None = None


def _get_default() -> SpladeEncoder:
    global _default_encoder
    if _default_encoder is None:
        _default_encoder = SpladeEncoder()
    return _default_encoder


def encode_texts(
    texts: Iterable[str],
    batch_size: int | None = None,
    top_k: int | None = None,
) -> list[dict[int, float]]:
    """Convenience function using the default singleton encoder."""
    return _get_default().encode_texts(texts, batch_size, top_k)


def encode_query(
    text: str,
    top_k: int | None = None,
) -> dict[int, float]:
    """Convenience function using the default singleton encoder."""
    return _get_default().encode_query(text, top_k)


# ---------------------------------------------------------------------------
# Smoke test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    setup_logging(logging.INFO)
    vec = encode_query("neural networks for protein folding")
    print(f"\nNon-zero tokens: {len(vec)}")
    print("Top 5 weights:")
    for tid, w in sorted(vec.items(), key=lambda x: -x[1])[:5]:
        print(f"  token_id={tid:>5}  weight={w:.4f}")