"""Dense vector retrieval using FAISS HNSW index.

Encodes the query with the same sentence-transformers model used at index time,
then searches the FAISS index and returns SearchResult objects.
"""

import numpy as np

from indexing.dense_encoder import DenseEncoder
from indexing.faiss_indexer import FaissIndex
from ingestion.loaders import Document
from retrieval.types import SearchResult


def search_dense(
    query_text: str,
    faiss_index: FaissIndex,
    corpus: dict[str, Document],
    encoder: DenseEncoder | None = None,
    top_k: int = 10,
) -> list[SearchResult]:
    """Search the dense FAISS index for documents matching the query.

    Args:
        query_text: Raw query string (e.g., a scientific claim).
        faiss_index: Pre-loaded FaissIndex instance.
        corpus: Mapping doc_id -> Document for metadata lookup.
                Must be provided -- FAISS stores only vectors.
        encoder: Optional pre-initialized DenseEncoder. If None, uses default.
        top_k: Number of top results to return.

    Returns:
        List of SearchResult objects, sorted by relevance score descending.

    Raises:
        ValueError: If corpus is not provided.
    """
    if corpus is None:
        raise ValueError(
            "corpus is required for dense retrieval -- FAISS stores only vectors, "
            "not document metadata."
        )

    # Phase 1: Encode query
    if encoder is None:
        encoder = DenseEncoder()
    query_vector = encoder.encode_query(query_text)
    query_batch = query_vector.reshape(1, -1)

    # Phase 2: Search
    doc_ids_list, scores = faiss_index.search(query_batch, top_k=top_k)

    # Phase 3: Parse results
    results: list[SearchResult] = []
    for doc_id, score in zip(doc_ids_list[0], scores[0]):
        doc = corpus[doc_id]
        results.append(
            SearchResult(
                doc_id=doc_id,
                score=float(score),
                title=doc.title,
                text=doc.text,
            )
        )

    return results
