"""FAISS HNSW index builder and loader.

Handles:
  - Building an HNSW index from dense vectors
  - Saving the index + ID mapping to disk
  - Loading the index + ID mapping from disk
  - Searching the index

The ID mapping is necessary because FAISS uses integer indices (0, 1, 2...)
internally, but our documents have string doc_ids.
"""

import json
import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FaissIndex:
    """Wrapper around a FAISS HNSW index with doc_id mapping."""

    def __init__(
        self,
        index: faiss.Index,
        id_map: list[str],
        dimension: int,
    ) -> None:
        self.index = index
        self.id_map = id_map
        self.dimension = dimension
        self._id_to_int = {doc_id: i for i, doc_id in enumerate(id_map)}

    @classmethod
    def build(
        cls,
        vectors: np.ndarray,
        doc_ids: list[str],
        M: int = 16,
        ef_construction: int = 200,
    ) -> "FaissIndex":
        """Build an HNSW index from dense vectors.

        Args:
            vectors: [N, D] numpy array of float32 vectors.
            doc_ids: List of N document IDs, in the same order as vectors.
            M: Number of bi-directional links per node.
            ef_construction: Build-time search depth.

        Returns:
            FaissIndex instance ready for search.

        Raises:
            ValueError: If vectors and doc_ids have mismatched lengths.
        """
        if len(vectors) != len(doc_ids):
            raise ValueError(
                f"Vectors ({len(vectors)}) and doc_ids ({len(doc_ids)}) "
                "must have the same length."
            )

        dimension = vectors.shape[1]
        logger.info(
            "Building HNSW index: %d vectors, %d dims, M=%d, efConstruction=%d",
            len(vectors),
            dimension,
            M,
            ef_construction,
        )

        # Use inner product metric (cosine similarity for normalized vectors)
        index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction

        # FAISS requires float32 contiguous arrays
        vectors = np.ascontiguousarray(vectors.astype("float32"))

        index.add(vectors)
        logger.info("HNSW index built: %d vectors indexed.", index.ntotal)

        return cls(index=index, id_map=list(doc_ids), dimension=dimension)

    def save(self, index_path: str | Path, ids_path: str | Path) -> None:
        """Save the FAISS index and ID mapping to disk.

        Args:
            index_path: Path to write the .faiss binary file.
            ids_path: Path to write the ID mapping JSON file.
        """
        index_path = Path(index_path)
        ids_path = Path(ids_path)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        ids_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        with ids_path.open("w", encoding="utf-8") as f:
            json.dump(self.id_map, f)

        logger.info(
            "Saved FAISS index to %s (%d vectors)",
            index_path,
            self.index.ntotal,
        )

    @classmethod
    def load(cls, index_path: str | Path, ids_path: str | Path) -> "FaissIndex":
        """Load a FAISS index and ID mapping from disk.

        Args:
            index_path: Path to the .faiss binary file.
            ids_path: Path to the ID mapping JSON file.

        Returns:
            FaissIndex instance ready for search.

        Raises:
            FileNotFoundError: If either file does not exist.
        """
        index_path = Path(index_path)
        ids_path = Path(ids_path)

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not ids_path.exists():
            raise FileNotFoundError(f"ID mapping not found: {ids_path}")

        index = faiss.read_index(str(index_path))
        with ids_path.open("r", encoding="utf-8") as f:
            id_map = json.load(f)

        dimension = index.d
        logger.info(
            "Loaded FAISS index from %s: %d vectors, %d dims",
            index_path,
            index.ntotal,
            dimension,
        )

        return cls(index=index, id_map=id_map, dimension=dimension)

    def search(
        self,
        query_vectors: np.ndarray,
        top_k: int = 10,
        ef_search: int = 128,
    ) -> tuple[list[list[str]], np.ndarray]:
        """Search the index for nearest neighbors.

        Args:
            query_vectors: [Q, D] numpy array of float32 query vectors.
            top_k: Number of nearest neighbors to return per query.
            ef_search: Query-time search depth.

        Returns:
            Tuple of (doc_ids_list, scores) where:
              - doc_ids_list: List of Q lists, each containing top_k doc_ids.
              - scores: [Q, top_k] numpy array of similarity scores.
                Higher is better (inner product / cosine similarity).
        """
        query_vectors = np.ascontiguousarray(query_vectors.astype("float32"))
        self.index.hnsw.efSearch = ef_search

        scores, indices = self.index.search(query_vectors, top_k)

        # Map integer indices back to doc_ids
        doc_ids_list: list[list[str]] = []
        for idx_row in indices:
            doc_ids = [self.id_map[i] for i in idx_row if i >= 0]
            doc_ids_list.append(doc_ids)

        return doc_ids_list, scores
