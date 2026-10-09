"""SPLADE sparse-vector retrieval against Elasticsearch.

Encodes the query with the same SPLADE model used at index time,
then searches the scifact_splade index using ES sparse_vector queries.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from retrieval.types import SearchResult

if TYPE_CHECKING:  # annotations only: keep torch out of import time
    from elasticsearch import Elasticsearch

    from indexing.splade_encoder import SpladeEncoder

logger = logging.getLogger(__name__)

def search_splade(
    client: Elasticsearch, 
    query_text: str,
    index_name: str = "scifact_splade",
    encoder: SpladeEncoder | None = None,
    top_k: int = 10,
) -> list[SearchResult]:
    """Search the SPLADE index for documents matching the query.

    Args:
        query_text: Raw query string (e.g., a scientific claim).
        index_name: Target Elasticsearch index.
        top_k: Number of top results to return.

    Returns:
        List of SearchResult objects, sorted by relevance score descending.
    """
    # Phase 1: Encode query into SPLADE sparse vector
    if encoder is None:
        from indexing.splade_encoder import SpladeEncoder

        encoder = SpladeEncoder()
    query_vector = encoder.encode_query(query_text)

    if not query_vector:
        return []

    # Elasticsearch sparse_vector expects string keys
    query_vector_str = {str(k): v for k, v in query_vector.items()}

    # Phase 2: Build and execute the sparse_vector query

    response = client.search(
        index=index_name,
        body={
            "query": {
                "sparse_vector": {
                    "field": "sparse_vector",
                    "query_vector": query_vector_str,
                }
            },
            "size": top_k,
        },
    )

    # Phase 3: Parse response into typed SearchResult objects
    hits = response["hits"]["hits"]
    results: list[SearchResult] = []

    for hit in hits:
        source = hit["_source"]
        results.append(
            SearchResult(
                doc_id=source["doc_id"],
                score=hit["_score"],
                title=source.get("title", ""),
                text=source.get("text", ""),
            )
        )

    return results

if __name__ == "__main__":
    from es_client import get_es_client

    query = "COVID-19 vaccines are effective against variants."
    results = search_splade(get_es_client(), query, top_k=5)
    for r in results:
        print(f"Doc ID: {r.doc_id}, Score: {r.score:.4f}, Title: {r.title}")
