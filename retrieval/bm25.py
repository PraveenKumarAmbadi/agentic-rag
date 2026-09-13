"""BM25 retrieval module — search documents in an Elasticsearch index."""

from elasticsearch import Elasticsearch

from retrieval.types import SearchResult

# Valid searchable fields in the scifact_bm25 mapping.
# doc_id is keyword (exact match), not text — BM25 match would tokenize it.
_SEARCHABLE_FIELDS = {"title", "text", "full_text"}


def _validate_fields(fields: list[str]) -> None:
    """Ensure all requested fields exist and are searchable."""
    invalid = set(fields) - _SEARCHABLE_FIELDS
    if invalid:
        raise ValueError(
            f"Invalid search field(s): {sorted(invalid)}. "
            f"Valid fields: {sorted(_SEARCHABLE_FIELDS)}"
        )


def search_bm25(
    client: Elasticsearch,
    index_name: str,
    query_text: str,
    top_k: int = 100,
    fields: list[str] | None = None,
) -> list[SearchResult]:
    """Search an Elasticsearch BM25 index and return ranked results.

    Args:
        client: Active Elasticsearch client.
        index_name: Name of the index to search.
        query_text: The query string.
        top_k: Number of top results to return.
        fields: List of fields to search. Defaults to ["full_text"].
                Valid values: "title", "text", "full_text".

    Returns:
        Ranked list of SearchResult objects, highest score first.
        Empty list if no results.

    Raises:
        ValueError: if fields contains invalid or non-searchable field names.
    """
    if fields is None:
        fields = ["full_text"]

    _validate_fields(fields)

    # Single field → match query
    # Multiple fields → multi_match query (best_fields by default)
    if len(fields) == 1:
        query_body = {
            "query": {
                "match": {
                    fields[0]: query_text,
                }
            },
            "size": top_k,
        }
    else:
        query_body = {
            "query": {
                "multi_match": {
                    "query": query_text,
                    "fields": fields,
                }
            },
            "size": top_k,
        }

    response = client.search(index=index_name, body=query_body)

    results: list[SearchResult] = []
    for hit in response["hits"]["hits"]:
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
    # Example usage
    es_client = Elasticsearch("http://localhost:9200")
    index = "scifact_bm25"
    query = "COVID-19 vaccine efficacy"
    top_k_results = 5
    search_fields = ["title", "text"]

    results = search_bm25(
        client=es_client,
        index_name=index,
        query_text=query,
        top_k=top_k_results,
        fields=search_fields,
    )

    for result in results:
        print(f"Doc ID: {result.doc_id}, Score: {result.score}, Title: {result.title}")