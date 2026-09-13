"""Elasticsearch index mapping for the SciFact BM25 index."""

SCIFACT_BM25_MAPPING: dict = {
    "properties": {
        "doc_id": {"type": "keyword"},
        "title": {"type": "text", "copy_to": "full_text"},
        "text": {"type": "text", "copy_to": "full_text"},
        "full_text": {"type": "text"},
    }
}

SCIFACT_SPLADE_MAPPING: dict = {
    "properties": {
        "doc_id": {"type": "keyword"},
        "title": {"type": "text"},
        "text": {"type": "text"},
        "sparse_vector": {
            "type": "sparse_vector",
        },
    }
}