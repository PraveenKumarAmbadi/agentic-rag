#!/usr/bin/env python3
"""Build every search index the API needs from the SciFact corpus.

  1. SciFact corpus             downloaded if missing (scripts.prepare_data)
  2. BM25 index                 Elasticsearch, bm25.index_name
  3. SPLADE index               Elasticsearch, splade.index_name (encodes every doc)
  4. Dense index                FAISS HNSW, dense.index_path + dense.ids_path

Idempotent: an index counts as built only if it holds exactly one entry per
corpus document. Anything else — missing, or partial from an interrupted
run — is rebuilt from scratch, so re-running after a failure is always safe.
``--recreate`` forces a rebuild of all three.

Runtime: about a minute with a GPU; 10-30 minutes CPU-only (SPLADE encoding
dominates). Models are downloaded from Hugging Face on first use.

Usage::
    python -m scripts.build_indexes [--recreate]
"""

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path

from configs.loader import load_config
from scripts.prepare_data import ensure_scifact

logger = logging.getLogger("build_indexes")

ES_CONNECT_ATTEMPTS = 30
ES_CONNECT_DELAY_S = 2.0
BULK_CHUNK = 500
SPLADE_ENCODE_CHUNK = 256
# SPLADE logits are [batch, seq, ~30k vocab] floats: ~0.5 GB per 8 docs of
# 512 tokens. GPUs take the configured batch; on CPU, RAM is shared with
# Elasticsearch and the OS, so keep the batch small.
SPLADE_CPU_MAX_BATCH = 8


class IndexBuildError(RuntimeError):
    pass


def _connect_es(cfg: dict):
    """Elasticsearch may still be starting (e.g. under docker compose)."""
    from es_client import get_es_client

    for attempt in range(1, ES_CONNECT_ATTEMPTS + 1):
        try:
            return get_es_client(cfg)
        except ConnectionError as exc:
            if attempt == ES_CONNECT_ATTEMPTS:
                raise IndexBuildError(str(exc)) from exc
            logger.info("waiting for Elasticsearch (%d/%d) ...", attempt, ES_CONNECT_ATTEMPTS)
            time.sleep(ES_CONNECT_DELAY_S)


def _es_index_complete(es, name: str, expected: int) -> bool:
    if not es.indices.exists(index=name):
        return False
    es.indices.refresh(index=name)
    count = es.count(index=name)["count"]
    if count == expected:
        return True
    logger.warning("index %s holds %d of %d documents; rebuilding", name, count, expected)
    return False


def _rebuild_es_index(es, name: str, mappings: dict, actions: Iterator[dict], expected: int) -> None:
    from elasticsearch.helpers import streaming_bulk

    if es.indices.exists(index=name):
        es.indices.delete(index=name)
    es.indices.create(index=name, mappings=mappings)
    indexed, failures = 0, []
    for ok, info in streaming_bulk(
        es, actions, chunk_size=BULK_CHUNK, max_retries=3, initial_backoff=1,
        raise_on_error=False, raise_on_exception=True,
    ):
        if ok:
            indexed += 1
        else:
            failures.append(info)
    if failures:
        raise IndexBuildError(f"{len(failures)} documents failed to index into {name}: {failures[:3]}")
    es.indices.refresh(index=name)
    count = es.count(index=name)["count"]
    if count != expected:
        raise IndexBuildError(f"{name}: expected {expected} documents, found {count}")
    logger.info("index %s built: %d documents", name, count)


def build_bm25(es, cfg: dict, docs: list, recreate: bool) -> str:
    from indexing.mapping import SCIFACT_BM25_MAPPING

    name = cfg["bm25"]["index_name"]
    if not recreate and _es_index_complete(es, name, len(docs)):
        return "present"
    actions = (
        {"_index": name, "_id": d.doc_id,
         "_source": {"doc_id": d.doc_id, "title": d.title, "text": d.text}}
        for d in docs
    )
    _rebuild_es_index(es, name, SCIFACT_BM25_MAPPING, actions, len(docs))
    return "built"


def build_splade(es, cfg: dict, docs: list, recreate: bool) -> str:
    from indexing.mapping import SCIFACT_SPLADE_MAPPING
    from indexing.splade_encoder import SpladeEncoder

    name = cfg["splade"]["index_name"]
    if not recreate and _es_index_complete(es, name, len(docs)):
        return "present"
    import torch

    scfg = cfg["splade"]
    batch_size = scfg["batch_size"]
    if not torch.cuda.is_available() and batch_size > SPLADE_CPU_MAX_BATCH:
        logger.info("no GPU: SPLADE batch size %d -> %d to bound RAM", batch_size, SPLADE_CPU_MAX_BATCH)
        batch_size = SPLADE_CPU_MAX_BATCH
    encoder = SpladeEncoder(
        model_name=scfg["model_name"], top_k=scfg["top_k"],
        batch_size=batch_size, max_seq_length=scfg["max_seq_length"],
    )

    def actions() -> Iterator[dict]:
        for start in range(0, len(docs), SPLADE_ENCODE_CHUNK):
            chunk = docs[start:start + SPLADE_ENCODE_CHUNK]
            vectors = encoder.encode_texts([d.full_text for d in chunk])
            logger.info("SPLADE encoded %d/%d documents", start + len(chunk), len(docs))
            for d, vec in zip(chunk, vectors):
                yield {
                    "_index": name, "_id": d.doc_id,
                    "_source": {
                        "doc_id": d.doc_id, "title": d.title, "text": d.text,
                        # sparse_vector fields need string keys
                        "sparse_vector": {str(k): v for k, v in vec.items()},
                    },
                }

    _rebuild_es_index(es, name, SCIFACT_SPLADE_MAPPING, actions(), len(docs))
    return "built"


def _dense_index_complete(index_path: Path, ids_path: Path, doc_ids: set[str]) -> bool:
    if not (index_path.is_file() and ids_path.is_file()):
        return False
    try:
        stored = json.loads(ids_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if len(stored) == len(doc_ids) and set(stored) == doc_ids:
        return True
    logger.warning("dense index does not match the corpus; rebuilding")
    return False


def build_dense(cfg: dict, docs: list, recreate: bool) -> str:
    dcfg = cfg["dense"]
    index_path, ids_path = Path(dcfg["index_path"]), Path(dcfg["ids_path"])
    if not recreate and _dense_index_complete(index_path, ids_path, {d.doc_id for d in docs}):
        return "present"

    from indexing.dense_encoder import DenseEncoder
    from indexing.faiss_indexer import FaissIndex

    encoder = DenseEncoder(
        model_name=dcfg["model_name"], batch_size=dcfg["batch_size"],
        max_seq_length=dcfg["max_seq_length"],
    )
    vectors = encoder.encode_texts([d.full_text for d in docs], show_progress_bar=False)
    index = FaissIndex.build(
        vectors=vectors, doc_ids=[d.doc_id for d in docs],
        M=dcfg["hnsw"]["M"], ef_construction=dcfg["hnsw"]["ef_construction"],
    )
    # Write next to the targets, then rename: readers never see half a file.
    tmp_index, tmp_ids = index_path.with_suffix(".tmp"), ids_path.with_suffix(".tmp")
    index.save(tmp_index, tmp_ids)
    os.replace(tmp_index, index_path)
    os.replace(tmp_ids, ids_path)
    logger.info("dense index built: %d vectors -> %s", len(docs), index_path)
    return "built"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--recreate", action="store_true", help="rebuild every index")
    args = parser.parse_args()

    from tracing import configure_logging

    configure_logging(json_format=False)
    started = time.perf_counter()
    try:
        cfg = load_config()
        from ingestion.loaders import load_corpus

        corpus = load_corpus(ensure_scifact(Path(cfg["data"]["corpus_path"])))
        docs = list(corpus.values())
        logger.info("corpus: %d documents", len(docs))
        es = _connect_es(cfg)
        status = {
            "bm25": build_bm25(es, cfg, docs, args.recreate),
            "splade": build_splade(es, cfg, docs, args.recreate),
            "dense": build_dense(cfg, docs, args.recreate),
        }
    except Exception as exc:
        logger.error("index build failed: %s", exc, exc_info=not isinstance(exc, IndexBuildError))
        return 1
    logger.info("indexes ready in %.0fs: %s", time.perf_counter() - started, status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
