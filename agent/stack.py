"""Builds the retrieval + generation stack from config.

The single place that loads the corpus, indexes and models, used by both the
API and the eval runner so they can never be wired differently. Heavy
imports (torch, faiss, transformers) happen inside build_pipeline(), keeping
`import agent.stack` cheap.
"""

import asyncio
import logging
from pathlib import Path

from agent.agent import AgentPipeline
from agent.llm import LLMClient
from agent.pipeline import RagPipeline

# Rough budget used only for a startup warning: ~3.5 chars/token for English
# scientific text, plus system prompt, question, and rescue/feedback text.
_CHARS_PER_TOKEN = 3.5
_PROMPT_OVERHEAD_TOKENS = 700


def check_context_budget(cfg: dict, logger: logging.Logger) -> None:
    """Warn at startup when the configured context cannot fit num_ctx."""
    agent, llm = cfg["agent"], cfg["llm"]
    estimate = int(
        agent["top_k"] * agent["context_chars"] / _CHARS_PER_TOKEN
        + llm["max_tokens"]
        + _PROMPT_OVERHEAD_TOKENS
    )
    if estimate > llm["num_ctx"]:
        logger.warning(
            "estimated prompt+answer size %d tokens exceeds llm.num_ctx=%d; "
            "Ollama will truncate prompts — raise llm.num_ctx or lower "
            "agent.top_k / agent.context_chars",
            estimate, llm["num_ctx"],
        )


async def build_pipeline(
    cfg: dict,
    *,
    llm: LLMClient,
    logger: logging.Logger,
    mode: str | None = None,
) -> RagPipeline:
    """Load every resource and return a ready pipeline.

    Args:
        mode: "baseline" or "agent"; defaults to cfg["agent"]["mode"].
    """
    from es_client import get_es_client
    from indexing.dense_encoder import DenseEncoder
    from indexing.faiss_indexer import FaissIndex
    from indexing.splade_encoder import SpladeEncoder
    from ingestion.loaders import load_corpus

    mode = mode or cfg["agent"]["mode"]
    check_context_budget(cfg, logger)

    corpus = await asyncio.to_thread(load_corpus, Path(cfg["data"]["corpus_path"]))
    faiss_index = await asyncio.to_thread(
        FaissIndex.load, cfg["dense"]["index_path"], cfg["dense"]["ids_path"]
    )
    dcfg, scfg = cfg["dense"], cfg["splade"]
    dense_encoder = DenseEncoder(
        model_name=dcfg["model_name"],
        batch_size=dcfg["batch_size"],
        max_seq_length=dcfg["max_seq_length"],
    )
    splade_encoder = SpladeEncoder(
        model_name=scfg["model_name"],
        top_k=scfg["top_k"],
        batch_size=scfg["batch_size"],
        max_seq_length=scfg["max_seq_length"],
    )
    # Warm up so the first request does not pay model-load latency.
    await asyncio.to_thread(dense_encoder.encode_query, "warmup")
    await asyncio.to_thread(splade_encoder.encode_query, "warmup")
    es = await asyncio.to_thread(get_es_client, cfg)

    common = {
        "cfg": cfg,
        "es": es,
        "corpus": corpus,
        "faiss_index": faiss_index,
        "dense_encoder": dense_encoder,
        "splade_encoder": splade_encoder,
        "llm": llm,
        "logger": logger,
    }
    if mode != "agent":
        return RagPipeline(**common)

    reranker = None
    if cfg["agent"]["rerank"]:
        from reranking.cross_encoder import CrossEncoderReranker
        from retrieval.types import FusedResult

        rcfg = cfg["reranker"]
        reranker = CrossEncoderReranker(
            model_name=rcfg["model_name"], max_seq_length=rcfg["max_seq_length"]
        )
        await asyncio.to_thread(
            reranker.rerank, "warmup", [FusedResult("warmup", 0.0, "", "warmup")]
        )
    return AgentPipeline(**common, reranker=reranker)
