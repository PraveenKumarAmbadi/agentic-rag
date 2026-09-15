#!/usr/bin/env python3
"""RAG eval runner — replays eval sets through the shared pipeline.

Metrics (per set):
  answerable    citation_precision (|cited ∩ gold| / |cited|, 0 if no
                citations), retrieval_hit_rate (any gold doc in context),
                false_refusal_rate, invalid_citation_rate, latency percentiles
  unanswerable  refusal_correctness (did it use the escape hatch), latency
  agent mode    + iterations/llm_calls means, rescue-override rate,
                critique pass rate, hedge rate

Results are written to results/<tag>_<timestamp>.json including per-question
records for error analysis. IMPORTANT: stop the API server before running —
both processes cannot hold the LLM in 8 GB VRAM at once.

Usage::
    python -m evaluation.evaluate_rag --mode baseline --tag baseline_phase1
    python -m evaluation.evaluate_rag --mode agent --tag agent_phase3
"""

import argparse
import asyncio
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from agent.agent import AgentPipeline
from agent.llm import LLMClient
from agent.pipeline import RagPipeline
from configs.loader import load_config
from es_client import get_es_client
from indexing.dense_encoder import DenseEncoder
from indexing.faiss_indexer import FaissIndex
from indexing.splade_encoder import SpladeEncoder
from ingestion.loaders import load_corpus
from tracing import JsonFormatter, setup_logger


def load_set(path: str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return round(sorted(values)[int(len(values) * p / 100) - 1 if p < 100 else -1], 1)


async def eval_set(pipe, records: list[dict]) -> list[dict]:
    rows = []
    for i, rec in enumerate(records, start=1):
        done = None
        async for ev in pipe.answer_stream(rec["question"]):
            if ev["type"] == "done":
                done = ev
        cited = list(done["citations"].values())
        gold = set(rec.get("gold_doc_ids", []))
        cited_set = set(cited)
        rows.append(
            {
                "query_id": rec.get("query_id", ""),
                "topic": rec.get("topic", ""),
                "question": rec["question"],
                "refusal": done["refusal"],
                "cited_doc_ids": cited,
                "invalid_citations": done["invalid_citations"],
                "gold_doc_ids": sorted(gold),
                "citation_precision": (
                    round(len(cited_set & gold) / len(cited_set), 4)
                    if cited_set
                    else 0.0
                ),
                "gold_doc_ids_retrieved": sorted(set(done["retrieved_doc_ids"]) & gold),
                "retrieved_doc_ids": done["retrieved_doc_ids"],
                "answer_text": done["text"],
                "timings": done["timings"],
                "agent": done.get("agent"),
            }
        )
        if i % 10 == 0:
            print(f"  progress: {i}/{len(records)}", flush=True)
    return rows


def summarize(rows: list[dict], has_gold: bool) -> dict:
    totals = [r["timings"]["total_ms"] for r in rows]
    s: dict = {
        "n": len(rows),
        "p50_total_ms": percentile(totals, 50),
        "p95_total_ms": percentile(totals, 95),
    }
    if has_gold:
        s["citation_precision"] = round(
            statistics.mean(r["citation_precision"] for r in rows), 4
        )
        s["false_refusal_rate"] = round(
            statistics.mean(r["refusal"] for r in rows), 4
        )
        s["retrieval_hit_rate"] = round(
            statistics.mean(bool(r["gold_doc_ids_retrieved"]) for r in rows), 4
        )
        s["invalid_citation_rate"] = round(
            statistics.mean(bool(r["invalid_citations"]) for r in rows), 4
        )
    else:
        s["refusal_correctness"] = round(
            statistics.mean(r["refusal"] for r in rows), 4
        )
    if rows and rows[0].get("agent"):
        ags = [r["agent"] for r in rows]
        s["agent_iterations_mean"] = round(
            statistics.mean(a["iterations"] for a in ags), 2
        )
        s["llm_calls_mean"] = round(
            statistics.mean(a["llm_calls"] for a in ags), 2
        )
        s["rescue_override_rate"] = round(
            statistics.mean(
                bool(a["rescue"] and a["rescue"]["overridden"]) for a in ags
            ),
            4,
        )
        s["critique_pass_rate"] = round(
            statistics.mean(
                bool(a["critique"] and a["critique"]["supported"]) for a in ags
            ),
            4,
        )
        s["hedge_rate"] = round(statistics.mean(a["hedged"] for a in ags), 4)
    return s


async def main_async(args) -> None:
    cfg = load_config()
    logger = setup_logger(name="eval_rag", formatter=JsonFormatter())
    loop = asyncio.get_running_loop()

    logger.info("Loading retrieval stack ...")
    corpus = await loop.run_in_executor(
        None, lambda: load_corpus(Path(cfg["data"]["corpus_path"]))
    )
    faiss_index = await loop.run_in_executor(
        None,
        lambda: FaissIndex.load(cfg["dense"]["index_path"], cfg["dense"]["ids_path"]),
    )
    dense_encoder = DenseEncoder()
    splade_encoder = SpladeEncoder()
    await loop.run_in_executor(None, lambda: dense_encoder.encode_query("warmup"))
    await loop.run_in_executor(None, lambda: splade_encoder.encode_query("warmup"))
    es = get_es_client()
    llm = LLMClient.from_config(cfg)
    pipe_cls = AgentPipeline if args.mode == "agent" else RagPipeline
    pipe = pipe_cls(
        cfg=cfg,
        es=es,
        corpus=corpus,
        faiss_index=faiss_index,
        dense_encoder=dense_encoder,
        splade_encoder=splade_encoder,
        llm=llm,
        logger=logger,
    )

    results: dict[str, list[dict]] = {}
    summary: dict[str, dict] = {}
    for set_name in args.sets.split(","):
        records = load_set(cfg["eval"][f"{set_name}_path"])
        if args.limit:
            records = records[: args.limit]
        print(f"Evaluating '{set_name}' ({len(records)} questions) ...", flush=True)
        rows = await eval_set(pipe, records)
        results[set_name] = rows
        has_gold = bool(records and records[0].get("gold_doc_ids"))
        summary[set_name] = summarize(rows, has_gold)
    await llm.close()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = Path(cfg["eval"]["results_dir"]) / f"{args.tag}_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"tag": args.tag, "mode": args.mode, "summary": summary, "results": results},
            f,
            indent=2,
        )

    print("\n" + "=" * 60)
    print(f"EVAL — {args.tag} (mode={args.mode})")
    print("=" * 60)
    for set_name, s in summary.items():
        print(f"\n  {set_name.upper()}:")
        for k, v in s.items():
            print(f"    {k:24} {v}")
    print("=" * 60)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", default="answerable,unanswerable")
    parser.add_argument("--limit", type=int, default=0, help="0 = full sets")
    parser.add_argument(
        "--mode",
        default=None,
        choices=["baseline", "agent"],
        help="pipeline to evaluate (default: cfg agent.mode)",
    )
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    if args.mode is None:
        cfg_for_mode = load_config()
        args.mode = cfg_for_mode["agent"].get("mode", "baseline")
    if args.tag is None:
        args.tag = args.mode
    asyncio.run(main_async(args))
