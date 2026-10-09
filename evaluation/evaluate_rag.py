#!/usr/bin/env python3
"""RAG eval runner — replays eval sets through the shared pipeline.

Metrics (per set):
  answerable    citation_precision (|cited ∩ gold| / |cited|; 0 if no
                citations; a refusal cites nothing — the "closest documents"
                its text lists are not answer citations), retrieval_hit_rate
                (any gold doc in context), false_refusal_rate,
                invalid_citation_rate, latency percentiles (nearest rank)
  unanswerable  refusal_correctness (did it use the escape hatch), latency
  agent mode    + iterations/llm_calls means, rescue-override rate,
                critique pass rate, hedge rate
  all           + n_errors (questions that raised; excluded from metrics),
                token usage means

Every finished question is appended to results/<tag>_<ts>.partial.jsonl
as it completes, so a crash or Ctrl-C keeps the work done so far. The final
results/<tag>_<ts>.json (summary + per-question records) replaces it.
The run aborts after --max-consecutive-errors failures in a row: that is an
outage (Ollama/ES down), not a property of the questions.

IMPORTANT: stop the API server before running — both processes cannot hold
the LLM in 8 GB VRAM at once.

Usage::
    python -m evaluation.evaluate_rag --mode baseline --tag baseline_phase1
    python -m evaluation.evaluate_rag --mode agent --tag agent_phase3
"""

import argparse
import asyncio
import json
import logging
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

from agent.llm import LLMClient
from agent.stack import build_pipeline
from configs.loader import load_config
from tracing import configure_logging

logger = logging.getLogger("agentic_rag.eval")


class EvalAborted(RuntimeError):
    """Too many consecutive failures: the stack is down, not the questions hard."""


def load_set(path: str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile: the smallest value with >= p% of data at or below it."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(len(ordered) * p / 100))
    return round(ordered[min(rank, len(ordered)) - 1], 1)


def score_row(rec: dict, done: dict) -> dict:
    """Per-question record from the pipeline's done event."""
    gold = set(rec.get("gold_doc_ids", []))
    all_cited = list(done["citations"].values())
    # A refusal's text names the closest documents ("[1] Title; ...") — that
    # is not answering with them, and the baseline's refusals cite nothing.
    cited = [] if done["refusal"] else all_cited
    cited_set = set(cited)
    return {
        "query_id": rec.get("query_id", ""),
        "topic": rec.get("topic", ""),
        "question": rec["question"],
        "refusal": done["refusal"],
        "cited_doc_ids": cited,
        "refusal_listed_doc_ids": all_cited if done["refusal"] else [],
        "invalid_citations": done["invalid_citations"],
        "gold_doc_ids": sorted(gold),
        "citation_precision": (
            round(len(cited_set & gold) / len(cited_set), 4) if cited_set else 0.0
        ),
        "gold_doc_ids_retrieved": sorted(set(done["retrieved_doc_ids"]) & gold),
        "retrieved_doc_ids": done["retrieved_doc_ids"],
        "answer_text": done["text"],
        "timings": done["timings"],
        "usage": done.get("usage"),
        "degraded": done.get("degraded", []),
        "truncated": done.get("truncated", False),
        "agent": done.get("agent"),
    }


async def eval_set(
    pipe,
    records: list[dict],
    set_name: str = "",
    checkpoint: IO[str] | None = None,
    max_consecutive_errors: int = 5,
) -> list[dict]:
    rows = []
    consecutive_errors = 0
    for i, rec in enumerate(records, start=1):
        try:
            done = None
            async for ev in pipe.answer_stream(rec["question"]):
                if ev["type"] == "done":
                    done = ev
            if done is None:
                raise RuntimeError("pipeline finished without a done event")
            row = score_row(rec, done)
            consecutive_errors = 0
        except Exception as exc:
            consecutive_errors += 1
            logger.error(
                "question %s failed", rec.get("query_id", i),
                exc_info=True, extra={"set": set_name},
            )
            row = {
                "query_id": rec.get("query_id", ""),
                "question": rec["question"],
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        if checkpoint is not None:
            checkpoint.write(json.dumps({"set": set_name, **row}) + "\n")
            checkpoint.flush()
        if consecutive_errors >= max_consecutive_errors:
            raise EvalAborted(
                f"{consecutive_errors} consecutive failures in '{set_name}' — "
                "is Ollama/Elasticsearch up? Partial results were kept."
            )
        if i % 10 == 0:
            print(f"  progress: {i}/{len(records)}", flush=True)
    return rows


def _mean(values) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def summarize(rows: list[dict], has_gold: bool) -> dict:
    ok = [r for r in rows if "error" not in r]
    totals = [r["timings"]["total_ms"] for r in ok]
    s: dict = {
        "n": len(ok),
        "n_errors": len(rows) - len(ok),
        "p50_total_ms": percentile(totals, 50),
        "p95_total_ms": percentile(totals, 95),
    }
    if has_gold:
        s["citation_precision"] = round(_mean(r["citation_precision"] for r in ok), 4)
        s["false_refusal_rate"] = round(_mean(r["refusal"] for r in ok), 4)
        s["retrieval_hit_rate"] = round(_mean(bool(r["gold_doc_ids_retrieved"]) for r in ok), 4)
        s["invalid_citation_rate"] = round(_mean(bool(r["invalid_citations"]) for r in ok), 4)
    else:
        s["refusal_correctness"] = round(_mean(r["refusal"] for r in ok), 4)
    s["degraded_rate"] = round(_mean(bool(r.get("degraded")) for r in ok), 4)
    s["truncated_rate"] = round(_mean(bool(r.get("truncated")) for r in ok), 4)
    usages = [r["usage"] for r in ok if r.get("usage")]
    if usages:
        s["prompt_tokens_mean"] = round(_mean(u["prompt_tokens"] for u in usages), 1)
        s["completion_tokens_mean"] = round(_mean(u["completion_tokens"] for u in usages), 1)
    ags = [r["agent"] for r in ok if r.get("agent")]
    if ags:
        s["agent_iterations_mean"] = round(_mean(a["iterations"] for a in ags), 2)
        s["llm_calls_mean"] = round(_mean(a["llm_calls"] for a in ags), 2)
        s["rescue_override_rate"] = round(
            _mean(bool(a["rescue"] and a["rescue"]["overridden"]) for a in ags), 4
        )
        s["critique_pass_rate"] = round(
            _mean(bool(a["critique"] and a["critique"]["supported"] is True) for a in ags), 4
        )
        s["hedge_rate"] = round(_mean(a["hedged"] for a in ags), 4)
    return s


async def main_async(args) -> None:
    configure_logging()
    cfg = load_config()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = Path(cfg["eval"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{args.tag}_{stamp}.json"
    partial_path = results_dir / f"{args.tag}_{stamp}.partial.jsonl"

    logger.info("loading retrieval stack ...")
    llm = LLMClient.from_config(cfg)
    results: dict[str, list[dict]] = {}
    summary: dict[str, dict] = {}
    try:
        pipe = await build_pipeline(cfg, llm=llm, logger=logging.getLogger("agentic_rag"), mode=args.mode)
        with partial_path.open("a", encoding="utf-8") as checkpoint:
            for set_name in args.sets.split(","):
                records = load_set(cfg["eval"][f"{set_name}_path"])
                if args.limit:
                    records = records[: args.limit]
                print(f"Evaluating '{set_name}' ({len(records)} questions) ...", flush=True)
                rows = await eval_set(
                    pipe, records, set_name, checkpoint, args.max_consecutive_errors
                )
                results[set_name] = rows
                has_gold = bool(records and records[0].get("gold_doc_ids"))
                summary[set_name] = summarize(rows, has_gold)
    finally:
        await llm.close()

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"tag": args.tag, "mode": args.mode, "summary": summary, "results": results},
            f,
            indent=2,
        )
    partial_path.unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print(f"EVAL — {args.tag} (mode={args.mode})")
    print("=" * 60)
    for set_name, s in summary.items():
        print(f"\n  {set_name.upper()}:")
        for k, v in s.items():
            print(f"    {k:24} {v}")
    print("=" * 60)
    print(f"Saved: {out_path}")


def main() -> None:
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
    parser.add_argument(
        "--max-consecutive-errors", type=int, default=5,
        help="abort after this many failures in a row (an outage)",
    )
    args = parser.parse_args()
    if args.mode is None:
        args.mode = load_config()["agent"]["mode"]
    if args.tag is None:
        args.tag = args.mode
    try:
        asyncio.run(main_async(args))
    except EvalAborted as exc:
        raise SystemExit(f"Eval aborted: {exc}") from None


if __name__ == "__main__":
    main()
