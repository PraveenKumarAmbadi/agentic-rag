#!/usr/bin/env python3
"""Build the answerable eval set from SciFact qrels.

Samples `--size` test queries that have gold relevance judgments and writes
data/eval/answerable.jsonl. The seed is fixed so the eval set is frozen:
numbers from different runs/phases are comparable only against the same set.

Unanswerable/adversarial sets are authored by hand (see data/eval/).

Usage::
    python -m scripts.build_eval_sets --size 100
"""

import argparse
import json
import random
from pathlib import Path

from configs.loader import load_config
from ingestion.loaders import load_qrels, load_queries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    queries = load_queries(Path(cfg["data"]["queries_path"]))
    qrels = load_qrels(Path(cfg["data"]["qrels_path"]))

    # Only queries that exist and have >=1 relevant doc with score > 0.
    eligible = sorted(qid for qid in qrels if qid in queries)
    if len(eligible) < args.size:
        raise SystemExit(
            f"Only {len(eligible)} eligible queries, need {args.size}."
        )

    rng = random.Random(args.seed)
    sampled = sorted(rng.sample(eligible, args.size))

    out_path = Path(cfg["eval"]["answerable_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for qid in sampled:
            record = {
                "query_id": qid,
                "question": queries[qid].text,
                "gold_doc_ids": sorted(qrels[qid].keys()),
                "set": "answerable",
            }
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(sampled)} answerable questions to {out_path}")


if __name__ == "__main__":
    main()
