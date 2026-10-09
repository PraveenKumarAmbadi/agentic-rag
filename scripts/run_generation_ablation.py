"""Mini-ablation for the 28 generation-side false refusals.

Reuses the documents the baseline already retrieved (no retrieval stack),
so the only variables are the answer prompt and the per-doc context size.

Arms:
  A  current prompt   @ 800 chars   (sanity check vs baseline)
  B  current prompt   @ 2400 chars  (does more context alone fix refusals?)
  C  claim prompt     @ 800 chars   (does better framing alone fix refusals?)
  D  claim prompt     @ 2400 chars  (do both together fix most refusals?)

A case is "fixed" when is_refusal() is False. Under the claim prompt,
SUPPORTED and CONTRADICTED both count as answered; only the
INSUFFICIENT_EVIDENCE escape phrase counts as a refusal, so the existing
refusal detector and citation parser stay valid without edits.

Stop the serving API before running — Ollama holds one model slot.
"""
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from agent.citations import is_refusal, parse_citations
from agent.llm import LLMClient
from agent.pipeline import CallSpecs
from agent.prompts import SYSTEM_PROMPT, build_context
from configs.loader import load_config
from ingestion.loaders import load_corpus

ABLATION_PATH = Path("data/eval/ablation_generation_false_refusals.jsonl")
OUT_PATH = Path("results/ablation_generation_false_refusals.json")

CONTEXT_SMALL = 800
CONTEXT_LARGE = 2400

CLAIM_SYSTEM_PROMPT = """
You are a careful scientific fact-checker. You are given a numbered set of
documents and ONE scientific claim. Using ONLY the documents (never your own
knowledge), decide whether the documents SUPPORT the claim, CONTRADICT the
claim, or do not contain enough information either way.

Rules:
- SUPPORTED means at least one document states the claim or something that
  directly implies it.
- CONTRADICTED means at least one document states the opposite of the claim
  or something that directly implies the opposite.
- Begin your reply with exactly one token: SUPPORTED, CONTRADICTED, or
  INSUFFICIENT_EVIDENCE.
- After that token, write one short sentence of justification and cite the
  document number(s) you relied on in square brackets, e.g. [2].
- If the claim is neither supported nor contradicted, reply with exactly one
  line starting with: INSUFFICIENT_EVIDENCE: followed by a short description
  of what is missing. Do not guess.
- Be concise: at most 80 words.
""".strip()

ARMS = [
    ("A_current_800", "current", CONTEXT_SMALL),
    ("B_current_2400", "current", CONTEXT_LARGE),
    ("C_claim_800", "claim", CONTEXT_SMALL),
    ("D_claim_2400", "claim", CONTEXT_LARGE),
]


def load_cases() -> list[dict]:
    with ABLATION_PATH.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def docs_from_ids(corpus: dict, doc_ids: list[str]) -> list[SimpleNamespace]:
    # Build prompt-order doc objects with the attributes build_context needs.
    # Missing ids become empty placeholders so [n] numbering never shifts.
    out = []
    for did in doc_ids:
        d = corpus.get(did)
        out.append(
            SimpleNamespace(
                doc_id=did,
                title=(getattr(d, "title", "") or "") if d is not None else "",
                text=(getattr(d, "text", "") or "") if d is not None else "",
            )
        )
    return out


def build_msgs(question: str, docs: list, ctx_chars: int, style: str) -> list[dict]:
    context = build_context(docs, ctx_chars)
    if style == "current":
        # Mirror baseline build_messages() exactly for a faithful arm A.
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\nQuestion: {question}"},
        ]
    return [
        {"role": "system", "content": CLAIM_SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}\nClaim: {question}"},
    ]


def score(text: str, docs: list, gold: set[str]) -> dict:
    doc_ids = [d.doc_id for d in docs]
    cites = parse_citations(text, len(docs))
    cited = [doc_ids[n - 1] for n in cites["markers"]]
    cited_set = set(cited)
    return {
        "refusal": is_refusal(text),
        "cited_doc_ids": cited,
        "invalid_citations": cites["invalid"],
        "citation_precision": (
            round(len(cited_set & gold) / len(cited_set), 4) if cited_set else 0.0
        ),
        "answer_text": text,
    }


async def main() -> None:
    cfg = load_config()
    cases = load_cases()
    corpus = load_corpus(Path(cfg["data"]["corpus_path"]))
    llm = LLMClient.from_config(cfg)
    spec = CallSpecs.from_config(cfg).answer  # same policy as served answers

    results: dict[str, dict[str, dict]] = {arm: {} for arm, _, _ in ARMS}

    for arm, style, ctx in ARMS:
        print(f"\n=== ARM {arm} (style={style}, context_chars={ctx}) ===", flush=True)
        for i, case in enumerate(cases, start=1):
            docs = docs_from_ids(corpus, case["retrieved_doc_ids"])
            gold = set(case["gold_doc_ids"])
            msgs = build_msgs(case["question"], docs, ctx, style)
            text = (await llm.complete(msgs, spec)).text
            results[arm][case["query_id"]] = score(text, docs, gold)
            flag = "REFUSE" if results[arm][case["query_id"]]["refusal"] else "answer"
            print(f"  [{i:>2}/{len(cases)}] qid={case['query_id']:>5} {flag}", flush=True)

    await llm.close()

    # ---- summary ----
    cats = {c["query_id"]: c["category"] for c in cases}
    summary = {}
    for arm, _, _ in ARMS:
        rows = list(results[arm].values())
        n = len(rows)
        summary[arm] = {
            "n": n,
            "refusal_rate": round(sum(r["refusal"] for r in rows) / n, 4),
            "mean_citation_precision": round(
                sum(r["citation_precision"] for r in rows) / n, 4
            ),
            "invalid_citation_rate": round(
                sum(bool(r["invalid_citations"]) for r in rows) / n, 4
            ),
        }

    # per-category refusal counts: {category: {arm: refused/total}}
    cat_matrix = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for arm, _, _ in ARMS:
        for qid, r in results[arm].items():
            cell = cat_matrix[cats[qid]][arm]
            cell[1] += 1
            if r["refusal"]:
                cell[0] += 1

    print("\n" + "=" * 64)
    print("ABLATION SUMMARY — 28 generation-side false refusals")
    print("=" * 64)
    print(f"{'arm':<16}{'refusal_rate':>14}{'cite_prec':>12}{'invalid':>10}")
    for arm, _, _ in ARMS:
        s = summary[arm]
        print(
            f"{arm:<16}{s['refusal_rate']:>14}{s['mean_citation_precision']:>12}"
            f"{s['invalid_citation_rate']:>10}"
        )

    print("\nPer-category refusals (refused/total):")
    header = f"{'category':<30}" + "".join(f"{a.split('_')[0]:>6}" for a, _, _ in ARMS)
    print(header)
    for cat in sorted(cat_matrix):
        line = f"{cat:<30}"
        for arm, _, _ in ARMS:
            refused, total = cat_matrix[cat][arm]
            line += f"{str(refused)+'/'+str(total):>6}"
        print(line)
    print("=" * 64)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "per_category_refusals": {
                    c: {a: tuple(v) for a, v in arms.items()}
                    for c, arms in cat_matrix.items()
                },
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
