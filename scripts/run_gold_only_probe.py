"""Decisive probe for the 13 all-arm-refused generation cases.

Re-run the claim-verification prompt with ONLY the gold document(s) in
context (full text, no distractors). This separates three causes that the
lexical check alone cannot:

  answer + real quote  -> evidence was usable; 9 distractor docs buried it
                          => fix = extractive context, NOT re-search
  still refuses        -> gold doc has no derivable support
                          => label noise / reasoning ceiling (document it)
  answers, no quote    -> parametric hallucination
                          => grounding prompt too weak on its own

`quoted_from_gold` is a deterministic substring test (>=40 chars of the
answer present in the gold text); it flags hallucination without an LLM judge.
Stop the serving API first (Ollama holds one model slot).
"""
import asyncio
import json
import re
from pathlib import Path

from agent.citations import is_refusal, parse_citations
from agent.llm import LLMClient
from agent.pipeline import CallSpecs
from agent.prompts import build_context
from configs.loader import load_config
from ingestion.loaders import load_corpus
from scripts.run_generation_ablation import CLAIM_SYSTEM_PROMPT  # reuse arm D prompt

ABLATION = Path("results/ablation_generation_false_refusals.json")
CASES = Path("data/eval/ablation_generation_false_refusals.jsonl")
OUT = Path("results/gold_only_probe.json")

GOLD_CTX_CHARS = 3000   # > median gold length (1282), so full abstract fits
QUOTE_MIN = 40          # min answer span that counts as a real extraction


def norm(s: str) -> str:
    return " ".join((s or "").split()).lower()


def quoted_from_gold(answer: str, gold_text: str) -> tuple[bool, str]:
    """Longest-prefix scan: does any >=QUOTE_MIN char window of the answer
    appear verbatim in the gold doc?  Deterministic hallucination check."""
    a, g = norm(answer), norm(gold_text)
    if len(a) < QUOTE_MIN:
        return False, ""
    for i in range(0, len(a) - QUOTE_MIN + 1):
        span = a[i:i + QUOTE_MIN]
        if span in g:
            return True, span
    return False, ""


def load_resistant() -> list[dict]:
    abl = json.loads(ABLATION.read_text(encoding="utf-8"))
    arms = list(abl["results"].keys())
    resistant = {
        qid for qid in abl["results"][arms[0]]
        if all(abl["results"][arm][qid]["refusal"] for arm in arms)
    }
    cases = {}
    with CASES.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["query_id"] in resistant:
                    cases[r["query_id"]] = r
    return [cases[q] for q in sorted(resistant, key=int)]


async def main() -> None:
    cfg = load_config()
    corpus = load_corpus(Path(cfg["data"]["corpus_path"]))
    llm = LLMClient.from_config(cfg)
    spec = CallSpecs.from_config(cfg).answer  # same policy as served answers
    cases = load_resistant()
    print(f"resistant_cases={len(cases)}  qids={[c['query_id'] for c in cases]}\n")

    rows = []
    for c in cases:
        gold_ids = c.get("gold_doc_ids_retrieved") or c.get("gold_doc_ids") or []
        docs = []
        for did in gold_ids:
            d = corpus.get(did)
            docs.append(type("D", (), {
                "doc_id": did,
                "title": (getattr(d, "title", "") or "") if d else "",
                "text": (getattr(d, "text", "") or "") if d else "",
            })())
        gold_text = " ".join(norm(d.text) for d in docs)

        # Reuse the exact arm-D builder so the ONLY change is doc count.
        from types import SimpleNamespace
        sd = [SimpleNamespace(doc_id=d.doc_id, title=d.title, text=d.text) for d in docs]
        msgs = [
            {"role": "system", "content": CLAIM_SYSTEM_PROMPT},
            {"role": "user", "content": f"{build_context(sd, GOLD_CTX_CHARS)}\nClaim: {c['question']}"},
        ]
        text = (await llm.complete(msgs, spec)).text
        cites = parse_citations(text, len(sd))
        refused = is_refusal(text)
        q, span = quoted_from_gold(text, gold_text)
        rows.append({
            "query_id": c["query_id"],
            "category": c.get("category"),
            "n_gold_docs": len(sd),
            "refusal": refused,
            "cited_any": bool(cites["markers"]),
            "quoted_from_gold": q,
            "answer_text": text,
        })
        verdict = (
            "REFUSES_WITH_GOLD_ONLY" if refused
            else ("REAL_EXTRACTION" if q else "HALLUCINATED_OR_UNQUOTED")
        )
        print(f"qid={c['query_id']:>5} gold_docs={len(sd)} {verdict:<26} "
              f"refusal={refused} quoted={q}")
        print(f"   CLAIM: {c['question'][:110]}")
        print(f"   ANS  : {text[:300].replace(chr(10),' ')}")
        if span:
            print(f"   QUOTE: ...{span[:120]}...")
        print()

    await llm.close()

    n = len(rows)
    fixed = sum(not r["refusal"] for r in rows)
    extracted = sum(r["quoted_from_gold"] and not r["refusal"] for r in rows)
    halluc = sum((not r["refusal"]) and (not r["quoted_from_gold"]) for r in rows)
    still = sum(r["refusal"] for r in rows)

    print("=" * 64)
    print("GOLD-ONLY PROBE — 13 resistant cases (arm D was 13/13 refused)")
    print("=" * 64)
    print(f"no_longer_refuses = {fixed}/{n}")
    print(f"  real_extraction = {extracted}/{n}   -> distractor dilution; fix = extractive context")
    print(f"  hallucinated    = {halluc}/{n}   -> prompt too weak; fix = grounding/few-shot")
    print(f"still_refuses     = {still}/{n}   -> label noise / reasoning ceiling; document it")
    print("=" * 64)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows, "n": n,
                               "fixed": fixed, "real_extraction": extracted,
                               "hallucinated": halluc, "still_refuses": still},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
