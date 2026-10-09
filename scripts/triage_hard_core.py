"""Deterministic triage of the 13 gold-only-refusing cases. No LLM.

For each case: does the claim's distinctive vocabulary appear in the GOLD
document, and where?  Four verdicts, four different owners:

  GOLD_IN_WINDOW       term in gold doc inside the probe window
                       -> reading/judgement failure      (generation bug)
  GOLD_OUT_OF_WINDOW   term in gold doc past char 3000
                       -> truncation beyond the probe    (context bug)
  TERM_ONLY_DISTRACTOR term only in a non-gold retrieved doc
                       -> gold doc is not the evidence   (retrieval/label)
  TERM_NOWHERE         term in no retrieved doc
                       -> claim typo / variant           (eval artefact)

difflib (stdlib fuzzy matcher) catches cases like claim "echo-domain" vs
doc "ectodomain", which a strict lexical check scores as zero overlap.
"""
import difflib
import json
import re
from pathlib import Path

from configs.loader import load_config
from ingestion.loaders import load_corpus

PROBE = Path("results/gold_only_probe.json")
CASES = Path("data/eval/ablation_generation_false_refusals.jsonl")
PROBE_WINDOW = 3000          # must match GOLD_CTX_CHARS in run_gold_only_probe.py
MIN_TERM_LEN = 5             # ignore short tokens; they match everything
STOP = {
    "the", "and", "for", "with", "from", "that", "this", "were", "are",
    "which", "their", "there", "been", "have", "has", "had", "not", "but",
    "can", "may", "will", "would", "could", "should", "does", "did",
    "into", "onto", "than", "then", "them", "they", "its", "his", "her",
    "such", "also", "only", "over", "under", "again", "once", "here",
    "where", "when", "what", "while", "about", "after", "before", "between",
    "among", "through", "during", "because", "although", "though", "unless",
    "until", "whether", "either", "neither", "both", "each", "every", "more",
    "most", "less", "least", "other", "others", "another", "same", "so",
    "some", "any", "all", "none", "one", "two", "three", "four", "five",
    "first", "second", "third", "studies", "study", "showed", "shows",
    "shown", "show", "based", "using", "used", "use", "via", "per",
}
WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-\.]*")


def norm(raw: str) -> str:
    return " ".join((raw or "").split())


def claim_terms(text: str) -> list[str]:
    seen, out = set(), []
    for w in WORD_RE.findall(norm(text)):
        if len(w) >= MIN_TERM_LEN and w not in STOP and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def find_term(term: str, text: str) -> tuple[int, str] | None:
    """Return (offset, how) for the first occurrence, word-boundary preferred."""
    m = re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text)
    if m:
        return m.start(), "exact"
    idx = text.find(term)
    return (idx, "substr") if idx >= 0 else None


def main() -> None:
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    refused_gold_only = {
        r["query_id"] for r in probe["rows"] if r["refusal"]
    }
    print(f"gold_only_refusals={len(refused_gold_only)} {sorted(refused_gold_only, key=int)}")

    cases = {}
    with CASES.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec["query_id"] in refused_gold_only:
                    cases[rec["query_id"]] = rec

    cfg = load_config()
    corpus = load_corpus(Path(cfg["data"]["corpus_path"]))

    tally = {}
    for qid in sorted(cases, key=int):
        c = cases[qid]
        terms = claim_terms(c["question"])
        gold = list(dict.fromkeys(c.get("gold_doc_ids_retrieved") or c.get("gold_doc_ids") or []))
        retrieved = list(dict.fromkeys(c.get("retrieved_doc_ids") or []))

        gold_texts = {
            d: norm(getattr(corpus.get(d), "text", "") if corpus.get(d) else "")
            for d in gold
        }
        all_texts = {
            d: norm(getattr(corpus.get(d), "text", "") if corpus.get(d) else "")
            for d in retrieved
        }
        all_vocab = set()
        for t in all_texts.values():
            all_vocab.update(WORD_RE.findall(t))

        hits = {}          # term -> (doc_id, offset, how, is_gold)
        fuzzy = {}         # term -> best near-match in corpus
        for term in terms:
            best = None
            for d, txt in all_texts.items():
                loc = find_term(term, txt)
                if loc and (best is None or (d in gold and best[3] is False)):
                    best = (d, loc[0], loc[1], d in gold)
            if best:
                hits[term] = best
                continue
            near = difflib.get_close_matches(term, all_vocab, n=1, cutoff=0.78)
            if near:
                fuzzy[term] = near[0]

        in_win = [(t, h) for t, h in hits.items() if h[3] and h[1] < PROBE_WINDOW]
        out_win = [(t, h) for t, h in hits.items() if h[3] and h[1] >= PROBE_WINDOW]
        distract = [(t, h) for t, h in hits.items() if not h[3]]

        if in_win:
            verdict = "GOLD_IN_WINDOW"
        elif out_win:
            verdict = "GOLD_OUT_OF_WINDOW"
        elif distract:
            verdict = "TERM_ONLY_DISTRACTOR"
        elif fuzzy:
            verdict = "TERM_NOWHERE_FUZZY_MATCH"
        else:
            verdict = "TERM_NOWHERE"
        tally[verdict] = tally.get(verdict, 0) + 1

        print("\n" + "=" * 72)
        print(f"qid={qid}  verdict={verdict}  category={c.get('category')}")
        print(f"CLAIM: {norm(c['question'])[:150]}")
        print(f"gold_docs={gold}  lens={{ {', '.join(f'{d}:{len(t)}' for d, t in gold_texts.items())} }}")
        print(f"terms_checked({len(terms)}): {terms[:12]}")
        for label, group in (("GOLD in-window", in_win), ("GOLD out-of-window", out_win),
                             ("distractor-only", distract)):
            for t, (d, off, how, is_gold) in group[:6]:
                print(f"  {label:<20} {t:<18} doc={d:<10} off={off:<5} {how}")
        for t, n in list(fuzzy.items())[:6]:
            print(f"  fuzzy              {t:<18} ~ {n}")
        print(f"REFUSAL: {norm(c.get('baseline_answer_text', ''))[:150]}")

    print("\n" + "=" * 72)
    print("TRIAGE SUMMARY — gold-only refusals")
    print("=" * 72)
    for v, n in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"{v:<24} {n}")
    print("=" * 72)
    print("GOLD_IN_WINDOW / GOLD_OUT_OF_WINDOW -> ours to fix (context or generation)")
    print("TERM_ONLY_DISTRACTOR / TERM_NOWHERE -> eval artefact; document, do not chase")


if __name__ == "__main__":
    main()
