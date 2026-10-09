"""Are the 13 all-arm-refused cases missing from the corpus text, or just
unread by the model?  Checks claim vocabulary against the FULL text of every
retrieved document (no truncation), and reports the best-matching sentence.

Category (a) -> overlap == 0 everywhere  -> corpus/label problem, needs re-search
Category (b) -> overlap > 0 in a gold doc -> model reading failure, needs
                                              an evidence-extraction step
"""
import json
import re
from pathlib import Path

from configs.loader import load_config
from ingestion.loaders import load_corpus

ABLATION = Path("results/ablation_generation_false_refusals.json")
CASES = Path("data/eval/ablation_generation_false_refusals.jsonl")

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "by", "for", "to", "with",
    "without", "within", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "and", "or", "not", "no", "nor", "but", "if", "then",
    "than", "that", "this", "these", "those", "there", "here", "where",
    "when", "which", "who", "whom", "whose", "what", "while", "during",
    "after", "before", "between", "among", "through", "over", "under",
    "again", "further", "once", "about", "against", "into", "onto", "via",
    "per", "plus", "minus", "have", "has", "had", "having", "do", "does",
    "did", "doing", "can", "could", "may", "might", "must", "shall",
    "should", "will", "would", "also", "such", "more", "most", "less",
    "least", "very", "too", "quite", "rather", "somewhat", "only",
    "own", "same", "other", "others", "another", "one", "two", "three",
    "four", "five", "six", "seven", "eight", "nine", "ten",
    "show", "shows", "shown", "showing", "demonstrate", "demonstrates",
    "demonstrated", "demonstrating", "indicate", "indicates", "indicated",
    "indicating", "suggest", "suggests", "suggested", "suggesting",
    "report", "reports", "reported", "reporting", "found", "find",
    "finds", "finding", "results", "result", "study", "studies",
    "paper", "papers", "article", "articles", "abstract", "abstracts",
    "conclusion", "conclusions", "method", "methods", "background",
    "objective", "objectives", "purpose", "purposes", "however",
    "therefore", "thus", "hence", "because", "since", "although",
    "though", "whereas", "whether", "unless", "until", "while",
}
WORD_RE = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> set[str]:
    return {w for w in WORD_RE.findall(text.lower()) if len(w) > 2 and w not in STOPWORDS}


def norm(raw: str) -> str:
    return " ".join((raw or "").split())


def sentences(text: str) -> list[str]:
    out = []
    for m in re.finditer(r"[^.!?]+[.!?]*(?:\s|$)", text):
        s = m.group().strip()
        if s:
            out.append(s)
    return out


def main() -> None:
    abl = json.loads(ABLATION.read_text(encoding="utf-8"))
    arms = list(abl["results"].keys())

    cases = {}
    with CASES.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                cases[rec["query_id"]] = rec

    resistant = [
        qid for qid in cases
        if all(abl["results"][arm][qid]["refusal"] for arm in arms)
    ]
    print(f"arms={arms}")
    print(f"resistant_count={len(resistant)} / {len(cases)}")
    print(f"resistant_qids={resistant}")

    cfg = load_config()
    corpus = load_corpus(Path(cfg["data"]["corpus_path"]))

    verdict: dict[str, list[str]] = {"absent_from_corpus": [], "present_but_unread": []}

    for qid in resistant:
        c = cases[qid]
        claim_tokens = tokens(c["question"])
        gold = set(c.get("gold_doc_ids_retrieved") or [])
        print("\n" + "=" * 70)
        print(f"qid={qid} category={c.get('category')}")
        print(f"CLAIM: {norm(c['question'])[:180]}")
        print(f"claim_terms({len(claim_tokens)}): {sorted(claim_tokens)[:14]}")

        best_gold_score = 0
        best_gold_sent = ""
        any_hit = False

        for rank, did in enumerate(c.get("retrieved_doc_ids") or [], start=1):
            doc = corpus.get(did)
            text = norm(getattr(doc, "text", "") if doc is not None else "")
            hits = claim_tokens & tokens(text)
            tag = "GOLD" if did in gold else "    "
            if hits:
                any_hit = True
                best_s, best_sc = "", 0
                for s in sentences(text):
                    sc = len(claim_tokens & tokens(s))
                    if sc > best_sc:
                        best_sc, best_s = sc, s
                print(
                    f"  [{rank:>2}] {tag} {did:>10} len={len(text):>5} "
                    f"overlap={len(hits):>2} top_sent_score={best_sc}"
                )
                print(f"        terms: {sorted(hits)[:10]}")
                if did in gold and best_sc > best_gold_score:
                    best_gold_score, best_gold_sent = best_sc, best_s
            else:
                print(
                    f"  [{rank:>2}] {tag} {did:>10} len={len(text):>5} overlap= 0"
                )

        if best_gold_score > 0:
            verdict["present_but_unread"].append(qid)
            print(f"  -> GOLD HAS EVIDENCE (score={best_gold_score}):")
            print(f"     {best_gold_sent[:240]}")
        elif any_hit:
            verdict["present_but_unread"].append(qid)
            print("  -> term present in a NON-gold retrieved doc only")
        else:
            verdict["absent_from_corpus"].append(qid)
            print("  -> VERDICT: claim vocabulary ABSENT from all 10 retrieved docs")

    print("\n" + "=" * 70)
    print("VERDICT SUMMARY")
    print("=" * 70)
    print(f"absent_from_corpus   = {len(verdict['absent_from_corpus'])} "
          f"{verdict['absent_from_corpus']}")
    print(f"present_but_unread   = {len(verdict['present_but_unread'])} "
          f"{verdict['present_but_unread']}")


if __name__ == "__main__":
    main()
