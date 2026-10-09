import json
import re
from glob import glob
from pathlib import Path
from collections import Counter

from configs.loader import load_config
from ingestion.loaders import load_corpus

RESULT_GLOB = "results/baseline_phase1_*.json"
MIN_OVERLAP = 2

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
    "though", "whereas", "whether", "unless", "until", "while"
}

WORD_RE = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> set[str]:
    return {
        word
        for word in WORD_RE.findall(text.lower())
        if len(word) > 2 and word not in STOPWORDS
    }


def split_sentences(text: str):
    # text should already be whitespace-normalized.
    for match in re.finditer(r"[^.!?]+[.!?]*(?:\s|$)", text):
        sentence = match.group().strip()
        if sentence:
            yield match.start(), sentence, match.end()


def norm_text(raw: str) -> str:
    return " ".join((raw or "").split())


result_files = sorted(glob(RESULT_GLOB))
if not result_files:
    raise SystemExit(f"No result files matched {RESULT_GLOB}")

result_path = Path(result_files[-1])
print("result_file=", result_path)

with result_path.open(encoding="utf-8") as f:
    res = json.load(f)

rows = res["results"]["answerable"]
gen_refusals = [
    row
    for row in rows
    if row.get("refusal") and row.get("gold_doc_ids_retrieved")
]

cfg = load_config()
ctx = int(cfg.get("agent", {}).get("context_chars", 800))
print("context_chars=", ctx)
print("generation_refusals=", len(gen_refusals))

corpus = load_corpus(Path(cfg["data"]["corpus_path"]))

counts = Counter()
examples = []

for row in gen_refusals:
    question = row["question"]
    question_tokens = tokens(question)
    gold_ids = row.get("gold_doc_ids_retrieved") or []

    best_before = 0
    best_after = 0
    best_cross = 0

    sent_before = ""
    sent_after = ""
    sent_cross = ""

    gid_before = ""
    gid_after = ""
    gid_cross = ""

    for gid in gold_ids:
        doc = corpus.get(gid)
        text = norm_text(getattr(doc, "text", "") if doc is not None else "")
        if not text:
            continue

        for start, sentence, end in split_sentences(text):
            score = len(question_tokens & tokens(sentence))
            if score <= 0:
                continue

            if end <= ctx:
                if score > best_before:
                    best_before = score
                    sent_before = sentence
                    gid_before = gid
            elif start >= ctx:
                if score > best_after:
                    best_after = score
                    sent_after = sentence
                    gid_after = gid
            else:
                if score > best_cross:
                    best_cross = score
                    sent_cross = sentence
                    gid_cross = gid

    best_any = max(best_before, best_after, best_cross)

    if best_any < MIN_OVERLAP:
        counts["weak_or_no_lexical_overlap"] += 1
        category = "weak"
    elif (
        best_after >= MIN_OVERLAP
        and best_after > best_before
        and best_after >= best_cross
    ):
        counts["after_800_stronger"] += 1
        category = "after_stronger"
    elif (
        best_cross >= MIN_OVERLAP
        and best_cross > best_before
        and best_cross > best_after
    ):
        counts["crosses_800_stronger"] += 1
        category = "cross_stronger"
    elif (
        best_after >= MIN_OVERLAP
        and best_before < MIN_OVERLAP
        and best_cross < MIN_OVERLAP
    ):
        counts["only_after_800"] += 1
        category = "only_after"
    elif (
        best_before >= MIN_OVERLAP
        and best_after < MIN_OVERLAP
        and best_cross < MIN_OVERLAP
    ):
        counts["before_800_only"] += 1
        category = "before_only"
    else:
        counts["mixed_before_and_after"] += 1
        category = "mixed"

    if category in {"after_stronger", "cross_stronger", "only_after", "mixed"} and len(examples) < 12:
        examples.append({
            "query_id": row.get("query_id"),
            "category": category,
            "best_before": best_before,
            "best_after": best_after,
            "best_cross": best_cross,
            "gid_after": gid_after,
            "gid_cross": gid_cross,
            "after_snippet": sent_after[:180],
            "cross_snippet": sent_cross[:180],
        })

print("counts=", dict(counts))
print(
    "truncation_suspects=",
    counts["only_after_800"]
    + counts["after_800_stronger"]
    + counts["crosses_800_stronger"]
    + counts["mixed_before_and_after"],
)
print("examples:")
for example in examples:
    print(example)
