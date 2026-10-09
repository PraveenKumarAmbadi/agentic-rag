import json
import re
import random
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path

from configs.loader import load_config
from ingestion.loaders import load_corpus


RESULT_GLOB = "results/baseline_phase1_*.json"
ANSWERABLE_PATH_KEY = "answerable_path"

OUT_JSONL = Path("data/eval/ablation_generation_false_refusals.jsonl")
OUT_META = Path("results/ablation_case_selection_generation_false_refusals.json")
OUT_IDS = Path("data/eval/ablation_generation_false_refusals_query_ids.txt")

SEED = 42
TARGET_TOTAL = 28
MIN_OVERLAP = 2

QUOTAS = {
    "after_800_stronger": 7,
    "crosses_800_stronger": 2,
    "mixed_before_and_after": 10,
    "before_800_only": 4,
    "weak_or_no_lexical_overlap": 5,
}

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
    return {
        word
        for word in WORD_RE.findall(text.lower())
        if len(word) > 2 and word not in STOPWORDS
    }


def norm_text(raw: str) -> str:
    return " ".join((raw or "").split())


def split_sentences(text: str):
    # Returns (start, sentence, end) using character offsets in normalized text.
    for match in re.finditer(r"[^.!?]+[.!?]*(?:\s|$)", text):
        sentence = match.group().strip()
        if sentence:
            yield match.start(), sentence, match.end()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def classify_row(row: dict, corpus: dict, context_chars: int) -> dict:
    question = row.get("question", "")
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

    doc_lengths = {}

    for gid in gold_ids:
        doc = corpus.get(gid)
        text = norm_text(getattr(doc, "text", "") if doc is not None else "")
        if not text:
            continue

        doc_lengths[gid] = len(text)

        for start, sentence, end in split_sentences(text):
            score = len(question_tokens & tokens(sentence))
            if score <= 0:
                continue

            if end <= context_chars:
                if score > best_before:
                    best_before = score
                    sent_before = sentence
                    gid_before = gid
            elif start >= context_chars:
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
        category = "weak_or_no_lexical_overlap"
    elif (
        best_after >= MIN_OVERLAP
        and best_after > best_before
        and best_after >= best_cross
    ):
        category = "after_800_stronger"
    elif (
        best_cross >= MIN_OVERLAP
        and best_cross > best_before
        and best_cross > best_after
    ):
        category = "crosses_800_stronger"
    elif (
        best_after >= MIN_OVERLAP
        and best_before < MIN_OVERLAP
        and best_cross < MIN_OVERLAP
    ):
        category = "after_800_stronger"
    elif (
        best_before >= MIN_OVERLAP
        and best_after < MIN_OVERLAP
        and best_cross < MIN_OVERLAP
    ):
        category = "before_800_only"
    else:
        category = "mixed_before_and_after"

    return {
        "category": category,
        "best_before": best_before,
        "best_after": best_after,
        "best_cross": best_cross,
        "best_before_doc_id": gid_before,
        "best_after_doc_id": gid_after,
        "best_cross_doc_id": gid_cross,
        "before_snippet": sent_before[:220],
        "after_snippet": sent_after[:220],
        "cross_snippet": sent_cross[:220],
        "gold_doc_lengths": doc_lengths,
    }


def main() -> None:
    cfg = load_config()
    answerable_path = Path(cfg["eval"][ANSWERABLE_PATH_KEY])
    eval_records = load_jsonl(answerable_path)
    eval_by_qid = {str(rec.get("query_id", "")): rec for rec in eval_records}

    result_files = sorted(glob(RESULT_GLOB))
    if not result_files:
        raise SystemExit(f"No result files matched {RESULT_GLOB}")

    result_path = Path(result_files[-1])
    print(f"result_file={result_path}")

    with result_path.open("r", encoding="utf-8") as f:
        res = json.load(f)

    rows = res["results"]["answerable"]
    refusals = [row for row in rows if row.get("refusal")]
    generation_refusals = [
        row for row in refusals if row.get("gold_doc_ids_retrieved")
    ]
    retrieval_side_refusals = [
        row for row in refusals if not row.get("gold_doc_ids_retrieved")
    ]

    context_chars = int(cfg.get("agent", {}).get("context_chars", 800))
    corpus = load_corpus(Path(cfg["data"]["corpus_path"]))

    classified = []

    for row in generation_refusals:
        diag = classify_row(row, corpus, context_chars)
        qid = str(row.get("query_id", ""))
        base = eval_by_qid.get(qid, {})

        item = {
            "query_id": qid,
            "question": row.get("question") or base.get("question", ""),
            "set": base.get("set", "answerable"),
            "gold_doc_ids": row.get("gold_doc_ids") or base.get("gold_doc_ids", []),
            "gold_doc_ids_retrieved": row.get("gold_doc_ids_retrieved", []),
            "retrieved_doc_ids": row.get("retrieved_doc_ids", []),
            "baseline_answer_text": row.get("answer_text", ""),
            "baseline_refusal": True,
            "failure_side": "generation",
        }
        item.update(diag)
        classified.append(item)

    by_category = defaultdict(list)
    for item in classified:
        by_category[item["category"]].append(item)

    # Make sampling deterministic independent of result-file ordering.
    for category in by_category:
        by_category[category].sort(key=lambda x: x["query_id"])

    rng = random.Random(SEED)
    selected = []
    selected_ids = set()

    for category, quota in QUOTAS.items():
        pool = by_category.get(category, [])[:]
        rng.shuffle(pool)

        chosen_in_category = 0
        for item in pool:
            if chosen_in_category >= quota:
                break
            if item["query_id"] in selected_ids:
                continue

            selected.append(item)
            selected_ids.add(item["query_id"])
            chosen_in_category += 1

    # If any category was smaller than its quota, fill the remainder deterministically.
    if len(selected) < TARGET_TOTAL:
        remaining = [
            item for item in classified if item["query_id"] not in selected_ids
        ]
        rng.shuffle(remaining)

        for item in remaining:
            if len(selected) >= TARGET_TOTAL:
                break
            selected.append(item)
            selected_ids.add(item["query_id"])

    category_order = {category: idx for idx, category in enumerate(QUOTAS)}
    selected.sort(
        key=lambda x: (
            category_order.get(x["category"], 99),
            x["query_id"],
        )
    )

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUT_META.parent.mkdir(parents=True, exist_ok=True)

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    meta = {
        "result_file": str(result_path),
        "answerable_path": str(answerable_path),
        "context_chars": context_chars,
        "seed": SEED,
        "target_total": TARGET_TOTAL,
        "quotas": QUOTAS,
        "answerable_refusals": len(refusals),
        "generation_refusals": len(generation_refusals),
        "retrieval_side_refusals_excluded": len(retrieval_side_refusals),
        "category_counts": dict(Counter(item["category"] for item in classified)),
        "selected_counts": dict(Counter(item["category"] for item in selected)),
        "selected_query_ids": [item["query_id"] for item in selected],
        "selected_cases": selected,
    }

    with OUT_META.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    with OUT_IDS.open("w", encoding="utf-8") as f:
        f.write("\n".join(item["query_id"] for item in selected) + "\n")

    print(f"context_chars={context_chars}")
    print(f"answerable_refusals={len(refusals)}")
    print(f"generation_refusals={len(generation_refusals)}")
    print(f"retrieval_side_refusals_excluded={len(retrieval_side_refusals)}")
    print(f"category_counts={dict(Counter(item['category'] for item in classified))}")
    print(f"selected_total={len(selected)}")
    print(f"selected_counts={dict(Counter(item['category'] for item in selected))}")
    print(f"selected_query_ids={[item['query_id'] for item in selected]}")
    print(f"wrote_jsonl={OUT_JSONL}")
    print(f"wrote_meta={OUT_META}")
    print(f"wrote_ids={OUT_IDS}")


if __name__ == "__main__":
    main()
