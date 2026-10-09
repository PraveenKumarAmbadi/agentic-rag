# Eval sets

Frozen inputs for `python -m evaluation.evaluate_rag`. Results are comparable
only between runs on the same version of these files (and the same model and
scoring code) — any change is recorded in the change log below.

| File | Rows | Source | Measures |
|---|---|---|---|
| `answerable.jsonl` | 100 | SciFact test claims with gold documents, sampled by `scripts/build_eval_sets.py` (seed 42) | citation precision, false-refusal rate, retrieval hit rate |
| `unanswerable.jsonl` | 50 | hand-written questions on 18 topics absent from SciFact | refusal correctness |
| `ablation_generation_false_refusals*.{jsonl,txt}` | 28 | subset of answerable cases from the false-refusal investigation | used by `scripts/run_generation_ablation.py` |

Fields: `query_id` (stable id; `u01`–`u50` for unanswerable), `question`,
`gold_doc_ids` (empty for unanswerable), `set`, `topic` (unanswerable only).

## How "unanswerable" was verified

Every question was run through the project's hybrid retrieval (BM25 + SPLADE +
dense) and its top documents inspected, plus a keyword scan of the corpus for
each topic. A question stays in the set only if no corpus document addresses
it — related-but-non-answering documents are allowed (e.g. u16 retrieves a
paper on the 2003 European heatwave, which does not discuss the AMOC).

## Known limitations

- **Out-of-domain by design.** All 50 questions are off-topic for a biomedical
  corpus, so retrieval returns clearly unrelated documents and refusing is
  easy. A refusal-correctness score of 1.0 here shows the system does not
  invent answers from unrelated evidence; it does not show it can refuse when
  near-miss documents are retrieved. That is the job of the planned
  adversarial set (`data/eval/adversarial.jsonl`, not yet written).
- **Form differs between sets.** Answerable items are SciFact *claims*
  (declarative sentences); unanswerable items are *questions*. A model could
  in principle key refusals on form rather than evidence. Rewriting the
  unanswerable items as claims would remove that confound.

## Change log

- **v2** (2026-10-09)
  - Added stable `query_id`s `u01`–`u50` to the unanswerable set (rows were
    previously unidentifiable in per-question results).
  - u44 replaced. The original — *"Can desalination brine be discharged
    without harming marine benthic ecosystems?"* — is partially answerable:
    corpus document 104143831 states that modern brine outfall designs have
    significantly limited the environmental impact of brine discharge. It is
    a good **adversarial** candidate (near-miss evidence) and should go there.
    New u44: *"Do microplastics in agricultural soils reduce earthworm growth
    and survival?"* (no corpus document mentions microplastics or earthworms).
  - u47 reworded for grammar (same meaning): *"Does space weather posing
    geomagnetic storm risk justify hardening all power grids?"* →
    *"Does the geomagnetic storm risk from space weather justify hardening all
    power grids?"*
- **v1** — original sets (baseline results from 2026-09-14 were measured on v1).
