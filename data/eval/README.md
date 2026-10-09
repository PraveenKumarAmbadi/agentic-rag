# Eval sets

Frozen inputs for `python -m evaluation.evaluate_rag`. Results are comparable
only between runs on the same version of these files (and the same model and
scoring code) — any change is recorded in the change log below.

| File | Rows | Source | Measures |
|---|---|---|---|
| `answerable.jsonl` | 100 | SciFact test claims with gold documents, sampled by `scripts/build_eval_sets.py` (seed 42) | citation precision, false-refusal rate, retrieval hit rate |
| `unanswerable.jsonl` | 50 | hand-written questions on 18 topics absent from SciFact | refusal correctness (floor test) |
| `adversarial.jsonl` | 43 | hand-written near-miss **claims**: each twists one detail of a real SciFact finding | refusal correctness under near-miss evidence, `near_miss_retrieved_rate`, `cited_near_miss_rate` |
| `ablation_generation_false_refusals*.{jsonl,txt}` | 28 | subset of answerable cases from the false-refusal investigation | used by `scripts/run_generation_ablation.py` |

Fields: `query_id` (stable id: SciFact ids for answerable, `u01`–`u50`,
`a01`–`a43`), `question`, `gold_doc_ids` (empty for unanswerable and
adversarial), `set`; unanswerable adds `topic`; adversarial adds
`near_miss_doc_ids`, `trap`, `rationale`, `paired_query_id`.

## The adversarial set

The out-of-domain set only shows that the system does not invent answers from
*unrelated* evidence. The adversarial set probes the hard case: evidence that
is on-topic and almost — but not — sufficient.

Each item takes a document that is the gold evidence for an answerable claim
(`paired_query_id`) and changes exactly one critical element so the document
no longer addresses it (`rationale` says what):

| `trap` | Example |
|---|---|
| population | trial enrolled only men → claim about women |
| outcome | study measured hip fractures → claim about vertebral fractures |
| species | mouse or fly result → claim about humans / another species |
| causal | observational association → claim that an intervention works |
| timeframe | 6-week trial → claim about 12 months |
| setting, disease, organ, cell-type, drug, mechanism, reverse-direction | same idea along other axes |

Claims are written as declarative sentences like SciFact's, so form gives no
hint. Per the system's contract (answer only from the documents, otherwise
refuse), the correct behaviour on every item is a refusal; a hedged answer
counts as a failure.

Verification, per item: (1) the near-miss document is in the hybrid
retriever's top 10 for the claim — otherwise it is not a trap (all 43 are:
41 at rank 1, one each at ranks 2 and 6); (2) no other top-10 document answers it (read by hand where
topical); (3) a corpus-wide keyword co-occurrence scan for the changed
element finds no document that addresses it. One drafted item was rejected:
"In humans, cold-induced beige adipocytes in subcutaneous fat arise mainly
from de novo adipocyte differentiation" — doc 17973161 studies the origin of
human beige adipocytes.

Paired analysis: for each `paired_query_id`, the same document should yield
an answer to the real claim and a refusal to the twisted one.

## How "unanswerable" was verified

Every question was run through the project's hybrid retrieval (BM25 + SPLADE +
dense) and its top documents inspected, plus a keyword scan of the corpus for
each topic. A question stays in the set only if no corpus document addresses
it — related-but-non-answering documents are allowed (e.g. u16 retrieves a
paper on the 2003 European heatwave, which does not discuss the AMOC).

## Known limitations

- **The unanswerable set is out-of-domain by design** and therefore easy; read
  it as a floor test. The adversarial set is the hard refusal test.
- **Form differs in the unanswerable set.** Its items are *questions* while
  answerable items are SciFact *claims*, so a model could key refusals on
  form. The adversarial set avoids this (claims only).
- **Single author, small n.** 43 adversarial items give a refusal rate with a
  standard error of up to ~7.6 points; treat differences smaller than ~15
  points between runs as noise.

## Change log

- **v3** (2026-10-09) — added `adversarial.jsonl` (43 items, `a01`–`a43`).
  Answerable and unanswerable sets unchanged.
  The former u44 (desalination brine) was not added: it is partially
  answerable, so it is not a clean unanswerable item in either set.
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
