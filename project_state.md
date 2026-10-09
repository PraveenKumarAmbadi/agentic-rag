# Project State — Agentic RAG Research Assistant

> **This is a living document.** Update the "Current Status" section at the end of
> every work session. If someone (a recruiter, an interviewer, or future-you)
> reads only this file, they should understand what exists, what's next, and why
> each decision was made.

---

## 1. Where this came from — Project 1: Hybrid Retrieval RAG

**Repo:** https://github.com/PraveenKumarAmbadi/rag-hybrid-retrieval

Built a production-grade hybrid retrieval system on the SciFact (BEIR) dataset:

| Component | What was done |
|---|---|
| Retrieval | BM25 (Elasticsearch) + SPLADE sparse (ES) + dense (FAISS HNSW, all-MiniLM-L6-v2) |
| Fusion | Reciprocal Rank Fusion — hand-rolled, pure function, unit-tested |
| Serving | FastAPI, async endpoint bridging to sync GPU code via `asyncio.to_thread`, lifespan warmup |
| Resilience | Hand-rolled circuit breakers per retrieval method, graceful degradation to fewer methods |
| Observability | Structured JSON logs with `trace_id` + per-step ms timings, 200 ms latency budget warning |
| Infra | Docker Compose, NVIDIA GPU passthrough, ES healthcheck, config-driven (YAML) |
| Evaluation | Hand-rolled IR metrics (Recall/Precision/nDCG/MRR/MAP@K) + unified benchmark across 5 methods |

**Results table (the strongest interview asset — memorize these):**

| Method | Hit Rate | Recall@100 | nDCG@10 |
|---|---|---|---|
| BM25 | 89.0% | 88.5% | 0.6606 |
| SPLADE | 95.0% | 95.0% | 0.7079 |
| Dense | 92.7% | 93.2% | 0.6451 |
| Hybrid (RRF) | 96.3% | 96.2% | 0.7097 |
| + BGE Reranker | 96.3% | 96.2% | 0.7158 |

**Reusable assets for this project (do NOT rebuild these):**
- `configs/` YAML pattern + loader
- `es_client.py`, `tracing.py` (TraceContext, JsonFormatter, CircuitBreaker)
- `ingestion/loaders.py`, `retrieval/*`, `indexing/*` — copy or vendor as needed
- Docker Compose pattern with ES healthcheck
- The evaluation discipline: hand-rolled metrics + before/after tables

---

## 2. Project 2 — Vision

**One-line pitch:** An AI research assistant that answers questions from a document
corpus with inline citations — and *knows when it doesn't know*. An agent decides
how to retrieve, generates, critiques its own answer for hallucination, and
refuses honestly when the corpus can't support an answer. Every decision, latency,
and cost is traced and evaluated.

**Why this project (interview narrative):**
- Project 1 answered "how do you retrieve well?" — this answers "how do you turn
  retrieval into a trustworthy product?" — the exact question interviewers ask next.
- Agent loops, evals, hallucination mitigation, and cost/latency engineering are
  the 2026 hiring hot buttons for applied AI engineers.
- Hand-rolling the agent loop and the evals (no LangChain/LlamaIndex) demonstrates
  understanding instead of framework gluing — same bet that made Project 1 stand out.

**Goals:**
1. Faithful, citation-backed answers streamed over an API.
2. A self-correcting agent: retrieval → answer → critique → retry/refuse.
3. A generation eval suite with numbers as concrete as Project 1's table.
4. Full observability: traces, token counts, cost per query, p50/p95 latency.

**Non-goals (guard the scope):**
- No multi-agent frameworks, no LangChain/LlamaIndex/CrewAI.
- No fine-tuning of the LLM (that's a possible Project 3).
- No web UI beyond a minimal page or Swagger — the product is the API + evals.
- No vector DB migration — reuse Elasticsearch + FAISS from Project 1.

---

## 3. Architecture

```
Client
  │  POST /answer  (SSE stream)
  ▼
FastAPI ──► Agent Loop (hand-rolled ReAct-style, bounded iterations)
              │
              ├── tool: hybrid_search()      ← Project 1 retrieval, unchanged
              ├── tool: rerank()             ← BGE cross-encoder
              ├── tool: refusal_check()      ← "can these docs answer this?"
              │
              ▼
            LLM (local via Ollama, API-swappable via config)
              │
              ├── generate answer with [n] citations
              ▼
            Self-Critique step (faithfulness check, second LLM call)
              │── fail → retry with reformulated query (max 2 retries)
              │── unsupported claims → refuse / hedge with citation gaps
              ▼
            Streaming response + trace {tool_calls, timings, tokens, cost}
```

---

## 4. Tech Decisions (with rationale)

| Decision | Choice | Rationale |
|---|---|---|
| LLM runtime | **Ollama, local**, model-swappable via config | Free, offline, repeatable for evals. RTX 5050 8 GB fits a 7–8B Q4 model (e.g. `qwen2.5:7b-instruct` ~4.7 GB). Config key `llm.provider` keeps an OpenAI/Groq API path open for comparison runs. |
| Agent loop | Hand-rolled while-loop, max N iterations, JSON-schema tool calls | Frameworks hide the failure modes interviews probe. A 200-line loop I can explain line-by-line is worth more. |
| Answer grounding | Prompt forces `[doc_id]` inline citations; post-parse validates every citation exists in retrieved set | Citation validation is cheap, deterministic, and evaluable — no LLM needed to check it. |
| Self-critique | Separate LLM call: "list claims not supported by the cited docs" | Separating generation and critique calls measurably reduces hallucination; it's also a clean A/B eval. |
| Evals | Hand-rolled (faithfulness, citation accuracy, refusal correctness) + RAGAS as a cross-check | Hand-rolled = understanding; RAGAS = external validity. Report both. |
| Tracing | Extend Project 1's `TraceContext` with tool-call spans + token/cost accounting | Continuity between repos shows system thinking. |
| Caching | SQLite table keyed on (query, config_hash) for answers; ES stays as-is | Cheap, demonstrable p95 win. Redis only if time permits. |
| Dataset | SciFact corpus (reuse) + new query sets (see §6) | Ground truth already exists; refusal set is easy to author. |

**LLM fallback plan:** if 8 GB VRAM proves too tight with SPLADE+BGE loaded,
offload reranker to CPU during generation tests, or use a 3–4B model
(`qwen2.5:3b-instruct`). Note actual VRAM numbers in §8 as they're measured.

---

## 5. Phase Plan

### Phase 0 — Scaffold (target: 2–3 sessions)
- [x] `git init`, repo skeleton: `configs/`, `serving/`, `agent/`, `retrieval/` (vendored from Project 1), `evaluation/`, `tests/`, `scripts/`
- [x] Copy `es_client.py`, `tracing.py`, `configs/` pattern, `ingestion/`, `retrieval/` from Project 1
- [x] `config.yaml`: add `llm:` section (provider, model, temperature, max_tokens), `agent:` (max_iterations, retry_limit), `eval:` paths
- [x] Ollama installed; `ollama pull qwen2.5:7b-instruct`; smoke: curl `/api/generate`
- [x] `/health` endpoint + baseline `/search` proxy to Project 1 code works

**Exit criteria:** ~~`python -m serving.main` boots, streams one hardcoded completion from the local LLM.~~ ✅ DONE 2026-09-13: `uvicorn serving.main:app` on :8001, `/health` = ES up + LLM up, `POST /answer` streams SSE tokens (`data: {"delta": ...}`) ending in a `done` event with `trace_id` + timings (first_token 163 ms, total 602 ms).

### Phase 1 — Baseline RAG generation (target: 3–4 sessions)
- [x] Prompt template: answer ONLY from provided docs, inline `[doc_id]` citations, "insufficient evidence" escape hatch (`agent/prompts.py`)
- [x] `/answer` endpoint, SSE streaming token-by-token
- [x] Citation parser + validator (`agent/citations.py`): markers `[n]` validated against context size, mapped to real doc_ids
- [x] Structured trace: retrieval ms, first-token ms, total ms

**Exit criteria:** ✅ DONE 2026-09-13: real question → hybrid retrieval (351 ms) → streamed grounded answer with 3 valid citations, 0 invalid. Timings: first_token 7.05 s (dominated by ~2k-token context ingestion — optimization target for Phase 4), total 9.3 s. Known polish item: model sometimes echoes the citation list ("[1] [4] [7]") as its first line — prompt tweak in Phase 2, not blocking.

### Phase 2 — Eval harness v1 (target: 3–4 sessions) ⭐ differentiator
- [~] Eval sets (see §6): answerable (100, seed 42, from qrels — `scripts/build_eval_sets.py`), unanswerable (50 authored, 18 domains — `data/eval/unanswerable.jsonl`). Adversarial (20) authored later.
- [ ] Metrics, hand-rolled first:
  - [ ] Citation accuracy: fraction of citations that are relevant docs (vs qrels)
  - [ ] Faithfulness: claim-level check — extract claims, verify each against cited docs (LLM-as-judge with a rubric; judge = same local model, temp 0)
  - [ ] Refusal correctness: on unanswerable set, did it refuse? (binary) + false-refusal rate on answerable set
  - [ ] Answer relevance: embedding cosine between answer and reference answer (Project 1's encoder reused)
- [ ] Runner: `python -m evaluation.evaluate_rag` → markdown table like Project 1's
- [ ] Baseline numbers captured BEFORE building the agent (this is the "before" column)

**Exit criteria:** a committed results table: baseline RAG faithfulness / citation accuracy / refusal rate.

### Phase 3 — Agent loop (target: 3–4 sessions)
- [ ] Tool registry with JSON-schema validated calls: `hybrid_search`, `rerank`, `insufficient_evidence`
- [ ] ReAct loop: think → act → observe, bounded (max 4 iterations), every step traced
- [ ] Self-critique step post-generation; retry with reformulated query on failure (max 2)
- [ ] Honest refusal path: cite what was found, state the gap explicitly

**Exit criteria:** on the unanswerable set, refusal correctness improves vs Phase 2 baseline; the delta is in the results table.

### Phase 4 — Guardrails + cost/latency (target: 2–3 sessions)
- [ ] Prompt-injection probes embedded in test docs; measure and document resistance; mitigate (input framing, output filtering)
- [ ] Answer cache (SQLite, keyed on query+config hash); measure hit-rate and p95 improvement
- [ ] Cost accounting: tokens per query → $ per 1k queries (API model) or tokens/s (local)
- [ ] p50/p95 latency breakdown per phase of the pipeline

**Exit criteria:** README table: latency & cost per query before/after caching; documented injection-test results.

### Phase 5 — Benchmark, README, polish (target: 2 sessions)
- [ ] Full eval run, final before/after table (Phase 2 baseline vs Phase 3 agent)
- [ ] README in Project 1's style: architecture diagram, eval tables, trade-offs, troubleshooting
- [ ] Record a short demo (terminal cast or Loom) — recruiters watch these
- [ ] Link the two repos to each other

---

## 6. Eval Data Plan

| Set | Size | Source | Purpose |
|---|---|---|---|
| Answerable | 100 | SciFact test queries with qrels; reference answers synthesized from gold docs (LLM, then human-reviewed) | faithfulness, citation accuracy, relevance |
| Unanswerable | 50 | Authored by hand: plausible questions about topics absent from SciFact (e.g. "What is the efficacy of drug X?") | refusal correctness |
| Adversarial | 20 | Questions with near-miss docs (right topic, wrong answer) | the hardest and most interview-valuable set |
| Injection | 10 | Docs with embedded "ignore instructions" text | guardrail testing |

---

## 7. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| **System RAM OOM during eval runs** | **Root cause found 2026-10-09:** Ollama ≥0.34 runs `llama-server`, which keeps a host-RAM *prompt cache* (default limit 8192 MiB, ~120 MiB per ~2k-token prompt). Eval prompts rarely share prefixes, so the cache only grows: it reached 44 prompts / 5.2 GiB and the kernel OOM-killed `llama-server` 130 questions into `baseline_v3`. The model itself needs little host RAM (weights on GPU, 292 MiB host buffer), so the earlier "7B ≈ 4.3 GB host RAM" explanation was the cache. **Fix:** `LLAMA_ARG_CACHE_RAM=0` in Ollama's environment (systemd override locally; set in `docker-compose.yml` for the bundled Ollama). The variable reaching the runner was checked (`prompt cache is disabled` in the log). Not yet verified: that a full eval then completes at default WSL RAM. **Fallback if it still OOMs:** temporarily create `C:\Users\<user>\.wslconfig` with `[wsl2]` / `memory=12GB` / `swap=4GB` / `autoMemoryReclaim=gradual`, run `wsl --shutdown` from PowerShell, restart, run the eval, then delete the file and restart again. Serving the API alone works at default RAM. |
| 8 GB VRAM can't fit LLM + encoders | 3–4B model; CPU-offload reranker during LLM calls; measure actual VRAM in Phase 0 and record in §8 |
| Local 7B model is weak at strict citation format | Few-shot prompt; retry-on-parse-fail; lower temperature; if still bad, that itself is a documented finding |
| LLM-as-judge is biased by same model | Cross-check a sample with RAGAS + a different judge model; keep judge rubric in repo |
| Scope creep into "chat product" | Non-goals in §2 are binding; every session ends by updating §8, not by starting new features |
| Eval runs too slow on CPU | Batch judge calls; cache judgments keyed on (answer_hash, rubric_hash) |

---

## 8. Current Status

**Last updated:** 2026-09-13 (Phase 0 complete)

- [x] Project 1 complete, pushed, smoke-tested end-to-end
- [x] This planning doc created
- [x] Phase 0 scaffold — server streams from local LLM; env: Ollama + qwen2.5:7b-instruct on RTX 5050, reusing `~/.venvs/rag-hybrid-retrieval` venv (all deps already present, nothing new installed)
- [x] Phase 1 generation — retrieval-wired `/answer` with validated citations; first commit pushed to local repo
- [x] Phase 2 evals — runner built (`evaluation/evaluate_rag.py`), baseline captured:
  false_refusal 0.69, citation_precision 0.3133, unanswerable refusal 1.0 (see §8)
- [ ] Phase 3 agent loop
- [ ] Phase 4 guardrails + cost
- [ ] Phase 5 benchmark + README

**Measured facts (fill as they're learned):**
- VRAM, LLM only (7B Q4 loaded): 5447 MiB / 8192 MiB — ~2.7 GB headroom for SPLADE + BGE; if tight, CPU-offload reranker
- System RAM: WSL raised to 12 GB temporarily for eval runs (default 7.6 GB OOM-killed Ollama — see §7 for the raise/lower procedure; reverted after baseline was captured). 2026-10-09: the OOMs came from Ollama's host-RAM prompt cache, not the model. Disable it with `LLAMA_ARG_CACHE_RAM=0` (§7).
- Phase 1 RAG path (single query): retrieval 351 ms, first_token 7.05 s cold, total 9.3 s
- **BASELINE (Phase 1 pipeline, frozen eval sets, 2026-09-14, results/baseline_phase1_*.json):**
  - Answerable (100): citation_precision **0.3133**, false_refusal_rate **0.69**, retrieval_hit_rate 0.87, invalid_citation_rate 0.0, p50 2.26 s, p95 3.38 s
  - Unanswerable (50): refusal_correctness **1.0**, p50 2.03 s, p95 2.50 s
  - - Key failure transcript: "Rapamycin … triacylglycerols in fruit flies" — gold doc WAS in context, model still refused → generation-side failure, not retrieval
  - BASELINE AUDIT (harness verified): 0.69 is REAL, no eval bug — refusal flag matches answer text 100/100; 31 answered + 69 refused; 31 + 56 = 87 = hit rate. Decomposition: 56/100 generation-side (gold in context; conditional refusal 56/87 = 0.64), 13/100 retrieval-side (gold never retrieved; refused 13/13). Truncation exonerated as primary driver: gold docs > 800 chars in 53/56 refused vs 28/31 answered (medians 1282 vs 1141 chars) — no discrimination. Phase 3 targets: rescue converts the 56, re-search covers the 13, unanswerable refusal must stay 1.0.
- Agent faithfulness (Phase 3): _TBD_
- p95 latency, uncached vs cached: _TBD_

**Next action:** Phase 3 — agent loop targeting the baseline's failure mode: 69% false refusals while unanswerable refusal is already 1.0. Self-critique must lower false refusals WITHOUT dropping unanswerable refusal below baseline.
