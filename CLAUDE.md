# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An agentic RAG research assistant over the SciFact (BEIR) corpus: answers with inline `[n]` citations and refuses when the corpus can't support an answer. Builds on Project 1 (`rag-hybrid-retrieval`); `retrieval/`, `indexing/`, `ingestion/`, `reranking/` are vendored from it and should be treated as stable.

`project_state.md` is the living plan (phases, design rationale, measured baselines, risks). Read it before starting work; update §8 "Current Status" when a session changes status or produces new numbers.

**Binding non-goals:** no LangChain/LlamaIndex/CrewAI or other agent frameworks, no `jsonschema` (tool-call validation is hand-rolled deliberately), no fine-tuning, no web UI, no vector-DB migration. The agent loop, tool validation, and eval metrics are intentionally hand-written.

## Environment & commands

- Python 3.12, WSL2. `requirements.txt` (full stack, includes `requirements-core.txt`) is an exact, pip-resolvable pin set — install it into a dedicated venv. `requirements-core.txt` alone is enough to run the unit tests. (The shared `~/.venvs/rag-hybrid-retrieval` currently has transformers 5.17 forced in by other projects; it runs, but is not reproducible.) No `python` on PATH: activate a venv first.
- CI (`.github/workflows/`): `tests.yml` runs the unit tests on every push/PR using only `requirements-core.txt`; `dependencies.yml` dry-run-resolves `requirements.txt` (fails on conflicts or yanked pins) when requirements files change.
- Run everything from the repo root.
- External services: Elasticsearch on `localhost:9200` (BM25 + SPLADE indexes; Docker container `rag-elasticsearch` from Project 1), Ollama on `localhost:11434` serving `qwen2.5:7b-instruct`.
- `data/` and `indexes/` are gitignored. `python scripts/prepare_data.py` restores the SciFact corpus; ES/FAISS indexes come from Project 1 (no index-build script here).
- `test.py` (repo root) is the owner's personal scratch file — never edit or delete it.

```bash
# API — auth is ON by default and the server refuses to start without a key
export AGENTIC_RAG_API_KEYS="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python -m serving.main            # binds serving.host:port (127.0.0.1:8001)
curl -N -X POST localhost:8001/answer -H "X-API-Key: $AGENTIC_RAG_API_KEYS" \
     -H 'content-type: application/json' -d '{"question": "..."}'

# Unit tests — no LLM/ES/GPU needed. Use -s tests (root test.py is not a test module)
python -m unittest discover -s tests -t .
python -m unittest tests.test_agent.TestAgentLoop.test_happy_path

# Eval runner -> results/<tag>_<ts>.json (checkpointed to .partial.jsonl as it runs)
python -m evaluation.evaluate_rag --mode agent --tag agent_phase3
python -m evaluation.evaluate_rag --mode baseline --sets answerable --limit 10
```

**Resource constraints (8 GB VRAM, ~7.6 GB default WSL RAM):** stop the API before running evals or `scripts/run_*` (both can't hold the LLM). Full evals have OOM-killed Ollama at default WSL RAM — see `project_state.md` §7 for the temporary `.wslconfig` raise. `serving.max_concurrent: 1` is deliberate: one GPU, one Ollama model slot.

## Configuration

`configs/schema.py` (pydantic) is the single source of defaults and constraints; `configs/loader.py::load_config()` validates `config.yaml` against it, rejects unknown keys, and returns a plain dict — so read keys directly (`cfg["agent"]["top_k"]`), never `.get(key, default)`. New keys go in the schema **and** `config.yaml`. Env overrides: `ES_HOST`, `ES_PORT`, `LLM_BASE_URL`, `LLM_MODEL`, `AGENTIC_RAG_CONFIG`. Secrets (API keys) are env-only and never enter the config dict. Tests build configs with `validate_config({...})`.

## Architecture

**One code path for serving and eval.** `agent/stack.py::build_pipeline()` is the only place that loads corpus/indexes/models; `serving/main.py` and `evaluation/evaluate_rag.py` both use it, then consume the same `answer_stream()`. RAG logic belongs in `agent/`, never in the API or eval layers.

**Event contract** (`agent/pipeline.py` docstring): `delta` events whose texts concatenate to the final answer, then exactly one `done` (text, refusal, citations, invalid_citations, retrieved_doc_ids, timings, step_timings, usage, degraded, truncated; agent adds `agent`). The agent also emits `step` and `reset` (client discards text so far). Every answer *and refusal* reaches the client as deltas — the SSE layer omits `text` from `done`. Failures are exceptions (`RetrievalUnavailable`, `OllamaError`), turned into an SSE `error` event by the API.

**LLM calls** (`agent/llm.py`): every call takes a `CallSpec`; `pipeline.CallSpecs.from_config()` defines answer/decide/rescue/critique policy in one place. `stream()` retries only before the first token (then `StreamInterrupted`); `complete()` retries whole attempts and applies `on_length`. Add new call types as new specs, not ad-hoc parameters.

**Agent loop** (`agent/agent.py`): pre-fetch → bounded ReAct decide turns (JSON tool calls validated by `agent/tools.py`: `answer`, `hybrid_search`, `insufficient_evidence`) → **rescue** (second-opinion call can override a refusal and hint generation at a doc) → streamed generation through `RefusalGate` (buffers a leading refusal so it can still be rescued) → **critique** with up to `agent.max_retries` regenerations → hedge with `HEDGE_PREFIX` if still unsupported. Degrade policy: internal-turn failures never sink the request (decide→generate, rescue→refusal stands, critique→answer marked `skipped`); `agent.request_timeout_s` is a soft budget that stops optional steps.

Design target (from `project_state.md`): baseline false-refusal **0.69** with unanswerable refusal already **1.0** — changes must lower the first without dropping the second. Scoring rules changed in the production pass (unified refusal detector, refusal "closest documents" no longer count as citations), so re-run both baseline and agent before comparing against older result files.

**Conventions:**
- Refusal is defined once: `agent/citations.py` (`classify_refusal_prefix`, `is_refusal`, `refusal_reason`) — case/underscore/markdown-insensitive. The gate, agent, baseline, and eval all use it.
- Parse failures are biased toward answering; `parse_action` returns `None` (no JSON) vs raises `ToolCallError` (bad JSON), counted separately in `done["agent"]`.
- Prompts live only in `agent/prompts.py`; its system prompts double as routing keys for the test `FakeLLM` ("controller", "second opinion", "fact-checker").
- Retrieval (`RagPipeline.retrieve`) returns `Retrieval(docs, elapsed_ms, failed_methods)`; per-method `CircuitBreaker`s (`tracing.py`) skip a failing retriever; partial failure → `done["degraded"]`, total failure → `RetrievalUnavailable`. Step timings are namespaced (`retrieve_2.bm25_retrieval`).
- Logging: `tracing.configure_logging()` routes everything (incl. uvicorn) to one JSON handler; `trace_id_var` (ContextVar) stamps the request's trace id on every line, including worker threads. Never log user query text.
- Heavy imports (torch, faiss, transformers) stay inside `agent/stack.py::build_pipeline` / lazy paths; `tests/test_imports.py` enforces that `serving`, `agent`, `evaluation`, `retrieval.fusion` import without them.
- API (`serving/`): `auth.py` (X-API-Key, SHA-256 digests, constant-time, fail closed), `limits.py` (concurrency limiter + bounded queue → 503, cached LLM probe). `create_app(services_factory)` lets tests inject fakes.
- Eval sets in `data/eval/*.jsonl` are frozen; results are comparable only on the same set and scoring code.
