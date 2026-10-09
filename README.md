# Agentic RAG Research Assistant

[![tests](https://github.com/PraveenKumarAmbadi/agentic-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/PraveenKumarAmbadi/agentic-rag/actions/workflows/tests.yml)
[![docker](https://github.com/PraveenKumarAmbadi/agentic-rag/actions/workflows/docker.yml/badge.svg)](https://github.com/PraveenKumarAmbadi/agentic-rag/actions/workflows/docker.yml)

An AI research assistant that answers questions from a scientific corpus
([SciFact](https://github.com/allenai/scifact), via BEIR) with inline `[n]`
citations — and **knows when it doesn't know**. A hand-rolled agent decides how
to retrieve, generates an answer, fact-checks its own draft, and refuses
honestly when the corpus cannot support an answer. Every step is traced and
measured.

It builds on [rag-hybrid-retrieval](https://github.com/PraveenKumarAmbadi/rag-hybrid-retrieval)
(BM25 + SPLADE + dense retrieval fused with RRF) and runs **entirely on your
own machine**: the LLM is served by [Ollama](https://ollama.com) locally — no
API keys for external services, no hosted backend.

> **Status:** Phase 3 (agent loop) with production hardening. Evaluation
> tables will be published with Phase 5.

## How it works

```
POST /answer ──► hybrid retrieval (BM25 + SPLADE + dense, RRF)
                   │
                   ▼
                 decide (LLM, JSON tool call): answer │ search again │ insufficient evidence
                   │                                          │
                   │                         rescue: a second opinion challenges the refusal
                   ▼
                 generate (streamed, refusal intercepted) ──► critique (LLM fact-check)
                   │                                              │ unsupported claims
                   ▼                                              ▼
                 SSE stream: answer text + citations      regenerate (≤ 2×), else hedge
```

- **Agent loop** — bounded ReAct loop, JSON-schema-validated tool calls, no agent framework.
- **Honest refusals** — the model may refuse only after a second-opinion check; a refusal names what *was* found.
- **Self-critique** — an LLM judge lists unsupported claims; the answer is rewritten or visibly hedged.
- **Production concerns** — API-key auth, admission control, timeouts, circuit breakers with graceful degradation, structured JSON logs with trace ids, token accounting, unit tests + CI.

---

## Requirements

| | Minimum | Recommended |
|---|---|---|
| RAM (for Docker) | 12 GB with a 3B model | 16 GB+ for the default 7B model |
| Disk | ~15 GB (images, model, indexes) | |
| GPU | not required | NVIDIA GPU (8 GB+ VRAM) for fast answers |
| Software | Docker Engine 24+ with **Docker Compose v2.24+** | |

Tested on Linux (WSL2, CPU and NVIDIA GPU). The image is built for amd64 and arm64, so macOS
(Intel and Apple Silicon) and Windows with Docker Desktop are expected to work as well.

**Speed expectations.** With an NVIDIA GPU an agent answer takes ~5–60 s.
CPU-only, a 7B model needs minutes per answer — use a smaller model
(see [CPU-only machines](#cpu-only-machines)) or a native Ollama with GPU/Metal
acceleration.

---

## Quick start (Docker — everything included)

This runs Elasticsearch, Ollama, the index builder and the API in containers.
Nothing else needs to be installed.

```bash
git clone https://github.com/PraveenKumarAmbadi/agentic-rag.git
cd agentic-rag
cp .env.example .env
```

Generate an API key and put it in `.env` as `AGENTIC_RAG_API_KEYS=...`:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Start everything:

```bash
docker compose up -d
```

The **first** start takes a while — it downloads the LLM (~4.7 GB for
`qwen2.5:7b-instruct`), the SciFact dataset and the retrieval models, then
builds the search indexes. With an NVIDIA GPU the build takes a few minutes.
On CPU it is heavy — encoding 5,183 documents with SPLADE can take an hour or
more. It uses 4 CPU threads by default so the machine stays usable; set
`INDEXER_THREADS` in `.env` to trade speed against heat. It is a one-time
cost: the indexes are kept in a Docker volume.
Follow progress with:

```bash
docker compose logs -f indexer ollama-pull
```

The API is ready when `docker compose ps` shows `api` as `healthy`.
Later starts reuse everything and take seconds.

### Ask a question

```bash
export API_KEY=...   # the key from your .env
curl -N http://127.0.0.1:8001/answer \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"question": "Does vitamin D supplementation reduce fracture risk?"}'
```

The response is a [Server-Sent Events](https://developer.mozilla.org/docs/Web/API/Server-sent_events)
stream, one JSON object per `data:` line:

| Event | Meaning |
|---|---|
| `{"delta": "..."}` | answer text; concatenate the deltas |
| `{"type": "step", ...}` | agent progress (retrieve, decide, rescue_override, correction) |
| `{"type": "reset"}` | discard the text received so far (a corrected answer follows) |
| `{"type": "done", ...}` | final metadata: `refusal`, `citations` (marker → doc id), `timings`, `usage`, `degraded`, `trace_id`, `agent` |
| `{"type": "error", "code", "message", "trace_id"}` | the request failed (`llm_unavailable`, `retrieval_unavailable`, `timeout`, `internal`) |

Health (no key needed): `curl http://127.0.0.1:8001/health` — `ok`, `degraded`
(e.g. Elasticsearch down, dense retrieval still serving) or `down` (HTTP 503).
Interactive API docs: http://127.0.0.1:8001/docs.

The API listens on `127.0.0.1` only. To expose it beyond your machine, put a
TLS-terminating reverse proxy in front of it — API keys sent over plain HTTP
can be read on the network.

### NVIDIA GPU

Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
and check it works: `docker run --rm --gpus all ubuntu nvidia-smi`.
Then add to `.env`:

```bash
COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml
TORCH_VARIANT=cu130
```

and rebuild: `docker compose up -d --build`. The LLM, the retrieval encoders
and the index build then all use the GPU.

### CPU-only machines

In `.env`, pick a smaller model and/or allow longer requests:

```bash
LLM_MODEL=qwen2.5:3b-instruct
AGENTIC_RAG__LLM__READ_TIMEOUT_S=300
AGENTIC_RAG__AGENT__REQUEST_TIMEOUT_S=600
AGENTIC_RAG__SERVING__HARD_TIMEOUT_S=900
```

Or switch the API to the single-call baseline pipeline:
`AGENTIC_RAG__AGENT__MODE=baseline`. Apply changes with `docker compose up -d`.

---

## Use a native Ollama (optional)

Running Ollama directly on your machine instead of in Docker is the fastest
option on a **Mac** (Docker on macOS has no GPU access; native Ollama uses
Metal) and lets you reuse models you already have.

**1. Install Ollama**

| OS | Install |
|---|---|
| macOS | Download the app from [ollama.com/download](https://ollama.com/download) (or `brew install ollama`) and start it |
| Linux | `curl -fsSL https://ollama.com/install.sh \| sh` (installs and starts a systemd service) |
| Windows | Run the installer from [ollama.com/download](https://ollama.com/download) |

**2. Download the model** (~4.7 GB):

```bash
ollama pull qwen2.5:7b-instruct
```

Check it: `ollama list` should list the model, and
`ollama run qwen2.5:7b-instruct "Say hi"` should answer.

**3. Point the stack at it** — in `.env`:

```bash
COMPOSE_PROFILES=
LLM_BASE_URL=http://host.docker.internal:11434
```

Then `docker compose up -d`. The bundled `ollama` containers are no longer started.

**Linux only:** Ollama listens on `127.0.0.1` by default, which containers
cannot reach. Make it listen on all interfaces:

```bash
sudo systemctl edit ollama     # add the two lines below, save
#   [Service]
#   Environment="OLLAMA_HOST=0.0.0.0"
sudo systemctl restart ollama
```

This exposes port 11434 to your network — block it in your firewall if the
machine is on an untrusted network. (Docker Desktop on macOS/Windows needs no
change.)

---

## Running the evaluation

The frozen eval sets live in `data/eval/` (100 answerable questions from
SciFact with gold documents, 50 hand-written unanswerable questions).

```bash
docker compose stop api      # the API and the eval should not share the LLM
docker compose run --rm api python -m evaluation.evaluate_rag --mode agent --tag agent
docker compose run --rm api python -m evaluation.evaluate_rag --mode baseline --tag baseline
docker compose start api
```

Results (summary + per-question records) are written to `./results/`.
Metrics: false-refusal rate and citation precision on answerable questions,
refusal correctness on unanswerable ones, latency percentiles, token usage,
and agent behaviour (rescue overrides, critique pass rate, hedges).
Numbers are only comparable on the same eval set, model, and code version.

---

## Configuration

All settings live in [`configs/config.yaml`](configs/config.yaml) and are
validated at startup against [`configs/schema.py`](configs/schema.py) —
unknown keys and invalid values are rejected with a clear message. Any key can
be overridden through the environment as `AGENTIC_RAG__<SECTION>__<KEY>`
(e.g. `AGENTIC_RAG__AGENT__TOP_K=5`); in Docker, put overrides in `.env`.

Key settings: `agent.mode` (`agent` | `baseline`), `llm.model`,
`agent.max_iterations` / `max_retries`, `agent.top_k`, `serving.max_concurrent`
(default 1 — one model serves one request at a time; extra requests queue
briefly, then get HTTP 503 with `Retry-After`).

## Day-to-day operations

| Task | Command |
|---|---|
| Stop (keep data) | `docker compose down` |
| Logs (JSON lines) | `docker compose logs -f api` |
| Rebuild indexes | `docker compose run --rm indexer python -m scripts.build_indexes --recreate` |
| Switch model | set `LLM_MODEL` in `.env`, then `docker compose up -d` |
| Start from scratch | `docker compose down -v` (deletes models, indexes and data volumes) |
| Verify a running deployment | `python3 scripts/smoke_api.py --key <your key>` |

---

## Local development (without Docker)

Requires Python 3.12, an NVIDIA GPU with CUDA 13 drivers, Elasticsearch 8.15
on `localhost:9200`, and Ollama on `localhost:11434`.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # pinned; includes the CUDA build of torch
python -m scripts.build_indexes          # downloads SciFact, builds all indexes
export AGENTIC_RAG_API_KEYS="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python -m serving.main                   # http://127.0.0.1:8001
```

Unit tests need no GPU, Elasticsearch or Ollama — only the light dependencies:

```bash
pip install -r requirements-core.txt
python -m unittest discover -s tests -t .
```

## Project structure

```
agent/        agent loop, baseline pipeline, LLM client, prompts, citation/refusal parsing, stack builder
serving/      FastAPI app (SSE), API-key auth, admission control
retrieval/    BM25, SPLADE, dense search and RRF fusion
indexing/     encoders, FAISS index, Elasticsearch mappings
evaluation/   eval runner and metrics
configs/      config.yaml + validated schema
scripts/      data download, index build, eval-set build, research diagnostics
data/eval/    frozen eval sets
tests/        unit tests (no external services needed)
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `required variable AGENTIC_RAG_API_KEYS is missing a value` | Create `.env` from `.env.example` and set a key (32+ characters). |
| `api` keeps restarting | `docker compose logs api` — the startup error names the problem (key too short, Elasticsearch unreachable, indexes missing). |
| `indexer` exited with an error | `docker compose logs indexer`, fix the cause, then `docker compose up -d` — the build resumes safely. |
| `{"type": "error", "code": "timeout"}` or very slow answers | CPU-only: use a smaller model or longer timeouts (see [CPU-only machines](#cpu-only-machines)). |
| `llm_unavailable` / `/health` reports `"llm": "down"` | Bundled: `docker compose logs ollama ollama-pull`. Native: is Ollama running, and (Linux) listening on `0.0.0.0`? |
| Elasticsearch exits with `max virtual memory areas vm.max_map_count is too low` (Linux) | `sudo sysctl -w vm.max_map_count=262144` |
| `could not select device driver "nvidia"` | The NVIDIA Container Toolkit is missing — install it or remove `COMPOSE_FILE`/`TORCH_VARIANT` from `.env`. |
| `docker compose` rejects `required: false` | Update Docker Compose to v2.24 or newer. |

## Acknowledgements

SciFact (Wadden et al., 2020) via [BEIR](https://github.com/beir-cellar/beir);
models: [Qwen2.5](https://ollama.com/library/qwen2.5),
[SPLADE](https://huggingface.co/naver/splade-cocondenser-ensembledistil),
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
[BGE reranker](https://huggingface.co/BAAI/bge-reranker-base).
