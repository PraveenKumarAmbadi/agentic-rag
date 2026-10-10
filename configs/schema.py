"""Typed schema for configs/config.yaml.

config.yaml stays the one file humans edit; this module is the one place
defaults and constraints live. ``load_config()`` validates against it, so a
typo or an out-of-range value fails at startup with a precise message instead
of a KeyError in the middle of a request.

Every section forbids unknown keys: a misspelled key is an error, never a
silently ignored setting.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Section):
    corpus_path: str = "data/scifact/corpus.jsonl"
    queries_path: str = "data/scifact/queries.jsonl"
    qrels_path: str = "data/scifact/qrels/test.tsv"


class ElasticsearchConfig(_Section):
    host: str = "localhost"
    port: int = Field(9200, ge=1, le=65535)


class BM25Config(_Section):
    index_name: str = "scifact_bm25"


class SpladeConfig(_Section):
    index_name: str = "scifact_splade"
    model_name: str = "naver/splade-cocondenser-ensembledistil"
    top_k: int = Field(256, ge=1)
    batch_size: int = Field(32, ge=1)
    max_seq_length: int = Field(512, ge=8)


class HNSWConfig(_Section):
    M: int = Field(16, ge=2)
    ef_construction: int = Field(200, ge=1)


class DenseConfig(_Section):
    index_path: str = "indexes/scifact_dense.faiss"
    ids_path: str = "indexes/scifact_dense_ids.json"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = Field(64, ge=1)
    max_seq_length: int = Field(512, ge=8)
    hnsw: HNSWConfig = Field(default_factory=HNSWConfig)


class FusionConfig(_Section):
    rrf_k: int = Field(60, ge=1)


class RerankerConfig(_Section):
    model_name: str = "BAAI/bge-reranker-base"
    max_seq_length: int = Field(512, ge=8)
    batch_size: int = Field(16, ge=1)


class ResilienceConfig(_Section):
    # Per-retriever circuit breakers (BM25, SPLADE, dense).
    breaker_failure_threshold: int = Field(3, ge=1)
    breaker_cooldown_s: float = Field(30.0, gt=0)


class LLMConfig(_Section):
    provider: Literal["ollama"] = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b-instruct"
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=16)
    judge_model: str = "qwen2.5:7b-instruct"
    judge_temperature: float = Field(0.0, ge=0.0, le=2.0)
    # Context window Ollama allocates per request. Prompts longer than this
    # are silently truncated from the FRONT (system prompt first).
    num_ctx: int = Field(4096, ge=512)
    keep_alive_s: float = Field(360.0, ge=0.0)
    connect_timeout_s: float = Field(5.0, gt=0.0)
    # Longest silent gap tolerated on a call, prompt prefill included.
    read_timeout_s: float = Field(60.0, gt=0.0)
    max_attempts: int = Field(3, ge=1, le=10)


class AgentConfig(_Section):
    mode: Literal["baseline", "agent"] = "agent"
    max_iterations: int = Field(4, ge=1, le=10)
    max_retries: int = Field(2, ge=0, le=5)
    rerank: bool = False
    rerank_candidates: int = Field(50, ge=1)
    per_method_top_k: int = Field(100, ge=1)
    # Citation markers are parsed as 1-2 digits, so the context must stay < 100.
    top_k: int = Field(10, ge=1, le=50)
    context_chars: int = Field(800, ge=100)
    # Soft budget: once spent, the agent stops starting optional steps
    # (more decide turns, rescue, critique retries) and finishes with what it has.
    request_timeout_s: float = Field(90.0, gt=0.0)

    @model_validator(mode="after")
    def _check_depths(self) -> "AgentConfig":
        if self.per_method_top_k < self.top_k:
            raise ValueError("agent.per_method_top_k must be >= agent.top_k")
        if self.rerank_candidates < self.top_k:
            raise ValueError("agent.rerank_candidates must be >= agent.top_k")
        return self


class ServingConfig(_Section):
    host: str = "127.0.0.1"
    port: int = Field(8001, ge=1, le=65535)
    # Fail closed: when true the API refuses to start without keys in
    # AGENTIC_RAG_API_KEYS. Keys never live in this file.
    auth_enabled: bool = True
    max_concurrent: int = Field(1, ge=1)
    max_queue: int = Field(8, ge=0)
    queue_timeout_s: float = Field(30.0, gt=0.0)
    # Hard ceiling per request; the stream ends with an error event after it.
    hard_timeout_s: float = Field(180.0, gt=0.0)
    max_question_chars: int = Field(2000, ge=1, le=20000)
    llm_health_ttl_s: float = Field(5.0, ge=0.0)


class EvalConfig(_Section):
    answerable_path: str = "data/eval/answerable.jsonl"
    unanswerable_path: str = "data/eval/unanswerable.jsonl"
    adversarial_path: str = "data/eval/adversarial.jsonl"
    dev_path: str = "data/eval/dev.jsonl"  # prompt-tuning set, disjoint from answerable
    results_dir: str = "results/"


class AppConfig(_Section):
    data: DataConfig = Field(default_factory=DataConfig)
    elasticsearch: ElasticsearchConfig = Field(default_factory=ElasticsearchConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    splade: SpladeConfig = Field(default_factory=SpladeConfig)
    dense: DenseConfig = Field(default_factory=DenseConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    resilience: ResilienceConfig = Field(default_factory=ResilienceConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    @model_validator(mode="after")
    def _check_timeouts(self) -> "AppConfig":
        if self.serving.hard_timeout_s < self.agent.request_timeout_s:
            raise ValueError(
                "serving.hard_timeout_s must be >= agent.request_timeout_s "
                "(the hard ceiling cannot be shorter than the soft budget)"
            )
        return self
