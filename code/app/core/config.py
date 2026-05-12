"""
Centralised application and RAG configuration via pydantic-settings.

``AppSettings`` covers infrastructure knobs (LLM backend, Qdrant, embeddings,
MCP transport, LangSmith).  ``RagSettings`` covers retrieval hyper-parameters
(weights, decay, graph-expansion, mastery).

Both classes read from the ``app/.env`` file and from the process
environment.  Environment variables always take precedence over ``.env``.

Usage::

    from app.core.config import get_settings, get_rag_settings

    s = get_settings()          # AppSettings singleton (lru_cache)
    r = get_rag_settings()      # RagSettings singleton (lru_cache)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to this file so it works regardless of cwd.
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class AppSettings(BaseSettings):
    """Infrastructure settings for LLM backends, vector store, embeddings and MCP."""

    # LLM backend selection
    llm_backend: Literal["groq", "openrouter", "ollama"] = "groq"

    # Groq
    groq_api_key: str = ""
    groq_model: str = "qwen/qwen3-32b"

    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_model: str = "openai/gpt-4o-mini"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_http_referer: str = "http://localhost"
    openrouter_app_title: str = "EGE RAG PoC"
    openrouter_timeout: float = 90.0
    openrouter_connect_timeout: float = 30.0
    openrouter_max_retries: int = 3

    # Ollama
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2"
    ollama_timeout: float = 120.0
    ollama_max_retries: int = 3

    # Embeddings
    embed_backend: Literal["local", "openrouter"] = "local"
    embed_model: str = "mixedbread-ai/mxbai-embed-large-v1"
    sparse_embed_model: str = "Qdrant/bm25"
    openrouter_embed_api_key: str = ""
    openrouter_embed_batch: int = 64
    openrouter_embed_provider_order: str = ""
    openrouter_embed_provider_json: str = ""
    openrouter_embed_model_fallbacks: str = ""

    # Reranker (Cohere-compatible via OpenRouter /v1/rerank)
    reranker_model: str = "cohere/rerank-v3.5"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "ege_rag_atoms"

    # ColBERT late-interaction model (fastembed LateInteractionTextEmbedding)
    colbert_model: str = "colbert-ir/colbertv2.0"

    # Postgres (platform integration)
    database_url: str = "postgresql+asyncpg://ege:ege@127.0.0.1:5432/ege_generator"
    db_pool_min_size: int = 2
    db_pool_max_size: int = 10
    db_pool_timeout_seconds: float = 30.0

    # Kafka (platform integration)
    kafka_bootstrap_servers: str = "127.0.0.1:9092"
    kafka_submission_topic: str = "ege.submissions.v1"
    kafka_consumer_group: str = "ege-rag-ingest"

    # Platform HTTP (for callbacks)
    platform_base_url: str = ""
    platform_auth_token: str = ""  # bearer для callback'ов

    # Retrieval judge model override (overrides the backend's default model for eval)
    judge_model: str = os.getenv("JUDGE_MODEL", "")

    # Conspect generation flags
    conspect_verify: str = ""  # "true"/"false" or "" (auto)
    conspect_math_verify: str = ""  # "true"/"false" or "" (auto)
    conspect_structured_output: str = ""

    # MCP server / client
    conspect_mcp_transport: str = "stdio"
    conspect_mcp_host: str = "127.0.0.1"
    conspect_mcp_port: int = 8001
    conspect_mcp_path: str = "/mcp"
    conspect_mcp_url: str = ""
    conspect_mcp_profiles_dir: str = ""

    # FIPI task bank JSON (default: sibling of app package directory)
    fipi_parsed_json: Path = (
        Path(__file__).resolve().parent.parent.parent / "fipi_parsed_katex.json"
    )

    # LangSmith tracing
    langsmith_api_key: str = ""
    langsmith_tracing: str = ""
    langsmith_project: str = ""
    langsmith_prompt_owner: str = ""

    # HuggingFace token (fastembed model downloads)
    hf_token: str = ""

    # Logging
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def profiles_dir(self) -> Path:
        """Absolute path to the JSON profiles directory."""
        return Path(__file__).resolve().parent.parent / "data" / "profiles"


class RagSettings(BaseSettings):
    """Retrieval hyper-parameters (all overridable via RAG_* env vars)."""

    # Retrieval weights (Blended RAG approach)
    dense_weight: float = 0.5
    bm25_weight: float = 0.5
    colbert_weight: float = 0.5
    profile_weight: float = 0.30
    skill_factor: float = 0.3

    # RRF constant (Cormack et al. 2009)
    rrf_k: int = 10

    # MMR diversity (lambda: 1=relevance only, 0=diversity only; 0.5 balanced)
    mmr_lambda: float = 0.6

    # Power-law decay exponent (KBS 2026: alpha=0.5)
    decay_alpha: float = 0.5

    # Graph expansion bonuses
    prereq_bonus: float = 0.05
    prereq_error_tag_bonus: float = 0.03
    dependent_bonus: float = 0.03
    cross_task_prereq_bonus: float = 0.5

    # Detail level accuracy thresholds
    accuracy_low: float = 0.3
    accuracy_mid: float = 0.6

    # Supported task numbers
    supported_tasks: frozenset[int] = frozenset({6, 10, 12})

    # Profile decay
    decay_half_life: float = 20.0
    cross_pollination: float = 0.3
    persistent_threshold: int = 3
    persistent_window: int = 12
    persistent_multiplier: float = 1.5
    recent_history_limit: int = 100
    error_events_limit: int = 300

    # Frontier selection
    frontier_weakness_threshold: float = 0.2
    frontier_prereq_weight: float = 0.5
    frontier_predicted_weight: float = 0.15
    frontier_max_per_task: int = 2

    # Qdrant prefetch multiplier
    qdrant_prefetch_factor: int = 1
    # Enable Cohere reranker (via OpenRouter /v1/rerank) instead of RRF fusion
    reranker_enabled: bool = False
    # BM25 sparse leg in hybrid search
    bm25_enabled: bool = False
    # ColBERT late-interaction leg in hybrid search
    colbert_enabled: bool = True

    # Mastery gating
    mastery_threshold: float = 0.8
    mastery_penalty: float = 0.15

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> AppSettings:
    """Return a cached AppSettings instance (loaded once per process)."""
    return AppSettings()


@lru_cache
def get_rag_settings() -> RagSettings:
    """Return a cached RagSettings instance (loaded once per process)."""
    return RagSettings()
