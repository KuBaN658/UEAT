"""
Text embeddings: local fastembed or OpenRouter (OpenAI-compatible /v1/embeddings).

Backend is selected via ``AppSettings.embed_backend``:
- ``"local"``      — fastembed ``TextEmbedding`` (default: mixedbread-ai/mxbai-embed-large-v1).
- ``"openrouter"`` — OpenRouter /v1/embeddings with fallback model support.

Sparse embeddings (BM25-style) always use the local fastembed
``SparseTextEmbedding`` regardless of the backend setting.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.request
from functools import lru_cache

import numpy as np
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding
from fastembed import SparseEmbedding as SparseResult

from app.core.config import get_settings
from app.infrastructure.llm.http_utils import urlopen_with_retries

log = logging.getLogger(__name__)

_DEFAULT_SPARSE_MODEL = "Qdrant/bm25"


# ── Backend / model selection ────────────────────────────────────────


def get_embed_backend() -> str:
    """Return the active embedding backend (``"local"`` or ``"openrouter"``)."""
    return get_settings().embed_backend


def get_embed_model_id() -> str:
    """Return the active dense embedding model identifier."""
    return get_settings().embed_model


def get_sparse_model_id() -> str:
    """Return the active sparse embedding model identifier."""
    return get_settings().sparse_embed_model or _DEFAULT_SPARSE_MODEL


def get_embed_display_for_meta() -> str:
    """Return a short label for the ``/meta/models`` UI."""
    b = get_embed_backend()
    if b == "openrouter":
        or_emb = _get_openrouter_embedder()
        m = or_emb.resolved_model or get_embed_model_id()
        return f"openrouter:{m}"
    return get_embed_model_id()


# ── Local fastembed ──────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_embedder(model_name: str) -> TextEmbedding:
    """Load (and cache) the local fastembed dense model."""
    return TextEmbedding(model_name=model_name)


@lru_cache(maxsize=1)
def _get_sparse_embedder(model_name: str) -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=model_name)


def _l2_normalize_rows(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (vecs / norms).astype(np.float32)


# ── OpenRouter embeddings ────────────────────────────────────────────


def _openrouter_embeddings_error_message(payload: dict) -> str | None:
    """Return an error string if the OpenRouter response contains an error field."""
    err = payload.get("error")
    if err is None:
        return None
    if isinstance(err, dict):
        return str(err.get("message") or err.get("code") or err)
    return str(err)


def _openrouter_embed_provider() -> dict:
    """Build provider routing dict; fixes «No successful provider responses» errors."""
    s = get_settings()
    raw = s.openrouter_embed_provider_json.strip()
    if raw:
        return json.loads(raw)
    order = s.openrouter_embed_provider_order.strip()
    if order:
        return {
            "order": [o.strip() for o in order.split(",") if o.strip()],
            "allow_fallbacks": True,
        }
    return {"allow_fallbacks": True}


def _openrouter_embed_models_to_try(primary: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    parts: list[str] = [primary]
    fb = get_settings().openrouter_embed_model_fallbacks.strip()
    if fb:
        parts.extend(fb.split(","))
    for p in parts:
        m = p.strip()
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _openrouter_embed_batch_once(chunk: list[str], model: str) -> np.ndarray:
    """Send a single embeddings request to OpenRouter; raises ``RuntimeError`` on failure."""
    s = get_settings()
    api_key = (s.openrouter_embed_api_key or s.openrouter_api_key).strip()
    if not api_key:
        raise RuntimeError(
            "Set OPENROUTER_API_KEY or OPENROUTER_EMBED_API_KEY when EMBED_BACKEND=openrouter"
        )
    base = s.openrouter_base_url.rstrip("/")
    timeout = max(s.openrouter_connect_timeout, s.openrouter_timeout)
    max_retries = s.openrouter_max_retries

    body_obj: dict = {
        "model": model,
        "input": chunk,
        "encoding_format": "float",
        "provider": _openrouter_embed_provider(),
    }
    body = json.dumps(body_obj).encode("utf-8")
    url = f"{base}/embeddings"

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": s.openrouter_http_referer,
            "X-Title": s.openrouter_app_title,
        },
        method="POST",
    )
    try:
        raw_bytes = urlopen_with_retries(
            req,
            timeout=timeout,
            max_retries=max_retries,
            url=url,
            log_prefix="OpenRouter-embed",
        )
    except RuntimeError as e:
        err_s = str(e)
        if "OpenRouter-embed HTTP" in err_s:
            raise
        raise RuntimeError(f"Cannot reach OpenRouter embeddings at {url}: {e}") from e

    data = json.loads(raw_bytes.decode())
    assert isinstance(data, dict)
    api_err = _openrouter_embeddings_error_message(data)
    if api_err:
        raise RuntimeError(
            f"OpenRouter embeddings failed (model={model!r}): {api_err}. "
            "Try OPENROUTER_EMBED_PROVIDER_ORDER=openai, "
            "OPENROUTER_EMBED_MODEL_FALLBACKS with another model, "
            "or EMBED_BACKEND=local with BAAI/bge-m3."
        )
    items = data.get("data") or []
    if len(items) != len(chunk):
        raise RuntimeError(
            f"OpenRouter embeddings: expected {len(chunk)} vectors, got {len(items)}; "
            f"body keys={list(data.keys())}"
        )
    items_sorted = sorted(items, key=lambda x: int(x.get("index", 0)))
    rows = []
    for it in items_sorted:
        emb = it.get("embedding")
        if not emb:
            raise RuntimeError(
                f"OpenRouter embeddings: missing embedding for index {it.get('index')}"
            )
        rows.append(np.asarray(emb, dtype=np.float32))
    return np.stack(rows, axis=0)


class OpenRouterEmbedder:
    """Thread-safe OpenRouter embedding client.

    The first successfully-resolved model is locked in for the process
    lifetime to avoid per-request negotiation overhead.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._resolved_model: str | None = None

    @property
    def resolved_model(self) -> str | None:
        """The model slug that was successfully used (or ``None`` before first call)."""
        return self._resolved_model

    def embed_batch(self, chunk: list[str], model_hint: str) -> np.ndarray:
        """Embed *chunk* using *model_hint* (or any configured fallback).

        Returns a float32 matrix of shape ``(len(chunk), dim)``.
        """
        with self._lock:
            already_resolved = self._resolved_model is not None
            primary = self._resolved_model or model_hint
            models = [primary] if already_resolved else _openrouter_embed_models_to_try(model_hint)
        last: Exception | None = None
        for cand in models:
            try:
                out = _openrouter_embed_batch_once(chunk, cand)
                with self._lock:
                    self._resolved_model = cand
                return out
            except RuntimeError as e:
                last = e
                msg = str(e)
                if already_resolved:
                    raise
                if "No successful provider" in msg or "provider responses" in msg.lower():
                    log.warning(
                        "OpenRouter embeddings: model %s failed (%s), trying next candidate",
                        cand,
                        msg[:120],
                    )
                    continue
                raise
        assert last is not None
        raise last


_openrouter_embedder_singleton: OpenRouterEmbedder | None = None
_embedder_singleton_lock = threading.Lock()


def _get_openrouter_embedder() -> OpenRouterEmbedder:
    global _openrouter_embedder_singleton
    with _embedder_singleton_lock:
        if _openrouter_embedder_singleton is None:
            _openrouter_embedder_singleton = OpenRouterEmbedder()
        return _openrouter_embedder_singleton


# ── Public API ───────────────────────────────────────────────────────


def embed_texts(texts: list[str], model_name: str | None = None) -> np.ndarray:
    """Embed a list of texts using the configured backend.

    Args:
        texts: Non-empty list of strings to embed.
        model_name: Override the model from settings.

    Returns:
        L2-normalised float32 matrix of shape ``(len(texts), dim)``.
    """
    if not texts:
        raise ValueError("embed_texts: empty texts")
    mid = model_name or get_embed_model_id()

    if get_embed_backend() == "openrouter":
        batch_size = get_settings().openrouter_embed_batch
        or_emb = _get_openrouter_embedder()
        out_parts: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            part = texts[i : i + batch_size]
            out_parts.append(or_emb.embed_batch(part, mid))
        vecs = np.concatenate(out_parts, axis=0)
        return _l2_normalize_rows(vecs)

    model = get_embedder(mid)
    vecs = np.array(list(model.embed(texts)), dtype=np.float32)
    return _l2_normalize_rows(vecs)


def embed_query(query: str, model_name: str | None = None) -> np.ndarray:
    """Embed a single query string.

    Returns a 1-D float32 vector.
    """
    return embed_texts([query], model_name=model_name)[0]


def embed_texts_sparse(
    texts: list[str],
    model_name: str | None = None,
) -> list[SparseResult]:
    """Compute sparse (BM25-style) embeddings for a list of texts.

    Always uses the local fastembed sparse model; ignores ``embed_backend``.

    Returns:
        List of ``SparseResult(indices, values)`` — one per text.
    """
    if not texts:
        raise ValueError("embed_texts_sparse: empty texts")
    mid = model_name or get_sparse_model_id()
    model = _get_sparse_embedder(mid)
    return list(model.embed(texts, batch_size=64))


def embed_query_sparse(
    query: str,
    model_name: str | None = None,
) -> SparseResult:
    """Compute a sparse embedding for a single query string."""
    return embed_texts_sparse([query], model_name=model_name)[0]


# ── ColBERT late-interaction (fastembed) ─────────────────────────────


@lru_cache(maxsize=1)
def _get_colbert_embedder(model_name: str) -> LateInteractionTextEmbedding:
    return LateInteractionTextEmbedding(model_name=model_name)


def embed_texts_colbert(
    texts: list[str],
    model_name: str | None = None,
) -> list[np.ndarray]:
    """Compute ColBERT multivector embeddings (one float32 matrix per text).

    Each matrix has shape ``(n_tokens, colbert_dim)``.

    Args:
        texts: Non-empty list of strings.
        model_name: Override the model from ``AppSettings.colbert_model``.

    Returns:
        List of float32 arrays, one per text.
    """
    if not texts:
        raise ValueError("embed_texts_colbert: empty texts")
    mid = model_name or get_settings().colbert_model
    model = _get_colbert_embedder(mid)
    return [np.array(e, dtype=np.float32) for e in model.embed(texts, batch_size=32)]


def embed_query_colbert(
    query: str,
    model_name: str | None = None,
) -> np.ndarray:
    """Compute a ColBERT multivector embedding for a single query string.

    Returns a float32 matrix of shape ``(n_query_tokens, colbert_dim)``.
    """
    return embed_texts_colbert([query], model_name=model_name)[0]
