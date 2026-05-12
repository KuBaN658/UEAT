"""
Qdrant-backed vector store for hybrid dense+sparse (+optional ColBERT) search.

Connection settings are read from ``AppSettings`` (``qdrant_url``,
``qdrant_collection``).  The collection uses named vectors:

- ``"dense"``   — float vector (cosine similarity).
- ``"sparse"``  — sparse vector (BM25-style term weights).
- ``"colbert"`` — multivector (token-level ColBERT embeddings, optional).

Hybrid search supports two fusion modes:

- **RRF** (default, ``query_text=None``): Qdrant server-side prefetch + RRF.
- **Reranker** (``query_text`` provided): dense + sparse candidates are merged
  client-side and reranked via Cohere ``/v1/rerank`` through OpenRouter.
  ColBERT is not used in the reranker path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import requests
from qdrant_client import QdrantClient, models

from app.core.config import get_settings

log = logging.getLogger(__name__)

_DENSE_NAME = "dense"
_SPARSE_NAME = "sparse"
_COLBERT_NAME = "colbert"
_CONTENT_HASH_KEY = "content_hash"


@dataclass
class IndexedDoc:
    """A document stored in the Qdrant collection."""

    doc_id: str
    text: str
    meta: dict


@dataclass
class QueryHit:
    """A single retrieval result from Qdrant."""

    doc: IndexedDoc
    score: float


class QdrantStore:
    """Persistent Qdrant collection with hybrid dense+sparse retrieval.

    Args:
        dim: Dimensionality of the dense embedding vectors.
    """

    def __init__(self, dim: int) -> None:
        s = get_settings()
        self.dim = dim
        self._collection = s.qdrant_collection
        self._client = QdrantClient(url=s.qdrant_url)

    # ── Collection lifecycle ─────────────────────────────────────────

    def _get_stored_hash(self) -> str | None:
        """Read the content hash stored in point payloads (first matching point)."""
        try:
            result = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=_CONTENT_HASH_KEY,
                            match=models.MatchExcept(**{"except": [""]}),
                        )
                    ]
                ),
                limit=1,
                with_payload=[_CONTENT_HASH_KEY],
            )
            points = result[0]
            if points:
                p = points[0].payload
                if p:
                    return p.get(_CONTENT_HASH_KEY)
        except Exception:
            log.error("Error getting stored hash for %r", self._collection, exc_info=True)
        return None

    def _stored_dense_dim(self) -> int | None:
        """Return the dense vector dimension of the existing collection, if any."""
        if not self._client.collection_exists(self._collection):
            return None
        try:
            info = self._client.get_collection(self._collection)
            vc = info.config.params.vectors
            if vc is None:
                return None
            if isinstance(vc, dict):
                vp = vc.get(_DENSE_NAME)
                return int(vp.size) if vp is not None else None
            return int(vc.size)
        except Exception:
            log.error("Error reading dense dim for %r", self._collection, exc_info=True)
            return None

    def _has_colbert_vector(self) -> bool:
        """Return True if the collection already has a ColBERT multivector configured."""
        if not self._client.collection_exists(self._collection):
            return False
        try:
            info = self._client.get_collection(self._collection)
            vc = info.config.params.vectors
            return isinstance(vc, dict) and _COLBERT_NAME in vc
        except Exception:
            log.error("Error checking ColBERT config for %r", self._collection, exc_info=True)
            return False

    def _has_sparse_vector(self) -> bool:
        """Return True if the collection already has a BM25 sparse vector configured."""
        if not self._client.collection_exists(self._collection):
            return False
        try:
            info = self._client.get_collection(self._collection)
            sv = info.config.params.sparse_vectors
            return sv is not None and _SPARSE_NAME in sv
        except Exception:
            log.error("Error checking BM25 sparse config for %r", self._collection, exc_info=True)
            return False

    def is_populated(
        self,
        expected_count: int,
        content_hash: str,
        bm25_enabled: bool = True,
        colbert_enabled: bool = False,
    ) -> bool:
        """Return True if the collection already has *expected_count* points
        with the given *content_hash*, the correct dense dimension and the
        expected BM25/ColBERT vector presence."""
        if not self._client.collection_exists(self._collection):
            return False
        stored_dim = self._stored_dense_dim()
        if stored_dim is not None and stored_dim != self.dim:
            log.info(
                "Qdrant %r dense dim %d != embedder %d; re-indexing",
                self._collection,
                stored_dim,
                self.dim,
            )
            return False
        has_sparse = self._has_sparse_vector()
        if bm25_enabled != has_sparse:
            log.info(
                "Qdrant %r BM25 sparse presence mismatch (stored=%s, wanted=%s); re-indexing",
                self._collection,
                has_sparse,
                bm25_enabled,
            )
            return False
        has_colbert = self._has_colbert_vector()
        if colbert_enabled != has_colbert:
            log.info(
                "Qdrant %r ColBERT presence mismatch (stored=%s, wanted=%s); re-indexing",
                self._collection,
                has_colbert,
                colbert_enabled,
            )
            return False
        try:
            info = self._client.get_collection(self._collection)
            if info.points_count != expected_count:
                return False
        except Exception:
            log.error("Error getting collection info for %r", self._collection, exc_info=True)
            return False
        stored = self._get_stored_hash()
        return stored == content_hash

    def ensure_collection(
        self,
        bm25_enabled: bool = True,
        colbert_dim: int | None = None,
    ) -> None:
        """Delete and recreate the collection with the requested schema.

        Args:
            bm25_enabled: When True (default), adds a ``"sparse"`` BM25 vector field.
            colbert_dim: When provided, adds a ``"colbert"`` multivector with this
                token-embedding dimensionality (e.g. 128 for colbert-ir/colbertv2.0).
        """
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)

        vectors_config: dict[str, models.VectorParams] = {
            _DENSE_NAME: models.VectorParams(
                size=self.dim,
                distance=models.Distance.COSINE,
            ),
        }
        if colbert_dim is not None:
            vectors_config[_COLBERT_NAME] = models.VectorParams(
                size=colbert_dim,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
            )

        sparse_vectors_config = (
            {_SPARSE_NAME: models.SparseVectorParams()} if bm25_enabled else None
        )
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )
        for fname, fschema in [
            ("task_number", models.PayloadSchemaType.INTEGER),
            ("subtypes", models.PayloadSchemaType.KEYWORD),
        ]:
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name=fname,
                field_schema=fschema,
            )
        log.info(
            "Created Qdrant collection %r (dim=%d, bm25=%s, colbert=%s)",
            self._collection,
            self.dim,
            bm25_enabled,
            colbert_dim,
        )

    # ── Upsert ───────────────────────────────────────────────────────

    def upsert_batch(
        self,
        docs: list[IndexedDoc],
        dense_vecs: np.ndarray,
        sparse_vecs: list[models.SparseVector] | None,
        content_hash: str,
        colbert_vecs: list[np.ndarray] | None = None,
        batch_size: int = 32,
    ) -> None:
        """Upsert documents into Qdrant in batches.

        Args:
            docs: Documents to store.
            dense_vecs: Dense vectors; shape ``(len(docs), dim)``.
            sparse_vecs: Sparse vectors; one per document, or ``None`` to skip BM25.
            content_hash: Hash stored in each point payload for staleness checks.
            colbert_vecs: Optional ColBERT multivectors; one float32 matrix
                ``(n_tokens, colbert_dim)`` per document.
            batch_size: Number of points per upsert request.
        """
        dense_vecs = np.asarray(dense_vecs, dtype=np.float32)
        n = len(docs)
        if dense_vecs.shape[0] != n:
            raise ValueError("docs / dense_vecs length mismatch")
        if sparse_vecs is not None and len(sparse_vecs) != n:
            raise ValueError("sparse_vecs length mismatch")
        if colbert_vecs is not None and len(colbert_vecs) != n:
            raise ValueError("colbert_vecs length mismatch")

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            points = []
            for i in range(start, end):
                doc = docs[i]
                vector: dict = {_DENSE_NAME: dense_vecs[i].tolist()}
                if sparse_vecs is not None:
                    vector[_SPARSE_NAME] = sparse_vecs[i]
                if colbert_vecs is not None:
                    # ColBERT multivector: list of token-level vectors
                    vector[_COLBERT_NAME] = colbert_vecs[i].tolist()
                points.append(
                    models.PointStruct(
                        id=i,
                        vector=vector,
                        payload={
                            "doc_id": doc.doc_id,
                            "text": doc.text,
                            **doc.meta,
                            _CONTENT_HASH_KEY: content_hash,
                        },
                    )
                )
            self._client.upsert(collection_name=self._collection, points=points)

        log.info("Upserted %d points into %r", n, self._collection)

    # ── Search ───────────────────────────────────────────────────────

    def hybrid_search(
        self,
        dense_vec: np.ndarray,
        sparse_vec: models.SparseVector | None,
        *,
        query_text: str | None = None,
        colbert_vec: np.ndarray | None = None,
        task_number: int | None = None,
        subtypes: list[str] | None = None,
        limit: int = 10,
        prefetch_factor: int = 1,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        colbert_weight: float = 1.0,
        rrf_k: int = 60,
    ) -> list[QueryHit]:
        """Hybrid dense+sparse (+ optional ColBERT) search with optional Cohere reranking.

        When *query_text* is provided the method fetches candidates from both
        the dense and sparse legs independently, deduplicates them, and
        reranks via ``AppSettings.reranker_model`` through OpenRouter
        ``/v1/rerank``.  On reranker failure it falls back silently to
        dense-score ordering.  ColBERT is not used in the reranker path.

        When *query_text* is ``None`` the original server-side RRF fusion is
        used.  Passing *colbert_vec* adds a third prefetch leg (ColBERT
        MaxSim) to the RRF fusion.

        Args:
            dense_vec: Query dense vector.
            sparse_vec: Query sparse vector.
            query_text: Raw query string — required to enable reranking.
            colbert_vec: Query ColBERT multivector ``(n_tokens, colbert_dim)``.
                When provided (and ``query_text`` is ``None``), adds a ColBERT
                prefetch leg to the RRF fusion.
            task_number: If set, filter results to this EGE task number.
            subtypes: If set, filter results to these subtypes.
            limit: Maximum number of results to return.
            prefetch_factor: Prefetch ``limit * prefetch_factor`` candidates per leg.
            dense_weight: RRF weight for the dense leg (RRF mode only).
            bm25_weight: RRF weight for the sparse leg (RRF mode only); ignored when ``sparse_vec`` is ``None``.
            colbert_weight: RRF weight for the ColBERT leg (RRF mode only).
            rrf_k: RRF constant *k* (Cormack et al. 2009) (RRF mode only).

        Returns:
            List of ``QueryHit`` sorted by score descending.
        """
        qf = self._build_filter(task_number, subtypes)
        prefetch_limit = limit * prefetch_factor

        d = np.asarray(dense_vec, dtype=np.float32)
        if d.ndim != 1:
            d = d.flatten()

        if query_text is not None:
            return self._hybrid_search_rerank(d, sparse_vec, qf, query_text, limit, prefetch_limit)

        prefetches: list[models.Prefetch] = [
            models.Prefetch(
                query=d.tolist(),
                using=_DENSE_NAME,
                limit=prefetch_limit,
                filter=qf,
            ),
        ]
        rrf_weights: list[float] = [dense_weight]

        if sparse_vec is not None:
            prefetches.append(
                models.Prefetch(
                    query=sparse_vec,
                    using=_SPARSE_NAME,
                    limit=prefetch_limit,
                    filter=qf,
                )
            )
            rrf_weights.append(bm25_weight)

        if colbert_vec is not None:
            colbert_mat = np.asarray(colbert_vec, dtype=np.float32)
            prefetches.append(
                models.Prefetch(
                    query=colbert_mat.tolist(),
                    using=_COLBERT_NAME,
                    limit=prefetch_limit,
                    filter=qf,
                )
            )
            rrf_weights.append(colbert_weight)

        result = self._client.query_points(
            collection_name=self._collection,
            prefetch=prefetches,
            query=models.RrfQuery(rrf=models.Rrf(k=rrf_k, weights=rrf_weights)),
            limit=limit,
            with_payload=True,
        )

        return self._hits_from_points(result.points)

    def _hybrid_search_rerank(
        self,
        dense_vec_1d: np.ndarray,
        sparse_vec: models.SparseVector | None,
        qf: models.Filter | None,
        query_text: str,
        limit: int,
        prefetch_limit: int,
    ) -> list[QueryHit]:
        """Fetch candidates from dense (+ optional sparse) legs, merge, and rerank with Cohere."""
        dense_res = self._client.query_points(
            collection_name=self._collection,
            query=dense_vec_1d.tolist(),
            using=_DENSE_NAME,
            query_filter=qf,
            limit=prefetch_limit,
            with_payload=True,
        )
        sparse_points = []
        if sparse_vec is not None:
            sparse_res = self._client.query_points(
                collection_name=self._collection,
                query=sparse_vec,
                using=_SPARSE_NAME,
                query_filter=qf,
                limit=prefetch_limit,
                with_payload=True,
            )
            sparse_points = sparse_res.points

        # Merge candidates, keep first occurrence (dense-leg ordering preserved).
        seen: dict[str, QueryHit] = {}
        for pt in dense_res.points + sparse_points:
            payload = pt.payload or {}
            doc_id = payload.get("doc_id", str(pt.id))
            if doc_id not in seen:
                doc = IndexedDoc(
                    doc_id=doc_id,
                    text=payload.get("text", ""),
                    meta={
                        k: v
                        for k, v in payload.items()
                        if k not in ("doc_id", "text", _CONTENT_HASH_KEY)
                    },
                )
                seen[doc_id] = QueryHit(doc=doc, score=float(pt.score))

        candidates = list(seen.values())
        if not candidates:
            return candidates

        try:
            return self._call_reranker(query_text, candidates, top_n=limit)
        except Exception:
            log.warning(
                "Cohere reranker failed; falling back to dense-score ordering",
                exc_info=True,
            )
            return sorted(candidates, key=lambda h: h.score, reverse=True)[:limit]

    def _call_reranker(
        self,
        query: str,
        hits: list[QueryHit],
        top_n: int,
    ) -> list[QueryHit]:
        """POST to OpenRouter /v1/rerank and return reranked QueryHits."""
        s = get_settings()
        url = s.openrouter_base_url.rstrip("/") + "/rerank"
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {s.openrouter_api_key}",
                "HTTP-Referer": s.openrouter_http_referer,
                "X-Title": s.openrouter_app_title,
                "Content-Type": "application/json",
            },
            json={
                "model": s.reranker_model,
                "query": query,
                "documents": [h.doc.text for h in hits],
                "top_n": top_n,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()

        # Cohere returns "results"; OpenRouter may use "results" or "data"
        items = data.get("results") or data.get("data")
        if not items:
            log.error(
                "Reranker response missing 'results'/'data' key. Full response: %s",
                data,
            )
            raise KeyError("results")

        reranked: list[QueryHit] = []
        for item in items:
            idx: int = item["index"]
            score: float = item["relevance_score"]
            reranked.append(QueryHit(doc=hits[idx].doc, score=score))
        log.debug("Reranker returned %d results (model=%s)", len(reranked), s.reranker_model)
        return reranked

    def dense_search(
        self,
        dense_vec: np.ndarray,
        *,
        task_number: int | None = None,
        limit: int = 48,
    ) -> list[QueryHit]:
        """Dense-only fallback search.

        Args:
            dense_vec: Query dense vector.
            task_number: Optional EGE task filter.
            limit: Maximum results.
        """
        qf = self._build_filter(task_number, None)

        d = np.asarray(dense_vec, dtype=np.float32)
        if d.ndim != 1:
            d = d.flatten()

        result = self._client.query_points(
            collection_name=self._collection,
            query=d.tolist(),
            using=_DENSE_NAME,
            query_filter=qf,
            limit=limit,
            with_payload=True,
        )

        return self._hits_from_points(result.points)

    def count(self) -> int:
        """Return the number of points in the collection (0 on error)."""
        try:
            info = self._client.get_collection(self._collection)
            return info.points_count or 0
        except Exception:
            return 0

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_filter(
        task_number: int | None,
        subtypes: list[str] | None,
    ) -> models.Filter | None:
        must: list[models.FieldCondition] = []
        if task_number is not None:
            must.append(
                models.FieldCondition(
                    key="task_number",
                    match=models.MatchValue(value=task_number),
                )
            )
        if subtypes:
            must.append(
                models.FieldCondition(
                    key="subtypes",
                    match=models.MatchAny(any=subtypes),
                )
            )
        return models.Filter(must=must) if must else None

    @staticmethod
    def _hits_from_points(points: list) -> list[QueryHit]:
        out: list[QueryHit] = []
        for pt in points:
            payload = pt.payload or {}
            doc = IndexedDoc(
                doc_id=payload.get("doc_id", ""),
                text=payload.get("text", ""),
                meta={
                    k: v
                    for k, v in payload.items()
                    if k not in ("doc_id", "text", _CONTENT_HASH_KEY)
                },
            )
            out.append(QueryHit(doc=doc, score=float(pt.score)))
        return out
