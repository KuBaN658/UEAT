"""
Conspect-worker -- polls ``conspect_jobs``, runs the RAG+LLM pipeline,
and POSTs the result back to the platform (Flow 2).

Per-job:
    1. claim row (SELECT FOR UPDATE SKIP LOCKED)
    2. materialize profile from job.payload.submissions
    3. RAG + LLM (existing pipeline)
    4. mark_done with result
    5. POST callback; on success -> mark_posted

Run with::

    uv run python -m app.workers.conspect_worker
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import signal
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import asyncpg

from app.api.schemas.jobs import Submission
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.domain.analysis import top_errors_with_scores
from app.domain.profile import StudentProfile
from app.infrastructure.http.platform_client import (
    PermanentDeliveryError,
    PlatformClient,
    TransientDeliveryError,
)
from app.infrastructure.llm.clients import get_llm_provider_and_model
from app.infrastructure.repositories import conspect_job_repo as repo
from app.infrastructure.repositories.db import close_pool, create_pool
from app.infrastructure.retrieval.engine import RagEngine
from app.services.conspect import generate_conspect_rag_answer
from app.services.profile_materializer import materialize_profile_from_history

log = logging.getLogger(__name__)

POLL_INTERVAL_SEC = 1.0
PROMPT_VERSION = "v1"  # TODO: derive from conspect_prompts.yaml hash


class _InMemoryProfileStore:
    """Duck-typed ProfileStore for one job -- returns the same materialized profile."""

    def __init__(self, profile: StudentProfile) -> None:
        self._profile = profile

    def load(self, student_id: str) -> StudentProfile:  # noqa: ARG002
        return self._profile

    def save(self, profile: StudentProfile) -> None:  # noqa: ARG002
        return None


def _submissions_hash(submissions: list[Submission]) -> str:
    """SHA-256 over sorted submission_ids -- used in source_signal_snapshot."""
    ids = sorted(str(s.submission_id) for s in submissions)
    h = hashlib.sha256()
    for sid in ids:
        h.update(sid.encode("ascii"))
    return f"sha256:{h.hexdigest()[:32]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_callback_payload(
    *,
    job_id: UUID,
    user_id: str,
    text: str,
    profile: StudentProfile,
    submissions: list[Submission],
    snapshot_at: str,
    model_name: str,
) -> dict[str, Any]:
    top_tags = [
        {"tag": tag, "score": round(score, 3)}
        for tag, score in top_errors_with_scores(profile, n=8, subtype=None)
    ]
    return {
        "job_id": str(job_id),
        "user_id": user_id,
        "text": text,
        "source_signal_snapshot": {
            "schema_version": "1.0",
            "snapshot_at": snapshot_at,
            "top_error_tags": top_tags,
            "submissions_count": len(submissions),
            "submissions_hash": _submissions_hash(submissions),
            "model": model_name,
            "prompt_version": PROMPT_VERSION,
        },
        # TODO: wire up real auto_check via app.rag.conspect_quality
        "auto_check_status": "PASSED",
        "auto_check_score": None,
        "auto_check_notes": None,
        "model_name": model_name,
        "prompt_version": PROMPT_VERSION,
        "generated_at": _now(),
    }


async def _process_job(
    job: repo.ConspectJob,
    *,
    pool: asyncpg.Pool,
    rag: RagEngine,
    platform: PlatformClient,
) -> None:
    snapshot_at = _now()

    submissions = [Submission(**s) for s in job.payload.get("submissions", [])]
    log.info(
        "processing job_id=%s user_id=%s submissions=%d",
        job.job_id, job.user_id, len(submissions),
    )

    profile = materialize_profile_from_history(job.user_id, submissions)
    profile_store = _InMemoryProfileStore(profile)

    # LangGraph is sync; run in executor so it doesn't block the event loop.
    loop = asyncio.get_running_loop()
    try:
        text, _retrieved, _frontier = await loop.run_in_executor(
            None, generate_conspect_rag_answer, job.user_id, profile_store, rag,
        )
    except Exception as exc:
        log.exception("LLM pipeline failed for job %s", job.job_id)
        async with pool.acquire() as conn:
            await repo.mark_failed(
                conn,
                job_id=job.job_id,
                error_code="LLM_PIPELINE_ERROR",
                error_message=str(exc)[:500],
            )
        return

    _, model_name = get_llm_provider_and_model()
    payload = _build_callback_payload(
        job_id=job.job_id,
        user_id=job.user_id,
        text=text,
        profile=profile,
        submissions=submissions,
        snapshot_at=snapshot_at,
        model_name=model_name,
    )

    async with pool.acquire() as conn:
        await repo.mark_done(conn, job_id=job.job_id, result=payload)

    cb = job.payload.get("callback") or {}
    callback_url = cb.get("url")
    if not callback_url:
        log.warning("job %s has no callback.url, skipping delivery", job.job_id)
        return

    try:
        await platform.post_conspect(
            callback_url=callback_url,
            callback_token=cb.get("auth"),
            job_id=job.job_id,
            payload=payload,
        )
    except PermanentDeliveryError as exc:
        log.error("permanent delivery error for job %s: %s", job.job_id, exc)
        async with pool.acquire() as conn:
            await repo.mark_failed(
                conn,
                job_id=job.job_id,
                error_code="CALLBACK_REJECTED",
                error_message=str(exc)[:500],
            )
        return
    except TransientDeliveryError as exc:
        # Job stays 'done' -- result is persisted; a sweep can re-deliver.
        log.error("transient delivery exhausted for job %s: %s", job.job_id, exc)
        async with pool.acquire() as conn:
            await repo.bump_retry(conn, job_id=job.job_id)
        return

    async with pool.acquire() as conn:
        await repo.mark_posted(conn, job_id=job.job_id)
    log.info("job %s delivered", job.job_id)


async def run_worker(*, poll_interval: float = POLL_INTERVAL_SEC) -> None:
    configure_logging()
    settings = get_settings()
    log.info("starting conspect-worker - db=%s", settings.database_url.rsplit("@", 1)[-1])

    pool = await create_pool(settings)
    rag = RagEngine()
    platform = PlatformClient(
        base_url=settings.platform_base_url,
        default_token=settings.platform_auth_token,
    )

    stop = asyncio.Event()
    if os.name != "nt":  # asyncio signal handlers are POSIX-only
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)

    try:
        while not stop.is_set():
            async with pool.acquire() as conn:
                job = await repo.claim_next_queued(conn)
            if job is None:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=poll_interval)
                except asyncio.TimeoutError:
                    pass
                continue
            try:
                await _process_job(job, pool=pool, rag=rag, platform=platform)
            except Exception:
                log.exception("unhandled error processing job %s", job.job_id)
                async with pool.acquire() as conn:
                    await repo.mark_failed(
                        conn,
                        job_id=job.job_id,
                        error_code="UNHANDLED",
                        error_message="see worker logs",
                    )
    finally:
        await close_pool(pool)
        await platform.aclose()
        log.info("conspect-worker stopped")


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
