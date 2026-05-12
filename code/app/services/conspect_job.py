"""Enqueue a conspect generation job. Workers pick it up from `conspect_jobs`."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import asyncpg

from app.api.schemas.jobs import JobCreateRequest
from app.infrastructure.repositories import conspect_job_repo as repo

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnqueueResult:
    job: repo.ConspectJob
    created: bool  # False if a row with the same job_id already existed


async def enqueue_conspect_job(
    *, request: JobCreateRequest, pool: asyncpg.Pool
) -> EnqueueResult:
    """Persist the request as a queued job. ``job_id`` is the idempotency key."""
    payload = request.model_dump(mode="json", exclude={"job_id", "user_id"})
    async with pool.acquire() as conn:
        job, created = await repo.insert_idempotent(
            conn,
            job_id=request.job_id,
            user_id=request.user_id,
            payload=payload,
        )
    if created:
        log.info(
            "conspect_job enqueued: job_id=%s user_id=%s submissions=%d",
            request.job_id, request.user_id, len(request.submissions),
        )
    else:
        log.info("conspect_job idempotency hit: job_id=%s status=%s", request.job_id, job.status)
    return EnqueueResult(job=job, created=created)
