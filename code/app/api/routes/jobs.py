"""POST /jobs -- enqueue a conspect generation job. GET /jobs/{id} -- poll its status."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from app.api.schemas.jobs import JobAccepted, JobCreateRequest, JobStatus
from app.core.config import get_settings
from app.infrastructure.repositories import conspect_job_repo as repo
from app.services.conspect_job import enqueue_conspect_job

log = logging.getLogger(__name__)

router = APIRouter()


def _require_bearer(request: Request) -> None:
    """When ``PLATFORM_AUTH_TOKEN`` is set, the Bearer must match it exactly.
    When unset (local dev), any Bearer is accepted."""
    configured = get_settings().platform_auth_token
    auth = request.headers.get("Authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authorization header required")
    if configured and auth[7:].strip() != configured:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")


@router.post(
    "/jobs",
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_200_OK: {"model": JobAccepted, "description": "Idempotency hit -- job already exists"},
        status.HTTP_202_ACCEPTED: {"model": JobAccepted, "description": "Job accepted"},
        status.HTTP_401_UNAUTHORIZED: {"description": "Auth missing or invalid"},
    },
    summary="Создать job на генерацию конспекта",
    operation_id="createJob",
)
async def create_job(
    body: JobCreateRequest,
    request: Request,
    response: Response,
    _: None = Depends(_require_bearer),
) -> JobAccepted:
    result = await enqueue_conspect_job(request=body, pool=request.app.state.db_pool)
    if not result.created:
        response.status_code = status.HTTP_200_OK  # duplicate job_id -> 200, not 202
    return JobAccepted(
        job_id=result.job.job_id,
        status=result.job.status,
        accepted_at=result.job.created_at or datetime.now(timezone.utc),
    )


@router.get("/jobs/{job_id}", summary="Статус job'а (polling)", operation_id="getJob")
async def get_job(
    job_id: UUID,
    request: Request,
    _: None = Depends(_require_bearer),
) -> JobStatus:
    async with request.app.state.db_pool.acquire() as conn:
        job = await repo.get_by_id(conn, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    return JobStatus(
        job_id=job.job_id,
        status=job.status,
        result=job.result,
        error_code=job.error_code,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )
