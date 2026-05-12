"""
Repository for ``conspect_jobs`` (Flow 2 queue + state).

State machine: queued -> running -> done | failed.
Raw asyncpg, no ORM -- keeps queue patterns (SKIP LOCKED, ON CONFLICT) explicit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

import asyncpg

JobStatus = Literal["queued", "running", "done", "failed"]


@dataclass(frozen=True)
class ConspectJob:
    job_id: UUID
    user_id: str
    status: JobStatus
    payload: dict
    result: dict | None
    error_code: str | None
    error_message: str | None
    retry_count: int
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    posted_at: datetime | None


def _parse_jsonb(v) -> dict | None:
    # asyncpg returns JSONB as a raw string unless a codec is registered.
    if v is None or isinstance(v, dict):
        return v
    return json.loads(v)


def _row_to_job(row: asyncpg.Record) -> ConspectJob:
    return ConspectJob(
        job_id=row["job_id"],
        user_id=row["user_id"],
        status=row["status"],
        payload=_parse_jsonb(row["payload"]) or {},
        result=_parse_jsonb(row["result"]),
        error_code=row["error_code"],
        error_message=row["error_message"],
        retry_count=row["retry_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        posted_at=row["posted_at"],
    )


async def insert_idempotent(
    conn: asyncpg.Connection, *, job_id: UUID, user_id: str, payload: dict,
) -> tuple[ConspectJob, bool]:
    """INSERT new job; on duplicate ``job_id`` return the existing row. ``(job, created)``."""
    row = await conn.fetchrow(
        """
        INSERT INTO conspect_jobs (job_id, user_id, status, payload)
        VALUES ($1, $2, 'queued', $3::jsonb)
        ON CONFLICT (job_id) DO NOTHING
        RETURNING *
        """,
        job_id, user_id, json.dumps(payload),
    )
    if row is not None:
        return _row_to_job(row), True

    existing = await conn.fetchrow("SELECT * FROM conspect_jobs WHERE job_id = $1", job_id)
    assert existing is not None, "ON CONFLICT yet no existing row -- race condition"
    return _row_to_job(existing), False


async def get_by_id(conn: asyncpg.Connection, job_id: UUID) -> ConspectJob | None:
    row = await conn.fetchrow("SELECT * FROM conspect_jobs WHERE job_id = $1", job_id)
    return _row_to_job(row) if row else None


async def claim_next_queued(conn: asyncpg.Connection) -> ConspectJob | None:
    """Atomically pick one queued job and mark it running. SKIP LOCKED -> many workers."""
    row = await conn.fetchrow(
        """
        WITH next_job AS (
            SELECT job_id FROM conspect_jobs
            WHERE status = 'queued'
            ORDER BY created_at
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        UPDATE conspect_jobs cj
        SET status = 'running', started_at = now(), updated_at = now()
        FROM next_job
        WHERE cj.job_id = next_job.job_id
        RETURNING cj.*
        """
    )
    return _row_to_job(row) if row else None


async def mark_done(conn: asyncpg.Connection, *, job_id: UUID, result: dict) -> None:
    await conn.execute(
        """
        UPDATE conspect_jobs
        SET status='done', result=$2::jsonb, completed_at=now(), updated_at=now()
        WHERE job_id=$1 AND status='running'
        """,
        job_id, json.dumps(result),
    )


async def mark_failed(
    conn: asyncpg.Connection, *, job_id: UUID, error_code: str, error_message: str,
) -> None:
    await conn.execute(
        """
        UPDATE conspect_jobs
        SET status='failed', error_code=$2, error_message=$3,
            completed_at=now(), updated_at=now()
        WHERE job_id=$1
        """,
        job_id, error_code, error_message,
    )


async def mark_posted(conn: asyncpg.Connection, *, job_id: UUID) -> None:
    await conn.execute(
        "UPDATE conspect_jobs SET posted_at=now(), updated_at=now() WHERE job_id=$1",
        job_id,
    )


async def bump_retry(
    conn: asyncpg.Connection, *, job_id: UUID, reason: str | None = None,
) -> int:
    row = await conn.fetchrow(
        """
        UPDATE conspect_jobs
        SET retry_count = retry_count + 1,
            error_message = COALESCE($2, error_message),
            updated_at = now()
        WHERE job_id = $1
        RETURNING retry_count
        """,
        job_id, reason[:500] if reason else None,
    )
    return int(row["retry_count"]) if row else 0
