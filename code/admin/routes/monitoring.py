"""Read-only endpoints that power the monitoring dashboard."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from admin.monitor import eval_worker_lag
from admin.store import _pct
from app.core.config import get_settings

router = APIRouter()


@router.get("/readyz")
async def readyz(request: Request):
    kafka_ok = request.app.state.kafka_producer is not None
    db_ok = request.app.state.db_pool is not None
    code = 200 if (kafka_ok and db_ok) else 503
    return JSONResponse({"kafka": kafka_ok, "db": db_ok}, status_code=code)


@router.get("/api/submissions")
async def api_submissions(request: Request, limit: int = 100):
    return request.app.state.submission_store.all()[:limit]


@router.get("/api/stats")
async def api_stats(request: Request):
    stats = request.app.state.submission_store.stats()
    stats["kafka_lag"] = await eval_worker_lag(get_settings())
    return stats


@router.get("/api/job-stats")
async def api_job_stats(request: Request):
    pool = request.app.state.db_pool
    if pool is None:
        return JSONResponse({"detail": "DB unavailable"}, status_code=503)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT status, started_at, completed_at, created_at
            FROM conspect_jobs
            ORDER BY created_at DESC
            LIMIT 500
            """
        )

    counts = {"queued": 0, "running": 0, "done": 0, "failed": 0}
    latencies_ms: list[float] = []
    oldest_queued_s: float | None = None
    now = datetime.now(timezone.utc)

    for r in rows:
        status = r["status"]
        if status in counts:
            counts[status] += 1
        if status == "done" and r["started_at"] and r["completed_at"]:
            latencies_ms.append((r["completed_at"] - r["started_at"]).total_seconds() * 1000)
        elif status == "queued" and r["created_at"]:
            age = (now - r["created_at"]).total_seconds()
            if oldest_queued_s is None or age > oldest_queued_s:
                oldest_queued_s = age

    return {
        "counts": {**counts, "total": sum(counts.values())},
        "latency_ms": {
            "avg": sum(latencies_ms) / len(latencies_ms) if latencies_ms else None,
            "p50": _pct(latencies_ms, 0.50),
            "p95": _pct(latencies_ms, 0.95),
            "samples": len(latencies_ms),
        },
        "oldest_queued_s": oldest_queued_s,
    }


@router.get("/api/jobs")
async def api_jobs(request: Request, limit: int = 50):
    pool = request.app.state.db_pool
    if pool is None:
        return JSONResponse({"detail": "DB unavailable"}, status_code=503)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT job_id, user_id, status, error_code, error_message,
                   retry_count, created_at, updated_at, started_at,
                   completed_at, posted_at
            FROM conspect_jobs
            ORDER BY created_at DESC
            LIMIT $1
            """,
            limit,
        )

    def _coerce(v):
        if isinstance(v, UUID):
            return str(v)
        if isinstance(v, datetime):
            return v.isoformat()
        return v

    return [{k: _coerce(v) for k, v in dict(r).items()} for r in rows]
