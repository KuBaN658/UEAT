"""``GET /healthz`` (liveness) and ``GET /readyz`` (readiness -- checks the DB)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/healthz", summary="Liveness probe", operation_id="healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz", summary="Readiness probe", operation_id="readyz")
async def readyz(request: Request) -> JSONResponse:
    components: dict[str, str] = {}

    pool = getattr(request.app.state, "db_pool", None)
    if pool is None:
        components["db"] = "unconfigured"
    else:
        try:
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            components["db"] = "ok"
        except Exception as exc:
            log.warning("readyz: db check failed: %s", exc)
            components["db"] = f"error: {type(exc).__name__}"

    healthy = all(v == "ok" for v in components.values())
    return JSONResponse(
        status_code=status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": "ready" if healthy else "degraded", "components": components},
    )
