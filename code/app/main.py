"""
FastAPI app factory for the EGE Generator API.

Routes:
- ``POST /jobs``, ``GET /jobs/{id}`` -- conspect generation
- ``GET /healthz``, ``GET /readyz`` -- liveness / readiness

Task serving and submission collection live in ``admin/app.py``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import asyncio

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import health as health_routes
from app.api.routes import jobs as jobs_routes
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.infrastructure.repositories.db import close_pool, create_pool

log = logging.getLogger(__name__)

_DB_RETRY_DELAYS = [2, 5, 10, 20]  # seconds between attempts (4 retries total)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    s = get_settings()
    log.info("Starting EGE RAG service (backend=%s)", s.llm_backend)

    app.state.db_pool = None
    for attempt, delay in enumerate([0] + _DB_RETRY_DELAYS, start=1):
        if delay:
            await asyncio.sleep(delay)
        try:
            app.state.db_pool = await create_pool(s)
            break
        except Exception as exc:
            if attempt <= len(_DB_RETRY_DELAYS):
                log.warning("DB pool attempt %d failed, retrying in %ds: %s", attempt, _DB_RETRY_DELAYS[attempt - 1], exc)
            else:
                log.warning("DB pool unavailable after %d attempts, /jobs endpoint will fail: %s", attempt, exc)

    yield

    if getattr(app.state, "db_pool", None) is not None:
        await close_pool(app.state.db_pool)
    log.info("EGE RAG service stopped.")


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, (HTTPException, RequestValidationError)):
        raise exc
    log.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "path": str(request.url.path)},
    )


def create_app() -> FastAPI:
    application = FastAPI(title="EGE Generator API", version="0.5.0", lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:8100", "http://localhost:8100"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    application.add_exception_handler(Exception, _unhandled_exception_handler)
    application.include_router(jobs_routes.router)
    application.include_router(health_routes.router)
    return application


app = create_app()

__all__ = ["app"]
