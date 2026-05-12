"""
Async Postgres connection pool. Lifecycle owned by ``app.main.lifespan``.

Raw asyncpg at runtime (not SQLAlchemy) -- lower overhead, and queue patterns
(SKIP LOCKED, ON CONFLICT, JSONB) are clearer in plain SQL. Migrations still
go through Alembic.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

import asyncpg

from app.core.config import AppSettings

log = logging.getLogger(__name__)


def _to_asyncpg_dsn(database_url: str) -> str:
    """Strip the ``+asyncpg`` SQLAlchemy driver suffix used by Alembic."""
    return database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def create_pool(settings: AppSettings) -> asyncpg.Pool:
    dsn = _to_asyncpg_dsn(settings.database_url)
    pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=settings.db_pool_min_size,
        max_size=settings.db_pool_max_size,
        timeout=settings.db_pool_timeout_seconds,
        command_timeout=30.0,
    )
    log.info(
        "Postgres pool ready: %s (min=%d, max=%d)",
        dsn.rsplit("@", 1)[-1],
        settings.db_pool_min_size,
        settings.db_pool_max_size,
    )
    return pool


async def close_pool(pool: asyncpg.Pool) -> None:
    await pool.close()


async def acquire(pool: asyncpg.Pool) -> AsyncIterator[asyncpg.Connection]:
    """FastAPI-style dependency: ``async for conn in acquire(pool)``."""
    async with pool.acquire() as conn:
        yield conn
