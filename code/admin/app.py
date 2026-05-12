"""
EGE Admin -- local platform simulator + monitoring dashboard.

Wires together:
- platform-side task endpoints (admin/routes/tasks.py)
- callback receivers from our workers (admin/routes/callbacks.py)
- read-only dashboard endpoints (admin/routes/monitoring.py)
- /jobs proxy to the generator API (admin/routes/proxy.py)
- a background Kafka consumer that mirrors submissions into an in-memory store

Run::

    uv run uvicorn admin.app:app --port 8100
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from aiokafka import AIOKafkaProducer
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from admin.monitor import run_monitor_consumer
from admin.routes import callbacks, monitoring, proxy, tasks
from admin.store import SubmissionStore
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.infrastructure.repositories.db import close_pool, create_pool
from app.infrastructure.repositories.task_repo import TaskBank, load_tasks

log = logging.getLogger(__name__)

_STATIC = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    s = get_settings()
    log.info("starting ege-admin, db=%s", s.database_url.rsplit("@", 1)[-1])

    app.state.submission_store = SubmissionStore()

    try:
        app.state.db_pool = await create_pool(s)
    except Exception as exc:
        log.warning("DB pool unavailable: %s", exc)
        app.state.db_pool = None

    try:
        app.state.task_bank = TaskBank(load_tasks(s.fipi_parsed_json), seed=42)
    except Exception as exc:
        log.warning("task bank unavailable: %s", exc)
        app.state.task_bank = None

    try:
        producer = AIOKafkaProducer(bootstrap_servers=s.kafka_bootstrap_servers)
        await producer.start()
        app.state.kafka_producer = producer
    except Exception as exc:
        log.warning("Kafka producer unavailable: %s", exc)
        app.state.kafka_producer = None

    monitor_task = asyncio.create_task(run_monitor_consumer(app.state.submission_store, s))

    api_base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    app.state.api_client = httpx.AsyncClient(base_url=api_base_url, timeout=300.0)

    yield

    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass

    await app.state.api_client.aclose()
    if app.state.kafka_producer is not None:
        await app.state.kafka_producer.stop()
    if app.state.db_pool is not None:
        await close_pool(app.state.db_pool)
    log.info("ege-admin stopped")


app = FastAPI(title="EGE Admin", lifespan=lifespan)
app.include_router(tasks.router)
app.include_router(callbacks.router)
app.include_router(monitoring.router)
app.include_router(proxy.router)


if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(_STATIC / "index.html"))
