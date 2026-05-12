"""
Direct Kafka consumer for submission diagnosis (Flow 1).

Reads ``ege.submissions.v1``, runs LLM diagnosis, POSTs diagnostic-tags
callback. Offset is committed after the batch finishes -- replay after a
crash is safe because diagnosis is deterministic.

Run with::

    uv run python -m app.workers.eval_worker
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from datetime import datetime, timezone
from uuid import UUID

from aiokafka import AIOKafkaConsumer

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.infrastructure.http.platform_client import (
    PermanentDeliveryError,
    PlatformClient,
    TransientDeliveryError,
)
from app.infrastructure.llm.clients import build_chat_client_optional, get_llm_provider_and_model
from app.services.diagnosis import diagnose_with_pipeline

log = logging.getLogger(__name__)

_SUPPORTED_TASK_NUMBERS = {6, 10, 12, 13, 16}
_REQUIRED_FIELDS = ("submission_id", "user_id", "task_id", "task_number", "task_text", "is_correct")


def _parse_event(raw: bytes) -> dict | None:
    """Return the submission dict, or None if the message should be skipped."""
    try:
        ev = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.warning("bad JSON in Kafka message: %s", exc)
        return None

    if not isinstance(ev, dict) or ev.get("event_type") != "submission.created":
        return None

    sub = ev.get("submission")
    if not isinstance(sub, dict):
        log.warning("event has no submission object")
        return None

    missing = [k for k in _REQUIRED_FIELDS if k not in sub]
    if missing:
        log.warning("submission missing fields: %s", missing)
        return None

    if int(sub["task_number"]) not in _SUPPORTED_TASK_NUMBERS:
        return None

    return sub


def _diagnose(sub: dict, llm) -> tuple[list[str], str | None]:
    hyps = diagnose_with_pipeline(
        task_number=int(sub["task_number"]),
        task_text=str(sub.get("task_text") or ""),
        correct=str(sub.get("correct_answer") or ""),
        student=str(sub.get("student_answer") or ""),
        llm=llm,
    )
    if not hyps:
        return [], None
    tags = [h.tag for h in hyps]
    rationale = " - ".join(h.reason for h in hyps if h.reason)[:1000] or None
    return tags, rationale


async def _process(sub: dict, *, platform: PlatformClient, llm, model_name: str) -> bool:
    """Process one message. Returns True if the message should be replayed (transient failure)."""
    sid = sub["submission_id"]
    if sub.get("is_correct"):
        return False

    loop = asyncio.get_running_loop()
    try:
        tags, rationale = await loop.run_in_executor(None, _diagnose, sub, llm)
    except Exception as exc:
        # Same input -> same failure, so replay would not help. Commit and move on.
        log.exception("LLM diagnosis failed for %s: %s", sid, exc)
        return False

    payload = {
        "submission_id": sid,
        "tags": tags,
        "rationale": rationale,
        "model": model_name,
        "diagnosed_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        await platform.post_diagnostic_tags(submission_id=UUID(sid), payload=payload)
        log.info("diagnosed+posted %s tags=%s", sid, tags)
        return False
    except PermanentDeliveryError as exc:
        log.error("permanent callback failure for %s: %s", sid, exc)
        return False
    except TransientDeliveryError as exc:
        log.warning("transient callback failure for %s: %s -- will replay", sid, exc)
        return True


async def run_worker() -> None:
    configure_logging()
    settings = get_settings()
    log.info(
        "starting eval-worker - topic=%s bootstrap=%s",
        settings.kafka_submission_topic,
        settings.kafka_bootstrap_servers,
    )

    consumer = AIOKafkaConsumer(
        settings.kafka_submission_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id="ege-eval-worker",
        enable_auto_commit=False,
        auto_offset_reset="latest",
    )
    platform = PlatformClient(
        base_url=settings.platform_base_url,
        default_token=settings.platform_auth_token,
    )
    llm = build_chat_client_optional()
    if llm is None:
        log.warning("no LLM client -- tags will be empty")
    _, model_name = get_llm_provider_and_model()

    stop = asyncio.Event()
    if os.name != "nt":  # asyncio signal handlers are POSIX-only
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)

    await consumer.start()
    log.info("consumer ready")
    try:
        while not stop.is_set():
            records = await consumer.getmany(max_records=8, timeout_ms=1000)
            subs = [s for msgs in records.values() for s in (_parse_event(m.value) for m in msgs) if s]
            replay = False
            if subs:
                results = await asyncio.gather(
                    *(_process(s, platform=platform, llm=llm, model_name=model_name) for s in subs)
                )
                replay = any(results)

            if replay:
                # Skip commit so Kafka redelivers. Successful messages are re-processed,
                # which is safe -- platform dedupes via Idempotency-Key=submission_id.
                log.warning("transient failure in batch -- skipping commit")
            else:
                await consumer.commit()
    finally:
        await consumer.stop()
        await platform.aclose()
        log.info("eval-worker stopped")


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
