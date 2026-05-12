"""
Background Kafka consumer that populates the SubmissionStore, plus a
helper that queries Kafka consumer-group lag for the eval-worker.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from aiokafka import AIOKafkaConsumer, TopicPartition
from aiokafka.admin import AIOKafkaAdminClient

from admin.store import SubmissionStore

log = logging.getLogger(__name__)

MONITOR_GROUP = "ege-admin-monitor"
EVAL_GROUP = "ege-eval-worker"
LAG_TTL_S = 5.0


async def run_monitor_consumer(store: SubmissionStore, settings) -> None:
    """Consume the submissions topic with a separate group; mirror events into the store."""
    consumer = AIOKafkaConsumer(
        settings.kafka_submission_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=MONITOR_GROUP,
        enable_auto_commit=True,
        auto_offset_reset="latest",
    )
    try:
        await consumer.start()
        log.info("monitor consumer started (group=%s)", MONITOR_GROUP)
        async for msg in consumer:
            try:
                ev = json.loads(msg.value)
            except json.JSONDecodeError:
                continue
            sub = ev.get("submission") if isinstance(ev, dict) else None
            # Correct submissions are skipped by the eval-worker, so showing them
            # would mean a permanent "pending" row in the UI.
            if isinstance(sub, dict) and not sub.get("is_correct"):
                store.add(sub)
    except asyncio.CancelledError:
        pass
    finally:
        await consumer.stop()
        log.info("monitor consumer stopped")


_lag_cache: dict = {"value": None, "at": 0.0}


async def eval_worker_lag(settings) -> int | None:
    """Sum of partition lags for the eval-worker group. Cached for 5s."""
    now = time.monotonic()
    if _lag_cache["value"] is not None and now - _lag_cache["at"] < LAG_TTL_S:
        return _lag_cache["value"]

    topic = settings.kafka_submission_topic
    consumer = AIOKafkaConsumer(
        topic, bootstrap_servers=settings.kafka_bootstrap_servers, group_id=None
    )
    admin = AIOKafkaAdminClient(bootstrap_servers=settings.kafka_bootstrap_servers)
    try:
        await consumer.start()
        await admin.start()
        partitions = consumer.partitions_for_topic(topic)
        if not partitions:
            return None
        tps = [TopicPartition(topic, p) for p in partitions]
        end_offsets = await consumer.end_offsets(tps)
        committed = await admin.list_consumer_group_offsets(EVAL_GROUP)
        total = sum(
            max(0, end_offsets.get(tp, 0) - (committed[tp].offset if tp in committed else 0))
            for tp in tps
        )
        _lag_cache["value"] = total
        _lag_cache["at"] = now
        return total
    except Exception as exc:
        log.warning("lag query failed: %s", exc)
        return None
    finally:
        await consumer.stop()
        await admin.close()
