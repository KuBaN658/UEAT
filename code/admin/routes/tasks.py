"""Platform-simulator task endpoints: serve a task, accept an answer, emit to Kafka."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request

from admin.schemas import NewTaskResponse, SubmitRequest, SubmitResponse
from app.core.config import get_settings, get_rag_settings
from app.domain.subtypes import classify_subtype
from app.infrastructure.repositories.task_repo import TaskBank
from app.services.diagnosis import is_correct

log = logging.getLogger(__name__)
router = APIRouter()


def _validated_task_number(task_number: int) -> int:
    supported = get_rag_settings().supported_tasks
    if task_number not in supported:
        raise HTTPException(400, f"Supported task numbers: {sorted(supported)}")
    return task_number


def _task_bank(request: Request) -> TaskBank:
    bank = request.app.state.task_bank
    if bank is None:
        raise HTTPException(503, "Task bank unavailable")
    return bank


@router.get("/task/{task_number}/new", response_model=NewTaskResponse)
def new_task(task_number: int, request: Request) -> NewTaskResponse:
    _validated_task_number(task_number)
    t = _task_bank(request).random_task(task_number)
    text = t.prompt_text()
    return NewTaskResponse(
        task_id=t.id,
        task_number=task_number,
        subtype=classify_subtype(text, task_number=task_number),
        text=text,
    )


@router.post("/task/{task_number}/submit", response_model=SubmitResponse)
async def submit(task_number: int, req: SubmitRequest, request: Request) -> SubmitResponse:
    _validated_task_number(task_number)
    if req.task_number != task_number:
        raise HTTPException(400, "Path task_number must match body task_number.")

    t = _task_bank(request).by_id(req.task_id)
    if t is None:
        raise HTTPException(404, f"Unknown task_id: {req.task_id!r}")
    if t.task_number != task_number:
        raise HTTPException(400, "task_id does not belong to the requested task_number")

    ok = is_correct(req.answer, t.answer)
    submission_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()

    event = {
        "event_id": str(uuid4()),
        "event_type": "submission.created",
        "schema_version": "1.0",
        "occurred_at": now,
        "submission": {
            "submission_id": submission_id,
            "user_id": req.student_id,
            "task_id": req.task_id,
            "task_number": task_number,
            "task_text": t.prompt_text(),
            "is_correct": ok,
            "student_answer": req.answer,
            "correct_answer": t.answer,
            "submitted_at": now,
        },
    }

    queued = False
    producer = request.app.state.kafka_producer
    if producer is not None:
        try:
            await producer.send_and_wait(
                get_settings().kafka_submission_topic,
                json.dumps(event).encode("utf-8"),
            )
            queued = True
        except Exception as exc:
            log.error("Kafka send failed for %s: %s", submission_id, exc)
    else:
        log.warning("Kafka unavailable, submission %s not queued", submission_id)

    return SubmitResponse(
        submission_id=submission_id, is_correct=ok, correct_answer=t.answer, queued=queued
    )
