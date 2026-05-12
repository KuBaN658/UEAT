"""Pydantic schemas for ``POST /jobs`` -- mirrors ``app/spec/generator.openapi.yaml``."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field

# Supported EGE task numbers (matches RagSettings.supported_tasks)
TaskNumber = Literal[6, 10, 12, 13, 16]


class Submission(BaseModel):
    model_config = ConfigDict(extra="ignore")

    submission_id: UUID
    task_id: str
    task_number: TaskNumber
    task_text: str
    is_correct: bool
    student_answer: str | None = None
    correct_answer: str | None = None
    diagnostic_tags: list[str] = Field(default_factory=list)
    submitted_at: datetime


class CallbackConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: AnyHttpUrl
    auth: str | None = None  # Bearer token for the callback


class JobCreateRequest(BaseModel):
    """Body of POST /jobs. ``job_id`` doubles as the idempotency key."""

    model_config = ConfigDict(extra="ignore")

    job_id: UUID
    user_id: str
    callback: CallbackConfig | None = None
    submissions: Annotated[list[Submission], Field(min_length=0, max_length=500)]


class JobAccepted(BaseModel):
    job_id: UUID
    status: Literal["queued", "running", "done", "failed"]
    accepted_at: datetime


class JobStatus(BaseModel):
    job_id: UUID
    status: Literal["queued", "running", "done", "failed"]
    result: dict | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: dict | None = None
