"""Pydantic models for the admin task endpoints (platform simulator)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class NewTaskResponse(BaseModel):
    task_id: str
    task_number: int
    subtype: str
    text: str


class SubmitRequest(BaseModel):
    student_id: str = Field(default="demo")
    task_number: int
    task_id: str
    answer: str


class SubmitResponse(BaseModel):
    submission_id: str
    is_correct: bool
    correct_answer: str
    queued: bool
