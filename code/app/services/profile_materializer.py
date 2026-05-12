"""
Replay a list of submissions into a transient ``StudentProfile``.

The platform owns submission history; our conspect pipeline expects a
fully-hydrated profile (error scores, attempt sequence, recent attempts).
This module bridges the two by replaying each submission in chronological
order via ``profile.record_attempt`` -- derived state falls out automatically.
"""

from __future__ import annotations

import logging
from typing import Iterable

from app.api.schemas.jobs import Submission
from app.domain.profile import StudentProfile
from app.domain.subtypes import classify_subtype

log = logging.getLogger(__name__)


def materialize_profile_from_history(
    user_id: str, submissions: Iterable[Submission]
) -> StudentProfile:
    """Replay submissions (chronologically) into a fresh, non-persisted profile."""
    profile = StudentProfile(student_id=user_id)
    for s in sorted(submissions, key=lambda s: s.submitted_at):
        subtype = classify_subtype(s.task_text, task_number=s.task_number)
        profile.record_attempt(
            task_id=s.task_id,
            task_number=s.task_number,
            subtype=subtype,
            ok=s.is_correct,
            error_tags=list(s.diagnostic_tags or []),
            student_answer=s.student_answer if not s.is_correct else None,
            correct_answer=s.correct_answer if not s.is_correct else None,
            task_text=s.task_text if not s.is_correct else None,
        )

    log.info(
        "materialized profile: user=%s submissions=%d wrong=%d error_events=%d",
        user_id, profile.attempt_seq, sum(profile.wrong.values()), len(profile.error_events),
    )
    return profile
