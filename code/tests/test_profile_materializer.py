"""
Tests for ``materialize_profile_from_history``.

Pure-function unit tests -- no DB, no LLM, no network.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from app.api.schemas.jobs import Submission
from app.services.profile_materializer import materialize_profile_from_history


def _sub(
    *,
    is_correct: bool,
    tags: list[str] | None = None,
    seconds_offset: int = 0,
    task_text: str = "Решите уравнение sin x = 1/2.",
    task_number: int = 13,
) -> Submission:
    """Build a submission with tunable fields for tests."""
    return Submission(
        submission_id=uuid4(),
        task_id=f"task-{seconds_offset}",
        task_number=task_number,  # type: ignore[arg-type]
        task_text=task_text,
        is_correct=is_correct,
        student_answer="x=π/3" if not is_correct else "x=π/6",
        correct_answer="x=π/6 + 2πk",
        diagnostic_tags=tags or [],
        submitted_at=datetime(2026, 4, 29, 12, 0, 0, tzinfo=timezone.utc)
        + timedelta(seconds=seconds_offset),
    )


def test_empty_history_yields_blank_profile():
    p = materialize_profile_from_history("user-x", [])
    assert p.student_id == "user-x"
    assert p.attempt_seq == 0
    assert p.attempts == {}
    assert p.error_events == []


def test_correct_submission_does_not_create_error_events():
    p = materialize_profile_from_history(
        "user-x", [_sub(is_correct=True)]
    )
    assert p.attempt_seq == 1
    assert p.error_events == []
    assert sum(p.attempts.values()) == 1


def test_wrong_submission_creates_error_event_per_tag():
    subs = [_sub(is_correct=False, tags=["trig_lost_solutions", "trig_extra_solutions"])]
    p = materialize_profile_from_history("user-x", subs)
    assert len(p.error_events) == 2
    assert {e["tag"] for e in p.error_events} == {
        "trig_lost_solutions",
        "trig_extra_solutions",
    }


def test_replay_is_chronological_regardless_of_input_order():
    subs_in_reverse = [
        _sub(is_correct=False, tags=["a"], seconds_offset=20),
        _sub(is_correct=False, tags=["a"], seconds_offset=10),
        _sub(is_correct=False, tags=["a"], seconds_offset=0),
    ]
    p = materialize_profile_from_history("user-x", subs_in_reverse)
    seqs = [e["attempt_seq"] for e in p.error_events]
    assert seqs == sorted(seqs), "events must replay in submitted_at order"


def test_idempotent_input_produces_same_profile_state():
    """Running the materializer twice on the same input gives identical state."""
    subs = [
        _sub(is_correct=True, seconds_offset=0),
        _sub(is_correct=False, tags=["trig_lost_solutions"], seconds_offset=10),
        _sub(is_correct=True, seconds_offset=20),
    ]
    p1 = materialize_profile_from_history("u", subs)
    p2 = materialize_profile_from_history("u", subs)
    # Compare the structurally meaningful pieces (timestamps differ by run).
    assert p1.attempt_seq == p2.attempt_seq
    assert p1.attempts == p2.attempts
    assert p1.wrong == p2.wrong
    assert [e["tag"] for e in p1.error_events] == [e["tag"] for e in p2.error_events]


def test_subtype_classified_per_submission():
    subs = [
        _sub(is_correct=False, tags=["x"], task_text="sin x = 1/2", task_number=13),
        _sub(is_correct=False, tags=["x"], task_text="кредит на 5 лет", task_number=16),
    ]
    p = materialize_profile_from_history("u", subs)
    assert set(p.attempts.keys()) >= {"simplest_trig", "annuity_credit"} or set(
        p.attempts.keys()
    ) >= {"simplest_trig", "other"}, p.attempts
