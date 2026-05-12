"""Tests for profile error ranking."""

from app.domain.analysis import top_errors
from app.domain.profile import StudentProfile


def test_top_errors_empty_profile():
    p = StudentProfile(student_id="t")
    assert top_errors(p, 5, None) == []


def test_top_errors_with_scores():
    p = StudentProfile(student_id="t")
    p.record_attempt(
        task_id="1",
        task_number=12,
        subtype="trig",
        ok=False,
        error_tags=["trig_deriv_error"],
        error_weights={"trig_deriv_error": 1.0},
    )
    te = top_errors(p, 3, None)
    assert "trig_deriv_error" in te
