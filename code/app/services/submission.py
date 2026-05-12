"""
Submit use-case: validate a student answer, run diagnosis, update profile.

This service owns the complete "submit answer" business flow that previously
lived in ``app.py``.  It has no HTTP knowledge; callers are responsible for
translating the result into HTTP responses.
"""

from __future__ import annotations

import logging

from app.domain.analysis import top_errors_with_scores
from app.domain.subtypes import classify_subtype
from app.infrastructure.llm.clients import build_chat_client_optional
from app.infrastructure.repositories.profile_repo import ProfileStore
from app.infrastructure.repositories.task_repo import TaskBank
from app.infrastructure.retrieval.engine import _human_error
from app.services.diagnosis import MistakeHypothesis, diagnose_with_pipeline, is_correct

log = logging.getLogger(__name__)


# ── Result dataclasses ────────────────────────────────────────────────


class DiagnosedMistake:
    """Fully resolved mistake including the per-profile error score.

    Attributes:
        tag: Error tag string.
        label: Human-readable Russian label for the tag.
        reason: LLM explanation of the error.
        source: Diagnosis source (``"llm"``).
        step_number: Optional step number in the student's solution.
        score: Profile cumulative error score for this tag.
    """

    __slots__ = ("tag", "label", "reason", "source", "step_number", "score")

    def __init__(
        self,
        tag: str,
        label: str,
        reason: str,
        source: str,
        step_number: int | None,
        score: float,
    ) -> None:
        self.tag = tag
        self.label = label
        self.reason = reason
        self.source = source
        self.step_number = step_number
        self.score = score


class SubmitResult:
    """Outcome of a submit operation returned to the API layer.

    Attributes:
        ok: Whether the student answer was correct.
        task_number: EGE task number.
        correct_answer: The expected answer string.
        subtype: Heuristic subtype label.
        mistakes: Diagnosed mistakes (empty when *ok* is True).
        profile_top_errors: Human-readable top error labels for the student.
        profile_top_errors_with_scores: Scored error list for detailed display.
    """

    __slots__ = (
        "ok",
        "task_number",
        "correct_answer",
        "subtype",
        "mistakes",
        "profile_top_errors",
        "profile_top_errors_with_scores",
    )

    def __init__(
        self,
        ok: bool,
        task_number: int,
        correct_answer: str,
        subtype: str,
        mistakes: list[DiagnosedMistake],
        profile_top_errors: list[str],
        profile_top_errors_with_scores: list[tuple[str, float]],
    ) -> None:
        self.ok = ok
        self.task_number = task_number
        self.correct_answer = correct_answer
        self.subtype = subtype
        self.mistakes = mistakes
        self.profile_top_errors = profile_top_errors
        self.profile_top_errors_with_scores = profile_top_errors_with_scores


# ── Helpers ───────────────────────────────────────────────────────────


def _profile_error_lists(profile) -> tuple[list[str], list[tuple[str, float]]]:
    """Return (human_labels, [(human_label, score)]) for the top 8 profile errors."""
    top_with_scores = top_errors_with_scores(profile, 8, subtype=None)
    labels = [_human_error(t) for t, _ in top_with_scores]
    scored = [((_human_error(tag)), round(sc, 2)) for tag, sc in top_with_scores]
    return labels, scored


def _diagnose_mistakes(
    task_number: int,
    text: str,
    correct: str,
    student: str,
    profile,
) -> tuple[list[DiagnosedMistake], list[str], list[MistakeHypothesis]]:
    """Run LLM diagnosis and enrich with per-profile scores.

    Args:
        task_number: EGE task number.
        text: Task prompt text.
        correct: Expected answer string.
        student: Student's submitted answer.
        profile: Loaded ``StudentProfile`` for score lookup.

    Returns:
        A 3-tuple of (diagnosed_mistakes, error_tag_list, raw_hypotheses).
    """
    llm = build_chat_client_optional()
    if llm is None:
        log.debug(
            "No LLM client for diagnosis (set GROQ_API_KEY, OPENROUTER_API_KEY, "
            "or LLM_BACKEND=ollama)"
        )

    hyps = diagnose_with_pipeline(
        task_number=task_number,
        task_text=text,
        correct=correct,
        student=student,
        llm=llm,
    )

    diagnosed: list[DiagnosedMistake] = []
    tags: list[str] = []
    for h in hyps[:4]:
        diagnosed.append(
            DiagnosedMistake(
                tag=h.tag,
                label=_human_error(h.tag),
                reason=h.reason,
                source=h.source,
                step_number=h.step_number,
                score=round(profile.error_score(h.tag), 2),
            )
        )
        tags.append(h.tag)
    return diagnosed, tags, hyps


# ── Public API ────────────────────────────────────────────────────────


def submit_answer(
    *,
    task_number: int,
    task_id: str,
    student_id: str,
    student_answer: str,
    task_bank: TaskBank,
    profile_store: ProfileStore,
) -> SubmitResult:
    """Execute the full submit flow.

    Looks up the task, classifies it, checks the answer, diagnoses any errors,
    updates the student profile, and returns a :class:`SubmitResult`.

    Args:
        task_number: EGE task number from the URL path.
        task_id: Unique task identifier returned by ``/task/{n}/new``.
        student_id: Student identifier (default ``"demo"``).
        student_answer: The student's submitted answer string.
        task_bank: Loaded task bank for task lookup.
        profile_store: Profile store for loading / saving the student profile.

    Returns:
        :class:`SubmitResult` with all outcome fields populated.

    Raises:
        ValueError: If *task_id* is unknown or belongs to a different task number.
        RuntimeError: If profile persistence fails.
    """
    t = task_bank.by_id(task_id)
    if t is None:
        raise ValueError(f"Unknown task_id: {task_id!r}")
    if t.task_number != task_number:
        raise ValueError("task_id does not belong to the requested task_number")

    text = t.prompt_text()
    subtype = classify_subtype(text, task_number=task_number)
    ok = is_correct(student_answer, t.answer)

    profile = profile_store.load(student_id)

    diagnosed: list[DiagnosedMistake] = []
    error_tags: list[str] = []
    hyps: list[MistakeHypothesis] = []

    if not ok:
        diagnosed, error_tags, hyps = _diagnose_mistakes(
            task_number, text, t.answer, student_answer, profile
        )

    weights = {h.tag: h.weight for h in hyps} if not ok else None
    profile.record_attempt(
        task_id=t.id,
        task_number=task_number,
        subtype=subtype,
        ok=ok,
        error_tags=error_tags,
        error_weights=weights,
        student_answer=student_answer if not ok else None,
        correct_answer=t.answer if not ok else None,
        task_text=text if not ok else None,
    )
    profile_store.save(profile)

    top_errs, top_errs_scored = _profile_error_lists(profile)
    return SubmitResult(
        ok=ok,
        task_number=task_number,
        correct_answer=t.answer,
        subtype=subtype,
        mistakes=diagnosed,
        profile_top_errors=top_errs,
        profile_top_errors_with_scores=top_errs_scored,
    )
