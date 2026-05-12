"""
Mistake-diagnosis service: LLM-first pipeline.

``diagnose_with_pipeline`` diagnoses student errors and returns a list of
``MistakeHypothesis`` objects.  The LLM backend is injected (not hardwired)
so callers can use any ``ChatClient`` implementation or pass ``None`` to skip.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from app.domain.atoms import ERROR_TAGS, ERROR_TAGS_BY_TASK
from app.infrastructure.llm.clients import ChatClient
from app.infrastructure.retrieval.engine import _ERROR_HINTS

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MistakeHypothesis:
    """A single diagnosed error hypothesis.

    Attributes:
        tag: Error tag string (from ``ERROR_TAGS``).
        reason: Short Russian explanation of the error.
        weight: Confidence weight used to boost the profile error score.
        source: Diagnosis source identifier (e.g. ``"llm"``).
        step_number: Optional step number in the student's solution.
    """

    tag: str
    reason: str
    weight: float = 1.0
    source: str = "llm"
    step_number: int | None = None


def _normalize_answer(s: str) -> str:
    s = s.strip().replace(",", ".")
    return re.sub(r"\s+", "", s)


def is_correct(student_answer: str, correct_answer: str) -> bool:
    """Return True if *student_answer* matches *correct_answer* after normalisation."""
    return _normalize_answer(student_answer) == _normalize_answer(correct_answer)


def _tags_for_task(task_number: int) -> list[str]:
    return list(ERROR_TAGS_BY_TASK.get(task_number, ()))


def _diagnose_with_llm(
    *,
    llm: ChatClient,
    task_number: int,
    task_text: str,
    correct: str,
    student: str,
) -> list[MistakeHypothesis]:
    """Ask the LLM to diagnose the student's error and return up to 3 hypotheses.

    The LLM is prompted in Russian to return structured JSON with 1–3 error tags
    chosen from the task-specific allowed list.

    Args:
        llm: Chat client to use for diagnosis.
        task_number: EGE task number (6, 10, or 12).
        task_text: Task prompt text.
        correct: Correct answer string.
        student: Student's answer string.

    Returns:
        List of ``MistakeHypothesis`` objects (empty on parsing failure).
    """
    tags = _tags_for_task(task_number)
    tag_descriptions = "\n".join(f"  - {tag}: {_ERROR_HINTS.get(tag, tag)}" for tag in tags)
    system = (
        "Ты — диагностический модуль для ЕГЭ по математике. "
        "Проанализируй ошибку ученика и определи наиболее вероятную причину. "
        "Верни ТОЛЬКО валидный JSON в формате: "
        '{"mistakes": [{"tag": "error_tag", "reason": "краткое объяснение по-русски"}]}. '
        "Выбери 1-3 тега из списка допустимых. Объяснение — 1-2 предложения, по-русски."
    )
    user = (
        f"Задание №{task_number} ЕГЭ (профильная математика).\n\n"
        f"Условие:\n{task_text[:1500]}\n\n"
        f"Правильный ответ: {correct}\n"
        f"Ответ ученика: {student}\n\n"
        f"Допустимые теги ошибок:\n{tag_descriptions}\n\n"
        "Выбери 1-3 наиболее вероятных тега и объясни ошибку кратко по-русски. "
        "Верни только JSON без markdown."
    )
    try:
        raw = llm.chat(system=system, user=user, temperature=0.1).text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```\s*$", "", raw)
        obj = json.loads(raw)
        mistakes = obj.get("mistakes", [])
        if not isinstance(mistakes, list):
            return []
        out: list[MistakeHypothesis] = []
        for m in mistakes[:3]:
            if not isinstance(m, dict):
                continue
            tag = str(m.get("tag", "")).strip()
            reason = str(m.get("reason", "")).strip()
            if tag in ERROR_TAGS and reason:
                out.append(MistakeHypothesis(tag=tag, reason=reason, weight=1.15, source="llm"))
        return out
    except Exception as exc:
        log.warning("LLM diagnosis JSON parse failed: %s", exc)
    return []


def diagnose_with_pipeline(
    *,
    task_number: int,
    task_text: str,
    correct: str,
    student: str,
    llm: ChatClient | None = None,
) -> list[MistakeHypothesis]:
    """Run the LLM-first diagnosis pipeline.

    Args:
        task_number: EGE task number.
        task_text: Task prompt text.
        correct: Correct answer string.
        student: Student's answer string.
        llm: Chat client; returns empty list if ``None``.

    Returns:
        Up to 4 ``MistakeHypothesis`` objects (empty if no LLM client).
    """
    if llm is None:
        return []
    return _diagnose_with_llm(
        llm=llm,
        task_number=task_number,
        task_text=task_text,
        correct=correct,
        student=student,
    )[:4]
