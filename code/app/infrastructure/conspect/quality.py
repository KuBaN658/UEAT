"""
Post-generation quality checks for EGE-format conspects (tasks 6, 10, and 12).

``scan_conspect_violations`` analyses the «Потренируйся» and «Разбор примера»
sections and returns human-readable violation codes.
``summarize_violations_for_llm`` converts those codes into Russian fix-hints.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

_PRACTICE_H1 = re.compile(r"(?m)^##\s*Потренируйся\s*$", re.IGNORECASE)
_NEXT_H1 = re.compile(r"(?m)^##\s+\S")


def _make_task_ref_re(n: int) -> re.Pattern[str]:
    return re.compile(
        rf"(?:задач[аи]|задани[ея])\s*:?\s*(?:№\s*|n\.?\s*)?{n}\b|(?<![\w/])(?:№|#)\s*{n}\b",
        re.IGNORECASE,
    )


_PRACTICE_REF = {t: _make_task_ref_re(t) for t in (6, 10, 12)}


def _extract_practice_section(text: str) -> str | None:
    m = _PRACTICE_H1.search(text)
    if not m:
        return None
    rest = text[m.end() :]
    for m2 in _NEXT_H1.finditer(rest):
        if m2.start() > 0:
            return rest[: m2.start()].strip()
    return rest.strip()


def _check_line_violations(line: str, issues: list[str]) -> None:
    """Append violation codes for a single practice/example line."""
    low = line.lower()
    if re.search(r"найдите\s+производную", low) or re.search(r"найти\s+производную", low):
        issues.append("practice_line: forbidden 'найдите производную'")
    if re.search(r"найдите\s+производные", low):
        issues.append("practice_line: forbidden 'найдите производные'")
    if re.search(r"реши(те)?\s+f['\u2032′]?", low) and re.search(r"=\s*0", low):
        if re.search(r"задач|упражнен|найди|найдите|потренируйся", low):
            issues.append("practice_line: solve f'=0 posed as exam question")
    if "нет экстремума" in low or "не существует экстремума" in low:
        issues.append("practice_line: non-numeric answer (no extremum)")
    if "√" in line or "\\sqrt" in line:
        issues.append("practice_line: sqrt in student-facing line")
    if "π" in line or "\\pi" in line:
        issues.append("practice_line: pi in student-facing line")
    if "≈" in line or "примерно" in low or "приблизительно" in low:
        issues.append("practice_line: approximation in practice / answer")


def scan_conspect_violations(text: str) -> list[str]:
    """Return human-readable violation codes for EGE Part 1 format.

    Scans ``## Потренируйся`` for content violations and
    ``## Разбор примера`` for forbidden final-answer patterns.

    Args:
        text: Full conspect markdown text.

    Returns:
        Deduplicated list of violation code strings.
    """
    issues: list[str] = []
    practice = _extract_practice_section(text)
    if practice:
        for line in practice.splitlines():
            _check_line_violations(line, issues)
        t6_n = len(_PRACTICE_REF[6].findall(practice))
        t10_n = len(_PRACTICE_REF[10].findall(practice))
        t12_n = len(_PRACTICE_REF[12].findall(practice))
        if t6_n < 1 or t10_n < 1 or t12_n < 1:
            issues.append(
                "practice_mix: require 1 task type 6 (equation), 1 task type 10 (text), 1 task type 12 (max/min on segment)"
            )
    else:
        issues.append("missing_practice_section")

    ex_m = re.search(r"(?m)^##\s*Разбор примера\s*$", text, re.IGNORECASE)
    if ex_m:
        start = ex_m.end()
        rest = text[start:]
        m2 = _NEXT_H1.search(rest[1:])
        ex_body = rest[: m2.start() + 1] if m2 else rest
        for line in ex_body.splitlines():
            low = line.lower()
            if "≈" in line or "примерно" in low or "приблизительно" in low:
                issues.append("example_section: approximation not allowed in writeup")
            if "ответ" in low or "**ответ**" in line or "**отв" in low or "итоговый ответ" in low:
                if "√" in line or "\\sqrt" in line:
                    issues.append("example_answer: sqrt in final answer line")
                if "π" in line or "\\pi" in line:
                    issues.append("example_answer: pi in final answer line")
                if "≈" in line:
                    issues.append("example_answer: approx in answer line")
            if re.search(r"почему\s+не\s+проверяем\s+концы", low):
                issues.append("example_text: wrong hint about skipping endpoints")

    seen: set[str] = set()
    out: list[str] = []
    for i in issues:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def summarize_violations_for_llm(violations: list[str]) -> str:
    """Convert violation codes into a Russian fix-hint string for the LLM repair prompt.

    Args:
        violations: Output of ``scan_conspect_violations``.

    Returns:
        Newline-joined Russian instructions, or a generic fallback message.
    """
    hints = []
    if any("practice_mix" in v for v in violations):
        hints.append(
            "В разделе «Потренируйся» должно быть: 1 задача №6 (уравнение), 1 задача №10 (текст) и 1 задача №12 (наибольшее/наименьшее на отрезке)."
        )
    if any("approximation" in v or "approx" in v for v in violations):
        hints.append(
            "Убери приближения (≈, «примерно») из разбора и ответов; подбери числа так, чтобы значения были точными десятичными."
        )
    if any("производн" in v or "f'" in v for v in violations):
        hints.append(
            "В «Потренируйся» и примерах нельзя спрашивать производную или решение f'(x)=0 как отдельную задачу — только формат №12: наибольшее/наименьшее на [a;b]."
        )
    if any("sqrt" in v or "pi" in v or "approx" in v for v in violations):
        hints.append(
            "Убери √, π, ≈ из финальных ответов; подбери числа так, чтобы ответ был целым или конечной десятичной дробью."
        )
    if any("non-numeric" in v or "экстрем" in v for v in violations):
        hints.append(
            "Ответ к задаче — всегда одно число для бланка, не формулировки «нет экстремума»."
        )
    if any("концы" in v for v in violations):
        hints.append(
            "Для отрезка [a;b] обязательно проверяй значения на концах и в критических точках; не пиши, что концы не нужны."
        )
    return "\n".join(hints) if hints else "Исправь нарушения формата ЕГЭ в конспекте."
