"""
Second focused LLM pass: verify arithmetic in conspect example sections.

``apply_conspect_math_verification`` re-checks the full conspect.
``verify_math_section`` re-checks a single section fragment.
"""

from __future__ import annotations

import logging

from app.infrastructure.conspect.queries import get_math_verify_system_prompt
from app.infrastructure.llm.clients import ChatClient
from app.infrastructure.retrieval.engine import sanitize_llm_output

log = logging.getLogger(__name__)


def apply_conspect_math_verification(llm: ChatClient, draft: str) -> str:
    """Re-check numeric answers in the full conspect draft.

    Returns the corrected conspect, or the original *draft* if the LLM
    returns an empty response.

    Args:
        llm: Configured chat client.
        draft: Full conspect markdown text.
    """
    system = get_math_verify_system_prompt()
    user = f"Проверь и при необходимости исправь этот конспект:\n\n{draft}"
    out = llm.chat(system=system, user=user, temperature=0.05, max_tokens=8192).text
    fixed = sanitize_llm_output(out) or ""
    if not fixed.strip():
        log.warning("Math verification returned empty; keeping draft")
        return draft
    return fixed


def verify_math_section(llm: ChatClient, section_heading: str, fragment: str) -> str:
    """Focused arithmetic pass on a single section fragment.

    Args:
        llm: Configured chat client.
        section_heading: Human-readable section label for the prompt (e.g. ``"«Разбор примера»"``).
        fragment: Raw markdown text of the section body.

    Returns:
        Corrected fragment, or the original *fragment* if the LLM returns empty.
    """
    if not fragment.strip():
        return fragment
    system = get_math_verify_system_prompt()
    user = (
        f"Проверь только этот фрагмент конспекта (раздел {section_heading}). "
        "Исправь арифметику и финальные числовые ответы для бланка; не меняй структуру списков и заголовков внутри фрагмента.\n\n"
        f"{fragment}"
    )
    out = llm.chat(system=system, user=user, temperature=0.05, max_tokens=6144).text
    fixed = sanitize_llm_output(out) or ""
    if not fixed.strip():
        log.warning(
            "Section math verify returned empty for %s; keeping fragment",
            section_heading,
        )
        return fragment
    return fixed
