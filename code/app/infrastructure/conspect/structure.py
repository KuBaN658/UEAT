"""
Parse structured JSON conspect output from the LLM and convert to markdown.

When the LLM returns a JSON object with section bodies, ``try_parse_conspect_json``
extracts the sections; ``conspect_dict_to_markdown`` renders them with standard
``##`` headings in the canonical section order.
"""

from __future__ import annotations

import json
import logging
import re

log = logging.getLogger(__name__)

_SECTION_ORDER: tuple[tuple[str, str], ...] = (
    ("what_to_remember", "## Что важно запомнить"),
    ("typical_errors", "## Типичные ошибки"),
    ("algorithm", "## Алгоритм решения"),
    ("example", "## Разбор примера"),
    ("find_error", "## Найди ошибку"),
    ("checklist", "## Чеклист перед ответом"),
)

# Keys whose content typically contains worked numeric examples.
MATH_SECTION_KEYS: tuple[str, ...] = ("example", "find_error")

MATH_SECTION_LABELS: dict[str, str] = {
    "example": "«Разбор примера»",
    "find_error": "«Найди ошибку»",
}


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```\w*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t)
    return t.strip()


def try_parse_conspect_json(raw: str) -> dict[str, str] | None:
    """Try to parse a structured JSON conspect from the LLM output.

    Returns a ``{section_key: body_text}`` dict if the response contains
    valid JSON with at least 3 recognised section keys, otherwise ``None``.

    Args:
        raw: Raw LLM response text (may be wrapped in code fences).

    Returns:
        Section dict or ``None`` if parsing fails / output is too sparse.
    """
    t = _strip_code_fences(raw)
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    out: dict[str, str] = {}
    for key, _ in _SECTION_ORDER:
        val = data.get(key)
        if val is not None and str(val).strip():
            out[key] = str(val).strip()
    if len(out) < 3:
        log.warning(
            "Structured conspect JSON too sparse (%s keys); falling back to markdown",
            len(out),
        )
        return None
    return out


def _strip_leading_h2_heading(body: str) -> str:
    """Remove one leading ``##`` line if the model duplicated the section title inside the JSON value."""
    b = body.strip()
    return re.sub(r"^##\s+\S[^\n]*\n?", "", b, count=1).strip()


def conspect_dict_to_markdown(sections: dict[str, str]) -> str:
    """Render a structured sections dict as a full markdown conspect.

    Args:
        sections: ``{section_key: body_text}`` dict from ``try_parse_conspect_json``.

    Returns:
        Markdown string with ``##`` headings in canonical section order.
    """
    parts: list[str] = []
    for key, heading in _SECTION_ORDER:
        body = sections.get(key)
        if not body:
            continue
        body = _strip_leading_h2_heading(body)
        parts.append(f"{heading}\n\n{body}")
    return "\n\n".join(parts)
