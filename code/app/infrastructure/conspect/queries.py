"""
Load conspect RAG query pools and verification prompts from YAML.

Data file: ``app/data/conspect_prompts.yaml``.

All getters are lazy-loaded and cached for the process lifetime.
"""

from __future__ import annotations

import hashlib
import logging
import random
from pathlib import Path
from typing import Any

import yaml

from app.domain.profile import StudentProfile

log = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_YAML_PATH = _DATA_DIR / "conspect_prompts.yaml"

_raw: dict[str, Any] = {}


def _load_yaml() -> dict[str, Any]:
    global _raw
    if _raw:
        return _raw
    if not _YAML_PATH.exists():
        log.warning("Missing %s — using empty conspect prompt config", _YAML_PATH)
        _raw = {}
        return _raw
    with open(_YAML_PATH, encoding="utf-8") as f:
        _raw = yaml.safe_load(f) or {}
    return _raw


_QUERY_POOL_KEYS: dict[int, str] = {
    6: "t6_query_pool",
    10: "t10_query_pool",
    12: "t12_query_pool",
}


def get_query_pool(task_number: int) -> list[str]:
    """Return the retrieval query pool for *task_number*."""
    key = _QUERY_POOL_KEYS.get(task_number)
    if not key:
        return []
    return list(_load_yaml().get(key) or [])


def get_subtype_focus_queries() -> dict[str, str]:
    """Return ``{subtype: query}`` mapping for subtype-focused retrieval."""
    d = _load_yaml().get("subtype_focus_queries") or {}
    return {str(k): str(v) for k, v in d.items()}


def get_relevant_subtypes() -> frozenset[str]:
    """Return the set of subtype strings that have dedicated focus queries."""
    sq = get_subtype_focus_queries()
    return frozenset(sq.keys()) | {
        "other",
        "no_derivative",
        "equation_setup",
        "circular_motion",
        "exponential_eq",
        "logarithmic_eq",
        "trigonometric_eq",
        "irrational_eq",
        "quadratic_eq",
        "combined_eq",
    }


def get_verify_system_prompt() -> str:
    """Return the LLM system prompt for the structural verify pass."""
    s = (_load_yaml().get("verify_system") or "").strip()
    if not s:
        raise RuntimeError(f"verify_system missing or empty in {_YAML_PATH}")
    return s


def get_math_verify_system_prompt() -> str:
    """Return the LLM system prompt for the arithmetic verify pass."""
    s = (_load_yaml().get("math_verify_system") or "").strip()
    if not s:
        raise RuntimeError(f"math_verify_system missing or empty in {_YAML_PATH}")
    return s


def get_conspect_system_prompt() -> str:
    """Return the main conspect generation system prompt."""
    s = (_load_yaml().get("conspect_system") or "").strip()
    if not s:
        raise RuntimeError(f"conspect_system missing or empty in {_YAML_PATH}")
    return s


def get_conspect_user_template() -> str:
    """Return the conspect user-message template string."""
    s = (_load_yaml().get("conspect_user_template") or "").strip()
    if not s:
        raise RuntimeError(f"conspect_user_template missing or empty in {_YAML_PATH}")
    return s


def get_conspect_json_output_instruction() -> str:
    """Return the optional JSON output instruction appended to the system prompt."""
    return (_load_yaml().get("conspect_json_output") or "").strip()


def get_what_to_remember_prompt() -> str:
    """Return the per-section «Что важно запомнить» prompt."""
    s = (_load_yaml().get("what_to_remember_prompt") or "").strip()
    if not s:
        raise RuntimeError(f"what_to_remember_prompt missing or empty in {_YAML_PATH}")
    return s


def get_typical_errors_prompt() -> str:
    """Return the per-section «Типичные ошибки» prompt."""
    s = (_load_yaml().get("typical_errors_prompt") or "").strip()
    if not s:
        raise RuntimeError(f"typical_errors_prompt missing or empty in {_YAML_PATH}")
    return s


def get_algorithm_prompt() -> str:
    """Return the per-section «Алгоритм решения» prompt."""
    s = (_load_yaml().get("algorithm_prompt") or "").strip()
    if not s:
        raise RuntimeError(f"algorithm_prompt missing or empty in {_YAML_PATH}")
    return s


def get_example_prompt() -> str:
    """Return the per-section «Разбор примера» prompt."""
    s = (_load_yaml().get("example_prompt") or "").strip()
    if not s:
        raise RuntimeError(f"example_prompt missing or empty in {_YAML_PATH}")
    return s


def get_find_error_prompt() -> str:
    """Return the per-section «Найди ошибку» prompt."""
    s = (_load_yaml().get("find_error_prompt") or "").strip()
    if not s:
        raise RuntimeError(f"find_error_prompt missing or empty in {_YAML_PATH}")
    return s


def get_checklist_prompt() -> str:
    """Return the per-section «Чеклист перед ответом» prompt."""
    s = (_load_yaml().get("checklist_prompt") or "").strip()
    if not s:
        raise RuntimeError(f"checklist_prompt missing or empty in {_YAML_PATH}")
    return s


def get_verify_system_prompt_v2() -> str:
    """Alias kept for backward compatibility."""
    return get_verify_system_prompt()


# ── Query builders ───────────────────────────────────────────────────


def cold_start_retrieval_queries(
    profile: StudentProfile, student_id: str | None = None
) -> list[str]:
    """Build a deterministically shuffled query list for zero-error / cold profiles.

    Args:
        profile: Student profile (used for gap detection).
        student_id: Optional override for the seeding hash.

    Returns:
        Mixed list of pool queries and subtype-gap queries.
    """
    t6_pool = get_query_pool(6)
    t10_pool = get_query_pool(10)
    t12_pool = get_query_pool(12)
    subtype_focus = get_subtype_focus_queries()
    sid = student_id or profile.student_id
    seed = int(hashlib.sha256(sid.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    t6 = list(t6_pool)
    t10 = list(t10_pool)
    t12 = list(t12_pool)
    rng.shuffle(t6)
    rng.shuffle(t10)
    rng.shuffle(t12)
    n_t6 = rng.randint(2, min(4, len(t6))) if t6 else 0
    n_t10 = rng.randint(3, min(5, len(t10))) if t10 else 0
    n_t12 = rng.randint(3, min(5, len(t12))) if t12 else 0
    queries = t6[:n_t6] + t10[:n_t10] + t12[:n_t12]
    gap_queries: list[str] = []
    for st in sorted(subtype_focus.keys()):
        if profile.attempts.get(st, 0) == 0:
            gap_queries.append(subtype_focus[st])
    rng.shuffle(gap_queries)
    for g in gap_queries[:3]:
        if g not in queries:
            queries.append(g)
    return queries


def micromodule_queries(profile: StudentProfile, task_number: int, n: int = 5) -> list[str]:
    """Build queries from ``task_text`` entries in the error-event log.

    Args:
        profile: Student profile.
        task_number: EGE task number to filter events by.
        n: Maximum number of queries to return.

    Returns:
        List of query strings prefixed with task context.
    """
    prefix = {
        6: "ЕГЭ задание 6 (уравнение / тип): ",
        10: "ЕГЭ задание 10 (текст / тип): ",
        12: "ЕГЭ задание 12 (микромодуль / тип задачи): ",
    }.get(task_number, f"ЕГЭ задание {task_number}: ")
    seen: set[str] = set()
    out: list[str] = []
    for ev in reversed(profile.error_events):
        if ev.get("task_number") != task_number:
            continue
        tt = (ev.get("task_text") or "").strip().replace("\n", " ")
        if len(tt) < 15:
            continue
        key = tt[:240]
        if key in seen:
            continue
        seen.add(key)
        out.append(f"{prefix}{tt[:320]}")
        if len(out) >= n:
            break
    return out


def subtype_seen_queries(profile: StudentProfile) -> list[str]:
    """Build queries biased toward subtypes the student has practiced.

    Args:
        profile: Student profile.

    Returns:
        Up to 4 focus-query strings for the most-attempted subtypes.
    """
    subtype_focus = get_subtype_focus_queries()
    relevant = get_relevant_subtypes()
    active = [(st, c) for st, c in profile.attempts.items() if st in relevant and c > 0]
    if not active:
        return []
    active.sort(key=lambda x: (-x[1], x[0]))
    queries: list[str] = []
    for st, _ in active[:4]:
        q = subtype_focus.get(st)
        if q:
            queries.append(q)
    return queries
