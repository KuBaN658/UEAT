"""
Conspect tools: local implementations + MCP server entry point.

``ConspectToolContext``  — runtime context (student ID + profile store + RAG engine).
``ConspectToolRegistry`` — dispatcher for local tool implementations.
``create_conspect_mcp_server`` — build a FastMCP server exposing all tools.
``run_conspect_mcp_server``    — start the MCP server from CLI.

Entry point (docker-compose / module run):
    python -m app.infrastructure.conspect.tools --mcp-server
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from fastmcp import FastMCP

from app.core.config import get_settings
from app.domain.analysis import next_frontier_atoms, top_errors_with_scores
from app.domain.atoms import ATOM_BY_ID
from app.domain.profile import StudentProfile
from app.domain.subtypes import SUBTYPE_LABELS
from app.infrastructure.repositories.profile_repo import ProfileStore
from app.infrastructure.retrieval.engine import RagEngine, _human_error

log = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_MISCONCEPTIONS_PATH = _DATA_DIR / "misconceptions.yaml"
_TASKS_WITH_SOLUTIONS_PATH = _DATA_DIR / "submissions_archive" / "tasks_with_solutions.csv"
_USERS_SUBMISSIONS_PATH = _DATA_DIR / "submissions_archive" / "users_submissions.json"
_MAX_TOOL_TEXT_CHARS = 900
_MAX_TOOL_LIST_ITEMS = 8


# ── Data types ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class ConspectToolContext:
    """Runtime context passed to each conspect tool invocation.

    Attributes:
        student_id: The student whose profile is used.
        profile_store: Repository for loading student profiles.
        rag: Retrieval engine for ``retrieve_learning_atoms``.
    """

    student_id: str
    profile_store: ProfileStore
    rag: RagEngine

    def load_profile(self) -> StudentProfile:
        """Load and return the student's current profile."""
        return self.profile_store.load(self.student_id)


@dataclass(frozen=True)
class ConspectTool:
    """Descriptor for a single locally-implemented conspect tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[[dict[str, Any]], dict[str, Any]]

    def schema(self) -> dict[str, Any]:
        """Return the OpenAI function-calling schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ── Argument coercion helpers ────────────────────────────────────────


def _limit_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _optional_int(value: Any) -> int | None:
    if value in (None, "", "null"):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed in (6, 10, 12) else None


def _clean_text(value: Any, limit: int = _MAX_TOOL_TEXT_CHARS) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _subtype_label(subtype: str) -> str:
    return SUBTYPE_LABELS.get(subtype, subtype.replace("_", " "))


def _profile_detail_level(profile: StudentProfile) -> str:
    from app.core.config import get_rag_settings

    cfg = get_rag_settings()
    total_att = sum(profile.attempts.values())
    total_wrong = sum(profile.wrong.values())
    accuracy = 0.0 if total_att == 0 else (total_att - total_wrong) / total_att
    if accuracy < cfg.accuracy_low:
        return "подробный"
    if accuracy < cfg.accuracy_mid:
        return "средний"
    return "краткий"


def _profile_summary(profile: StudentProfile) -> dict[str, Any]:
    top_scored = top_errors_with_scores(profile, 8, subtype=None)
    attempts = sorted(profile.attempts.items(), key=lambda item: (-item[1], item[0]))
    wrong = sorted(profile.wrong.items(), key=lambda item: (-item[1], item[0]))
    recent_tasks: list[dict[str, Any]] = []
    for item in reversed(profile.recent[-8:]):
        recent_tasks.append(
            {
                "task_number": item.get("task_number"),
                "subtype": _subtype_label(str(item.get("subtype") or "other")),
                "ok": bool(item.get("ok")),
                "errors": [_human_error(str(tag)) for tag in item.get("error_tags", [])],
            }
        )
    return {
        "student_id": profile.student_id,
        "attempts_total": sum(profile.attempts.values()),
        "wrong_total": sum(profile.wrong.values()),
        "detail_level": _profile_detail_level(profile),
        "top_errors": [
            {"error": _human_error(tag), "score": round(score, 2)} for tag, score in top_scored
        ],
        "practiced_subtypes": [
            {"subtype": _subtype_label(subtype), "attempts": count}
            for subtype, count in attempts[:_MAX_TOOL_LIST_ITEMS]
        ],
        "weak_subtypes": [
            {"subtype": _subtype_label(subtype), "wrong": count}
            for subtype, count in wrong[:_MAX_TOOL_LIST_ITEMS]
            if count > 0
        ],
        "recent_tasks": recent_tasks,
    }


# ── Data loaders ─────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _load_misconceptions() -> tuple[dict[str, Any], ...]:
    if not _MISCONCEPTIONS_PATH.exists():
        return ()
    raw = yaml.safe_load(_MISCONCEPTIONS_PATH.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        return ()
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "title": str(item.get("title") or "").strip(),
                "text": _clean_text(item.get("text"), limit=700),
                "error_tag": str(item.get("error_tag") or "").strip(),
                "task_number": _optional_int(item.get("task_number")),
            }
        )
    return tuple(out)


@lru_cache(maxsize=1)
def _load_solution_rows() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    if _USERS_SUBMISSIONS_PATH.exists():
        try:
            raw = json.loads(_USERS_SUBMISSIONS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raw = []
        if isinstance(raw, list):
            for user in raw[:80]:
                for solution in (user.get("solutions") or [])[:80]:
                    if not isinstance(solution, dict):
                        continue
                    rows.append(
                        {
                            "task_number": _optional_int(solution.get("task_number")),
                            "problem": solution.get("problem_katex") or "",
                            "solution": solution.get("solution") or "",
                            "source": "archive_submission",
                        }
                    )
                    if len(rows) >= 2500:
                        return tuple(rows)
    if _TASKS_WITH_SOLUTIONS_PATH.exists():
        with _TASKS_WITH_SOLUTIONS_PATH.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    {
                        "task_number": None,
                        "problem": row.get("problem_katex") or "",
                        "solution": row.get("solution") or "",
                        "source": "tasks_with_solutions",
                    }
                )
                if len(rows) >= 3000:
                    break
    return tuple(rows)


def _tokenize_query(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zа-яё0-9]+", text.lower()) if len(token) > 2}


def _score_solution_row(row: dict[str, Any], query_tokens: set[str]) -> int:
    haystack = f"{row.get('problem', '')} {row.get('solution', '')}".lower()
    return sum(1 for token in query_tokens if token in haystack)


def _bool_arg(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _canonical_tool_argument_key(key: Any) -> str:
    return re.sub(r"\s+", "_", str(key).strip())


def _normalize_argument_keys(arguments: dict[str, Any]) -> dict[str, Any]:
    return {_canonical_tool_argument_key(k): v for k, v in arguments.items()}


def _string_list_arg(value: Any) -> list[str]:
    if value in (None, "", "null"):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        return [stripped]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _normalize_tool_arguments(name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
    normalized = _normalize_argument_keys(dict(arguments or {}))
    if "task_number" in normalized:
        normalized["task_number"] = _optional_int(normalized.get("task_number"))
    if name == "retrieve_learning_atoms":
        if "k" in normalized:
            normalized["k"] = _limit_int(normalized.get("k"), default=4, minimum=1, maximum=6)
        if "subtype" in normalized:
            subtype = _clean_text(normalized.get("subtype"), limit=80)
            normalized["subtype"] = subtype or None
    if (
        name
        in {
            "get_recent_wrong_examples",
            "get_recent_attempts",
            "get_misconception_hints",
            "get_frontier_topics",
        }
        and "limit" in normalized
    ):
        normalized["limit"] = _limit_int(normalized.get("limit"), default=3, minimum=1, maximum=6)
    if name == "get_similar_solved_tasks" and "limit" in normalized:
        normalized["limit"] = _limit_int(normalized.get("limit"), default=2, minimum=1, maximum=4)
    if name == "get_recent_attempts" and "only_wrong" in normalized:
        normalized["only_wrong"] = _bool_arg(normalized.get("only_wrong"))
    if name == "get_misconception_hints" and "error_tags" in normalized:
        normalized["error_tags"] = _string_list_arg(normalized.get("error_tags"))
    return normalized


# ── Tool registry ────────────────────────────────────────────────────


class ConspectToolRegistry:
    """Registry and dispatcher for conspect-generation tools.

    Args:
        context: Runtime context providing profile store and RAG engine.
    """

    def __init__(self, context: ConspectToolContext) -> None:
        self.context = context
        self._tools = {
            "retrieve_learning_atoms": ConspectTool(
                name="retrieve_learning_atoms",
                description=(
                    "Найти справочные микротемы по запросу для ЕГЭ 6, 10 или 12. "
                    "Используй, когда не хватает правила, алгоритма или материала."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "task_number": {"type": "integer", "enum": [6, 10, 12]},
                        "subtype": {"type": "string"},
                        "k": {"type": "integer", "minimum": 1, "maximum": 6},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                func=self._retrieve_learning_atoms,
            ),
            "get_student_profile_summary": ConspectTool(
                name="get_student_profile_summary",
                description="Получить краткую сводку по текущему ученику: ошибки, подтипы, уровень детализации.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                func=self._get_student_profile_summary,
            ),
            "get_recent_wrong_examples": ConspectTool(
                name="get_recent_wrong_examples",
                description="Получить последние ошибочные попытки ученика с условием, решением и типом ошибки.",
                parameters={
                    "type": "object",
                    "properties": {
                        "task_number": {"type": "integer", "enum": [6, 10, 12]},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 5},
                    },
                    "additionalProperties": False,
                },
                func=self._get_recent_wrong_examples,
            ),
            "get_recent_attempts": ConspectTool(
                name="get_recent_attempts",
                description="Получить последние попытки ученика с условиями и решениями.",
                parameters={
                    "type": "object",
                    "properties": {
                        "task_number": {"type": "integer", "enum": [6, 10, 12]},
                        "only_wrong": {"type": "boolean"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 5},
                    },
                    "additionalProperties": False,
                },
                func=self._get_recent_attempts,
            ),
            "get_misconception_hints": ConspectTool(
                name="get_misconception_hints",
                description="Получить подсказки по типичным заблуждениям ученика.",
                parameters={
                    "type": "object",
                    "properties": {
                        "error_tags": {"type": "array", "items": {"type": "string"}},
                        "task_number": {"type": "integer", "enum": [6, 10, 12]},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 6},
                    },
                    "additionalProperties": False,
                },
                func=self._get_misconception_hints,
            ),
            "get_similar_solved_tasks": ConspectTool(
                name="get_similar_solved_tasks",
                description="Найти реальные похожие задачи с решениями для опоры при разборе примера.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "task_number": {"type": "integer", "enum": [6, 10, 12]},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 4},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                func=self._get_similar_solved_tasks,
            ),
            "get_frontier_topics": ConspectTool(
                name="get_frontier_topics",
                description="Получить ближайшие темы для изучения по графу пререквизитов и слабым местам ученика.",
                parameters={
                    "type": "object",
                    "properties": {
                        "task_number": {"type": "integer", "enum": [6, 10, 12]},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 6},
                    },
                    "additionalProperties": False,
                },
                func=self._get_frontier_topics,
            ),
        }

    def schemas_for(self, names: list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
        """Return OpenAI-compatible tool schemas for the given tool names."""
        return [self._tools[name].schema() for name in names if name in self._tools]

    def call(self, name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        """Dispatch a tool call by name, normalising arguments first.

        Args:
            name: Tool function name.
            arguments: Raw arguments dict (may have un-normalised keys).

        Returns:
            Tool result dict or ``{"error": "..."}`` on failure.
        """
        normalized_arguments = _normalize_tool_arguments(name, arguments)
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            return tool.func(normalized_arguments)
        except Exception as exc:
            log.warning("Conspect tool %s failed: %s", name, exc, exc_info=True)
            return {"error": f"{name} failed"}

    # ── Local tool implementations ───────────────────────────────────

    @staticmethod
    def recent_wrong_attempts(
        profile: StudentProfile,
        n: int = 5,
    ) -> list[dict[str, Any]]:
        """Return recent unique wrong tasks with as much context as the profile stores."""
        seen: set[str] = set()
        attempts: list[dict[str, Any]] = []

        for ev in reversed(profile.error_events):
            fallback_id = f"{ev.get('task_number')}:{ev.get('task_text', '')}"
            task_id = str(ev.get("task_id") or fallback_id)
            if task_id in seen:
                continue
            seen.add(task_id)

            task_text = ev.get("problem_katex") or ev.get("task_text") or ""
            attempts.append(
                {
                    "task_id": task_id,
                    "task_number": ev.get("task_number"),
                    "subtype": ev.get("subtype"),
                    "error_tag": ev.get("tag"),
                    "task_text": str(task_text).strip(),
                    "solution": str(ev.get("solution") or "").strip(),
                    "student_answer": ev.get("student_answer"),
                    "correct_answer": ev.get("correct_answer"),
                }
            )
            if len(attempts) >= n:
                break

        return attempts

    @staticmethod
    def build_queries(profile: StudentProfile) -> list[str]:
        """Mirror retrieval query-building logic using recent wrong tasks."""
        wrong_attempts = ConspectToolRegistry.recent_wrong_attempts(profile)
        queries: list[str] = []
        if wrong_attempts:
            for m in wrong_attempts:
                tt = m.get("solution") or m.get("task_text")
                sa = m.get("student_answer")
                ca = m.get("correct_answer")
                query = f"{tt}".strip()
                if sa is not None:
                    query += f" Ученик ответил {sa}."
                if ca is not None:
                    query += f" Правильный ответ {ca}."
                queries.append(query)
        return queries

    def _retrieve_learning_atoms(self, args: dict[str, Any]) -> dict[str, Any]:
        profile = self.context.load_profile()
        k = _limit_int(args.get("k"), default=4, minimum=1, maximum=6)
        task_number = _optional_int(args.get("task_number"))
        subtype = _clean_text(args.get("subtype"), limit=80) or None
        queries = ConspectToolRegistry.build_queries(profile)
        retrieved = self.context.rag.retrieve(
            query=queries, task_number=task_number, subtype=subtype, profile=profile, k=k
        )
        return {
            "items": [
                {
                    "title": item.atom.title,
                    "text": _clean_text(item.atom.text),
                    "task_number": item.atom.task_number,
                    "subtypes": [_subtype_label(st) for st in item.atom.subtypes[:4]],
                    "score": round(item.score, 4),
                }
                for item in retrieved[:k]
            ]
        }

    def _get_student_profile_summary(self, _args: dict[str, Any]) -> dict[str, Any]:
        return _profile_summary(self.context.load_profile())

    def _get_recent_wrong_examples(self, args: dict[str, Any]) -> dict[str, Any]:
        profile = self.context.load_profile()
        task_number = _optional_int(args.get("task_number"))
        limit = _limit_int(args.get("limit"), default=3, minimum=1, maximum=5)
        examples: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in reversed(profile.recent):
            if item.get("ok") is not False:
                continue
            if task_number is not None and item.get("task_number") != task_number:
                continue
            task_id = str(item.get("task_id") or "")
            if task_id in seen:
                continue
            seen.add(task_id)
            examples.append(
                {
                    "task_number": item.get("task_number"),
                    "subtype": _subtype_label(str(item.get("subtype") or "other")),
                    "problem": _clean_text(item.get("problem_katex") or item.get("task_text")),
                    "solution": _clean_text(item.get("solution"), limit=1200),
                    "errors": [_human_error(str(tag)) for tag in item.get("error_tags", [])],
                }
            )
            if len(examples) >= limit:
                break
        if len(examples) < limit:
            for ev in reversed(profile.error_events):
                if task_number is not None and ev.get("task_number") != task_number:
                    continue
                task_id = str(ev.get("task_id") or "")
                if task_id in seen:
                    continue
                seen.add(task_id)
                examples.append(
                    {
                        "task_number": ev.get("task_number"),
                        "subtype": _subtype_label(str(ev.get("subtype") or "other")),
                        "problem": _clean_text(ev.get("problem_katex") or ev.get("task_text")),
                        "student_answer": _clean_text(ev.get("student_answer"), limit=180),
                        "correct_answer": _clean_text(ev.get("correct_answer"), limit=180),
                        "errors": [_human_error(str(ev.get("tag") or "unknown"))],
                    }
                )
                if len(examples) >= limit:
                    break
        return {"items": examples}

    def _get_recent_attempts(self, args: dict[str, Any]) -> dict[str, Any]:
        profile = self.context.load_profile()
        task_number = _optional_int(args.get("task_number"))
        only_wrong = bool(args.get("only_wrong", False))
        limit = _limit_int(args.get("limit"), default=3, minimum=1, maximum=5)
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in reversed(profile.recent):
            if only_wrong and item.get("ok") is not False:
                continue
            if task_number is not None and item.get("task_number") != task_number:
                continue
            task_id = str(item.get("task_id") or "")
            if task_id in seen:
                continue
            seen.add(task_id)
            items.append(
                {
                    "task_number": item.get("task_number"),
                    "subtype": _subtype_label(str(item.get("subtype") or "other")),
                    "ok": bool(item.get("ok")),
                    "problem": _clean_text(item.get("problem_katex") or item.get("task_text")),
                    "solution": _clean_text(item.get("solution"), limit=1200),
                    "errors": [_human_error(str(tag)) for tag in item.get("error_tags", [])],
                }
            )
            if len(items) >= limit:
                break
        return {"items": items}

    def _get_misconception_hints(self, args: dict[str, Any]) -> dict[str, Any]:
        profile = self.context.load_profile()
        task_number = _optional_int(args.get("task_number"))
        limit = _limit_int(args.get("limit"), default=4, minimum=1, maximum=6)
        raw_tags = args.get("error_tags") or []
        tags = {str(tag) for tag in raw_tags if str(tag).strip()}
        if not tags:
            tags = {tag for tag, _score in top_errors_with_scores(profile, 8, subtype=None)}
        items: list[dict[str, Any]] = []
        for item in _load_misconceptions():
            if task_number is not None and item.get("task_number") != task_number:
                continue
            tag = str(item.get("error_tag") or "")
            if tags and tag not in tags:
                continue
            items.append(
                {
                    "title": item.get("title"),
                    "hint": item.get("text"),
                    "error": _human_error(tag),
                    "task_number": item.get("task_number"),
                }
            )
            if len(items) >= limit:
                break
        return {"items": items}

    def _get_similar_solved_tasks(self, args: dict[str, Any]) -> dict[str, Any]:
        profile = self.context.load_profile()
        query = _clean_text(args.get("query"), limit=320)
        task_number = _optional_int(args.get("task_number"))
        limit = _limit_int(args.get("limit"), default=2, minimum=1, maximum=4)
        query_tokens = _tokenize_query(query)
        candidates: list[tuple[int, dict[str, Any]]] = []
        for item in reversed(profile.recent):
            if task_number is not None and item.get("task_number") != task_number:
                continue
            problem = item.get("problem_katex") or item.get("task_text") or ""
            solution = item.get("solution") or ""
            if not problem or not solution:
                continue
            row = {
                "task_number": item.get("task_number"),
                "problem": problem,
                "solution": solution,
                "source": "student_recent",
            }
            candidates.append((_score_solution_row(row, query_tokens) + 3, row))
        for row in _load_solution_rows():
            if task_number is not None and row.get("task_number") != task_number:
                continue
            score = _score_solution_row(row, query_tokens)
            if score > 0 or not query_tokens:
                candidates.append((score, row))
        candidates.sort(key=lambda item: item[0], reverse=True)
        results: list[dict[str, Any]] = []
        seen_problems: set[str] = set()
        for score, row in candidates:
            problem = _clean_text(row.get("problem"), limit=650)
            if not problem or problem in seen_problems:
                continue
            seen_problems.add(problem)
            results.append(
                {
                    "task_number": row.get("task_number"),
                    "problem": problem,
                    "solution": _clean_text(row.get("solution"), limit=1300),
                    "source": row.get("source"),
                    "match_score": score,
                }
            )
            if len(results) >= limit:
                break
        return {"items": results}

    def _get_frontier_topics(self, args: dict[str, Any]) -> dict[str, Any]:
        profile = self.context.load_profile()
        task_number = _optional_int(args.get("task_number"))
        limit = _limit_int(args.get("limit"), default=5, minimum=1, maximum=6)
        atom_ids = next_frontier_atoms(profile, task_number=task_number, n=limit)
        return {
            "items": [
                {
                    "title": ATOM_BY_ID[atom_id].title,
                    "task_number": ATOM_BY_ID[atom_id].task_number,
                    "subtypes": [_subtype_label(st) for st in ATOM_BY_ID[atom_id].subtypes[:4]],
                    "text": _clean_text(ATOM_BY_ID[atom_id].text, limit=650),
                }
                for atom_id in atom_ids
                if atom_id in ATOM_BY_ID
            ]
        }


# ── MCP server factory ───────────────────────────────────────────────


@lru_cache(maxsize=1)
def _default_mcp_rag() -> RagEngine:
    return RagEngine()


class _LazyMcpRag:
    """Lazy wrapper so the MCP server creates the RagEngine on first tool call."""

    def retrieve(self, **kwargs):
        return _default_mcp_rag().retrieve(**kwargs)


def _default_mcp_context(student_id: str) -> ConspectToolContext:
    profiles_dir = Path(
        get_settings().conspect_mcp_profiles_dir or str(_DATA_DIR / "profiles")
    ).resolve()
    return ConspectToolContext(
        student_id=student_id,
        profile_store=ProfileStore(profiles_dir),
        rag=_LazyMcpRag(),  # type: ignore[arg-type]
    )


def _dispatch_mcp_tool(
    context_factory: Callable[[str], ConspectToolContext],
    student_id: str,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    registry = ConspectToolRegistry(context_factory(student_id))
    return registry.call(name, arguments)


def create_conspect_mcp_server(
    context_factory: Callable[[str], ConspectToolContext] = _default_mcp_context,
) -> FastMCP:
    """Build and return a FastMCP server exposing all conspect tools.

    Args:
        context_factory: Callable ``(student_id) -> ConspectToolContext``.
                         Defaults to ``_default_mcp_context`` which reads
                         ``CONSPECT_MCP_PROFILES_DIR`` from settings.
    """
    mcp = FastMCP("ConspectTools")

    @mcp.tool(
        description="Найти справочные микротемы по запросу для ЕГЭ 6, 10 или 12. Используй, когда не хватает правила, алгоритма или материала."
    )
    def retrieve_learning_atoms(
        query: str,
        task_number: Literal[6, 10, 12] | None = None,
        subtype: str | None = None,
        k: int = 4,
        student_id: str = "demo",
    ) -> dict[str, Any]:
        return _dispatch_mcp_tool(
            context_factory,
            student_id,
            "retrieve_learning_atoms",
            {"query": query, "task_number": task_number, "subtype": subtype, "k": k},
        )

    @mcp.tool(
        description="Получить краткую сводку по текущему ученику: ошибки, подтипы, уровень детализации."
    )
    def get_student_profile_summary(student_id: str = "demo") -> dict[str, Any]:
        return _dispatch_mcp_tool(context_factory, student_id, "get_student_profile_summary", {})

    @mcp.tool(
        description="Получить последние ошибочные попытки ученика с условием, решением и типом ошибки."
    )
    def get_recent_wrong_examples(
        task_number: Literal[6, 10, 12] | None = None,
        limit: int = 3,
        student_id: str = "demo",
    ) -> dict[str, Any]:
        return _dispatch_mcp_tool(
            context_factory,
            student_id,
            "get_recent_wrong_examples",
            {"task_number": task_number, "limit": limit},
        )

    @mcp.tool(description="Получить последние попытки ученика с условиями и решениями.")
    def get_recent_attempts(
        task_number: Literal[6, 10, 12] | None = None,
        only_wrong: bool = False,
        limit: int = 3,
        student_id: str = "demo",
    ) -> dict[str, Any]:
        return _dispatch_mcp_tool(
            context_factory,
            student_id,
            "get_recent_attempts",
            {"task_number": task_number, "only_wrong": only_wrong, "limit": limit},
        )

    @mcp.tool(description="Получить подсказки по типичным заблуждениям ученика.")
    def get_misconception_hints(
        error_tags: list[str] | None = None,
        task_number: Literal[6, 10, 12] | None = None,
        limit: int = 4,
        student_id: str = "demo",
    ) -> dict[str, Any]:
        return _dispatch_mcp_tool(
            context_factory,
            student_id,
            "get_misconception_hints",
            {
                "error_tags": error_tags or [],
                "task_number": task_number,
                "limit": limit,
            },
        )

    @mcp.tool(
        description="Найти реальные похожие задачи с решениями для опоры при разборе примера."
    )
    def get_similar_solved_tasks(
        query: str,
        task_number: Literal[6, 10, 12] | None = None,
        limit: int = 2,
        student_id: str = "demo",
    ) -> dict[str, Any]:
        return _dispatch_mcp_tool(
            context_factory,
            student_id,
            "get_similar_solved_tasks",
            {"query": query, "task_number": task_number, "limit": limit},
        )

    @mcp.tool(
        description="Получить ближайшие темы для изучения по графу пререквизитов и слабым местам ученика."
    )
    def get_frontier_topics(
        task_number: Literal[6, 10, 12] | None = None,
        limit: int = 5,
        student_id: str = "demo",
    ) -> dict[str, Any]:
        return _dispatch_mcp_tool(
            context_factory,
            student_id,
            "get_frontier_topics",
            {"task_number": task_number, "limit": limit},
        )

    return mcp


def run_conspect_mcp_server() -> None:
    """Start the MCP server with transport settings from ``AppSettings``."""
    s = get_settings()
    transport = s.conspect_mcp_transport.strip().lower().replace("_", "-")
    if transport == "stdio":
        create_conspect_mcp_server().run(transport="stdio", show_banner=False)
        return
    if transport not in ("http", "sse", "streamable-http"):
        raise ValueError(
            f"Unsupported CONSPECT_MCP_TRANSPORT {transport!r}; expected stdio, http, sse, or streamable-http."
        )
    create_conspect_mcp_server().run(
        transport=transport,
        host=s.conspect_mcp_host,
        port=s.conspect_mcp_port,
        path=s.conspect_mcp_path,
        show_banner=False,
    )


if __name__ == "__main__" and "--mcp-server" in sys.argv:
    run_conspect_mcp_server()
