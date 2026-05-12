"""
Retrieval evaluation with a direct LLM judge.

Evaluates whether each retrieved atom is useful for solving tasks
where the student made mistakes. Prompts are loaded from
app/data/conspect_prompts.yaml.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from app.core.config import get_rag_settings
from app.domain.profile import StudentProfile
from app.infrastructure.llm.clients import build_chat_client_for_judge
from app.infrastructure.repositories.profile_repo import ProfileStore
from app.infrastructure.retrieval.engine import RagEngine, Retrieved
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

SETTINGS = get_rag_settings()

PROFILES_DIR = DATA_DIR / "profiles"
PROMPTS_PATH = DATA_DIR / "conspect_prompts.yaml"
STUDENT_ID = "archive_u7"
MAX_WRONG_TASKS = 5

ALLOWED_SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)

# Concurrency for judge LLM calls; override with JUDGE_CONCURRENCY env var.
# Use 1 for Ollama (local GPU/CPU), 6-8 for Groq/OpenRouter.
JUDGE_CONCURRENCY = int(os.environ.get("JUDGE_CONCURRENCY", "8"))

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class JudgeResult:
    atom_id: str
    atom_title: str
    retrieval_score: float
    judge_score: float
    is_relevant: bool
    matched_task_numbers: list[int]
    rationale: str
    raw_response: str


def load_prompt(key: str) -> str:
    data = yaml.safe_load(PROMPTS_PATH.read_text(encoding="utf-8")) or {}
    prompt = str(data.get(key) or "").strip()
    if not prompt:
        raise RuntimeError(f"{key} missing or empty in {PROMPTS_PATH}")
    return prompt


def recent_wrong_attempts(
    profile: StudentProfile,
    n: int = MAX_WRONG_TASKS,
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


def build_queries(profile: StudentProfile) -> list[str]:
    """Mirror retrieval query-building logic using recent wrong tasks."""
    wrong_attempts = recent_wrong_attempts(profile)
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


def format_wrong_tasks(wrong_attempts: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for i, m in enumerate(wrong_attempts, 1):
        lines = [
            f"Задача {i}:",
        ]
        if m.get("solution"):
            lines.append(f"- Условие задачи с примером решения: {m['solution']}")
        elif m.get("task_text"):
            lines.append(f"- Условие задачи: {m['task_text']}")
        if m.get("student_answer") is not None:
            lines.append(f"- Ответ ученика: {m['student_answer']}")
        if m.get("correct_answer") is not None:
            lines.append(f"- Правильный ответ: {m['correct_answer']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_atom_judge_prompt(
    relevance_prompt: str,
    wrong_attempts: list[dict[str, Any]],
    item: Retrieved,
) -> str:
    atom = item.atom
    return relevance_prompt.format(
        wrong_tasks=format_wrong_tasks(wrong_attempts),
        atom_text=atom.title + " " + atom.text,
    )


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def normalize_judge_result(item: Retrieved, raw_response: str) -> JudgeResult:
    raw = raw_response or ""
    if not raw.strip():
        data: dict[str, Any] = {
            "score": 0.0,
            "matched_task_numbers": [],
            "rationale": "empty_llm_response",
        }
    else:
        try:
            data = extract_json_object(raw)
        except json.JSONDecodeError:
            data = {
                "score": 0.0,
                "matched_task_numbers": [],
                "rationale": "invalid_json_from_llm",
            }
    raw_score = float(data.get("score", 0.0))
    judge_score = min(ALLOWED_SCORES, key=lambda score: abs(score - raw_score))
    matched_task_numbers = [
        int(x)
        for x in data.get("matched_task_numbers", [])
        if isinstance(x, int) or (isinstance(x, str) and x.isdigit())
    ]

    return JudgeResult(
        atom_id=item.atom.id,
        atom_title=item.atom.title,
        retrieval_score=float(item.score),
        judge_score=judge_score,
        is_relevant=judge_score >= 0.2,
        matched_task_numbers=matched_task_numbers,
        rationale=str(data.get("rationale") or "").strip(),
        raw_response=raw_response,
    )


def _judge_single(
    client: Any,
    system_prompt: str,
    relevance_prompt: str,
    wrong_attempts: list[dict[str, Any]],
    item: Retrieved,
) -> JudgeResult:
    user_prompt = build_atom_judge_prompt(relevance_prompt, wrong_attempts, item)
    response = client.chat(
        system=system_prompt,
        user=user_prompt,
        temperature=0.0,
        max_tokens=2000,
    )
    return normalize_judge_result(item, response.text)


def judge_atoms(
    retrieved: list[Retrieved],
    wrong_attempts: list[dict[str, Any]],
) -> list[JudgeResult]:
    system_prompt = load_prompt("judge_system_prompt")
    relevance_prompt = load_prompt("retrieval_relevance_check_prompt")
    client = build_chat_client_for_judge()

    results: dict[int, JudgeResult] = {}
    with ThreadPoolExecutor(max_workers=JUDGE_CONCURRENCY) as pool:
        futures = {
            pool.submit(
                _judge_single,
                client,
                system_prompt,
                relevance_prompt,
                wrong_attempts,
                item,
            ): idx
            for idx, item in enumerate(retrieved)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                item = retrieved[idx]
                log.warning("Judge failed for atom %s: %s", item.atom.id, exc)
                results[idx] = JudgeResult(
                    atom_id=item.atom.id,
                    atom_title=item.atom.title,
                    retrieval_score=float(item.score),
                    judge_score=0.0,
                    is_relevant=False,
                    matched_task_numbers=[],
                    rationale=f"judge_error: {type(exc).__name__}",
                    raw_response=str(exc),
                )
    return [results[idx] for idx in range(len(retrieved))]


def judge_result_to_dict(r: JudgeResult) -> dict[str, Any]:
    d = asdict(r)
    # dataclass field names match JSON-friendly types
    return d


def run_judge_retrieval() -> dict[str, Any]:
    """
    Run LLM relevance evaluation for all archive profiles with wrong tasks.
    Returns a JSON-serializable dict.
    """
    rag = RagEngine()
    score = 0.0
    count = 0
    students_out: list[dict[str, Any]] = []

    for user_path in PROFILES_DIR.glob("archive_u[0-9]*.json"):
        student_id = user_path.stem

        profile_store = ProfileStore(root_dir=PROFILES_DIR)
        profile = profile_store.load(student_id)
        wrong_attempts = recent_wrong_attempts(profile)

        if not wrong_attempts:
            print(f"У профиля {student_id!r} нет ошибочных задач для relevance-eval")
            print()
            continue

        print(f"Студент: {student_id}")
        print(f"Ошибочных задач для judge: {len(wrong_attempts)}\n")

        queries = build_queries(profile)

        retrieved = rag.retrieve(
            query=queries,
            task_number=None,
            subtype=None,
            profile=profile,
            k=SETTINGS.rrf_k,
        )

        results = judge_atoms(retrieved, wrong_attempts)
        avg = sum(r.judge_score for r in results) / len(results) if results else 0.0
        score += avg
        count += 1
        relevant_count = sum(1 for r in results if r.is_relevant)

        print(f"=== Итог для студента {student_id} ===")
        print(f"Средний judge score: {avg:.3f}")
        print(f"Релевантных атомов: {relevant_count}/{len(results)}")
        print()

        students_out.append(
            {
                "student_id": student_id,
                "wrong_tasks_count": len(wrong_attempts),
                "queries": queries,
                "avg_judge_score": avg,
                "relevant_count": relevant_count,
                "retrieved_count": len(results),
                "results": [judge_result_to_dict(r) for r in results],
            }
        )

    overall = score / count if count else None
    if overall is not None:
        print(f"Средний score по всем студентам: {overall:.3f}")
    return {
        "config": {
            "profiles_dir": str(PROFILES_DIR),
            "retrieval_k": SETTINGS.rrf_k,
            "max_wrong_tasks": MAX_WRONG_TASKS,
        },
        "students": students_out,
        "student_count": count,
        "overall_avg_judge_score": overall,
    }


def main() -> None:
    print("=== LLM Retrieval Relevance Evaluation ===\n")
    run_judge_retrieval()


if __name__ == "__main__":
    main()
