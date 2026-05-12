"""
G-Eval conspect quality evaluation.

Evaluates a generated conspect against three whole-conspect metrics:
  - personalization  — how well the conspect targets the specific student's weaknesses
  - school_format    — ЕГЭ/school format compliance, section completeness and purpose match
  - math_correctness — correctness of all formulas, expressions, and calculations

overall_score = mean(personalization, school_format, math_correctness)

Conspect markdowns and their student context metadata are read from
``app/rag/eval/conspects/``.

Run via the runner:
    uv run python -m app.rag.eval.run_conspect_benchmarks
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from app.rag.eval.deepeval_llm import JudgeLLM

log = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
CONSPECTS_DIR = EVAL_DIR / "conspects"
RULES_PATH = Path(__file__).resolve().parents[3] / "app" / "data" / "conspect_eval_rules.yaml"

# Concurrency for judge LLM calls; override with JUDGE_CONCURRENCY env var.
JUDGE_CONCURRENCY = int(os.environ.get("JUDGE_CONCURRENCY", "4"))

METRIC_KEYS = ("personalization", "school_format", "math_correctness")


# ── YAML rules ───────────────────────────────────────────────────────


def load_eval_rules() -> dict[str, Any]:
    """Load and return the conspect evaluation rules YAML."""
    data = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8")) or {}
    if "metrics" not in data:
        raise RuntimeError(f"'metrics' key missing in {RULES_PATH}")
    for key in METRIC_KEYS:
        if key not in data["metrics"]:
            raise RuntimeError(f"metric '{key}' missing in {RULES_PATH}")
    return data


# ── Personalization input formatter ─────────────────────────────────


def format_personalization_input(meta: dict[str, Any]) -> str:
    """Format student context metadata into a readable text block for G-Eval.

    The text is passed as the ``input`` field of the test case, so the judge
    model can reference the student's profile while evaluating the full conspect.

    Args:
        meta: Loaded ``.meta.json`` dict for the student.

    Returns:
        A compact text block describing the student's errors and frontier topics.
    """
    lines: list[str] = []

    top_err = str(meta.get("top_err") or "").strip()
    if top_err:
        lines.append(f"Главная ошибка ученика (top_err): {top_err}")

    top_errors = meta.get("top_errors") or []
    if top_errors:
        lines.append("Топ ошибок по тегам: " + ", ".join(str(t) for t in top_errors[:6]))

    frontier = meta.get("frontier_atoms") or []
    if frontier:
        lines.append("Frontier-темы (ещё не изучены): " + "; ".join(frontier))

    recent_wrong: list[dict] = meta.get("recent_wrong") or []
    if recent_wrong:
        lines.append("Недавние неверные задачи:")
        for i, rw in enumerate(recent_wrong, 1):
            tn = rw.get("task_number", "?")
            txt = str(rw.get("task_text") or "").strip()[:120]
            sa = rw.get("student_answer")
            ca = rw.get("correct_answer")
            parts = [f"  {i}. Задание №{tn}"]
            if txt:
                parts.append(f": {txt}")
            if sa is not None:
                parts.append(f" | ответил: {sa}")
            if ca is not None:
                parts.append(f" | правильно: {ca}")
            lines.append("".join(parts))

    retrieved = meta.get("retrieved_titles") or []
    if retrieved:
        lines.append("Использованные материалы: " + "; ".join(retrieved[:6]))

    return "\n".join(lines) if lines else "Данных о профиле ученика нет."


# ── Single GEval runner ──────────────────────────────────────────────


_INVALID_JSON_MARKER = "invalid json"

# Number of retries when the judge outputs malformed JSON; override with JUDGE_JSON_RETRIES.
JUDGE_JSON_RETRIES = int(os.environ.get("JUDGE_JSON_RETRIES", "3"))


def _run_geval(
    name: str,
    evaluation_steps: list[str],
    criteria: str,
    input_text: str,
    actual_output: str,
    judge_llm: JudgeLLM,
    include_input: bool = False,
) -> dict[str, Any]:
    """Run a single GEval metric and return {score, reason, success}.

    Retries up to ``JUDGE_JSON_RETRIES`` times when the judge outputs invalid JSON.

    Args:
        name: Metric name.
        evaluation_steps: Steps for the G-Eval judge.
        criteria: Short criteria string.
        input_text: Test-case ``input`` field (student profile for personalization,
            empty string for other metrics).
        actual_output: Full conspect markdown text to evaluate.
        judge_llm: DeepEval-compatible judge model.
        include_input: If True, passes INPUT alongside ACTUAL_OUTPUT to the judge
            (used for personalization so the judge sees the student profile).

    Returns:
        Dict with ``score`` (float 0–1), ``reason`` (str), ``success`` (bool).
    """
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, SingleTurnParams

    params = [SingleTurnParams.ACTUAL_OUTPUT]
    if include_input:
        params.append(SingleTurnParams.INPUT)

    test_case = LLMTestCase(input=input_text, actual_output=actual_output)

    last_exc: Exception | None = None
    max_attempts = 1 + JUDGE_JSON_RETRIES

    for attempt in range(1, max_attempts + 1):
        metric = GEval(
            name=name,
            criteria=criteria,
            evaluation_steps=evaluation_steps,
            evaluation_params=params,
            model=judge_llm,
            threshold=0.7,
            async_mode=False,
        )
        try:
            metric.measure(test_case, _show_indicator=False)
            return {
                "score": float(metric.score or 0.0),
                "reason": str(metric.reason or ""),
                "success": bool(metric.is_successful()),
                "error": False,
            }
        except Exception as exc:
            last_exc = exc
            if _INVALID_JSON_MARKER in str(exc).lower():
                log.warning(
                    "GEval '%s' invalid JSON on attempt %d/%d, retrying…",
                    name,
                    attempt,
                    max_attempts,
                )
                continue
            # Non-JSON error — no point retrying
            log.warning("GEval '%s' failed: %s", name, exc)
            return {
                "score": 0.0,
                "reason": f"eval_error: {type(exc).__name__}: {exc}",
                "success": False,
                "error": True,
            }

    log.warning("GEval '%s' failed after %d attempts: %s", name, max_attempts, last_exc)
    return {
        "score": 0.0,
        "reason": f"eval_error: {type(last_exc).__name__}: {last_exc}",
        "success": False,
        "error": True,
    }


# ── Single conspect evaluation ───────────────────────────────────────


def evaluate_conspect(
    student_id: str,
    md: str,
    meta: dict[str, Any],
    rules: dict[str, Any],
    judge_llm: JudgeLLM,
) -> dict[str, Any]:
    """Evaluate a single conspect across all three whole-conspect metrics.

    Args:
        student_id: Student identifier (used for logging).
        md: Full conspect markdown text.
        meta: Loaded student meta dict.
        rules: Loaded eval rules.
        judge_llm: DeepEval judge model.

    Returns:
        Dict ``{"metrics": {key: result}, "overall_score": float}``
        where each ``result`` has ``score``, ``reason``, ``success``.
    """
    metrics_rules: dict[str, dict] = rules["metrics"]
    persona_input = format_personalization_input(meta)

    tasks: list[tuple[str, str, list[str], str, str, bool]] = []
    for key in METRIC_KEYS:
        cfg = metrics_rules[key]
        include_input = bool(cfg.get("include_input", False))
        input_text = persona_input if include_input else ""
        tasks.append(
            (
                key,
                cfg["criteria"],
                cfg["evaluation_steps"],
                input_text,
                md,
                include_input,
            )
        )

    results: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=min(JUDGE_CONCURRENCY, len(tasks))) as pool:
        future_to_key = {
            pool.submit(
                _run_geval,
                key,
                steps,
                criteria,
                input_text,
                actual_output,
                judge_llm,
                include_input,
            ): key
            for key, criteria, steps, input_text, actual_output, include_input in tasks
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                log.warning(
                    "Evaluation task failed for '%s' (student=%s): %s", key, student_id, exc
                )
                results[key] = {
                    "score": 0.0,
                    "reason": f"task_error: {exc}",
                    "success": False,
                    "error": True,
                }

    metric_results = {
        k: results.get(k, {"score": 0.0, "reason": "", "success": False, "error": True})
        for k in METRIC_KEYS
    }
    has_error = any(v.get("error", False) for v in metric_results.values())
    overall_score = sum(v["score"] for v in metric_results.values()) / len(metric_results)

    log.info(
        "Student %s | personalization=%.3f | school_format=%.3f | math_correctness=%.3f | overall=%.3f%s",
        student_id,
        metric_results["personalization"]["score"],
        metric_results["school_format"]["score"],
        metric_results["math_correctness"]["score"],
        overall_score,
        " [EXCLUDED — eval error]" if has_error else "",
    )

    return {"metrics": metric_results, "overall_score": overall_score, "has_error": has_error}


# ── Multi-student runner ─────────────────────────────────────────────


def run_judge_conspect(
    conspects_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate all conspects found in *conspects_dir*.

    Expects pairs of files: ``{student_id}.md`` and ``{student_id}.meta.json``.
    Students with no ``.meta.json`` are evaluated with an empty meta dict.

    Args:
        conspects_dir: Directory to scan (default: ``app/rag/eval/conspects/``).

    Returns:
        JSON-serialisable dict with ``students``, ``student_count``,
        ``overall_avg_score``, and per-metric averages.
    """
    cdir = conspects_dir or CONSPECTS_DIR
    rules = load_eval_rules()
    judge_llm = JudgeLLM()

    md_files = sorted(cdir.glob("*.md"))
    if not md_files:
        log.warning("No .md files found in %s", cdir)
        return {
            "students": [],
            "student_count": 0,
            "overall_avg_score": None,
            "per_metric_avg": {k: None for k in METRIC_KEYS},
        }

    students_out: list[dict[str, Any]] = []

    for md_path in md_files:
        student_id = md_path.stem
        meta_path = md_path.with_suffix(".meta.json")

        if not meta_path.exists():
            log.warning("No meta.json for %s — evaluating without student profile", student_id)
            meta: dict[str, Any] = {}
        else:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                log.warning("Failed to load meta for %s: %s", student_id, exc)
                meta = {}

        md = md_path.read_text(encoding="utf-8")

        print(f"\nОцениваю конспект: {student_id}")
        eval_result = evaluate_conspect(student_id, md, meta, rules, judge_llm)

        m = eval_result["metrics"]
        has_error = eval_result.get("has_error", False)

        if has_error:
            failed_metrics = [k for k in METRIC_KEYS if m[k].get("error", False)]
            print(f"  [ИСКЛЮЧЁН из статистики — ошибка оценки в: {', '.join(failed_metrics)}]")
        else:
            print(
                f"  Персонализация={m['personalization']['score']:.3f}"
                f"  Школьный формат={m['school_format']['score']:.3f}"
                f"  Матем. корректность={m['math_correctness']['score']:.3f}"
                f"  Итог={eval_result['overall_score']:.3f}"
            )

        students_out.append(
            {
                "student_id": student_id,
                "overall_score": eval_result["overall_score"],
                "metrics": eval_result["metrics"],
                "has_error": has_error,
            }
        )

    # Only include error-free conspects in aggregate statistics
    valid_students = [s for s in students_out if not s.get("has_error", False)]
    excluded_count = len(students_out) - len(valid_students)

    student_count = len(valid_students)
    overall_avg = (
        sum(s["overall_score"] for s in valid_students) / student_count if valid_students else None
    )

    per_metric_avg: dict[str, float | None] = {}
    for key in METRIC_KEYS:
        scores = [s["metrics"][key]["score"] for s in valid_students if key in s["metrics"]]
        per_metric_avg[key] = sum(scores) / len(scores) if scores else None

    print("\n=== Итог по всем конспектам ===")
    print(f"Оценено (без ошибок): {student_count}")
    if excluded_count:
        print(f"Исключено из статистики (ошибка оценки): {excluded_count}")
    if overall_avg is not None:
        for key in METRIC_KEYS:
            avg = per_metric_avg[key]
            print(f"  {key}: {avg:.3f}" if avg is not None else f"  {key}: —")
        print(f"Средний итоговый score: {overall_avg:.3f}")
    else:
        print("Нет валидных конспектов для подсчёта статистики.")

    return {
        "config": {
            "conspects_dir": str(cdir),
            "rules_path": str(RULES_PATH),
            "judge_concurrency": JUDGE_CONCURRENCY,
        },
        "students": students_out,
        "student_count": student_count,
        "excluded_count": excluded_count,
        "overall_avg_score": overall_avg,
        "per_metric_avg": per_metric_avg,
    }


def main() -> None:
    print("=== G-Eval Conspect Quality Evaluation ===\n")
    run_judge_conspect()


if __name__ == "__main__":
    main()
