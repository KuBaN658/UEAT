"""
Run IR benchmark (benchmark_retrieval) and LLM-judge eval (judge_retrieval);
write results as JSON in this directory.

From repo root:
    uv run python -m app.rag.eval.run_retrieval_benchmarks
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_BENCHMARK_JSON = EVAL_DIR / "metrics" / "benchmark_retrieval_results.json"
DEFAULT_JUDGE_JSON = EVAL_DIR / "metrics" / "judge_retrieval_results.json"
DEFAULT_README_PATH = EVAL_DIR.parent.parent / "README.md"
README_METRICS_START = "<!-- AUTO_METRICS:START -->"
README_METRICS_END = "<!-- AUTO_METRICS:END -->"
METRIC_ORDER = ("precision", "recall", "ndcg", "mrr", "map", "hitrate")


def _write_json(path: Path, payload: object) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    print(f"Wrote {path}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_retrieval_metrics(benchmark_payload: dict[str, Any]) -> list[str]:
    benchmark = benchmark_payload.get("benchmark") or {}
    per_task = benchmark.get("per_task") or {}
    task_numbers = benchmark.get("task_numbers") or []
    all_tasks = benchmark.get("all_tasks") or {}
    lines: list[str] = []

    def append_task_block(title: str, task_data: dict[str, Any]) -> None:
        evaluated = int(task_data.get("evaluated", 0))
        macro = task_data.get("macro_averages") or {}
        lines.append(f"### {title}: evaluated={evaluated}")
        lines.append("")
        for k in benchmark.get("k_values") or []:
            metrics = macro.get(str(k)) or {}
            metric_values = " ".join(
                f"{name}={float(metrics.get(name, 0.0)):.4f}" for name in METRIC_ORDER
            )
            lines.append(f"@{k}: {metric_values}")
        lines.append("")

    for task_number in task_numbers:
        task_data = per_task.get(str(task_number)) or {}
        append_task_block(f"Task {task_number}", task_data)

    append_task_block("All tasks", all_tasks)
    return lines


def _format_judge_metrics(judge_payload: dict[str, Any]) -> list[str]:
    judge = judge_payload.get("judge") or {}
    students = judge.get("students") or []
    student_count = int(judge.get("student_count", len(students)))
    overall = judge.get("overall_avg_judge_score")
    cfg = judge.get("config") or {}
    retrieval_k = int(cfg.get("retrieval_k", 0))
    total_relevant = sum(int(student.get("relevant_count", 0)) for student in students)
    total_retrieved = sum(int(student.get("retrieved_count", 0)) for student in students)
    avg_relevant = (total_relevant / total_retrieved) if total_retrieved else 0.0

    lines = [
        "## Метрики LLM judge",
        "",
        f"- Оценено студентов: {student_count}",
        (
            f"- Overall avg judge score: {float(overall):.4f}"
            if overall is not None
            else "- Overall avg judge score: n/a"
        ),
        f"- Средняя доля релевантных атомов@{retrieval_k}: {avg_relevant:.2%} ({total_relevant}/{total_retrieved})",
        "",
    ]
    return lines


def _build_readme_metrics_block(
    benchmark_payload: dict[str, Any], judge_payload: dict[str, Any]
) -> str:
    lines = [
        "## Метрики retrieval",
        "",
        README_METRICS_START,
        "",
        "_Источник: `rag/eval/benchmark_retrieval_results.json` и `rag/eval/judge_retrieval_results.json`._",
        "",
    ]
    lines.extend(_format_retrieval_metrics(benchmark_payload))
    lines.extend(_format_judge_metrics(judge_payload))
    lines.append(README_METRICS_END)
    return "\n".join(lines)


def _replace_metrics_section(readme_text: str, new_section: str) -> str:
    if README_METRICS_START in readme_text and README_METRICS_END in readme_text:
        start_idx = readme_text.index(README_METRICS_START)
        end_idx = readme_text.index(README_METRICS_END) + len(README_METRICS_END)
        prefix = readme_text[:start_idx]
        suffix = readme_text[end_idx:]
        replacement = new_section[new_section.index(README_METRICS_START) :]
        return prefix + replacement + suffix

    start_anchor = "\n## Метрики retrieval\n"
    end_anchor = "\n## Тесты\n"
    start_idx = readme_text.find(start_anchor)
    if start_idx == -1:
        raise RuntimeError("Could not find '## Метрики retrieval' section in README.")
    end_idx = readme_text.find(end_anchor, start_idx)
    if end_idx == -1:
        raise RuntimeError("Could not find '## Тесты' section after metrics in README.")
    return readme_text[: start_idx + 1] + new_section + "\n\n" + readme_text[end_idx + 1 :]


def _update_readme_metrics(
    readme_path: Path, benchmark_payload: dict[str, Any], judge_payload: dict[str, Any]
) -> None:
    readme_text = readme_path.read_text(encoding="utf-8")
    new_section = _build_readme_metrics_block(benchmark_payload, judge_payload)
    updated_text = _replace_metrics_section(readme_text, new_section)
    readme_path.write_text(updated_text, encoding="utf-8")
    print(f"Updated metrics in {readme_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run retrieval benchmarks and save JSON in app/rag/eval/."
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Only run the LLM judge eval.",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Only run the IR / hybrid benchmark.",
    )
    parser.add_argument(
        "--benchmark-out",
        type=Path,
        default=DEFAULT_BENCHMARK_JSON,
        help=f"Path for IR benchmark JSON (default: {DEFAULT_BENCHMARK_JSON})",
    )
    parser.add_argument(
        "--judge-out",
        type=Path,
        default=DEFAULT_JUDGE_JSON,
        help=f"Path for judge eval JSON (default: {DEFAULT_JUDGE_JSON})",
    )
    parser.add_argument(
        "--skip-readme-update",
        action="store_true",
        help="Do not update metrics block in README.",
    )
    parser.add_argument(
        "--readme-path",
        type=Path,
        default=DEFAULT_README_PATH,
        help=f"Path to README with metrics block (default: {DEFAULT_README_PATH})",
    )
    args = parser.parse_args(argv)

    if args.skip_benchmark and args.skip_judge and args.skip_readme_update:
        print(
            "Nothing to do: both --skip-benchmark and --skip-judge set.",
            file=sys.stderr,
        )
        return 2

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    benchmark_payload: dict[str, Any] | None = None
    judge_payload: dict[str, Any] | None = None

    if not args.skip_benchmark:
        from app.rag.eval.benchmark_retrieval import run_benchmark

        data = run_benchmark()
        out = {**meta, "benchmark": data}
        _write_json(args.benchmark_out, out)
        benchmark_payload = out

    if not args.skip_judge:
        from app.rag.eval.judge_retrieval import run_judge_retrieval

        data = run_judge_retrieval()
        out = {**meta, "judge": data}
        _write_json(args.judge_out, out)
        judge_payload = out

    if not args.skip_readme_update:
        if benchmark_payload is None:
            if args.benchmark_out.exists():
                benchmark_payload = _read_json(args.benchmark_out)
            else:
                print(
                    f"Skip README update: benchmark JSON not found at {args.benchmark_out}.",
                    file=sys.stderr,
                )
                return 1
        if judge_payload is None:
            if args.judge_out.exists():
                judge_payload = _read_json(args.judge_out)
            else:
                print(
                    f"Skip README update: judge JSON not found at {args.judge_out}.",
                    file=sys.stderr,
                )
                return 1
        _update_readme_metrics(args.readme_path, benchmark_payload, judge_payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
