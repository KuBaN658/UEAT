"""
Run G-Eval conspect quality benchmark and write results as JSON.

From repo root:
    uv run python -m app.rag.eval.run_conspect_benchmarks

Options:
    --conspects-dir   Path to the directory with *.md + *.meta.json files.
                      Default: app/rag/eval/conspects/
    --out             Path for the output JSON.
                      Default: app/rag/eval/metrics/conspect_geval_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_CONSPECTS_DIR = EVAL_DIR / "conspects"
DEFAULT_OUT_JSON = EVAL_DIR / "metrics" / "conspect_geval_results.json"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    print(f"Wrote {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run G-Eval conspect quality benchmark.")
    parser.add_argument(
        "--conspects-dir",
        type=Path,
        default=DEFAULT_CONSPECTS_DIR,
        help=f"Directory with *.md + *.meta.json conspect files (default: {DEFAULT_CONSPECTS_DIR})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_JSON,
        help=f"Output JSON path (default: {DEFAULT_OUT_JSON})",
    )
    args = parser.parse_args(argv)

    if not args.conspects_dir.exists():
        print(
            f"Conspects directory does not exist: {args.conspects_dir}",
            file=sys.stderr,
        )
        return 1

    from app.rag.eval.judge_conspect import run_judge_conspect

    data = run_judge_conspect(conspects_dir=args.conspects_dir)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "geval": data,
    }
    _write_json(args.out, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
