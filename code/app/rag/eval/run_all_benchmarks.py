"""
Run retrieval benchmarks, conspect G-Eval, and render the metrics dashboard
into a timestamped directory under app/rag/eval/metrics/.

From repo root:
    uv run python -m app.rag.eval.run_all_benchmarks
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
METRICS_ROOT = EVAL_DIR / "metrics"


def main(argv: list[str] | None = None) -> int:
    _ = argv
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = METRICS_ROOT / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    from app.rag.eval.render_metrics_dashboard import (
        BENCHMARK_JSON,
        CONSPECT_JSON,
        JUDGE_RETRIEVAL_JSON,
        write_dashboard,
    )
    from app.rag.eval.run_conspect_benchmarks import main as run_conspect_main
    from app.rag.eval.run_retrieval_benchmarks import main as run_retrieval_main

    benchmark_out = run_dir / BENCHMARK_JSON
    judge_out = run_dir / JUDGE_RETRIEVAL_JSON

    rc = run_retrieval_main(
        [
            "--benchmark-out",
            str(benchmark_out),
            "--judge-out",
            str(judge_out),
            "--skip-readme-update",
        ]
    )
    if rc != 0:
        return rc

    conspect_out = run_dir / CONSPECT_JSON
    rc = run_conspect_main(["--out", str(conspect_out)])
    if rc != 0:
        return rc

    write_dashboard(metrics_dir=run_dir, out_path=run_dir / "dashboard.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
