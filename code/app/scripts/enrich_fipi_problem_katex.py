"""
One-time / batch: read FIPI JSON, convert MathML in `problem` to LaTeX, set `problem_katex`.

Usage (repo root):
  uv run python -m app.scripts.enrich_fipi_problem_katex --input fipi_parsed_katex.json
  uv run python -m app.scripts.enrich_fipi_problem_katex --input fipi_parsed_katex.json --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.infrastructure.conspect.mathml import problem_html_to_katex


def main() -> None:
    ap = argparse.ArgumentParser(description="Populate problem_katex from problem MathML.")
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("fipi_parsed_katex.json"),
        help="Path to FIPI tasks JSON",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write here (default: overwrite --input)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Count only, do not write")
    args = ap.parse_args()
    inp = args.input.resolve()
    if not inp.is_file():
        raise SystemExit(f"Missing file: {inp}")

    raw = json.loads(inp.read_text(encoding="utf-8"))
    updated = 0
    skipped_has = 0
    skipped_no_problem = 0
    for item in raw:
        pk = item.get("problem_katex")
        if pk and str(pk).strip():
            skipped_has += 1
            continue
        prob = item.get("problem")
        if not prob or not str(prob).strip():
            skipped_no_problem += 1
            continue
        katex = problem_html_to_katex(str(prob))
        if katex:
            item["problem_katex"] = katex
            updated += 1

    out = args.output.resolve() if args.output else inp
    if not args.dry_run:
        out.write_text(json.dumps(raw, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")

    print(
        f"Updated problem_katex: {updated} | already had: {skipped_has} | no problem field: {skipped_no_problem} | total: {len(raw)}"
    )
    if args.dry_run:
        print("Dry run — file not written.")


if __name__ == "__main__":
    main()
