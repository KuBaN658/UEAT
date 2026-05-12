"""
Run the same RAG conspect pipeline as GET /conspect on imported archive students.

Flow:
  1. import_submissions_archive — builds profiles via ProfileStore.record_attempt (same as API)
  2. This script — calls run_generate_conspect_rag_answer() from app (identical to production)

Usage (from repo root, after `uv sync`):
  uv run python -m app.scripts.import_submissions_archive
  uv run python -m app.scripts.generate_archive_conspects
  uv run python -m app.scripts.generate_archive_conspects --users 20,15,11
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")

from app.main import run_generate_conspect_rag_answer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch RAG conspects (same engine as /conspect)")
    ap.add_argument("--prefix", default="archive_u", help="Profile stem prefix")
    ap.add_argument(
        "--profiles-dir",
        type=Path,
        default=_ROOT / "data" / "profiles",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "data" / "archive_conspects",
        help="Output .md + .meta.json per student",
    )
    ap.add_argument("--limit", type=int, default=0, help="Process only first N profiles (0 = all)")
    ap.add_argument(
        "--users",
        type=str,
        default="",
        help="Comma-separated user numbers (e.g. 20,15,11). Only those archive_u{N}.json are processed.",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.users.strip():
        nums: list[int] = []
        for part in args.users.split(","):
            part = part.strip()
            if not part:
                continue
            nums.append(int(part))
        profiles = []
        for n in nums:
            p = args.profiles_dir / f"{args.prefix}{n}.json"
            if p.is_file():
                profiles.append(p)
            else:
                print(f"  WARN: missing profile {p.name}, skip", flush=True)
        if not profiles:
            raise SystemExit(f"No profiles found for --users {args.users!r} in {args.profiles_dir}")
    else:
        profiles = sorted(
            args.profiles_dir.glob(f"{args.prefix}*.json"),
            key=lambda p: p.stem,
        )
        if args.limit and args.limit > 0:
            profiles = profiles[: args.limit]
    if not profiles:
        raise SystemExit(f"No profiles matching {args.prefix}*.json in {args.profiles_dir}")

    print(f"Found {len(profiles)} profiles. Running RAG pipeline (same as API /conspect)...")
    ok_n = 0
    for p in profiles:
        student_id = p.stem
        print(f"  -> {student_id} ...", flush=True)
        try:
            text, retrieved_titles, frontier = run_generate_conspect_rag_answer(student_id)
        except ValueError as exc:
            print(f"     SKIP: {exc}", flush=True)
            continue
        except Exception as exc:
            print(f"     FAILED: {exc}", flush=True)
            traceback.print_exc()
            continue
        md_path = args.out / f"{student_id}.md"
        meta_path = args.out / f"{student_id}.meta.json"
        if not (text or "").strip():
            print(f"     FAILED: empty LLM output for {student_id}", flush=True)
            continue
        md_path.write_text(text, encoding="utf-8")
        meta_path.write_text(
            json.dumps(
                {
                    "student_id": student_id,
                    "retrieved_titles": retrieved_titles,
                    "frontier_atoms": frontier,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        ok_n += 1
        print(f"     wrote {md_path.name}", flush=True)

    print(f"Done. Wrote {ok_n}/{len(profiles)} conspects. Output: {args.out.resolve()}")


if __name__ == "__main__":
    main()
