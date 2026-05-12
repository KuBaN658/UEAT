"""
Import depersonalized submission archive into RAG student profiles (tasks 6, 10, and 12).

CSV columns (semicolon, UTF-8 BOM): user_task_submissions.csv from platform export.

Usage (from repo root that contains app/, after `uv sync`):
  uv run python -m app.scripts.import_submissions_archive
  uv run python -m app.scripts.import_submissions_archive --archive path/to/archive --dry-run

Profiles are written to app/data/profiles/ as archive_u{user_number}.json
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from app.domain.atoms import ERROR_TAGS_BY_TASK
from app.domain.profile import StudentProfile
from app.domain.subtypes import classify_subtype
from app.infrastructure.repositories.profile_repo import ProfileStore


def _parse_bool(s: str) -> bool:
    return str(s).strip().lower() in ("true", "1", "yes")


def _parse_time(s: str) -> datetime:
    s = str(s).strip()
    # "2026-02-04 22:23:20.665872+00:00"
    if "+" in s:
        s = s.split("+")[0].strip()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00").split("+")[0])
    except ValueError:
        return datetime.min


def infer_error_tags(task_number: int, node_name: str, task_name: str) -> list[str]:
    """
    Map platform micromodule labels to internal ERROR_TAGS (heuristic; no LLM).
    """
    text = f"{node_name} {task_name}".lower()
    allowed = set(ERROR_TAGS_BY_TASK.get(task_number, ()))

    def pick(*candidates: str) -> list[str]:
        for c in candidates:
            if c in allowed:
                return [c]
        return []

    if task_number == 6:
        if any(k in text for k in ("лог", "ln ", "ln(", "логарифм")):
            return pick("log_domain_missed", "log_property_error")
        if any(k in text for k in ("sin", "cos", "tg", "тригонометр", "синус", "косинус")):
            return pick("trig_lost_solutions", "trig_extra_solutions")
        if any(k in text for k in ("корень", "sqrt", "иррацион")):
            return pick("irrational_no_check")
        if any(k in text for k in ("степен", "показател", "^", "a^x")):
            return pick("wrong_base_reduction", "exp_property_error")
        if any(k in text for k in ("квадрат", "дискриминант", "биквадрат")):
            return pick("quadratic_discriminant_error", "forgot_back_substitution")
        return pick("sign_error_in_equation", "forgot_back_substitution")

    if task_number == 10:
        if any(
            k in text
            for k in (
                "течени",
                "по течению",
                "против течения",
                "теплоход",
                "катер",
                "лодк",
                "плот",
            )
        ):
            return pick(
                "river_swap_plus_minus",
                "river_forgot_v_gt_u",
                "motion_wrong_relative_speed",
            )
        if any(k in text for k in ("прогресс", "арифмет", "геометр", "член прогрессии")):
            return pick("progression_wrong_formula", "progression_off_by_one")
        if any(k in text for k in ("процент", "раствор", "смес", "концентрац", "сплав")):
            return pick(
                "percent_wrong_base",
                "percent_used_as_number",
                "mixture_not_using_balance",
            )
        if any(k in text for k in ("работ", "насос", "труба", "бак", "выполнит", "детал")):
            return pick("work_wrong_rate_equation", "work_added_times_instead_of_rates")
        if any(k in text for k in ("км/ч", "скорост", "встрет", "догнал", "выехал", "прибыл")):
            return pick("motion_wrong_relative_speed", "motion_mixed_time_and_distance")
        if "средн" in text:
            return pick("avg_speed_arithmetic_mean", "avg_speed_used_wrong_total_time")
        if "уравнен" in text or "неизвест" in text:
            return pick("picked_wrong_variable", "lost_parentheses_or_sign")
        return pick("picked_wrong_variable", "lost_parentheses_or_sign")

    # task 12
    if any(k in text for k in ("sin", "cos", "tg", "ctg", "триг", "синус", "косинус", "тангенс")):
        return pick("trig_deriv_error")
    if any(k in text for k in ("лог", "ln ", "ln(", "натуральн")):
        return pick("log_deriv_error")
    if any(k in text for k in ("показател", "e^", "exp")):
        return pick("exp_deriv_error")
    if "частн" in text or "отношен" in text or "quotient" in text:
        return pick("quotient_rule_error")
    if "произвед" in text or "product" in text:
        return pick("product_rule_error")
    if any(k in text for k in ("экстремум", "минимум", "максимум", "критич", "производн")):
        return pick(
            "critical_point_wrong_eq",
            "minmax_forgot_endpoints",
            "minmax_wrong_sign_analysis",
        )
    return pick("deriv_wrong_rule", "deriv_sign_error")


def _fix_fallback_tags(task_number: int, tags: list[str]) -> list[str]:
    """Remove invalid tag names from heuristic fallback."""
    allowed = set(ERROR_TAGS_BY_TASK.get(task_number, ()))
    out = [t for t in tags if t in allowed]
    return out if out else ["unknown"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Import archive CSV into RAG profiles (tasks 6/10/12)."
    )
    ap.add_argument(
        "--archive",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "submissions_archive",
        help="Folder containing user_task_submissions.csv",
    )
    ap.add_argument(
        "--profiles-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "profiles",
        help="Directory for JSON profiles",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and count only; do not write files",
    )
    ap.add_argument("--prefix", default="archive_u", help="student_id = {prefix}{user_number}")
    args = ap.parse_args()

    csv_path = args.archive / "user_task_submissions.csv"
    if not csv_path.is_file():
        raise SystemExit(f"Missing file: {csv_path}")

    rows: list[dict[str, str]] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            try:
                tn = int(float(row["task_number"]))
            except (KeyError, ValueError):
                continue
            if tn not in (6, 10, 12):
                continue
            rows.append(row)

    # Per user, chronological order
    by_user: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        u = str(row["user_number"]).strip()
        by_user.setdefault(u, []).append(row)
    for u in by_user:
        by_user[u].sort(key=lambda r: _parse_time(r.get("submission_time", "")))

    store = ProfileStore(root_dir=args.profiles_dir.resolve())
    total_attempts = 0
    total_wrong = 0

    for user_number, urows in sorted(by_user.items(), key=lambda x: int(x[0])):
        student_id = f"{args.prefix}{user_number}"
        # Fresh profile each run so re-import replaces archive data (no duplicate attempts).
        profile = StudentProfile(student_id=student_id) if not args.dry_run else None

        for row in urows:
            task_number = int(float(row["task_number"]))
            node_name = (row.get("node_name") or "").strip()
            task_name = (row.get("task_name") or "").strip()
            task_text = f"{task_name}. {node_name}".strip() or task_name or node_name or "task"
            subtype = classify_subtype(task_text, task_number=task_number)
            ok = _parse_bool(row.get("is_correct", "False"))
            task_id = (row.get("task_id") or row.get("submission_id") or "unknown").strip()

            if not ok:
                raw_tags = infer_error_tags(task_number, node_name, task_name)
                tags = _fix_fallback_tags(task_number, raw_tags)
                total_wrong += 1
            else:
                tags = []

            total_attempts += 1
            if not args.dry_run and profile is not None:
                weights = {t: 1.0 for t in tags} if tags else None
                profile.record_attempt(
                    task_id=task_id,
                    task_number=task_number,
                    subtype=subtype,
                    ok=ok,
                    error_tags=tags,
                    error_weights=weights,
                    student_answer=None,
                    correct_answer=None,
                    task_text=task_text[:500] if not ok else None,
                )
        if not args.dry_run and profile is not None:
            store.save(profile)

    print(
        f"Users: {len(by_user)} | Rows (tasks 6/10/12): {len(rows)} | "
        f"attempts: {total_attempts} | wrong: {total_wrong}"
    )
    if args.dry_run:
        print("Dry run — no profiles written.")
    else:
        print(f"Profiles saved under {args.profiles_dir.resolve()}")


if __name__ == "__main__":
    main()
