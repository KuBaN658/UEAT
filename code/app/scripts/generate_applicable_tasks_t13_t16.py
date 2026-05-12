"""
Build atoms_t{13,16}_applicable_tasks.yaml from atoms_t{N}.yaml using subtype heuristics
on real FIPI task statements (`fipi_parsed_katex.json`).

For each atom with `subtypes: [...]`, we collect every FIPI task of the same task_number
whose classify_subtype() result intersects the atom's subtypes. Atoms tagged with universal
subtypes (`interval_selection`, `other`, or anything in T16) are also linked to all
tasks of that task_number — they apply broadly.

This is a v0 labeling — useful for retrieval benchmarks but not a substitute for
human / LLM ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from app.domain.subtypes import classify_subtype

REPO = Path(__file__).resolve().parents[2]
FIPI = REPO / "fipi_parsed_katex.json"
DATA = REPO / "app" / "data"

# Subtypes that are universal within their task — applicable to every task of that number.
UNIVERSAL_SUBTYPES_BY_TASK: dict[int, set[str]] = {
    13: {"interval_selection", "other"},
    16: {"other"},
}


def task_text(t: dict) -> str:
    return str(t.get("problem_katex") or t.get("question") or "")


def build(task_number: int) -> None:
    atoms = yaml.safe_load((DATA / f"atoms_t{task_number}.yaml").read_text(encoding="utf-8")) or []
    fipi = json.loads(FIPI.read_text(encoding="utf-8"))
    tasks = [t for t in fipi if int(t.get("task_number", 0)) == task_number]

    # Pre-classify each FIPI task once.
    classified: list[tuple[str, str]] = []
    for t in tasks:
        tid = str(t.get("id", "")).strip()
        if not tid:
            continue
        subtype = classify_subtype(task_text(t), task_number=task_number)
        classified.append((tid, subtype))

    universal = UNIVERSAL_SUBTYPES_BY_TASK.get(task_number, set())

    out: list[dict] = []
    for atom in atoms:
        subs = set(atom.get("subtypes") or [])
        is_universal = bool(subs & universal) or not subs

        applicable: list[str] = []
        if is_universal:
            applicable = [tid for tid, _ in classified]
        else:
            applicable = [tid for tid, sub in classified if sub in subs]

        # Fallback: an atom whose subtype heuristic doesn't match any real task
        # is still relevant as background theory — link to all tasks of that number.
        # This avoids zero-recall "dead" atoms in the benchmark; better to over-attach
        # than to silently exclude.
        if not applicable:
            applicable = [tid for tid, _ in classified]

        out.append(
            {
                "id": atom["id"],
                "title": atom["title"],
                "text": atom["text"],
                "task_number": task_number,
                "subtypes": atom.get("subtypes") or [],
                "error_tags": atom.get("error_tags") or [],
                "prerequisites": atom.get("prerequisites") or [],
                "shared_skills": atom.get("shared_skills") or [],
                "applicable_tasks": applicable,
            }
        )

    target = DATA / f"atoms_t{task_number}_applicable_tasks.yaml"
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False, width=10_000)

    total_links = sum(len(a["applicable_tasks"]) for a in out)
    print(
        f"Task {task_number}: atoms={len(out)}, fipi_tasks={len(classified)}, "
        f"applicable links={total_links}, avg per atom={total_links / max(1, len(out)):.1f}"
    )


if __name__ == "__main__":
    for n in (13, 16):
        build(n)
