"""
Save a generated conspect markdown + personalization context to the eval archive.

Writes two files per student under ``output_dir`` (default: ``app/rag/eval/conspects/``):
  - ``{student_id}.md``        — raw markdown text
  - ``{student_id}.meta.json`` — student context needed for G-Eval personalization scoring
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.domain.analysis import (
    get_recent_wrong_attempts,
    next_frontier_atoms,
    top_errors,
)
from app.domain.atoms import ATOM_BY_ID
from app.infrastructure.repositories.profile_repo import ProfileStore
from app.infrastructure.retrieval.engine import _human_error

log = logging.getLogger(__name__)

_DEFAULT_EVAL_DIR = Path(__file__).resolve().parents[3] / "app" / "rag" / "eval" / "conspects"


def build_personalization_context(
    profile_store: ProfileStore,
    student_id: str,
) -> dict[str, Any]:
    """Collect all student-specific data needed for personalization evaluation.

    Args:
        profile_store: Profile store used to load the student profile.
        student_id: Student identifier.

    Returns:
        Dict with keys: ``top_errors``, ``top_err``, ``frontier_atoms``,
        ``recent_wrong``, ``n_err_items``.
    """
    profile = profile_store.load(student_id)

    error_tags = top_errors(profile, 8, subtype=None)
    top_err_raw = error_tags[0] if error_tags else ""
    top_err_human = _human_error(top_err_raw) if top_err_raw else ""

    frontier_ids = next_frontier_atoms(profile, task_number=None, n=6)
    frontier_titles = [ATOM_BY_ID[aid].title for aid in frontier_ids if aid in ATOM_BY_ID]

    recent_wrong_raw = get_recent_wrong_attempts(profile, task_number=None, n=5)
    recent_wrong = [
        {
            "task_number": r.get("task_number"),
            "task_text": str(r.get("task_text") or r.get("solution") or "")[:200],
            "student_answer": r.get("student_answer"),
            "correct_answer": r.get("correct_answer"),
        }
        for r in recent_wrong_raw
    ]

    n_err_items = min(len(error_tags), 4) or 2

    return {
        "top_errors": error_tags,
        "top_err": top_err_human,
        "frontier_atoms": frontier_titles,
        "recent_wrong": recent_wrong,
        "n_err_items": n_err_items,
    }


def save_conspect_for_eval(
    markdown_text: str,
    student_id: str,
    profile_store: ProfileStore,
    retrieved_titles: list[str] | None = None,
    frontier_atoms: list[str] | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Write markdown and meta.json to the eval archive directory.

    Args:
        markdown_text: Full conspect markdown.
        student_id: Student identifier (used for filenames).
        profile_store: Profile store for building personalization context.
        retrieved_titles: RAG-retrieved atom titles from the generation run.
        frontier_atoms: Frontier atom IDs from the generation run.
        output_dir: Override for the output directory.

    Returns:
        Path to the saved ``.md`` file.
    """
    out = output_dir or _DEFAULT_EVAL_DIR
    out.mkdir(parents=True, exist_ok=True)

    md_path = out / f"{student_id}.md"
    md_path.write_text(markdown_text, encoding="utf-8")

    persona_ctx = build_personalization_context(profile_store, student_id)

    meta: dict[str, Any] = {
        "student_id": student_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "retrieved_titles": retrieved_titles or [],
        "frontier_atoms": frontier_atoms or [],
        **persona_ctx,
    }

    meta_path = out / f"{student_id}.meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("Eval archive saved: %s + meta", md_path)
    return md_path
