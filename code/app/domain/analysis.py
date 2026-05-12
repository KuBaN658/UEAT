"""
Profile analysis functions: error scoring, mastery, frontier selection.

These are pure functions over ``StudentProfile`` + the ATOMS knowledge base.
They have no I/O or infrastructure dependencies.
"""

from __future__ import annotations

from collections import defaultdict, deque

from app.core.config import get_rag_settings
from app.domain.atoms import ATOM_BY_ID, ATOMS, get_dependents
from app.domain.profile import StudentProfile


def apply_cross_task_skill_transfer(profile: StudentProfile, tag: str, base_weight: float) -> None:
    """Propagate an error signal to cross-task skill scores.

    When an error is diagnosed in one task, the shared skills associated with
    atoms that carry that error tag are boosted in the profile.

    Args:
        profile: The student profile to update in-place.
        tag: The diagnosed error tag.
        base_weight: Weight of the error event (from diagnosis confidence).
    """
    cross = get_rag_settings().cross_pollination
    for atom in ATOMS:
        if tag in atom.error_tags:
            for skill in atom.shared_skills:
                profile.skill_scores[skill] = (
                    profile.skill_scores.get(skill, 0.0) + cross * base_weight
                )


def top_errors(profile: StudentProfile, n: int = 5, subtype: str | None = None) -> list[str]:
    """Return the top *n* error tags by decayed score.

    Args:
        profile: Student profile.
        n: Maximum number of tags to return.
        subtype: If set, restrict tags to those associated with this subtype.

    Returns:
        List of error-tag strings sorted by descending decayed score.
    """
    all_tags = {t for a in ATOMS for t in a.error_tags}
    scores: list[tuple[str, float]] = [(t, profile.error_score(t)) for t in all_tags]

    if subtype:
        subtype_tags: set[str] = set()
        for atom in ATOMS:
            if subtype in atom.subtypes:
                subtype_tags.update(atom.error_tags)
        scores = [(tag, sc) for tag, sc in scores if tag in subtype_tags]

    scores = [(tag, sc) for tag, sc in scores if sc > 0]
    scores.sort(key=lambda kv: kv[1], reverse=True)
    return [k for k, _v in scores[:n]]


def top_errors_with_scores(
    profile: StudentProfile, n: int = 5, subtype: str | None = None
) -> list[tuple[str, float]]:
    """Return the top *n* ``(tag, score)`` pairs by decayed error score."""
    all_tags = {t for a in ATOMS for t in a.error_tags}
    scores: list[tuple[str, float]] = [(t, profile.error_score(t)) for t in all_tags]

    if subtype:
        subtype_tags: set[str] = set()
        for atom in ATOMS:
            if subtype in atom.subtypes:
                subtype_tags.update(atom.error_tags)
        scores = [(tag, sc) for tag, sc in scores if tag in subtype_tags]

    scores = [(tag, sc) for tag, sc in scores if sc > 0]
    scores.sort(key=lambda kv: kv[1], reverse=True)
    return scores[:n]


def atom_weakness(profile: StudentProfile, atom_id: str) -> float:
    """Average decayed error score across all error tags of *atom_id*.

    Returns 0.0 if the atom is unknown or has no error tags.
    """
    atom = ATOM_BY_ID.get(atom_id)
    if not atom:
        return 0.0
    own = sum(profile.error_score(tag) for tag in atom.error_tags)
    return own / max(1, len(atom.error_tags))


def mastery_level(profile: StudentProfile, atom_id: str) -> float:
    """Estimated mastery level in [0, 1] (higher = more mastered).

    Derived from ``atom_weakness``: ``max(0, 1 - weakness)``.
    Used to down-rank retrieval of atoms the student no longer needs.
    """
    w = atom_weakness(profile, atom_id)
    return max(0.0, min(1.0, 1.0 - w))


def predicted_weakness(profile: StudentProfile, atom_id: str, depth: int = 2) -> float:
    """Propagate weakness signal down the prerequisite graph with factor 0.7/edge.

    Args:
        profile: Student profile.
        atom_id: Root atom ID.
        depth: Maximum BFS depth.

    Returns:
        Accumulated weakness including dependents.
    """
    base = atom_weakness(profile, atom_id)
    if depth <= 0:
        return base
    accum = base
    frontier: deque[tuple[str, int, float]] = deque([(atom_id, 0, base)])
    seen: set[str] = {atom_id}
    while frontier:
        aid, d, w = frontier.popleft()
        if d >= depth:
            continue
        for child in get_dependents(aid):
            if child.id in seen:
                continue
            seen.add(child.id)
            child_w = w * 0.7
            accum += child_w
            frontier.append((child.id, d + 1, child_w))
    return accum


def get_recent_wrong_attempts(
    profile: StudentProfile, task_number: int | None = None, n: int = 5
) -> list[dict]:
    """Extract recent wrong attempts that include task text, student answer, and correct answer.

    Used to build personalised retrieval queries for conspect generation.

    Args:
        profile: Student profile.
        task_number: If set, filter to this EGE task number.
        n: Maximum number of records to return.

    Returns:
        List of dicts with keys ``task_text``, ``student_answer``,
        ``correct_answer``, ``task_number``.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for ev in reversed(profile.error_events):
        tn = ev.get("task_number")
        if task_number is not None and tn != task_number:
            continue
        task_id = ev.get("task_id", "")
        if task_id in seen:
            continue
        seen.add(task_id)
        out.append(
            {
                "task_text": ev.get("problem_katex", ""),
                "solution": ev.get("solution", ""),
                "student_answer": ev.get("student_answer"),
                "correct_answer": ev.get("correct_answer"),
                "task_number": tn,
            }
        )
        if len(out) >= n:
            break
    return out


def next_frontier_atoms(
    profile: StudentProfile, task_number: int | None = None, n: int = 5
) -> list[str]:
    """Return atom IDs at the learning frontier: weak but prerequisites are ready.

    Args:
        profile: Student profile.
        task_number: If set, restrict candidates to this task.
        n: Maximum number of atom IDs to return.

    Returns:
        List of atom IDs sorted by frontier priority (descending).
    """
    cfg = get_rag_settings()
    candidates: list[tuple[str, float]] = []
    for atom in ATOMS:
        if (task_number is not None and atom.task_number != task_number) or atom.id.startswith(
            "mc_"
        ):
            continue
        own = atom_weakness(profile, atom.id)
        if own < cfg.frontier_weakness_threshold:
            continue
        prereq_avg = 0.0
        if atom.prerequisites:
            prereq_avg = sum(atom_weakness(profile, pid) for pid in atom.prerequisites) / len(
                atom.prerequisites
            )
        pred = predicted_weakness(profile, atom.id)
        priority = (
            own + cfg.frontier_predicted_weight * pred - cfg.frontier_prereq_weight * prereq_avg
        )
        candidates.append((atom.id, priority))
    candidates.sort(key=lambda x: x[1], reverse=True)
    if task_number is not None:
        return [aid for aid, _ in candidates[:n]]
    per_task: dict[int, int] = defaultdict(int)
    cap = cfg.frontier_max_per_task
    selected: list[str] = []
    for aid, _ in candidates:
        if len(selected) >= n:
            break
        atom = ATOM_BY_ID.get(aid)
        if not atom:
            continue
        tn = atom.task_number
        if per_task[tn] >= cap:
            continue
        selected.append(aid)
        per_task[tn] += 1
    return selected
