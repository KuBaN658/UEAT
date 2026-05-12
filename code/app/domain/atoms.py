"""
Knowledge-base atoms for EGE Tasks 6, 10, 12, 13, and 16.

Atoms are loaded once at import time from the YAML files in ``data/``.
The module also exposes error-tag registries and prerequisite-graph helpers.

Public singletons
-----------------
- ``ATOMS``        — ordered list of all ``Atom`` objects (content + misconceptions).
- ``ATOM_BY_ID``   — mapping ``{atom_id: Atom}``.
- ``DEPENDENTS_OF`` — reverse prerequisite index ``{atom_id: [Atom, ...]}``.
- ``ERROR_TAGS``   — tuple of all known error-tag strings.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import yaml

from .atom import Atom

# ── Error-tag registries ─────────────────────────────────────────────

ERROR_TAGS_T10: tuple[str, ...] = (
    "units_mismatch",
    "lost_parentheses_or_sign",
    "picked_wrong_variable",
    "did_not_check_constraints",
    "percent_used_as_number",
    "percent_wrong_base",
    "percent_chain_changes",
    "mixture_not_using_balance",
    "mixture_percent_not_to_fraction",
    "mixture_wrong_total_mass",
    "work_added_times_instead_of_rates",
    "work_wrong_rate_equation",
    "motion_mixed_time_and_distance",
    "motion_wrong_relative_speed",
    "motion_forgot_delay_or_stop",
    "river_swap_plus_minus",
    "river_forgot_v_gt_u",
    "avg_speed_arithmetic_mean",
    "avg_speed_used_wrong_total_time",
    "progression_wrong_formula",
    "progression_off_by_one",
)

ERROR_TAGS_T12: tuple[str, ...] = (
    "deriv_wrong_rule",
    "deriv_forgot_chain_rule",
    "deriv_sign_error",
    "critical_point_wrong_eq",
    "critical_point_missed_domain",
    "minmax_forgot_endpoints",
    "minmax_wrong_sign_analysis",
    "minmax_confused_min_max",
    "trig_deriv_error",
    "log_deriv_error",
    "exp_deriv_error",
    "quotient_rule_error",
    "product_rule_error",
)

ERROR_TAGS_T6: tuple[str, ...] = (
    "wrong_base_reduction",
    "log_domain_missed",
    "log_property_error",
    "exp_property_error",
    "trig_lost_solutions",
    "trig_extra_solutions",
    "irrational_no_check",
    "quadratic_discriminant_error",
    "forgot_back_substitution",
    "sign_error_in_equation",
)

ERROR_TAGS_T13: tuple[str, ...] = (
    # ОДЗ
    "trig_forgot_odz_function",
    "trig_forgot_odz_denominator",
    # Loss / introduction of roots (also in T6 — shared by trig nature)
    "trig_lost_solutions",
    "trig_extra_solutions",
    # Identity / formula errors
    "trig_wrong_reduction",
    "trig_wrong_double_angle",
    "trig_wrong_arc_inverse",
    # Substitution / series handling
    "trig_substitution_no_check",
    "trig_lost_series_subst",
    "trig_merging_series_wrong",
    # Interval root selection
    "trig_interval_endpoint",
    "trig_selection_per_series_skip",
    "trig_selection_circle_misread",
    "trig_enumeration_no_outside_check",
)

ERROR_TAGS_T16: tuple[str, ...] = (
    # Compound / simple
    "fin_simple_vs_compound",
    "fin_wrong_base_for_percent",
    "fin_wrong_growth_factor",
    # Periods / order
    "fin_off_by_one_periods",
    "fin_order_charge_payment",
    "fin_monthly_yearly_mismatch",
    # Equations / sums
    "fin_geometric_sum_index_off",
    "fin_boundary_debt_not_zero",
    "fin_overpayment_calc_error",
    # Schemes
    "fin_table_share_misread",
    "fin_annuity_vs_diff_confusion",
    "fin_replenishment_timing",
    # Constraints
    "fin_integer_constraint_ignored",
)


def _dedup_tags(*groups: tuple[str, ...]) -> tuple[str, ...]:
    """Concatenate while preserving order and dropping duplicates across tasks."""
    seen: set[str] = set()
    out: list[str] = []
    for g in groups:
        for t in g:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return tuple(out)


ERROR_TAGS: tuple[str, ...] = _dedup_tags(
    ERROR_TAGS_T10, ERROR_TAGS_T12, ERROR_TAGS_T6, ERROR_TAGS_T13, ERROR_TAGS_T16
)

ERROR_TAGS_BY_TASK: dict[int, tuple[str, ...]] = {
    6: ERROR_TAGS_T6,
    10: ERROR_TAGS_T10,
    12: ERROR_TAGS_T12,
    13: ERROR_TAGS_T13,
    16: ERROR_TAGS_T16,
}

# ── Shared cross-task skills ─────────────────────────────────────────

SHARED_SKILLS: dict[str, list[str]] = {
    "equation_setup": [
        "t10_mm07",
        "t10_mm08",
        "t10_mm09",
        "t10_mm11",
        "t12_mm09",
        "t12_mm10",
        "t12_mm22",
        "t12_mm28",
        "t06_mm01",
        "t06_mm04",
        "t06_mm18",
        # T13: substitution, factorisation
        "t13_mm03",
        "t13_mm16",
        # T16: debt evolution as recurrence, optimisation
        "t16_mm03",
        "t16_mm17",
        "t16_mm31",
    ],
    "constraint_checking": [
        "t10_mm53",
        "t10_mm55",
        "t12_mm34",
        "t12_mm36",
        "t06_mm05",
        "t06_mm13",
        "t06_mm26",
        # T13: ODZ for trig fractions, log of trig, selection with ODZ
        "t13_mm21",
        "t13_mm22",
        "t13_mm35",
    ],
    "sign_analysis": [
        "t10_mm18",
        "t10_mm19",
        "t12_mm09",
        "t12_mm11",
        "t06_mm12",
        "t06_mm20",
        # T13: factorisation cases
        "t13_mm20",
    ],
    "answer_plausibility": [
        "t10_mm01",
        "t10_mm03",
        "t12_mm01",
        "t12_mm09",
        "t06_mm26",
    ],
}

# ── YAML loaders ─────────────────────────────────────────────────────

_REQUIRED_ATOM_KEYS = ("id", "title", "text", "task_number")


def _load_atoms_from_yaml(data_dir: Path) -> list[Atom]:
    """Load content atoms from the three per-task YAML files."""
    atoms: list[Atom] = []
    for name in ("atoms_t10", "atoms_t12", "atoms_t6", "atoms_t13", "atoms_t16"):
        path = data_dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Atoms file not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"Expected list of atoms in {path}")
        for i, d in enumerate(raw):
            if not isinstance(d, dict):
                raise ValueError(f"Atom at index {i} in {path} must be a dict")
            for key in _REQUIRED_ATOM_KEYS:
                if key not in d:
                    raise ValueError(f"Atom at index {i} in {path} missing required key: {key}")
            d = dict(d)
            atoms.append(
                Atom(
                    id=str(d["id"]),
                    title=str(d["title"]),
                    text=str(d["text"]),
                    task_number=int(d["task_number"]),
                    subtypes=tuple(d.get("subtypes") or []),
                    error_tags=tuple(d.get("error_tags") or []),
                    prerequisites=tuple(d.get("prerequisites") or []),
                    shared_skills=tuple(d.get("shared_skills") or []),
                )
            )
    return atoms


def _load_misconceptions_from_yaml(data_dir: Path) -> list[Atom]:
    """Load misconception atoms (human-authored buggy rules) for RAG retrieval."""
    path = data_dir / "misconceptions.yaml"
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []
    atoms: list[Atom] = []
    known = set(ERROR_TAGS)
    for i, d in enumerate(raw):
        if not isinstance(d, dict):
            continue
        tag = d.get("error_tag")
        if not tag or tag not in known:
            continue
        atoms.append(
            Atom(
                id=str(d["id"]),
                title=str(d["title"]),
                text=str(d["text"]).strip(),
                task_number=int(d.get("task_number", 10)),
                subtypes=tuple(),
                error_tags=(tag,),
                prerequisites=tuple(),
                shared_skills=tuple(),
            )
        )
    return atoms


def _data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


# Module-level singletons populated at import time.
ATOMS: list[Atom] = _load_atoms_from_yaml(_data_dir()) + _load_misconceptions_from_yaml(_data_dir())
ATOM_BY_ID: dict[str, Atom] = {a.id: a for a in ATOMS}


def _build_dependents_index(atoms: list[Atom]) -> dict[str, list[Atom]]:
    idx: dict[str, list[Atom]] = {}
    for a in atoms:
        for p in a.prerequisites:
            idx.setdefault(p, []).append(a)
    return idx


DEPENDENTS_OF: dict[str, list[Atom]] = _build_dependents_index(ATOMS)


# ── Validation ───────────────────────────────────────────────────────


def validate_error_tags() -> None:
    """Raise ValueError if any atom references an unknown error tag.

    Called once during ``RagEngine.__init__`` to catch YAML authoring mistakes.
    """
    known = set(ERROR_TAGS)
    for a in ATOMS:
        for tag in a.error_tags:
            if tag not in known:
                raise ValueError(f"Unknown error tag in atom {a.id}: {tag}")


# ── Prerequisite-graph helpers ───────────────────────────────────────


def get_prerequisites(atom_id: str, depth: int = 2) -> list[Atom]:
    """Return prerequisite atoms up to *depth* hops away (BFS)."""
    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(atom_id, 0)])
    result: list[Atom] = []
    while queue:
        aid, d = queue.popleft()
        if aid in visited or d > depth:
            continue
        visited.add(aid)
        atom = ATOM_BY_ID.get(aid)
        if atom and aid != atom_id:
            result.append(atom)
        if atom and d < depth:
            for pid in atom.prerequisites:
                if pid not in visited:
                    queue.append((pid, d + 1))
    return result


def get_dependents(atom_id: str) -> list[Atom]:
    """Return atoms that directly depend on *atom_id* (children in DAG)."""
    return DEPENDENTS_OF.get(atom_id, [])


def topological_order(atoms: list[Atom]) -> list[Atom]:
    """Sort *atoms* in prerequisite-first (topological) order."""
    ids = {a.id for a in atoms}
    by_id = {a.id: a for a in atoms}
    visited: set[str] = set()
    order: list[Atom] = []

    def visit(aid: str) -> None:
        if aid in visited or aid not in ids:
            return
        visited.add(aid)
        atom = by_id[aid]
        for pid in atom.prerequisites:
            if pid in ids:
                visit(pid)
        order.append(atom)

    for a in atoms:
        visit(a.id)
    return order
