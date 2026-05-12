"""
Atom dataclass — the atomic unit of educational content.

Intentionally has no imports from other application modules so it can be
used as a pure value object anywhere in the codebase without circular deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Atom:
    """An immutable micro-module (knowledge atom) in the EGE knowledge base.

    Atoms form a directed prerequisite DAG and are indexed in the Qdrant
    vector store for hybrid retrieval.

    Attributes:
        id: Unique identifier (e.g. ``"t10_mm07"``).
        title: Short human-readable title used in retrieval and prompts.
        text: Full educational content; indexed for dense and sparse search.
        task_number: EGE task this atom belongs to (6, 10, or 12; 0 = cross-task).
        subtypes: Tuple of subtype tags (e.g. ``("water_motion",)``).
        error_tags: Known error tags that this atom helps remedy.
        prerequisites: IDs of atoms that must be understood first.
        shared_skills: Cross-task skill identifiers for skill-transfer scoring.
    """

    id: str
    title: str
    text: str
    task_number: int = 0
    subtypes: tuple[str, ...] = field(default_factory=tuple)
    error_tags: tuple[str, ...] = field(default_factory=tuple)
    prerequisites: tuple[str, ...] = field(default_factory=tuple)
    shared_skills: tuple[str, ...] = field(default_factory=tuple)
