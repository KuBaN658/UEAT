"""
Task repository: load and serve EGE task items from a parsed FIPI JSON file.

``load_tasks`` parses the raw JSON.
``TaskBank`` wraps a list of tasks with per-task random pickers.
"""

from __future__ import annotations

import html
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from random import Random

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Task:
    """An immutable EGE task item.

    Attributes:
        id: Unique task identifier (string).
        task_number: EGE task number (6, 10, or 12).
        question: Plain-text question (may contain HTML entities).
        problem_katex: KaTeX-enriched problem text preferred over *question*.
        answer: Correct answer string.
    """

    id: str
    task_number: int
    question: str
    problem_katex: str | None
    answer: str

    def prompt_text(self) -> str:
        """Return the best available text representation of this task prompt."""
        raw = (self.problem_katex or "").strip() or self.question.strip()
        return normalize_task_text(raw)


def normalize_task_text(text: str) -> str:
    """Clean HTML entities and normalise whitespace / punctuation artefacts."""
    t = html.unescape(text or "")
    t = (
        t.replace("\u200a", " ")
        .replace("\u2009", " ")
        .replace("\u00a0", " ")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
    )
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"(?:\s*[.;,:]){2,}\s*$", "", t)
    t = re.sub(r"\s*;\s*;\s*$", "", t)
    t = re.sub(r"\s+\)", ")", t)
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"\s+([,.:;!?])", r"\1", t)
    return t.strip()


def load_tasks(path: Path) -> list[Task]:
    """Parse a FIPI JSON file and return a list of ``Task`` objects.

    Malformed items are skipped with a warning.

    Args:
        path: Path to the JSON file.

    Returns:
        List of valid ``Task`` instances.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: list[Task] = []
    for item in raw:
        try:
            out.append(
                Task(
                    id=str(item.get("id", "")),
                    task_number=int(item.get("task_number", -1)),
                    question=str(item.get("question", "")),
                    problem_katex=item.get("problem_katex"),
                    answer=str(item.get("answer", "")).strip(),
                )
            )
        except Exception as exc:
            log.warning("Skipping invalid task item: %s", exc)
            continue
    return out


def only_task_number(tasks: list[Task], task_number: int) -> list[Task]:
    """Filter tasks to those with the given task number and a non-empty answer."""
    return [t for t in tasks if t.task_number == task_number and t.id and t.answer]


class TaskPicker:
    """Randomly selects tasks from a fixed pool using a seeded RNG.

    Args:
        tasks: Pool of tasks to pick from.
        seed: RNG seed for reproducibility.
    """

    def __init__(self, tasks: list[Task], seed: int = 42) -> None:
        self.tasks = tasks
        self.rng = Random(seed)

    def random_task(self) -> Task:
        """Return a uniformly random task."""
        return self.rng.choice(self.tasks)


class TaskBank:
    """In-memory store of tasks with per-task-number random selection.

    Args:
        tasks: All loaded tasks (all task numbers).
        seed: Base RNG seed; per-task seed is ``seed + task_number``.
    """

    def __init__(self, tasks: list[Task], seed: int = 42) -> None:
        self._all = tasks
        self._pickers: dict[int, TaskPicker] = {}
        from app.core.config import get_rag_settings

        for n in sorted(get_rag_settings().supported_tasks):
            subset = only_task_number(tasks, n)
            if subset:
                self._pickers[n] = TaskPicker(subset, seed=seed + n)

    def task_numbers(self) -> list[int]:
        """Return sorted list of task numbers that have at least one task."""
        return sorted(self._pickers.keys())

    def random_task(self, task_number: int) -> Task:
        """Return a random task for *task_number*.

        Raises:
            ValueError: If no tasks are available for *task_number*.
        """
        p = self._pickers.get(task_number)
        if not p:
            raise ValueError(f"No tasks available for task_number={task_number}")
        return p.random_task()

    def by_id(self, task_id: str) -> Task | None:
        """Look up a task by its string identifier."""
        for t in self._all:
            if t.id == task_id:
                return t
        return None
