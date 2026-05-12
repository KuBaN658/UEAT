"""
StudentProfile — the mutable student learning state.

Contains attempt history, decayed error scores, and cross-task skill ratings.
The ``ProfileStore`` that persists profiles lives in
``infrastructure/repositories/profile_repo.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.core.config import get_rag_settings


def utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StudentProfile:
    """Mutable learning profile for a single student.

    Attributes:
        student_id: Unique identifier (e.g. ``"demo"``).
        created_at: ISO-8601 timestamp of first profile creation.
        updated_at: ISO-8601 timestamp of last write.
        attempts: ``{subtype: count}`` — total attempts per subtype.
        wrong: ``{subtype: count}`` — wrong attempts per subtype.
        error_scores: ``{error_tag: raw_score}`` — kept for backward compatibility;
            decayed scores are computed on the fly from ``error_events``.
        error_events: Ordered event log ``[{tag, ts, attempt_seq, weight, …}]``.
        recent: Last ``recent_history_limit`` attempt records.
        skill_scores: Cross-task skill rating ``{skill_id: score}``.
        decay_half_life: Half-life parameter passed through from config.
        attempt_seq: Monotonically increasing attempt counter (used for decay).
    """

    student_id: str
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    attempts: dict[str, int] = field(default_factory=dict)
    wrong: dict[str, int] = field(default_factory=dict)
    error_scores: dict[str, float] = field(default_factory=dict)
    error_events: list[dict] = field(default_factory=list)
    recent: list[dict] = field(default_factory=list)
    skill_scores: dict[str, float] = field(default_factory=dict)
    decay_half_life: float = 20.0
    attempt_seq: int = 0

    # ── Internal helpers ─────────────────────────────────────────────

    def _is_persistent(self, tag: str) -> bool:
        """Return True if *tag* appears at least ``persistent_threshold`` times
        in the last ``persistent_window`` error events."""
        cfg = get_rag_settings()
        last = self.error_events[-cfg.persistent_window :]
        cnt = sum(1 for e in last if e.get("tag") == tag)
        return cnt >= cfg.persistent_threshold

    # ── Public API ───────────────────────────────────────────────────

    def record_attempt(
        self,
        *,
        task_id: str,
        task_number: int,
        subtype: str,
        ok: bool,
        error_tags: list[str],
        error_weights: dict[str, float] | None = None,
        student_answer: str | None = None,
        correct_answer: str | None = None,
        task_text: str | None = None,
    ) -> None:
        """Record one task attempt and update all derived scores.

        Args:
            task_id: Unique task identifier.
            task_number: EGE task number (6, 10, or 12).
            subtype: Classified subtype of the task.
            ok: Whether the answer was correct.
            error_tags: Diagnosed error tags (empty if correct).
            error_weights: Optional per-tag confidence weights from the LLM.
            student_answer: Raw student answer (stored for context when incorrect).
            correct_answer: Correct answer (stored for context when incorrect).
            task_text: Task prompt text (stored for context, truncated to 500 chars).
        """
        self.attempt_seq += 1
        self.attempts[subtype] = self.attempts.get(subtype, 0) + 1
        if not ok:
            self.wrong[subtype] = self.wrong.get(subtype, 0) + 1
            tags_to_record = error_tags if error_tags else ["unknown"]
            for tag in tags_to_record:
                base = 1.0
                if error_weights and tag in error_weights:
                    base = float(error_weights[tag])
                if tag != "unknown" and self._is_persistent(tag):
                    base *= get_rag_settings().persistent_multiplier
                self.error_scores[tag] = self.error_scores.get(tag, 0.0) + base
                ev: dict = {
                    "tag": tag,
                    "ts": utc_now_iso(),
                    "attempt_seq": self.attempt_seq,
                    "weight": base,
                    "task_number": task_number,
                    "task_id": task_id,
                    "subtype": subtype,
                }
                if student_answer is not None and correct_answer is not None:
                    ev["student_answer"] = student_answer
                    ev["correct_answer"] = correct_answer
                if task_text is not None:
                    ev["task_text"] = task_text[:500]
                self.error_events.append(ev)
                if tag != "unknown":
                    from app.domain.analysis import apply_cross_task_skill_transfer

                    apply_cross_task_skill_transfer(self, tag, base)

        self.recent.append(
            {
                "task_id": task_id,
                "task_number": task_number,
                "subtype": subtype,
                "ok": ok,
                "error_tags": error_tags,
                "ts": utc_now_iso(),
            }
        )
        cfg = get_rag_settings()
        self.recent = self.recent[-cfg.recent_history_limit :]
        self.error_events = self.error_events[-cfg.error_events_limit :]
        self.updated_at = utc_now_iso()

    def error_score(self, tag: str) -> float:
        """Compute power-law decayed error score for *tag*.

        Uses KBS 2026 formula: ``sum(w * (1 + delta)^(-alpha))``.
        """
        alpha = get_rag_settings().decay_alpha
        score = 0.0
        for ev in self.error_events:
            if ev.get("tag") != tag:
                continue
            seq = int(ev.get("attempt_seq", 0))
            delta = max(0, self.attempt_seq - seq)
            w = float(ev.get("weight", 1.0))
            score += w * (1.0 + delta) ** (-alpha)
        return max(0.0, score)

    def mastery(self, subtype: str) -> float:
        """Return mastery level in [0, 1] for a subtype based on accuracy."""
        a = self.attempts.get(subtype, 0)
        if a <= 0:
            return 0.0
        w = self.wrong.get(subtype, 0)
        return max(0.0, min(1.0, 1.0 - w / a))

    # ── Serialisation ────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialise the profile to a JSON-compatible dict."""
        return {
            "student_id": self.student_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "attempts": self.attempts,
            "wrong": self.wrong,
            "error_scores": self.error_scores,
            "error_events": self.error_events,
            "recent": self.recent,
            "skill_scores": self.skill_scores,
            "decay_half_life": self.decay_half_life,
            "attempt_seq": self.attempt_seq,
        }

    @staticmethod
    def from_dict(d: dict) -> "StudentProfile":
        """Deserialise a profile from a JSON-compatible dict."""
        return StudentProfile(
            student_id=d["student_id"],
            created_at=d.get("created_at", utc_now_iso()),
            updated_at=d.get("updated_at", utc_now_iso()),
            attempts=dict(d.get("attempts", {})),
            wrong=dict(d.get("wrong", {})),
            error_scores=dict(d.get("error_scores", {})),
            error_events=list(d.get("error_events", [])),
            recent=list(d.get("recent", [])),
            skill_scores=dict(d.get("skill_scores", {})),
            decay_half_life=float(d.get("decay_half_life", 20.0)),
            attempt_seq=int(d.get("attempt_seq", 0)),
        )
