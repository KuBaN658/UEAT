"""In-memory store of recent submissions, used by the monitoring UI."""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone

_STORE_MAX = 500


def _parse_iso(v: str | None) -> datetime | None:
    if not v:
        return None
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        return None


def _pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((len(xs) - 1) * p))))
    return xs[k]


class SubmissionStore:
    """Recent submissions for the monitoring UI. Single-process, bounded."""

    def __init__(self) -> None:
        self._data: OrderedDict[str, dict] = OrderedDict()

    def add(self, sub: dict) -> None:
        sid = str(sub.get("submission_id") or "")
        if not sid:
            return
        self._data[sid] = {
            "submission_id": sid,
            "user_id": sub.get("user_id", ""),
            "task_number": sub.get("task_number"),
            "status": "pending",
            "payload": sub,
            "tags": [],
            "last_error": None,
            "created_at": sub.get("submitted_at") or datetime.now(timezone.utc).isoformat(),
        }
        while len(self._data) > _STORE_MAX:
            self._data.popitem(last=False)

    def patch_tags(self, submission_id: str, tags: list[str]) -> None:
        row = self._data.get(submission_id)
        if row is not None:
            row["status"] = "done"
            row["tags"] = tags
            row["diagnosed_at"] = datetime.now(timezone.utc).isoformat()

    def all(self) -> list[dict]:
        return list(reversed(self._data.values()))

    def stats(self) -> dict:
        rows = list(self._data.values())
        counts = {"pending": 0, "done": 0, "failed": 0}
        latencies_ms: list[float] = []
        recent_done = 0
        oldest_pending_s: float | None = None
        now = datetime.now(timezone.utc)

        for r in rows:
            status = r["status"]
            if status in counts:
                counts[status] += 1

            if status == "done":
                created = _parse_iso(r.get("created_at"))
                diagnosed = _parse_iso(r.get("diagnosed_at"))
                if created and diagnosed:
                    latencies_ms.append((diagnosed - created).total_seconds() * 1000)
                    if (now - diagnosed).total_seconds() <= 60:
                        recent_done += 1
            elif status == "pending":
                created = _parse_iso(r.get("created_at"))
                if created:
                    age = (now - created).total_seconds()
                    if oldest_pending_s is None or age > oldest_pending_s:
                        oldest_pending_s = age

        return {
            "counts": {**counts, "total": len(rows)},
            "latency_ms": {
                "avg": sum(latencies_ms) / len(latencies_ms) if latencies_ms else None,
                "p50": _pct(latencies_ms, 0.50),
                "p95": _pct(latencies_ms, 0.95),
                "samples": len(latencies_ms),
            },
            "throughput_per_min": recent_done,
            "oldest_pending_s": oldest_pending_s,
        }
