"""
Unit tests for the in-memory ``SubmissionStore`` used by the admin monitor.

Covers add/patch_tags/all/stats and the LRU eviction behaviour.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from admin.store import SubmissionStore


def _sub(
    sid: str,
    *,
    user: str = "demo",
    task: int = 10,
    answer: str = "5",
    submitted_at: str | None = None,
) -> dict:
    return {
        "submission_id": sid,
        "user_id": user,
        "task_number": task,
        "task_text": "...",
        "is_correct": False,
        "student_answer": answer,
        "correct_answer": "8",
        "submitted_at": submitted_at or datetime.now(timezone.utc).isoformat(),
    }


# -- add / patch_tags / all --------------------------------------------


def test_add_creates_pending_row():
    store = SubmissionStore()
    store.add(_sub("a"))
    rows = store.all()
    assert len(rows) == 1
    r = rows[0]
    assert r["submission_id"] == "a"
    assert r["status"] == "pending"
    assert r["tags"] == []


def test_add_without_submission_id_is_noop():
    store = SubmissionStore()
    store.add({"user_id": "demo"})  # no submission_id
    assert store.all() == []


def test_all_returns_newest_first():
    store = SubmissionStore()
    store.add(_sub("first"))
    store.add(_sub("second"))
    store.add(_sub("third"))
    sids = [r["submission_id"] for r in store.all()]
    assert sids == ["third", "second", "first"]


def test_patch_tags_marks_done_and_records_timestamp():
    store = SubmissionStore()
    store.add(_sub("a"))
    store.patch_tags("a", ["units_mismatch"])
    r = store.all()[0]
    assert r["status"] == "done"
    assert r["tags"] == ["units_mismatch"]
    assert r["diagnosed_at"] is not None


def test_patch_tags_unknown_id_is_silent():
    store = SubmissionStore()
    store.patch_tags("missing", ["x"])  # must not raise
    assert store.all() == []


# -- eviction ----------------------------------------------------------


def test_store_evicts_oldest_beyond_limit(monkeypatch):
    """When over _STORE_MAX entries, oldest are dropped (LRU-like)."""
    monkeypatch.setattr("admin.store._STORE_MAX", 3)
    store = SubmissionStore()
    for sid in ("a", "b", "c", "d"):
        store.add(_sub(sid))
    sids = [r["submission_id"] for r in store.all()]
    assert sids == ["d", "c", "b"]  # 'a' evicted


# -- stats ------------------------------------------------------------


def test_stats_counts_by_status():
    store = SubmissionStore()
    store.add(_sub("a"))
    store.add(_sub("b"))
    store.add(_sub("c"))
    store.patch_tags("a", ["t1"])
    store.patch_tags("b", ["t2"])

    s = store.stats()
    assert s["counts"]["pending"] == 1
    assert s["counts"]["done"] == 2
    assert s["counts"]["failed"] == 0
    assert s["counts"]["total"] == 3


def test_stats_latency_uses_created_to_diagnosed_delta():
    store = SubmissionStore()
    started = datetime.now(timezone.utc) - timedelta(seconds=10)
    store.add(_sub("a", submitted_at=started.isoformat()))
    store.patch_tags("a", ["t"])

    s = store.stats()
    assert s["latency_ms"]["samples"] == 1
    # Roughly 10s = 10000ms -- give generous tolerance for clock variance.
    assert 9000 <= s["latency_ms"]["avg"] <= 12000


def test_stats_latency_null_when_no_done_rows():
    store = SubmissionStore()
    store.add(_sub("a"))
    s = store.stats()
    assert s["latency_ms"]["avg"] is None
    assert s["latency_ms"]["samples"] == 0


def test_stats_oldest_pending_age_reflects_oldest_only():
    store = SubmissionStore()
    older = datetime.now(timezone.utc) - timedelta(seconds=120)
    newer = datetime.now(timezone.utc) - timedelta(seconds=5)
    store.add(_sub("old", submitted_at=older.isoformat()))
    store.add(_sub("new", submitted_at=newer.isoformat()))

    s = store.stats()
    assert s["oldest_pending_s"] is not None
    assert s["oldest_pending_s"] >= 100  # closer to 120


def test_stats_throughput_counts_recent_done():
    store = SubmissionStore()
    store.add(_sub("a"))
    store.add(_sub("b"))
    store.patch_tags("a", ["t"])
    store.patch_tags("b", ["t"])

    s = store.stats()
    assert s["throughput_per_min"] == 2


def test_stats_p95_returns_largest_for_small_samples():
    """With 2 samples, P95 should pick the larger one."""
    store = SubmissionStore()
    fast = datetime.now(timezone.utc) - timedelta(seconds=2)
    slow = datetime.now(timezone.utc) - timedelta(seconds=20)
    store.add(_sub("fast", submitted_at=fast.isoformat()))
    store.add(_sub("slow", submitted_at=slow.isoformat()))
    store.patch_tags("fast", ["t"])
    store.patch_tags("slow", ["t"])

    s = store.stats()
    assert s["latency_ms"]["p95"] >= s["latency_ms"]["p50"]
