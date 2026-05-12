"""Unit tests for the direct-Kafka eval-worker (LLM and platform mocked)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.workers import eval_worker
from app.infrastructure.http.platform_client import (
    PermanentDeliveryError,
    TransientDeliveryError,
)


def _valid_event() -> dict:
    return {
        "event_id": str(uuid4()),
        "event_type": "submission.created",
        "schema_version": "1.0",
        "occurred_at": "2026-05-07T22:00:00Z",
        "submission": {
            "submission_id": str(uuid4()),
            "user_id": "user-1",
            "task_id": "T10-1",
            "task_number": 10,
            "task_text": "Найдите...",
            "is_correct": False,
            "student_answer": "5",
            "correct_answer": "8",
        },
    }


# -- _parse_event ------------------------------------------------------


def test_parse_event_returns_submission_dict():
    sub = eval_worker._parse_event(json.dumps(_valid_event()).encode("utf-8"))
    assert sub is not None
    assert sub["task_number"] == 10


def test_parse_event_rejects_bad_json():
    assert eval_worker._parse_event(b"not-json") is None


def test_parse_event_rejects_wrong_event_type():
    ev = _valid_event()
    ev["event_type"] = "submission.deleted"
    assert eval_worker._parse_event(json.dumps(ev).encode()) is None


def test_parse_event_rejects_missing_submission():
    ev = _valid_event()
    del ev["submission"]
    assert eval_worker._parse_event(json.dumps(ev).encode()) is None


def test_parse_event_rejects_missing_required_fields():
    ev = _valid_event()
    del ev["submission"]["task_text"]
    assert eval_worker._parse_event(json.dumps(ev).encode()) is None


def test_parse_event_rejects_unsupported_task_number():
    ev = _valid_event()
    ev["submission"]["task_number"] = 99
    assert eval_worker._parse_event(json.dumps(ev).encode()) is None


def test_parse_event_accepts_each_supported_task_number():
    for n in (6, 10, 12, 13, 16):
        ev = _valid_event()
        ev["submission"]["task_number"] = n
        assert eval_worker._parse_event(json.dumps(ev).encode()) is not None, n


# -- _diagnose ---------------------------------------------------------


def test_diagnose_returns_empty_when_no_hypotheses(monkeypatch):
    monkeypatch.setattr(eval_worker, "diagnose_with_pipeline", lambda **kw: [])
    tags, rationale = eval_worker._diagnose(_valid_event()["submission"], llm=object())
    assert tags == []
    assert rationale is None


def test_diagnose_extracts_tags_and_joins_rationale(monkeypatch):
    hyps = [
        MagicMock(tag="tag_a", reason="reason a"),
        MagicMock(tag="tag_b", reason="reason b"),
        MagicMock(tag="tag_c", reason=""),  # empty reason filtered out
    ]
    monkeypatch.setattr(eval_worker, "diagnose_with_pipeline", lambda **kw: hyps)
    tags, rationale = eval_worker._diagnose(_valid_event()["submission"], llm=object())
    assert tags == ["tag_a", "tag_b", "tag_c"]
    assert rationale == "reason a - reason b"


def test_diagnose_truncates_long_rationale(monkeypatch):
    big = MagicMock(tag="t", reason="x" * 2000)
    monkeypatch.setattr(eval_worker, "diagnose_with_pipeline", lambda **kw: [big])
    _, rationale = eval_worker._diagnose(_valid_event()["submission"], llm=object())
    assert len(rationale) == 1000


# -- _process ----------------------------------------------------------


def _mock_platform() -> MagicMock:
    p = MagicMock()
    p.post_diagnostic_tags = AsyncMock()
    return p


@pytest.mark.asyncio
async def test_process_skips_correct_submissions(monkeypatch):
    sub = _valid_event()["submission"]
    sub["is_correct"] = True

    diagnosed = MagicMock()
    monkeypatch.setattr(eval_worker, "diagnose_with_pipeline", diagnosed)
    platform = _mock_platform()

    assert await eval_worker._process(sub, platform=platform, llm=None, model_name="m") is False
    diagnosed.assert_not_called()
    platform.post_diagnostic_tags.assert_not_called()


@pytest.mark.asyncio
async def test_process_calls_diagnose_then_posts(monkeypatch):
    sub = _valid_event()["submission"]
    h = MagicMock(tag="lost_parentheses_or_sign", reason="r")
    monkeypatch.setattr(eval_worker, "diagnose_with_pipeline", lambda **kw: [h])
    platform = _mock_platform()

    assert await eval_worker._process(sub, platform=platform, llm=object(), model_name="qwen") is False
    platform.post_diagnostic_tags.assert_awaited_once()
    payload = platform.post_diagnostic_tags.await_args.kwargs["payload"]
    assert payload["tags"] == ["lost_parentheses_or_sign"]
    assert payload["model"] == "qwen"
    assert payload["submission_id"] == sub["submission_id"]


@pytest.mark.asyncio
async def test_process_swallows_llm_exception(monkeypatch):
    sub = _valid_event()["submission"]

    def boom(**kw):
        raise RuntimeError("LLM boom")

    monkeypatch.setattr(eval_worker, "diagnose_with_pipeline", boom)
    platform = _mock_platform()

    # Same input would fail again -- no replay.
    assert await eval_worker._process(sub, platform=platform, llm=object(), model_name="m") is False
    platform.post_diagnostic_tags.assert_not_called()


@pytest.mark.asyncio
async def test_process_no_replay_on_permanent_callback_failure(monkeypatch):
    sub = _valid_event()["submission"]
    monkeypatch.setattr(
        eval_worker, "diagnose_with_pipeline",
        lambda **kw: [MagicMock(tag="t", reason="r")],
    )
    platform = _mock_platform()
    platform.post_diagnostic_tags.side_effect = PermanentDeliveryError(400, "bad")

    assert await eval_worker._process(sub, platform=platform, llm=object(), model_name="m") is False
    platform.post_diagnostic_tags.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_signals_replay_on_transient_failure(monkeypatch):
    sub = _valid_event()["submission"]
    monkeypatch.setattr(
        eval_worker, "diagnose_with_pipeline",
        lambda **kw: [MagicMock(tag="t", reason="r")],
    )
    platform = _mock_platform()
    platform.post_diagnostic_tags.side_effect = TransientDeliveryError("retry me")

    # True -> run_worker skips commit so Kafka redelivers the batch.
    assert await eval_worker._process(sub, platform=platform, llm=object(), model_name="m") is True
    platform.post_diagnostic_tags.assert_awaited_once()
