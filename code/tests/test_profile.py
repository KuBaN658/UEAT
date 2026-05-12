"""Profile store tests."""

import json
from pathlib import Path

from app.domain.profile import StudentProfile
from app.infrastructure.repositories.profile_repo import ProfileStore


def test_profile_save_load_roundtrip(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path)
    p = StudentProfile(student_id="test_student")
    p.attempts["trig"] = 1
    store.save(p)
    p2 = store.load("test_student")
    assert p2.attempts.get("trig") == 1
    assert p2.student_id == "test_student"


def test_profile_json_is_valid(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path)
    p = StudentProfile(student_id="u1")
    store.save(p)
    data = json.loads((tmp_path / "u1.json").read_text(encoding="utf-8"))
    assert data["student_id"] == "u1"
