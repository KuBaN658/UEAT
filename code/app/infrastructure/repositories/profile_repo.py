"""
File-based student profile repository.

Profiles are stored as JSON files under a configurable root directory.
Writes are atomic: data is written to a temp file then ``os.replace``-d
into place so a crash during save never corrupts an existing profile.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from app.domain.profile import StudentProfile

log = logging.getLogger(__name__)


class ProfileStore:
    """Persists ``StudentProfile`` objects as JSON files.

    Args:
        root_dir: Directory where ``<student_id>.json`` files are stored.
                  Created automatically if it does not exist.
    """

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, student_id: str) -> Path:
        safe = "".join(ch for ch in student_id if ch.isalnum() or ch in ("-", "_"))
        return self.root_dir / f"{safe}.json"

    def load(self, student_id: str) -> StudentProfile:
        """Load a profile by *student_id*, or return a fresh profile if not found.

        Args:
            student_id: Unique student identifier.

        Returns:
            Loaded or newly-created ``StudentProfile``.
        """
        p = self._path(student_id)
        if not p.exists():
            return StudentProfile(student_id=student_id)
        data = json.loads(p.read_text(encoding="utf-8"))
        return StudentProfile.from_dict(data)

    def save(self, profile: StudentProfile) -> None:
        """Atomically persist *profile* to disk.

        Args:
            profile: Profile to write.

        Raises:
            OSError: If the write or rename fails.
        """
        p = self._path(profile.student_id)
        data = json.dumps(profile.to_dict(), ensure_ascii=False, indent=2)
        fd, tmp = tempfile.mkstemp(suffix=".json", dir=self.root_dir, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, p)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
