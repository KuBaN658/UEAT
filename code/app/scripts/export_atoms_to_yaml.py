"""
Re-export atoms from YAML for validation or format updates.
Run from app: python scripts/export_atoms_to_yaml.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from app.domain.atom import Atom

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))


def atom_to_dict(a: Atom) -> dict[str, str | int | list[str]]:
    d = {
        "id": a.id,
        "title": a.title,
        "text": a.text,
        "task_number": a.task_number,
        "subtypes": list(a.subtypes),
        "error_tags": list(a.error_tags),
        "prerequisites": list(a.prerequisites),
        "shared_skills": list(a.shared_skills),
    }
    return {k: v for k, v in d.items() if v}


def main() -> None:
    from app.domain.atoms import ATOMS

    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    t6 = [a for a in ATOMS if a.task_number == 6]
    t10 = [a for a in ATOMS if a.task_number == 10]
    t12 = [a for a in ATOMS if a.task_number == 12]

    for name, atoms in [("atoms_t6", t6), ("atoms_t10", t10), ("atoms_t12", t12)]:
        out = [atom_to_dict(a) for a in atoms]
        path = data_dir / f"{name}.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(out, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"Wrote {path} ({len(out)} atoms)")


if __name__ == "__main__":
    main()
