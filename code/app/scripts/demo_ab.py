"""
Quick A/B demo (baseline vs RAG) without running the API.

Usage (PowerShell):
  python -m app.scripts.demo_ab --subtype river --student demo

If an LLM is available (GROQ_API_KEY, OPENROUTER_API_KEY, or LLM_BACKEND=ollama), generates the RAG conspect.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from app.domain.analysis import next_frontier_atoms, top_errors
from app.domain.atoms import ATOM_BY_ID
from app.domain.profile import StudentProfile
from app.infrastructure.conspect.queries import (
    get_conspect_json_output_instruction,
    get_conspect_system_prompt,
    get_conspect_user_template,
)
from app.infrastructure.llm.clients import build_chat_client_for_conspect
from app.infrastructure.repositories.profile_repo import ProfileStore
from app.infrastructure.retrieval.engine import (
    RagEngine,
    _extract_concrete_mistakes,
    _human_error,
)
from app.services.conspect import conspect_structured_output_enabled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--subtype",
        required=True,
        help="percent|mixture|work|motion|river|avg_speed|quad|other",
    )
    ap.add_argument("--student", default="demo")
    ap.add_argument("--task", type=int, default=10, help="Task number (6, 10, or 12)")
    ap.add_argument(
        "--profiles_dir",
        default=str(os.path.join(os.path.dirname(__file__), "..", "data", "profiles")),
    )
    args = ap.parse_args()

    store = ProfileStore(root_dir=Path(args.profiles_dir).resolve())
    profile: StudentProfile = store.load(args.student)
    rag = RagEngine()

    error_tags = top_errors(profile, 6, args.subtype)
    human_errors = [_human_error(t) for t in error_tags]
    queries = [f"ЕГЭ задание {args.task}: {h}" for h in human_errors[:5]] + [
        f"ЕГЭ профильная математика задание {args.task} обзор"
    ]
    retrieved = rag.retrieve(
        query=queries,
        task_number=args.task,
        subtype=args.subtype,
        profile=profile,
        k=8,
    )
    frontier_ids = next_frontier_atoms(profile, args.task, 4)
    frontier_titles = [ATOM_BY_ID[aid].title for aid in frontier_ids if aid in ATOM_BY_ID]
    concrete_mistakes = _extract_concrete_mistakes(profile, args.task)
    bundle = rag.build_personalized_conspect_prompt(
        task_number=args.task,
        subtype=args.subtype,
        profile=profile,
        retrieved=retrieved,
        frontier_atoms=frontier_titles,
        concrete_mistakes=concrete_mistakes,
    )
    system_prompt = get_conspect_system_prompt()
    if conspect_structured_output_enabled():
        json_instr = get_conspect_json_output_instruction()
        if json_instr:
            system_prompt = f"{system_prompt}\n\n{json_instr}"
    user_prompt = get_conspect_user_template().format(**bundle)

    print("=== RETRIEVED TITLES ===")
    for r in retrieved:
        extra = ""
        print(f"- {r.atom.title}{extra}")
    print()

    try:
        client = build_chat_client_for_conspect()
    except ValueError as e:
        print(f"Skip LLM generation: {e}")
        print(
            "Prompt is ready; set GROQ_API_KEY, OPENROUTER_API_KEY, or LLM_BACKEND=ollama with Ollama running."
        )
    else:
        ans = client.chat(system=system_prompt, user=user_prompt, temperature=0.25).text
        print("=== RAG CONSPECT (LLM) ===")
        print(ans)


if __name__ == "__main__":
    main()
