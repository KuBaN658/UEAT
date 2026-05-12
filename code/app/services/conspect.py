"""
Conspect service: build personalized prompt bundle and run the full LLM pipeline.

Merges ``rag/conspect_build.py`` (retrieval / prompt assembly) and
``rag/conspect_generation.py`` (LLM generation + verify passes).
"""

from __future__ import annotations

import os

from app.domain.analysis import (
    get_recent_wrong_attempts,
    next_frontier_atoms,
    top_errors,
)
from app.domain.atoms import ATOM_BY_ID
from app.infrastructure.conspect.graph import run_conspect_generation_graph
from app.infrastructure.conspect.queries import (
    cold_start_retrieval_queries,
    get_conspect_json_output_instruction,
    micromodule_queries,
    subtype_seen_queries,
)
from app.infrastructure.conspect.tools import ConspectToolContext
from app.infrastructure.llm.clients import (
    build_chat_client_for_conspect,
    get_llm_backend,
)
from app.infrastructure.repositories.profile_repo import ProfileStore
from app.infrastructure.retrieval.engine import (
    RagEngine,
    _extract_concrete_mistakes,
    _human_error,
)

# ── Feature flags ────────────────────────────────────────────────────


def conspect_verify_enabled() -> bool:
    """Return True if the second LLM verify pass is enabled.

    Off by default for local Ollama (small models).
    Override with ``CONSPECT_VERIFY=true/false`` env var.
    """
    raw = os.getenv("CONSPECT_VERIFY", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return get_llm_backend() != "ollama"


def conspect_math_verify_enabled() -> bool:
    """Return True if the focused arithmetic verification pass is enabled.

    Falls back to ``conspect_verify_enabled()`` when the env var is not set.
    Override with ``CONSPECT_MATH_VERIFY=true/false`` env var.
    """
    raw = os.getenv("CONSPECT_MATH_VERIFY", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return conspect_verify_enabled()


def conspect_structured_output_enabled() -> bool:
    """Return True if JSON per-section output is enabled.

    Requires the YAML prompt file to supply a JSON output instruction.
    Override with ``CONSPECT_STRUCTURED_OUTPUT=false`` to disable.
    """
    raw = os.getenv("CONSPECT_STRUCTURED_OUTPUT", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    return bool(get_conspect_json_output_instruction())


# ── Prompt assembly ──────────────────────────────────────────────────


def build_conspect_prompt_bundle(
    profile_store: ProfileStore,
    rag: RagEngine,
    student_id: str = "demo",
) -> dict[str, str | int]:
    """Assemble a personalized conspect context dict without calling the LLM.

    Performs hybrid RAG retrieval, selects frontier atoms and concrete
    mistake snippets, then delegates to the engine's prompt builder.

    Args:
        profile_store: Profile store for loading student history.
        rag: RAG engine for retrieval and prompt construction.
        student_id: Student identifier (default ``"demo"``).

    Returns:
        Context dict with keys ``task_scope``, ``detail``, ``errors_block``,
        ``frontier_block``, ``diversity_block``, ``context_section``,
        ``n_err_items``, ``top_err``.
    """
    profile = profile_store.load(student_id)
    recent_wrong = get_recent_wrong_attempts(profile, task_number=None, n=5)

    if recent_wrong:
        queries: list[str] = []
        for m in recent_wrong:
            tt = (m.get("task_text") or "").replace("\n", " ")[:100]
            sa = m.get("student_answer", "?")
            ca = m.get("correct_answer", "?")
            tn = m.get("task_number", "")
            queries.append(f"ЕГЭ задание {tn}: {tt} ученик ответил {sa}, правильно {ca}")
        queries.extend(micromodule_queries(profile, 6, n=3))
        queries.extend(micromodule_queries(profile, 10, n=3))
        queries.extend(micromodule_queries(profile, 12, n=3))
        queries.extend(subtype_seen_queries(profile)[:2])
    else:
        relevant_error_tags = top_errors(profile, 8, subtype=None)
        human_errors = [_human_error(t) for t in relevant_error_tags[:5]]
        if human_errors:
            queries = [f"ЕГЭ задания 6, 10 и 12: {h}" for h in human_errors]
            queries.extend(micromodule_queries(profile, 6, n=5))
            queries.extend(micromodule_queries(profile, 10, n=5))
            queries.extend(micromodule_queries(profile, 12, n=5))
            queries.extend(cold_start_retrieval_queries(profile, student_id)[:4])
        else:
            queries = cold_start_retrieval_queries(profile, student_id)
            queries.extend(subtype_seen_queries(profile))

    queries.append("ЕГЭ профильная математика задания 6, 10 и 12 обзор")

    seen: set[str] = set()
    deduped: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            deduped.append(q)
    queries = deduped

    retrieved = rag.retrieve(
        query=queries,
        task_number=None,
        subtype=None,
        profile=profile,
        k=12,
    )
    frontier_ids = next_frontier_atoms(profile, task_number=None, n=6)
    frontier_titles = [ATOM_BY_ID[aid].title for aid in frontier_ids if aid in ATOM_BY_ID]
    concrete_mistakes = _extract_concrete_mistakes(profile, task_number=None)
    return rag.build_personalized_conspect_prompt(
        task_number=None,
        subtype="all_subtypes",
        profile=profile,
        retrieved=retrieved,
        frontier_atoms=frontier_titles,
        concrete_mistakes=concrete_mistakes,
        recent_wrong=recent_wrong,
    )


# ── Generation ───────────────────────────────────────────────────────


def generate_conspect_rag_answer(
    student_id: str,
    profile_store: ProfileStore,
    rag: RagEngine,
) -> tuple[str, list[str], list[str]]:
    """Run the full conspect generation pipeline and return the result.

    Pipeline: personalized retrieval → LangGraph (generate → verify → repair
    → math verify).

    Args:
        student_id: Student identifier.
        profile_store: Profile store injected from app state.
        rag: RAG engine injected from app state.

    Returns:
        A 3-tuple ``(rag_answer, retrieved_titles, frontier_atom_ids)``.

    Raises:
        ValueError: If the LLM client is unavailable (no API key).
        Exception: Propagates any LLM or graph execution error.
    """

    def _build_prompt(sid: str) -> dict[str, str | int]:
        return build_conspect_prompt_bundle(profile_store, rag, sid)

    tool_context = ConspectToolContext(
        student_id=student_id,
        profile_store=profile_store,
        rag=rag,
    )
    llm = build_chat_client_for_conspect()
    return run_conspect_generation_graph(
        student_id,
        _build_prompt,
        verify_enabled=conspect_verify_enabled(),
        math_verify_enabled=conspect_math_verify_enabled(),
        structured_output_enabled=conspect_structured_output_enabled(),
        llm=llm,
        tool_context=tool_context,
    )
