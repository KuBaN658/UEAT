"""
LangGraph orchestration for parallel conspect section generation.

The compiled graph ``_GRAPH``: START → parallel ``Send`` into ``section_agent``
→ ``build_conspect`` → END. Tool-assisted ReAct runs inside ``section_agent``
when the LLM client implements ``chat_with_tools``.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from contextlib import nullcontext
from functools import wraps
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.infrastructure.conspect.queries import (
    get_algorithm_prompt,
    get_checklist_prompt,
    get_conspect_system_prompt,
    get_example_prompt,
    get_find_error_prompt,
    get_typical_errors_prompt,
    get_what_to_remember_prompt,
)
from app.infrastructure.conspect.tools import ConspectToolContext, ConspectToolRegistry
from app.infrastructure.llm.clients import (
    ChatClient,
    ToolCall,
    build_chat_client_for_conspect,
    serialize_tool_calls,
)
from app.infrastructure.retrieval.engine import sanitize_llm_output

log = logging.getLogger(__name__)

_MAX_SECTION_TOOL_ROUNDS = 4

_TRACE_FLAGS = ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2")
_TRACE_BOOL_TRUE = {"1", "true", "yes", "on"}
_TRACE_STATE_KEYS = (
    "student_id",
    "verify_enabled",
    "math_verify_enabled",
    "structured_output_enabled",
)

try:
    from langsmith.run_helpers import trace as _langsmith_trace
except Exception:
    _langsmith_trace = None


def _merge_section_drafts(a: dict[str, str], b: dict[str, str]) -> dict[str, str]:
    """Reducer that merges section draft dicts from parallel Send invocations."""
    return {**a, **b}


def _last_value(a: Any, b: Any) -> Any:
    """Reducer that keeps the latest write.

    Applied to keys shared between ConspectGraphState and SectionAgentState so
    that parallel Send invocations (all writing the same value) don't trigger
    LangGraph's INVALID_CONCURRENT_GRAPH_UPDATE error.
    """
    return b


class SectionAgentState(TypedDict, total=False):
    """State for the per-section ReAct subgraph.

    Populated via ``Send`` from ``_fan_out_sections``; the ``draft_sections``
    key is written back into the parent ``ConspectGraphState`` via its reducer.
    """

    # Inputs (set by Send)
    section_key: str
    user_prompt: str
    system_prompt: str
    llm: ChatClient
    tool_context: ConspectToolContext | None

    # Internal ReAct loop state
    messages: list[dict[str, Any]]
    round_idx: int
    pending_tool_calls: list[Any]  # list[ToolCall]
    result: str

    # Output – merged into parent ConspectGraphState via _merge_section_drafts
    draft_sections: dict[str, str]


class ConspectGraphState(TypedDict, total=False):
    """Shared mutable state passed between LangGraph nodes.

    Each node reads a subset of keys and returns incremental updates that
    LangGraph merges into the accumulated state.
    """

    # Immutable request / runtime inputs — Annotated with _last_value so that
    # 6 parallel Send invocations writing back the same value don't conflict.
    student_id: str
    context: dict[str, str | int]
    system_prompt: Annotated[str, _last_value]
    llm: Annotated[ChatClient, _last_value]
    tool_context: Annotated[ConspectToolContext | None, _last_value]

    # Accumulated section drafts; reducer merges outputs of parallel Send calls
    draft_sections: Annotated[dict[str, str], _merge_section_drafts]

    # Passed through initial state / tracing metadata (unused by compile path)
    verify_enabled: bool
    math_verify_enabled: bool
    structured_output_enabled: bool

    rag_answer: str
    retrieved_titles: list[str]
    frontier_ids: list[str]


TOOLS = (
    "get_student_profile_summary",
    "retrieve_learning_atoms",
    "get_frontier_topics",
    "get_recent_wrong_examples",
    "get_misconception_hints",
    "get_similar_solved_tasks",
    "get_recent_attempts",
)


# ── Tracing helpers ──────────────────────────────────────────────────


def _langsmith_tracing_enabled() -> bool:
    return any(os.getenv(flag, "").strip().lower() in _TRACE_BOOL_TRUE for flag in _TRACE_FLAGS)


def _truncate_trace_text(value: str, limit: int = 220) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def _trace_payload_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 4:
        return _truncate_trace_text(str(value), limit=500)
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return _truncate_trace_text(value.replace("\n", "\\n"), limit=500)
    if isinstance(value, dict):
        items = list(value.items())
        payload = {
            str(key): _trace_payload_value(item, depth=depth + 1) for key, item in items[:12]
        }
        if len(items) > 12:
            payload["..."] = f"{len(items) - 12} more keys"
        return payload
    if isinstance(value, list | tuple):
        items_out = [_trace_payload_value(item, depth=depth + 1) for item in value[:8]]
        if len(value) > 8:
            items_out.append(f"... {len(value) - 8} more items")
        return items_out
    return _truncate_trace_text(str(value), limit=500)


def _state_trace_payload(state: dict[str, Any] | None) -> dict[str, Any]:
    if not state:
        return {}
    payload: dict[str, Any] = {}
    for key in _TRACE_STATE_KEYS:
        if key in state:
            payload[key] = state[key]
    for key in ("rag_answer",):
        value = state.get(key)
        if isinstance(value, str) and value:
            payload[f"{key}_len"] = len(value)
            payload[f"{key}_preview"] = _truncate_trace_text(value.replace("\n", "\\n"))
    return payload


def _trace_context(
    name: str,
    *,
    run_type: str = "chain",
    inputs: dict[str, Any],
    tags: list[str],
    metadata: dict[str, Any] | None = None,
):
    if _langsmith_trace is None or not _langsmith_tracing_enabled():
        return nullcontext(None)
    merged_metadata = {"component": "conspect_generation_graph", **(metadata or {})}
    return _langsmith_trace(
        name=name,
        run_type=run_type,
        inputs=inputs,
        tags=tags,
        metadata=merged_metadata,
    )


def _trace_node(
    node_name: str,
    node_fn: Callable[[ConspectGraphState], ConspectGraphState],
) -> Callable[[ConspectGraphState], ConspectGraphState]:
    """Wrap a node function with LangSmith trace spans."""

    @wraps(node_fn)
    def _wrapped(state: ConspectGraphState) -> ConspectGraphState:
        with _trace_context(
            f"conspect_graph.node.{node_name}",
            inputs={"state": _state_trace_payload(state)},
            tags=["conspect", "langgraph-node", node_name],
            metadata={"node": node_name},
        ) as run:
            updates = node_fn(state)
            if run is not None:
                run.end(outputs={"updates": _state_trace_payload(updates)})
            return updates

    return _wrapped


def _call_tool_with_trace(
    registry: ConspectToolRegistry,
    *,
    section_key: str,
    round_idx: int,
    tool_call: ToolCall,
) -> dict[str, Any]:
    """Execute a conspect tool call as a visible LangSmith tool run."""
    with _trace_context(
        f"conspect_tool.{tool_call.name}",
        run_type="tool",
        inputs={"arguments": _trace_payload_value(tool_call.arguments)},
        tags=["conspect", "conspect-tool", section_key, tool_call.name],
        metadata={
            "section": section_key,
            "round": round_idx,
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
        },
    ) as run:
        result = registry.call(tool_call.name, tool_call.arguments)
        if run is not None:
            run.end(outputs={"result": _trace_payload_value(result)})
        return result


# ── LLM helpers ──────────────────────────────────────────────────────


_SECTION_HEADERS: dict[str, str] = {
    "what_to_remember": "## Что важно запомнить",
    "typical_errors": "## Типичные ошибки",
    "algorithm": "## Алгоритм решения",
    "example": "## Разбор примера",
    "find_error": "## Найди ошибку",
    "checklist": "## Чеклист перед ответом",
}

_TOOL_USE_INSTRUCTION = (
    "\n\nНе придумывай и не угадывай профиль ученика, ошибки, примеры и "
    "справочный материал. Выбери и вызови подходящие доступные инструменты. "
    "Не вызывай один и тот же инструмент с одинаковыми параметрами дважды. "
    "В финальном тексте не упоминай инструменты, внутренние идентификаторы, "
    "JSON или технические источники."
)

_FINALIZE_INSTRUCTION = (
    "Сформируй финальный текст раздела на основе исходного задания и уже "
    "полученных результатов. Верни только содержимое раздела без заголовка."
)


# ── Section-agent subgraph nodes ─────────────────────────────────────


def _node_section_call_model(state: SectionAgentState) -> SectionAgentState:
    """LLM inference step of the section ReAct loop.

    On the first invocation, builds the initial messages list.  On the final
    forced round (round_idx >= MAX), appends a plain-language finalization
    prompt and calls without tools so the model must produce a text answer.
    Falls back to a plain ``llm.chat`` call when tools are unavailable.
    """
    llm = state["llm"]
    system_prompt = state["system_prompt"]
    user_prompt = state["user_prompt"]
    section_key = state.get("section_key", "")
    tool_context = state.get("tool_context")
    round_idx = state.get("round_idx", 0)
    messages: list[dict[str, Any]] = list(state.get("messages") or [])

    has_tools = bool(tool_context) and hasattr(llm, "chat_with_tools")

    if not messages:
        user_content = user_prompt + (_TOOL_USE_INSTRUCTION if has_tools else "")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    if not has_tools:
        resp = llm.chat(system=system_prompt, user=user_prompt, temperature=0.25, max_tokens=8192)
        return {
            "messages": messages,
            "round_idx": round_idx + 1,
            "pending_tool_calls": [],
            "result": sanitize_llm_output(resp.text),
        }

    registry = ConspectToolRegistry(tool_context)
    tools = registry.schemas_for(list(TOOLS))
    if not tools:
        resp = llm.chat(system=system_prompt, user=user_prompt, temperature=0.25, max_tokens=8192)
        return {
            "messages": messages,
            "round_idx": round_idx + 1,
            "pending_tool_calls": [],
            "result": sanitize_llm_output(resp.text),
        }

    # On the final forced round: inject a finalization prompt, pass no tools
    if round_idx >= _MAX_SECTION_TOOL_ROUNDS:
        messages = messages + [{"role": "user", "content": _FINALIZE_INSTRUCTION}]
        tools_to_pass: list[Any] = []
    else:
        tools_to_pass = tools

    try:
        chat_with_tools = llm.chat_with_tools  # type: ignore[attr-defined]
        response = chat_with_tools(
            messages=messages,
            tools=tools_to_pass,
            temperature=0.25,
            max_tokens=8192,
        )
        pending = list(response.tool_calls or [])
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": response.text or "",
        }
        if pending:
            assistant_msg["tool_calls"] = serialize_tool_calls(pending)
        return {
            "messages": messages + [assistant_msg],
            "round_idx": round_idx + 1,
            "pending_tool_calls": pending,
            "result": sanitize_llm_output(response.text) if not pending else "",
        }
    except Exception as exc:
        log.warning(
            "Tool-assisted generation failed for section %s: %s; falling back to plain chat",
            section_key,
            exc,
            exc_info=True,
        )
        resp = llm.chat(system=system_prompt, user=user_prompt, temperature=0.25, max_tokens=8192)
        return {
            "messages": messages,
            "round_idx": round_idx + 1,
            "pending_tool_calls": [],
            "result": sanitize_llm_output(resp.text),
        }


def _node_section_call_tools(state: SectionAgentState) -> SectionAgentState:
    """Execute all pending tool calls and append results to the message history."""
    tool_context = state.get("tool_context")
    section_key = state.get("section_key", "")
    round_idx = state.get("round_idx", 1)
    messages: list[dict[str, Any]] = list(state.get("messages") or [])
    pending = state.get("pending_tool_calls") or []

    if not tool_context or not pending:
        return {"pending_tool_calls": []}

    registry = ConspectToolRegistry(tool_context)
    tool_messages: list[dict[str, Any]] = []
    for tool_call in pending:
        tool_result = _call_tool_with_trace(
            registry,
            section_key=section_key,
            round_idx=round_idx - 1,
            tool_call=tool_call,
        )
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.name,
                "content": json.dumps(tool_result, ensure_ascii=False),
            }
        )

    return {
        "messages": messages + tool_messages,
        "pending_tool_calls": [],
    }


def _node_section_finalize(state: SectionAgentState) -> SectionAgentState:
    """Wrap the generated text in a section header and write to draft_sections."""
    result = state.get("result") or ""
    if not result:
        for msg in reversed(state.get("messages") or []):
            if msg.get("role") == "assistant" and msg.get("content"):
                result = sanitize_llm_output(msg["content"])
                break
    section_key = state.get("section_key", "unknown")
    header = _SECTION_HEADERS.get(section_key, f"## {section_key}")
    return {
        "result": result,
        "draft_sections": {section_key: f"{header}\n\n{result}"},
    }


def _route_section_after_call_model(state: SectionAgentState) -> str:
    """Route to tool execution while rounds remain; otherwise finalize."""
    pending = state.get("pending_tool_calls") or []
    round_idx = state.get("round_idx", 0)
    if pending and round_idx <= _MAX_SECTION_TOOL_ROUNDS:
        return "call_tools"
    return "finalize_section"


def _tool_driven_conspect_context() -> dict[str, str | int]:
    """Return prompt scaffolding for tool-driven generation without eager retrieval."""
    return {
        "task_scope": "заданиям №6, №10 и №12",
        "detail": "выбери по профилю ученика через инструменты; если профиль недоступен, используй средний уровень подробности",
        "errors_block": "Не подставлен заранее. При необходимости получи профиль, недавние ошибки или подсказки по заблуждениям через доступные инструменты.",
        "frontier_block": "Не подставлен заранее. При необходимости получи ближайшие темы через доступные инструменты.",
        "diversity_block": "Подбери пример по данным профиля и не повторяй однотипные формулировки.",
        "context_section": "Справочный контекст не собран заранее. Если он нужен для точности раздела, вызови инструмент поиска учебных атомов с собственным запросом.",
        "n_err_items": 3,
        "top_err": "ключевая ошибка ученика из профиля",
    }


# ── Fan-out and section-agent subgraph ───────────────────────────────


def _fan_out_sections(state: ConspectGraphState) -> list[Send]:
    """Build one Send per section so all six run in parallel inside the subgraph."""
    ctx = state["context"]
    system_prompt = state["system_prompt"]
    llm = state["llm"]
    tool_context = state.get("tool_context")

    sections: list[tuple[str, str]] = [
        (
            "what_to_remember",
            get_what_to_remember_prompt().format(
                detail=ctx["detail"],
                errors_block=ctx["errors_block"],
                frontier_block=ctx["frontier_block"],
                context_section=ctx["context_section"],
                n_err_items=ctx["n_err_items"],
                top_err=ctx["top_err"],
            ),
        ),
        (
            "typical_errors",
            get_typical_errors_prompt().format(
                detail=ctx["detail"],
                errors_block=ctx["errors_block"],
                frontier_block=ctx["frontier_block"],
                context_section=ctx["context_section"],
                n_err_items=ctx["n_err_items"],
                top_err=ctx["top_err"],
            ),
        ),
        (
            "algorithm",
            get_algorithm_prompt().format(
                task_scope=ctx["task_scope"],
                detail=ctx["detail"],
                errors_block=ctx["errors_block"],
                frontier_block=ctx["frontier_block"],
                context_section=ctx["context_section"],
                top_err=ctx["top_err"],
            ),
        ),
        (
            "example",
            get_example_prompt().format(
                task_scope=ctx["task_scope"],
                detail=ctx["detail"],
                errors_block=ctx["errors_block"],
                frontier_block=ctx["frontier_block"],
                diversity_block=ctx["diversity_block"],
                context_section=ctx["context_section"],
                top_err=ctx["top_err"],
            ),
        ),
        (
            "find_error",
            get_find_error_prompt().format(
                task_scope=ctx["task_scope"],
                detail=ctx["detail"],
                errors_block=ctx["errors_block"],
                frontier_block=ctx["frontier_block"],
                context_section=ctx["context_section"],
                top_err=ctx["top_err"],
            ),
        ),
        (
            "checklist",
            get_checklist_prompt().format(
                task_scope=ctx["task_scope"],
                detail=ctx["detail"],
                errors_block=ctx["errors_block"],
                frontier_block=ctx["frontier_block"],
                context_section=ctx["context_section"],
                top_err=ctx["top_err"],
            ),
        ),
    ]

    return [
        Send(
            "section_agent",
            {
                "section_key": key,
                "user_prompt": prompt,
                "system_prompt": system_prompt,
                "llm": llm,
                "tool_context": tool_context,
                "messages": [],
                "round_idx": 0,
                "pending_tool_calls": [],
                "result": "",
                "draft_sections": {},
            },
        )
        for key, prompt in sections
    ]


def _node_build_conspect(state: ConspectGraphState) -> ConspectGraphState:
    """Concatenate section drafts (from parallel Send calls) into the final conspect."""
    sections = state.get("draft_sections") or {}
    ordered = [
        sections.get("what_to_remember", ""),
        sections.get("typical_errors", ""),
        sections.get("algorithm", ""),
        sections.get("example", ""),
        sections.get("find_error", ""),
        sections.get("checklist", ""),
    ]
    return {"rag_answer": "\n\n".join(s for s in ordered if s)}


# ── Graph construction ───────────────────────────────────────────────


def _build_section_agent_graph():
    """Compile the per-section ReAct subgraph (call_model ⇄ call_tools loop)."""
    g = StateGraph(SectionAgentState)
    g.add_node("call_model", _node_section_call_model)
    g.add_node("call_tools", _node_section_call_tools)
    g.add_node("finalize_section", _node_section_finalize)

    g.add_edge(START, "call_model")
    g.add_conditional_edges(
        "call_model",
        _route_section_after_call_model,
        path_map=["call_tools", "finalize_section"],
    )
    g.add_edge("call_tools", "call_model")
    g.add_edge("finalize_section", END)

    return g.compile(name="section_agent")


_SECTION_AGENT = _build_section_agent_graph()


def _generate_section_with_tools(
    state: dict[str, Any],
    *,
    section_key: str,
    user_prompt: str,
) -> str:
    """Run the compiled ``section_agent`` subgraph once (tests / ad-hoc use)."""
    initial: SectionAgentState = {
        "section_key": section_key,
        "user_prompt": user_prompt,
        "system_prompt": state["system_prompt"],
        "llm": state["llm"],
        "tool_context": state.get("tool_context"),
        "messages": [],
        "round_idx": 0,
        "pending_tool_calls": [],
        "result": "",
        "draft_sections": {},
    }
    final = _SECTION_AGENT.invoke(initial)
    return str(final.get("result") or "")


def _build_graph():
    """Construct and compile the conspect generation state machine.

    START → fan-out (Send × 6) → section_agent subgraph (call_model ⇄ call_tools)
          → build_conspect → END
    """
    graph = StateGraph(ConspectGraphState)
    graph.add_node("section_agent", _SECTION_AGENT)
    graph.add_node("build_conspect", _trace_node("build_conspect", _node_build_conspect))

    # path_map tells LangGraph which nodes _fan_out_sections can route to (via Send)
    graph.add_conditional_edges(START, _fan_out_sections, path_map=["section_agent"])
    graph.add_edge("section_agent", "build_conspect")
    graph.add_edge("build_conspect", END)

    return graph.compile(name="conspect_generation_graph")


_GRAPH = _build_graph()


def run_conspect_generation_graph(
    student_id: str,
    build_prompt_bundle: Callable[[str], dict[str, Any]],
    *,
    verify_enabled: bool,
    math_verify_enabled: bool,
    structured_output_enabled: bool,
    llm: ChatClient | None = None,
    tool_context: ConspectToolContext | None = None,
) -> tuple[str, list[str], list[str]]:
    """Execute the compiled graph and return the final conspect with metadata.

    Args:
        student_id: Student identifier.
        build_prompt_bundle: Callable that builds the RAG prompt context dict.
        verify_enabled,
        math_verify_enabled,
        structured_output_enabled: Forwarded into initial graph state and run
            metadata for observability only (no verify/repair nodes on ``_GRAPH``).
        llm: Optional pre-built LLM client; built from settings if ``None``.
        tool_context: Optional tool context for per-section tool calls.

    Returns:
        ``(rag_answer, retrieved_titles, frontier_ids)`` tuple.
    """
    llm_client = llm or build_chat_client_for_conspect()
    tools_available = tool_context is not None and hasattr(llm_client, "chat_with_tools")
    if tools_available:
        context = _tool_driven_conspect_context()
    else:
        try:
            context = build_prompt_bundle(student_id)
        except TypeError:
            context = build_prompt_bundle()  # type: ignore[call-arg]
    initial_state: ConspectGraphState = {
        "student_id": student_id,
        "system_prompt": get_conspect_system_prompt(),
        "context": context,
        "draft_sections": {},
        "llm": llm_client,
        "tool_context": tool_context,
        "verify_enabled": verify_enabled,
        "math_verify_enabled": math_verify_enabled,
        "structured_output_enabled": structured_output_enabled,
    }
    graph_config: dict[str, Any] = {
        "run_name": "conspect_generation_graph",
        "tags": ["conspect", "langgraph", "conspect-generation"],
        "metadata": {
            "student_id": student_id,
            "verify_enabled": verify_enabled,
            "math_verify_enabled": math_verify_enabled,
            "structured_output_enabled": structured_output_enabled,
        },
    }
    final_state = _GRAPH.invoke(initial_state, config=graph_config)
    rag_answer = str(final_state.get("rag_answer") or "")
    titles = list(final_state.get("retrieved_titles") or [])
    frontier_ids = list(final_state.get("frontier_ids") or [])
    return rag_answer, titles, frontier_ids
