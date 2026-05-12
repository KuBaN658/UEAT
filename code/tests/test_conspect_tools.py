from types import SimpleNamespace

import app.infrastructure.conspect.graph as conspect_generation_graph
from app.domain.atom import Atom
from app.domain.profile import StudentProfile
from app.infrastructure.conspect.graph import (
    _generate_section_with_tools,
    run_conspect_generation_graph,
)
from app.infrastructure.conspect.tools import (
    ConspectToolContext,
    ConspectToolRegistry,
    _normalize_tool_arguments,
)
from app.infrastructure.llm.clients import LlmResponse, ToolCall


class FakeProfileStore:
    def __init__(self, profile: StudentProfile) -> None:
        self.profile = profile

    def load(self, student_id: str) -> StudentProfile:
        assert student_id == self.profile.student_id
        return self.profile


class FakeRag:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def retrieve(self, **_kwargs):
        self.calls.append(_kwargs)
        atom = Atom(
            id="t06_mm_test",
            title="Показательные уравнения",
            text="Приведи обе части к одному основанию и сравни показатели.",
            task_number=6,
            subtypes=("exponential_eq",),
        )
        return [SimpleNamespace(atom=atom, score=0.42)]


def _make_test_profile() -> StudentProfile:
    profile = StudentProfile(student_id="u1")
    profile.attempts = {"exponential_eq": 3}
    profile.wrong = {"exponential_eq": 1}
    profile.recent = [
        {
            "task_id": "task-1",
            "task_number": 6,
            "subtype": "exponential_eq",
            "ok": False,
            "error_tags": ["wrong_base_reduction"],
            "problem_katex": "Найдите корень уравнения $3^x=9$.",
            "solution": "Ответ: 2.",
        }
    ]
    profile.error_events = [
        {
            "tag": "wrong_base_reduction",
            "task_number": 6,
            "task_id": "task-1",
            "subtype": "exponential_eq",
            "attempt_seq": 1,
            "weight": 1.0,
        }
    ]
    profile.attempt_seq = 1
    return profile


def _context(rag: FakeRag | None = None) -> ConspectToolContext:
    return ConspectToolContext(
        student_id="u1",
        profile_store=FakeProfileStore(_make_test_profile()),  # type: ignore[arg-type]
        rag=rag or FakeRag(),  # type: ignore[arg-type]
    )


def test_registry_returns_profile_summary_without_internal_ids():
    registry = ConspectToolRegistry(_context())

    result = registry.call("get_student_profile_summary", {})

    assert result["student_id"] == "u1"
    assert result["attempts_total"] == 3
    assert "t06_mm" not in str(result)


def test_registry_retrieves_learning_atoms_without_atom_ids():
    registry = ConspectToolRegistry(_context())

    result = registry.call(
        "retrieve_learning_atoms",
        {"query": "показательное уравнение", "task_number": 6, "k": 1},
    )

    assert result["items"][0]["title"] == "Показательные уравнения"
    assert "t06_mm_test" not in str(result)


def test_registry_passes_subtype_to_learning_atom_retrieval():
    rag = FakeRag()
    registry = ConspectToolRegistry(_context(rag))

    registry.call(
        "retrieve_learning_atoms",
        {
            "query": "показательное уравнение",
            "task_number": 6,
            "subtype": "exponential_eq",
        },
    )

    assert rag.calls[0]["subtype"] == "exponential_eq"


def test_tool_arguments_normalize_string_task_number_for_mcp_validation():
    result = _normalize_tool_arguments(
        "retrieve_learning_atoms",
        {"query": "логарифмы", "task_number": "12", "k": "2"},
    )

    assert result["task_number"] == 12
    assert result["k"] == 2


def test_tool_arguments_normalize_spaced_task_number_key_for_mcp_validation():
    result = _normalize_tool_arguments(
        "get_frontier_topics",
        {"task number": "10", "limit": "2"},
    )

    assert result["task_number"] == 10
    assert "task number" not in result
    assert result["limit"] == 2


def test_tool_arguments_parse_json_string_error_tags_for_mcp_validation():
    result = _normalize_tool_arguments(
        "get_misconception_hints",
        {"error_tags": '["wrong_base_reduction", ""]', "limit": "3"},
    )

    assert result["error_tags"] == ["wrong_base_reduction"]
    assert result["limit"] == 3


def test_registry_returns_recent_attempts():
    registry = ConspectToolRegistry(_context())

    result = registry.call(
        "get_recent_attempts",
        {"task_number": 6, "only_wrong": True, "limit": 1},
    )

    assert result["items"][0]["ok"] is False
    assert "Найдите корень" in result["items"][0]["problem"]


class ToolCallingFakeLlm:
    def __init__(self) -> None:
        self.messages_seen: list[list[dict]] = []

    def chat(
        self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 8192
    ) -> LlmResponse:
        return LlmResponse(text="plain fallback")

    def chat_with_tools(self, messages, tools, **_kwargs) -> LlmResponse:
        self.messages_seen.append(list(messages))
        if tools and not any(msg.get("role") == "tool" for msg in messages):
            return LlmResponse(
                text="",
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="get_student_profile_summary",
                        arguments={},
                    ),
                ),
            )
        return LlmResponse(text="Итоговый раздел")


def test_section_generation_executes_tool_call_loop():
    llm = ToolCallingFakeLlm()
    state = {
        "llm": llm,
        "system_prompt": "system",
        "tool_context": _context(),
    }

    result = _generate_section_with_tools(
        state,  # type: ignore[arg-type]
        section_key="what_to_remember",
        user_prompt="Сделай раздел.",
    )

    assert result == "Итоговый раздел"
    assert any(msg.get("role") == "tool" for msg in llm.messages_seen[-1])


class FakeTraceRun:
    def __init__(self, kwargs: dict) -> None:
        self.kwargs = kwargs
        self.outputs: dict | None = None

    def end(self, *, outputs: dict | None = None) -> None:
        self.outputs = outputs


class FakeTraceContext:
    def __init__(self, runs: list[FakeTraceRun], kwargs: dict) -> None:
        self.run = FakeTraceRun(kwargs)
        runs.append(self.run)

    def __enter__(self) -> FakeTraceRun:
        return self.run

    def __exit__(self, *_args) -> bool:
        return False


def test_section_generation_traces_tool_calls(monkeypatch):
    runs: list[FakeTraceRun] = []

    def fake_trace(**kwargs):
        return FakeTraceContext(runs, kwargs)

    monkeypatch.setattr(conspect_generation_graph, "_langsmith_trace", fake_trace)  # type: ignore[attr-defined]
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    result = _generate_section_with_tools(
        {
            "llm": ToolCallingFakeLlm(),
            "system_prompt": "system",
            "tool_context": _context(),
        },  # type: ignore[arg-type]
        section_key="what_to_remember",
        user_prompt="Сделай раздел.",
    )

    tool_runs = [run for run in runs if run.kwargs["run_type"] == "tool"]
    assert result == "Итоговый раздел"
    assert len(tool_runs) == 1
    assert tool_runs[0].kwargs["name"] == "conspect_tool.get_student_profile_summary"
    assert tool_runs[0].kwargs["metadata"]["tool_call_id"] == "call-1"
    assert tool_runs[0].outputs is not None
    assert tool_runs[0].outputs["result"]["student_id"] == "u1"


class PlainFakeLlm:
    def chat(
        self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 8192
    ) -> LlmResponse:
        return LlmResponse(text="Обычный раздел")


def test_section_generation_falls_back_without_tool_context():
    result = _generate_section_with_tools(
        {
            "llm": PlainFakeLlm(),
            "system_prompt": "system",
            "tool_context": None,
        },  # type: ignore[arg-type]
        section_key="what_to_remember",
        user_prompt="Сделай раздел.",
    )

    assert result == "Обычный раздел"


class ToolAwareNoopLlm:
    def chat(
        self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 8192
    ) -> LlmResponse:
        return LlmResponse(text="Не должен использоваться")

    def chat_with_tools(self, messages, tools, **_kwargs) -> LlmResponse:
        assert tools
        assert "выбери и вызови" in messages[1]["content"].lower()
        return LlmResponse(text="Содержимое раздела")


def test_generation_skips_prompt_bundle_when_tools_available():
    def build_prompt_bundle(_student_id: str):
        raise AssertionError("prompt bundle should not be built for tool-driven nodes")

    answer, titles, frontier_ids = run_conspect_generation_graph(
        "u1",
        build_prompt_bundle,
        verify_enabled=False,
        math_verify_enabled=False,
        structured_output_enabled=False,
        llm=ToolAwareNoopLlm(),  # type: ignore[arg-type]
        tool_context=_context(),
    )

    assert "## Что важно запомнить" in answer
    assert "Содержимое раздела" in answer
    assert titles == []
    assert frontier_ids == []


def test_generation_uses_prompt_bundle_without_tools():
    calls: list[str] = []

    def build_prompt_bundle(student_id: str):
        calls.append(student_id)
        return {
            "task_scope": "заданиям №6, №10 и №12",
            "detail": "средний",
            "errors_block": "ошибки",
            "frontier_block": "",
            "diversity_block": "",
            "context_section": "контекст",
            "n_err_items": 3,
            "top_err": "ошибка",
        }

    answer, _titles, _frontier_ids = run_conspect_generation_graph(
        "u1",
        build_prompt_bundle,
        verify_enabled=False,
        math_verify_enabled=False,
        structured_output_enabled=False,
        llm=PlainFakeLlm(),  # type: ignore[arg-type]
        tool_context=None,
    )

    assert calls == ["u1"]
    assert "Обычный раздел" in answer
