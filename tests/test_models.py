"""Tests for core data models."""

from datetime import datetime

from helios.core.models import (
    CompletionResponse,
    CostRecord,
    Message,
    Role,
    Session,
    SessionStatus,
    Skill,
    SubTask,
    SubTaskStatus,
    TaskExchange,
    TokenUsage,
    ToolCall,
    ToolResult,
)


class TestToolCall:
    def test_create_with_defaults(self):
        tc = ToolCall(name="create_subtask", arguments={"description": "test"})
        assert tc.name == "create_subtask"
        assert tc.arguments == {"description": "test"}
        assert len(tc.id) == 12

    def test_create_with_explicit_id(self):
        tc = ToolCall(id="custom_id", name="complete_objective", arguments={})
        assert tc.id == "custom_id"


class TestToolResult:
    def test_success_result(self):
        tr = ToolResult(tool_call_id="tc_001", content="Done")
        assert tr.is_error is False

    def test_error_result(self):
        tr = ToolResult(tool_call_id="tc_001", content="Failed", is_error=True)
        assert tr.is_error is True


class TestMessage:
    def test_user_message(self):
        msg = Message(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.tool_calls == []
        assert msg.tool_results == []

    def test_assistant_message_with_tool_calls(self):
        tc = ToolCall(name="web_search", arguments={"query": "python async"})
        msg = Message(role=Role.ASSISTANT, tool_calls=[tc])
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "web_search"

    def test_tool_result_message(self):
        tr = ToolResult(tool_call_id="tc_001", content="search results")
        msg = Message(role=Role.TOOL_RESULT, tool_results=[tr])
        assert len(msg.tool_results) == 1


class TestTokenUsage:
    def test_defaults(self):
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.model == ""

    def test_with_values(self):
        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            model="claude-haiku-4-5-20251001",
            provider="anthropic",
        )
        assert usage.input_tokens == 1000
        assert usage.provider == "anthropic"


class TestCostRecord:
    def test_cost_calculation(self, sample_usage):
        cost = CostRecord(
            usage=sample_usage,
            input_cost=0.0005,
            output_cost=0.001,
            total_cost=0.0015,
        )
        assert cost.total_cost == 0.0015
        assert cost.usage.model == "llama3:instruct"


class TestCompletionResponse:
    def test_text_only_response(self):
        resp = CompletionResponse(content="Hello world", stop_reason="end_turn")
        assert resp.content == "Hello world"
        assert resp.tool_calls == []

    def test_tool_use_response(self):
        tc = ToolCall(name="create_subtask", arguments={"description": "step 1"})
        resp = CompletionResponse(
            content="",
            tool_calls=[tc],
            stop_reason="tool_use",
        )
        assert len(resp.tool_calls) == 1
        assert resp.stop_reason == "tool_use"


class TestSubTask:
    def test_defaults(self):
        st = SubTask(description="Write unit tests")
        assert st.status == SubTaskStatus.PENDING
        assert st.priority == 3
        assert st.result is None
        assert len(st.id) == 8

    def test_with_search_query(self):
        st = SubTask(
            description="Research async patterns",
            search_query="python asyncio best practices",
        )
        assert st.search_query is not None


class TestTaskExchange:
    def test_create(self):
        subtask = SubTask(description="Build API endpoints")
        exchange = TaskExchange(subtask=subtask, prompt="Create REST endpoints")
        assert exchange.result == ""
        assert exchange.cost is None


class TestSession:
    def test_defaults(self):
        session = Session(objective="Build a Flask TODO API")
        assert session.status == SessionStatus.PENDING
        assert session.exchanges == []
        assert session.total_cost == 0.0
        assert session.file_content is None
        assert len(session.id) == 32
        assert isinstance(session.created_at, datetime)

    def test_with_exchanges(self):
        subtask = SubTask(description="Setup project")
        exchange = TaskExchange(
            subtask=subtask,
            prompt="Initialize Flask project",
            result="Flask project initialized",
        )
        session = Session(
            objective="Build API",
            exchanges=[exchange],
            total_cost=0.05,
        )
        assert len(session.exchanges) == 1
        assert session.total_cost == 0.05

    def test_serialization_roundtrip(self):
        session = Session(objective="Test roundtrip")
        data = session.model_dump()
        restored = Session.model_validate(data)
        assert restored.objective == session.objective
        assert restored.id == session.id


class TestSkill:
    def test_defaults(self):
        skill = Skill(name="Flask API setup")
        assert skill.usage_count == 0
        assert skill.procedure_steps == []
        assert skill.trigger_keywords == []

    def test_with_data(self):
        skill = Skill(
            name="REST API Design",
            description="How to design RESTful APIs with Flask",
            trigger_keywords=["flask", "api", "rest", "endpoint"],
            procedure_steps=[
                "Create Flask app",
                "Define routes",
                "Add error handling",
            ],
        )
        assert len(skill.trigger_keywords) == 4
        assert len(skill.procedure_steps) == 3
