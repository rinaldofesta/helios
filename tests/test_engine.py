"""Tests for the core orchestration engine."""

from __future__ import annotations

from typing import Any

import pytest

from helios.core.engine import HeliosEngine
from helios.core.models import (
    CompletionResponse,
    Message,
    SessionStatus,
    TokenUsage,
    ToolCall,
)
from helios.providers.base import ToolDefinition


# --------------------------------------------------------------------------- #
#  Mock provider
# --------------------------------------------------------------------------- #


class MockProvider:
    """A mock provider that returns scripted responses."""

    def __init__(self, responses: list[CompletionResponse]) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self.calls: list[dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return "mock-model"

    def supports_tools(self) -> bool:
        return True

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> CompletionResponse:
        self.calls.append({
            "messages": messages,
            "tools": tools,
            "system": system,
            "max_tokens": max_tokens,
        })
        if self._call_index < len(self._responses):
            response = self._responses[self._call_index]
            self._call_index += 1
            return response
        # Default: return empty text response
        return CompletionResponse(
            content="Done",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            stop_reason="end_turn",
        )


def _usage(inp: int = 100, out: int = 50) -> TokenUsage:
    return TokenUsage(input_tokens=inp, output_tokens=out, model="mock-model", provider="mock")


# --------------------------------------------------------------------------- #
#  Tests
# --------------------------------------------------------------------------- #


class TestHeliosEngine:
    @pytest.mark.asyncio
    async def test_simple_objective_with_text_completion(self):
        """Orchestrator returns text without tools → treated as completion."""
        orchestrator = MockProvider([
            CompletionResponse(
                content="The objective is simple, here's the answer.",
                usage=_usage(),
                stop_reason="end_turn",
            ),
        ])
        sub_agent = MockProvider([])
        refiner = MockProvider([
            CompletionResponse(
                content="Refined output.",
                usage=_usage(),
                stop_reason="end_turn",
            ),
        ])

        engine = HeliosEngine(orchestrator=orchestrator, sub_agent=sub_agent, refiner=refiner)
        session = await engine.run("What is 2+2?")

        assert session.status == SessionStatus.COMPLETED
        assert len(session.exchanges) == 0  # No subtasks created

    @pytest.mark.asyncio
    async def test_one_subtask_then_complete(self):
        """Orchestrator creates one subtask, then completes."""
        orchestrator = MockProvider([
            # Round 1: create a subtask
            CompletionResponse(
                content="",
                tool_calls=[ToolCall(
                    id="tc_1",
                    name="create_subtask",
                    arguments={"description": "Research the topic"},
                )],
                usage=_usage(),
                stop_reason="tool_use",
            ),
            # Round 2: complete
            CompletionResponse(
                content="",
                tool_calls=[ToolCall(
                    id="tc_2",
                    name="complete_objective",
                    arguments={"summary": "Research completed"},
                )],
                usage=_usage(),
                stop_reason="tool_use",
            ),
        ])
        sub_agent = MockProvider([
            CompletionResponse(
                content="Here is the research result about the topic.",
                usage=_usage(),
                stop_reason="end_turn",
            ),
        ])
        refiner = MockProvider([
            CompletionResponse(
                content="",
                tool_calls=[ToolCall(
                    id="tc_3",
                    name="output_synthesis",
                    arguments={"content": "Synthesized research", "is_code_project": False},
                )],
                usage=_usage(),
                stop_reason="tool_use",
            ),
            CompletionResponse(
                content="",
                usage=_usage(),
                stop_reason="end_turn",
            ),
        ])

        events: list[tuple[str, dict]] = []
        engine = HeliosEngine(
            orchestrator=orchestrator,
            sub_agent=sub_agent,
            refiner=refiner,
            on_event=lambda t, d: events.append((t, d)),
        )
        session = await engine.run("Research AI")

        assert session.status == SessionStatus.COMPLETED
        assert len(session.exchanges) == 1
        assert session.exchanges[0].subtask.description == "Research the topic"
        assert "research result" in session.exchanges[0].result
        assert session.synthesis == "Synthesized research"
        assert session.total_cost >= 0

        # Check events were emitted
        event_types = [e[0] for e in events]
        assert "session_started" in event_types
        assert "subtask_created" in event_types
        assert "subtask_completed" in event_types
        assert "session_completed" in event_types

    @pytest.mark.asyncio
    async def test_code_project_with_structure_and_files(self):
        """Refiner creates project structure and code files."""
        orchestrator = MockProvider([
            CompletionResponse(
                content="",
                tool_calls=[ToolCall(
                    id="tc_1",
                    name="create_subtask",
                    arguments={"description": "Create Flask app"},
                )],
                usage=_usage(),
                stop_reason="tool_use",
            ),
            CompletionResponse(
                content="",
                tool_calls=[ToolCall(
                    id="tc_2",
                    name="complete_objective",
                    arguments={"summary": "Flask app created"},
                )],
                usage=_usage(),
                stop_reason="tool_use",
            ),
        ])
        sub_agent = MockProvider([
            CompletionResponse(
                content="Flask app code here",
                usage=_usage(),
                stop_reason="end_turn",
            ),
        ])
        refiner = MockProvider([
            # Define structure
            CompletionResponse(
                content="",
                tool_calls=[ToolCall(
                    id="tc_3",
                    name="define_project_structure",
                    arguments={
                        "project_name": "flask-todo",
                        "structure": {"src": {"app.py": None}, "tests": {}},
                    },
                )],
                usage=_usage(),
                stop_reason="tool_use",
            ),
            # Create code file
            CompletionResponse(
                content="",
                tool_calls=[ToolCall(
                    id="tc_4",
                    name="create_code_file",
                    arguments={
                        "filename": "app.py",
                        "content": "from flask import Flask\napp = Flask(__name__)",
                        "language": "python",
                    },
                )],
                usage=_usage(),
                stop_reason="tool_use",
            ),
            # Synthesis
            CompletionResponse(
                content="",
                tool_calls=[ToolCall(
                    id="tc_5",
                    name="output_synthesis",
                    arguments={"content": "Flask TODO API", "is_code_project": True},
                )],
                usage=_usage(),
                stop_reason="tool_use",
            ),
            # Done
            CompletionResponse(content="", usage=_usage(), stop_reason="end_turn"),
        ])

        engine = HeliosEngine(orchestrator=orchestrator, sub_agent=sub_agent, refiner=refiner)
        session = await engine.run("Build a Flask TODO API")

        assert session.project_name == "flask-todo"
        assert session.project_structure is not None
        assert len(session.code_files) == 1
        assert session.code_files[0]["filename"] == "app.py"
        assert session.synthesis == "Flask TODO API"

    @pytest.mark.asyncio
    async def test_multiple_subtasks(self):
        """Orchestrator creates multiple subtasks before completing."""
        orchestrator = MockProvider([
            CompletionResponse(
                content="",
                tool_calls=[ToolCall(id="t1", name="create_subtask", arguments={"description": "Step 1"})],
                usage=_usage(), stop_reason="tool_use",
            ),
            CompletionResponse(
                content="",
                tool_calls=[ToolCall(id="t2", name="create_subtask", arguments={"description": "Step 2"})],
                usage=_usage(), stop_reason="tool_use",
            ),
            CompletionResponse(
                content="",
                tool_calls=[ToolCall(id="t3", name="complete_objective", arguments={"summary": "Done"})],
                usage=_usage(), stop_reason="tool_use",
            ),
        ])
        sub_agent = MockProvider([
            CompletionResponse(content="Result 1", usage=_usage(), stop_reason="end_turn"),
            CompletionResponse(content="Result 2", usage=_usage(), stop_reason="end_turn"),
        ])
        refiner = MockProvider([
            CompletionResponse(content="Final output", usage=_usage(), stop_reason="end_turn"),
        ])

        engine = HeliosEngine(orchestrator=orchestrator, sub_agent=sub_agent, refiner=refiner)
        session = await engine.run("Multi-step task")

        assert len(session.exchanges) == 2
        assert session.exchanges[0].result == "Result 1"
        assert session.exchanges[1].result == "Result 2"

    @pytest.mark.asyncio
    async def test_file_content_passed_to_first_subtask(self):
        """File content is included in the first sub-agent call only."""
        orchestrator = MockProvider([
            CompletionResponse(
                content="",
                tool_calls=[ToolCall(id="t1", name="create_subtask", arguments={"description": "Analyze file"})],
                usage=_usage(), stop_reason="tool_use",
            ),
            CompletionResponse(
                content="",
                tool_calls=[ToolCall(id="t2", name="complete_objective", arguments={"summary": "Done"})],
                usage=_usage(), stop_reason="tool_use",
            ),
        ])
        sub_agent = MockProvider([
            CompletionResponse(content="Analysis complete", usage=_usage(), stop_reason="end_turn"),
        ])
        refiner = MockProvider([
            CompletionResponse(content="Refined", usage=_usage(), stop_reason="end_turn"),
        ])

        engine = HeliosEngine(orchestrator=orchestrator, sub_agent=sub_agent, refiner=refiner)
        await engine.run("Analyze this code", file_content="def hello(): pass")

        # The sub-agent should have received file content
        sub_call = sub_agent.calls[0]
        assert "def hello(): pass" in sub_call["messages"][0].content

    @pytest.mark.asyncio
    async def test_max_subtasks_limit(self):
        """Engine stops after max_subtasks even without complete_objective."""
        from helios.config.schema import GeneralSettings, HeliosSettings

        settings = HeliosSettings(general=GeneralSettings(max_subtasks=2))

        # Orchestrator keeps creating subtasks forever
        responses = []
        for i in range(10):
            responses.append(CompletionResponse(
                content="",
                tool_calls=[ToolCall(id=f"t{i}", name="create_subtask", arguments={"description": f"Task {i}"})],
                usage=_usage(), stop_reason="tool_use",
            ))

        orchestrator = MockProvider(responses)
        sub_agent = MockProvider([
            CompletionResponse(content=f"Result {i}", usage=_usage(), stop_reason="end_turn")
            for i in range(10)
        ])
        refiner = MockProvider([
            CompletionResponse(content="Refined", usage=_usage(), stop_reason="end_turn"),
        ])

        engine = HeliosEngine(
            orchestrator=orchestrator, sub_agent=sub_agent, refiner=refiner, settings=settings
        )
        session = await engine.run("Infinite loop test")

        assert len(session.exchanges) == 2  # Capped at max_subtasks
        assert session.status == SessionStatus.COMPLETED
