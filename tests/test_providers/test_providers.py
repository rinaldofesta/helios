"""Tests for providers and shared utilities."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from helios.core.models import Message, Role, ToolCall, ToolResult
from helios.providers.base import ToolDefinition
from helios.providers.openai_compatible import (
    OpenAICompatibleProvider,
    parse_openai_response,
    to_openai_messages,
)


# --------------------------------------------------------------------------- #
#  ToolDefinition conversion tests
# --------------------------------------------------------------------------- #


class TestToolDefinition:
    def test_to_openai_format(self):
        td = ToolDefinition(
            name="web_search",
            description="Search the web",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        )
        result = td.to_openai()
        assert result["type"] == "function"
        assert result["function"]["name"] == "web_search"
        assert result["function"]["parameters"]["type"] == "object"

    def test_to_openai_with_required(self):
        td = ToolDefinition(
            name="create_subtask",
            description="Create a new sub-task",
            input_schema={
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                },
                "required": ["description"],
            },
        )
        result = td.to_openai()
        assert result["function"]["name"] == "create_subtask"
        assert result["function"]["parameters"]["required"] == ["description"]


# --------------------------------------------------------------------------- #
#  Message conversion tests (OpenAI format — used by Ollama, Groq, etc.)
# --------------------------------------------------------------------------- #


class TestMessageConversion:
    def test_user_message(self):
        msgs = [Message(role=Role.USER, content="Hello")]
        result = to_openai_messages(msgs)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_system_message(self):
        result = to_openai_messages([], system="You are helpful")
        assert result == [{"role": "system", "content": "You are helpful"}]

    def test_assistant_message(self):
        msgs = [Message(role=Role.ASSISTANT, content="Hi there")]
        result = to_openai_messages(msgs)
        assert result == [{"role": "assistant", "content": "Hi there"}]

    def test_system_in_messages(self):
        msgs = [
            Message(role=Role.SYSTEM, content="You are helpful"),
            Message(role=Role.USER, content="Hello"),
        ]
        result = to_openai_messages(msgs)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    def test_tool_call_message(self):
        tc = ToolCall(id="tc_001", name="create_subtask", arguments={"description": "test"})
        msgs = [Message(role=Role.ASSISTANT, content="Let me create a task", tool_calls=[tc])]
        result = to_openai_messages(msgs)
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Let me create a task"
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["function"]["name"] == "create_subtask"

    def test_tool_result_message(self):
        tr = ToolResult(tool_call_id="tc_001", content="Task created")
        msgs = [Message(role=Role.TOOL_RESULT, tool_results=[tr])]
        result = to_openai_messages(msgs)
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "tc_001"
        assert result[0]["content"] == "Task created"


# --------------------------------------------------------------------------- #
#  Response parsing tests
# --------------------------------------------------------------------------- #


class TestResponseParsing:
    def test_text_response(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="Hello world", tool_calls=None),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        )
        result = parse_openai_response(response, "test-model")
        assert result.content == "Hello world"
        assert result.tool_calls == []
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50
        assert result.usage.provider == "openai_compatible"
        assert result.stop_reason == "stop"

    def test_tool_use_response(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content="Creating a subtask",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_001",
                            function=SimpleNamespace(
                                name="create_subtask",
                                arguments='{"description": "Write unit tests"}',
                            ),
                        ),
                    ],
                ),
                finish_reason="tool_calls",
            )],
            usage=SimpleNamespace(prompt_tokens=200, completion_tokens=100),
        )
        result = parse_openai_response(response, "test-model")
        assert result.content == "Creating a subtask"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "create_subtask"
        assert result.tool_calls[0].id == "call_001"
        assert result.tool_calls[0].arguments == {"description": "Write unit tests"}

    def test_multiple_tool_calls(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_001",
                            function=SimpleNamespace(
                                name="create_code_file",
                                arguments='{"filename": "main.py", "content": "print(1)"}',
                            ),
                        ),
                        SimpleNamespace(
                            id="call_002",
                            function=SimpleNamespace(
                                name="create_code_file",
                                arguments='{"filename": "test.py", "content": "assert True"}',
                            ),
                        ),
                    ],
                ),
                finish_reason="tool_calls",
            )],
            usage=SimpleNamespace(prompt_tokens=300, completion_tokens=150),
        )
        result = parse_openai_response(response, "test-model")
        assert len(result.tool_calls) == 2

    def test_no_usage(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="hi", tool_calls=None),
                finish_reason="stop",
            )],
            usage=None,
        )
        result = parse_openai_response(response, "test-model")
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0


# --------------------------------------------------------------------------- #
#  Provider integration tests (mocked)
# --------------------------------------------------------------------------- #


class TestOpenAICompatibleProvider:
    @pytest.mark.asyncio
    async def test_complete_text_response(self):
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="The answer is 42", tool_calls=None),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=50, completion_tokens=10),
        )

        provider = OpenAICompatibleProvider(model="test-model")
        provider._client = AsyncMock()
        provider._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await provider.complete([Message(role=Role.USER, content="What is 6*7?")])
        assert result.content == "The answer is 42"
        assert result.usage.model == "test-model"

    @pytest.mark.asyncio
    async def test_complete_with_tools(self):
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_abc",
                            function=SimpleNamespace(
                                name="create_subtask",
                                arguments='{"description": "Step 1"}',
                            ),
                        ),
                    ],
                ),
                finish_reason="tool_calls",
            )],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        )

        provider = OpenAICompatibleProvider(model="test-model")
        provider._client = AsyncMock()
        provider._client.chat.completions.create = AsyncMock(return_value=mock_response)

        tools = [
            ToolDefinition(
                name="create_subtask",
                description="Create a subtask",
                input_schema={"type": "object", "properties": {"description": {"type": "string"}}},
            )
        ]

        result = await provider.complete(
            [Message(role=Role.USER, content="Break this down")],
            tools=tools,
        )
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "create_subtask"

    def test_supports_tools(self):
        provider = OpenAICompatibleProvider()
        assert provider.supports_tools() is True

    def test_model_name(self):
        provider = OpenAICompatibleProvider(model="llama3")
        assert provider.model_name == "llama3"
