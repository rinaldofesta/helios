"""Shared test fixtures for Helios."""

import pytest

from helios.config.schema import HeliosSettings
from helios.core.models import (
    CostRecord,
    Session,
    TokenUsage,
    ToolCall,
    ToolResult,
)


@pytest.fixture
def sample_tool_call() -> ToolCall:
    return ToolCall(id="tc_001", name="create_subtask", arguments={"description": "Write tests"})


@pytest.fixture
def sample_tool_result() -> ToolResult:
    return ToolResult(tool_call_id="tc_001", content="Subtask created")


@pytest.fixture
def sample_usage() -> TokenUsage:
    return TokenUsage(
        input_tokens=500,
        output_tokens=200,
        model="llama3:instruct",
        provider="ollama",
    )


@pytest.fixture
def sample_cost(sample_usage: TokenUsage) -> CostRecord:
    return CostRecord(
        usage=sample_usage,
        input_cost=0.0005,
        output_cost=0.001,
        total_cost=0.0015,
    )


@pytest.fixture
def sample_session() -> Session:
    return Session(objective="Build a Flask TODO API")


@pytest.fixture
def default_settings() -> HeliosSettings:
    return HeliosSettings()
