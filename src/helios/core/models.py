"""Core data models for Helios."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
#  Enums
# --------------------------------------------------------------------------- #


class Role(str, Enum):
    """Message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"


class SessionStatus(str, Enum):
    """Session lifecycle status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class SubTaskStatus(str, Enum):
    """Sub-task lifecycle status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# --------------------------------------------------------------------------- #
#  Message & Tool primitives
# --------------------------------------------------------------------------- #


class ToolCall(BaseModel):
    """A tool invocation requested by the model."""

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result of executing a tool."""

    tool_call_id: str
    content: str
    is_error: bool = False


class Message(BaseModel):
    """A single message in a conversation."""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
#  Usage & Cost
# --------------------------------------------------------------------------- #


class TokenUsage(BaseModel):
    """Token usage for a single provider call."""

    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    provider: str = ""


class CostRecord(BaseModel):
    """Cost breakdown for a single provider call."""

    usage: TokenUsage
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0


# --------------------------------------------------------------------------- #
#  Provider response
# --------------------------------------------------------------------------- #


class CompletionResponse(BaseModel):
    """Normalized response from any provider."""

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    stop_reason: str | None = None


# --------------------------------------------------------------------------- #
#  Sub-task & Exchange
# --------------------------------------------------------------------------- #


class SubTask(BaseModel):
    """A sub-task created by the orchestrator."""

    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    description: str
    priority: int = 3
    depends_on: list[str] = Field(default_factory=list)
    search_query: str | None = None
    status: SubTaskStatus = SubTaskStatus.PENDING
    result: str | None = None


class TaskExchange(BaseModel):
    """Record of a sub-task execution cycle."""

    subtask: SubTask
    prompt: str
    result: str = ""
    cost: CostRecord | None = None


# --------------------------------------------------------------------------- #
#  Session
# --------------------------------------------------------------------------- #


class Session(BaseModel):
    """A complete Helios session."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    objective: str
    status: SessionStatus = SessionStatus.PENDING
    exchanges: list[TaskExchange] = Field(default_factory=list)
    total_cost: float = 0.0
    file_content: str | None = None
    project_name: str | None = None
    project_structure: dict[str, Any] | None = None
    code_files: list[dict[str, str]] = Field(default_factory=list)
    synthesis: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


# --------------------------------------------------------------------------- #
#  Skills
# --------------------------------------------------------------------------- #


class Skill(BaseModel):
    """A learned skill extracted from a completed session."""

    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    name: str
    description: str = ""
    trigger_keywords: list[str] = Field(default_factory=list)
    procedure_steps: list[str] = Field(default_factory=list)
    usage_count: int = 0
    success_count: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
