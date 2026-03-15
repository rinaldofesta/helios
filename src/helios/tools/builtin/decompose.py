"""CreateSubtask tool — used by the orchestrator to create sub-tasks."""

from __future__ import annotations

from typing import Any

from helios.core.models import SubTask
from helios.providers.base import ToolDefinition


class CreateSubtaskTool:
    """Tool for the orchestrator to create a new sub-task."""

    def __init__(self) -> None:
        self._definition = ToolDefinition(
            name="create_subtask",
            description=(
                "Create a new sub-task for the sub-agent to execute. "
                "Break complex objectives into focused, actionable tasks."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Detailed description of the sub-task to execute.",
                    },
                    "priority": {
                        "type": "integer",
                        "description": "Priority from 1 (highest) to 5 (lowest).",
                        "minimum": 1,
                        "maximum": 5,
                        "default": 3,
                    },
                    "search_query": {
                        "type": "string",
                        "description": "Optional web search query to gather information for this task.",
                    },
                },
                "required": ["description"],
            },
        )
        self.last_subtask: SubTask | None = None

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments: dict[str, Any]) -> str:
        subtask = SubTask(
            description=arguments["description"],
            priority=arguments.get("priority", 3),
            search_query=arguments.get("search_query"),
        )
        self.last_subtask = subtask
        return f"Sub-task created: {subtask.description}"
