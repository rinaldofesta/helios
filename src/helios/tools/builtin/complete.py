"""CompleteObjective tool — used by the orchestrator to signal completion."""

from __future__ import annotations

from typing import Any

from helios.providers.base import ToolDefinition


class CompleteObjectiveTool:
    """Tool for the orchestrator to signal that the objective is fully achieved."""

    def __init__(self) -> None:
        self._definition = ToolDefinition(
            name="complete_objective",
            description=(
                "Signal that the objective has been fully achieved. "
                "Call this when all sub-tasks are done and the objective is complete."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of what was accomplished.",
                    },
                },
                "required": ["summary"],
            },
        )
        self.completed = False
        self.summary: str = ""

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments: dict[str, Any]) -> str:
        self.completed = True
        self.summary = arguments["summary"]
        return f"Objective complete: {self.summary}"
