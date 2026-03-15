"""OutputSynthesis tool — used by the refiner."""

from __future__ import annotations

from typing import Any

from helios.providers.base import ToolDefinition


class OutputSynthesisTool:
    """Tool for the refiner to output the final synthesized result."""

    def __init__(self) -> None:
        self._definition = ToolDefinition(
            name="output_synthesis",
            description=(
                "Output the final synthesized result. Use this to provide "
                "the refined summary of all sub-task results."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The synthesized final output in markdown format.",
                    },
                    "is_code_project": {
                        "type": "boolean",
                        "description": "Whether this output includes a code project.",
                        "default": False,
                    },
                },
                "required": ["content"],
            },
        )
        self.content: str | None = None
        self.is_code_project: bool = False

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments: dict[str, Any]) -> str:
        self.content = arguments["content"]
        self.is_code_project = arguments.get("is_code_project", False)
        return "Synthesis output recorded."
