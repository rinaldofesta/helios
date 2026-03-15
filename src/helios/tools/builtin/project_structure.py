"""DefineProjectStructure tool — used by the refiner."""

from __future__ import annotations

from typing import Any

from helios.providers.base import ToolDefinition


class DefineProjectStructureTool:
    """Tool for the refiner to define the project folder structure."""

    def __init__(self) -> None:
        self._definition = ToolDefinition(
            name="define_project_structure",
            description=(
                "Define the project folder structure as a JSON object. "
                "Keys are folder/file names, nested dicts are subfolders, null values are files."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "Name of the project (max 20 characters).",
                    },
                    "structure": {
                        "type": "object",
                        "description": (
                            "Folder structure as JSON. Keys are names, "
                            "nested objects are subfolders, null values are files."
                        ),
                    },
                },
                "required": ["project_name", "structure"],
            },
        )
        self.project_name: str | None = None
        self.structure: dict[str, Any] | None = None

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments: dict[str, Any]) -> str:
        self.project_name = arguments["project_name"]
        self.structure = arguments["structure"]
        return f"Project structure defined for '{self.project_name}'"
