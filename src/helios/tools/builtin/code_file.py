"""CreateCodeFile tool — used by the refiner."""

from __future__ import annotations

from typing import Any

from helios.providers.base import ToolDefinition


class CreateCodeFileTool:
    """Tool for the refiner to create a code file."""

    def __init__(self) -> None:
        self._definition = ToolDefinition(
            name="create_code_file",
            description="Create a code file with the given filename and content.",
            input_schema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The filename (e.g., 'main.py', 'utils/helpers.py').",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full source code content of the file.",
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language (e.g., 'python', 'javascript').",
                    },
                },
                "required": ["filename", "content"],
            },
        )
        self.files: list[dict[str, str]] = []

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments: dict[str, Any]) -> str:
        self.files.append({
            "filename": arguments["filename"],
            "content": arguments["content"],
            "language": arguments.get("language", ""),
        })
        return f"File created: {arguments['filename']}"
