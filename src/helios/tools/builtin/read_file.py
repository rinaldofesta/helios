"""ReadFile tool — reads local files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from helios.providers.base import ToolDefinition


class ReadFileTool:
    """Tool for reading local files."""

    def __init__(self) -> None:
        self._definition = ToolDefinition(
            name="read_file",
            description="Read the contents of a local file.",
            input_schema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to read.",
                    },
                },
                "required": ["file_path"],
            },
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments: dict[str, Any]) -> str:
        file_path = Path(arguments["file_path"])
        if not file_path.exists():
            return f"Error: File not found: {file_path}"
        if not file_path.is_file():
            return f"Error: Not a file: {file_path}"
        try:
            content = file_path.read_text(encoding="utf-8")
            if len(content) > 100_000:
                content = content[:100_000] + "\n... (truncated)"
            return content
        except Exception as e:
            return f"Error reading file: {e}"
