"""Tool registry for discovering and dispatching tools."""

from __future__ import annotations

from helios.providers.base import ToolDefinition
from helios.tools.base import Tool


# Which tools are available to each role
ROLE_TOOLS: dict[str, list[str]] = {
    "orchestrator": ["create_subtask", "complete_objective"],
    "sub_agent": [],  # Sub-agent uses free text, no tools
    "refiner": ["define_project_structure", "create_code_file", "output_synthesis"],
}


class ToolRegistry:
    """Registry for tool discovery and dispatch."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.definition.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_all(self) -> list[ToolDefinition]:
        return [t.definition for t in self._tools.values()]

    def get_for_role(self, role: str) -> list[ToolDefinition]:
        """Return tool definitions available for a given role."""
        allowed = ROLE_TOOLS.get(role, [])
        return [t.definition for name, t in self._tools.items() if name in allowed]
