"""Tool protocol and base types."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from helios.providers.base import ToolDefinition


@runtime_checkable
class Tool(Protocol):
    """Protocol that all Helios tools must satisfy."""

    @property
    def definition(self) -> ToolDefinition:
        """Return the tool's schema definition."""
        ...

    async def execute(self, arguments: dict[str, Any]) -> str:
        """Execute the tool with the given arguments and return a text result."""
        ...
