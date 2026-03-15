"""WebSearch tool — Tavily integration."""

from __future__ import annotations

from typing import Any

from helios.providers.base import ToolDefinition


class WebSearchTool:
    """Tool for searching the web via Tavily."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._definition = ToolDefinition(
            name="web_search",
            description="Search the web for information relevant to the current task.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                },
                "required": ["query"],
            },
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments: dict[str, Any]) -> str:
        query = arguments["query"]
        if not self._api_key:
            return f"Web search unavailable (no API key). Query was: {query}"

        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=self._api_key)
            result = client.qna_search(query=query)
            return str(result)
        except Exception as e:
            return f"Search failed: {e}"
