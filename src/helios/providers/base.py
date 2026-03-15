"""Provider protocol and shared types."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from helios.core.models import CompletionResponse, Message


# --------------------------------------------------------------------------- #
#  Tool definition (canonical format)
# --------------------------------------------------------------------------- #


class ToolDefinition:
    """A tool that can be sent to a provider."""

    __slots__ = ("name", "description", "input_schema")

    def __init__(self, name: str, description: str, input_schema: dict) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema

    def to_openai(self) -> dict:
        """Convert to OpenAI-compatible tool format.

        Used by Ollama, Groq, OpenAI-compatible servers, and any provider
        that follows the OpenAI function-calling convention.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


# --------------------------------------------------------------------------- #
#  Provider protocol
# --------------------------------------------------------------------------- #


@runtime_checkable
class Provider(Protocol):
    """Protocol that all LLM providers must satisfy."""

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> CompletionResponse:
        """Send messages to the model and get a completion."""
        ...

    def supports_tools(self) -> bool:
        """Whether this provider supports native tool calling."""
        ...

    @property
    def model_name(self) -> str:
        """The model identifier."""
        ...
