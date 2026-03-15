"""Groq provider implementation — fast inference for open-weight models."""

from __future__ import annotations

import json
import os
from typing import Any

from groq import AsyncGroq

from helios.core.models import CompletionResponse, Message, TokenUsage, ToolCall
from helios.providers.base import ToolDefinition
from helios.providers.openai_compatible import to_openai_messages


class GroqProvider:
    """Provider for Groq's fast inference API (runs open-weight models)."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._client = AsyncGroq(
            api_key=api_key or os.environ.get("GROQ_API_KEY"),
        )

    @property
    def model_name(self) -> str:
        return self._model

    def supports_tools(self) -> bool:
        return True

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> CompletionResponse:
        api_messages = to_openai_messages(messages, system)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": api_messages,
        }
        if tools:
            kwargs["tools"] = [t.to_openai() for t in tools]

        response = await self._client.chat.completions.create(**kwargs)
        return _parse_groq_response(response, self._model)


def _parse_groq_response(response: Any, model: str) -> CompletionResponse:
    """Parse a Groq API response."""
    choice = response.choices[0]
    message = choice.message

    content = message.content or ""
    tool_calls: list[ToolCall] = []

    if message.tool_calls:
        for tc in message.tool_calls:
            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                arguments = {"raw": tc.function.arguments}
            tool_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments=arguments)
            )

    usage = TokenUsage(
        input_tokens=response.usage.prompt_tokens if response.usage else 0,
        output_tokens=response.usage.completion_tokens if response.usage else 0,
        model=model,
        provider="groq",
    )

    return CompletionResponse(
        content=content,
        tool_calls=tool_calls,
        usage=usage,
        stop_reason=choice.finish_reason,
    )
