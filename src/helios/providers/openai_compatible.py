"""OpenAI-compatible provider for local servers (LM Studio, vLLM, LocalAI, etc.)."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import AsyncOpenAI

from helios.core.models import CompletionResponse, Message, Role, TokenUsage, ToolCall
from helios.providers.base import ToolDefinition


class OpenAICompatibleProvider:
    """Provider for any OpenAI-compatible API server.

    Works with LM Studio, vLLM, LocalAI, text-generation-webui, and
    any server that exposes an OpenAI-compatible ``/v1/chat/completions``
    endpoint.
    """

    def __init__(
        self,
        model: str = "default",
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "not-needed",
    ) -> None:
        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
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
        return parse_openai_response(response, self._model, provider="openai_compatible")


# --------------------------------------------------------------------------- #
#  Shared OpenAI-format helpers (used by Groq as well)
# --------------------------------------------------------------------------- #


def to_openai_messages(messages: list[Message], system: str | None = None) -> list[dict[str, Any]]:
    """Convert Helios Messages to OpenAI-compatible API format."""
    api_messages: list[dict[str, Any]] = []

    if system:
        api_messages.append({"role": "system", "content": system})

    for msg in messages:
        if msg.role == Role.SYSTEM:
            api_messages.append({"role": "system", "content": msg.content})

        elif msg.role == Role.TOOL_RESULT:
            for tr in msg.tool_results:
                api_messages.append({
                    "role": "tool",
                    "tool_call_id": tr.tool_call_id,
                    "content": tr.content,
                })

        elif msg.role == Role.ASSISTANT and msg.tool_calls:
            tool_calls = []
            for tc in msg.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                })
            api_messages.append({
                "role": "assistant",
                "content": msg.content or None,
                "tool_calls": tool_calls,
            })

        elif msg.role == Role.ASSISTANT:
            api_messages.append({"role": "assistant", "content": msg.content})

        else:
            api_messages.append({"role": "user", "content": msg.content})

    return api_messages


def parse_openai_response(
    response: Any, model: str, provider: str = "openai_compatible"
) -> CompletionResponse:
    """Parse an OpenAI-compatible API response into a CompletionResponse."""
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
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments,
                )
            )

    usage = TokenUsage(
        input_tokens=response.usage.prompt_tokens if response.usage else 0,
        output_tokens=response.usage.completion_tokens if response.usage else 0,
        model=model,
        provider=provider,
    )

    return CompletionResponse(
        content=content,
        tool_calls=tool_calls,
        usage=usage,
        stop_reason=choice.finish_reason,
    )
