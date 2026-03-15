"""Ollama provider implementation."""

from __future__ import annotations

import json
import logging
from typing import Any

import ollama as ollama_lib
from ollama import AsyncClient

from helios.core.models import CompletionResponse, Message, Role, TokenUsage, ToolCall
from helios.providers.base import ToolDefinition

logger = logging.getLogger(__name__)


class OllamaProvider:
    """Provider for local Ollama models."""

    def __init__(
        self,
        model: str = "llama3:instruct",
        host: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._client = AsyncClient(host=host)
        self._host = host

    @property
    def model_name(self) -> str:
        return self._model

    def supports_tools(self) -> bool:
        return True

    async def ensure_model(self) -> None:
        """Pull the model if it's not available locally."""
        try:
            await self._client.show(self._model)
        except Exception:
            logger.info("Model %s not found, pulling...", self._model)
            await self._client.pull(self._model)

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> CompletionResponse:
        api_messages = _to_ollama_messages(messages, system)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "options": {"num_predict": max_tokens},
        }
        if tools:
            kwargs["tools"] = [t.to_openai() for t in tools]

        response = await self._client.chat(**kwargs)
        return _parse_response(response, self._model)


def _to_ollama_messages(messages: list[Message], system: str | None = None) -> list[dict[str, Any]]:
    """Convert Helios Messages to Ollama format."""
    api_messages: list[dict[str, Any]] = []

    if system:
        api_messages.append({"role": "system", "content": system})

    for msg in messages:
        if msg.role == Role.SYSTEM:
            api_messages.append({"role": "system", "content": msg.content})
        elif msg.role == Role.TOOL_RESULT:
            for tr in msg.tool_results:
                api_messages.append({"role": "tool", "content": tr.content})
        elif msg.role == Role.ASSISTANT:
            api_messages.append({"role": "assistant", "content": msg.content})
        else:
            api_messages.append({"role": "user", "content": msg.content})

    return api_messages


def _parse_response(response: Any, model: str) -> CompletionResponse:
    """Parse an Ollama API response."""
    message = response.get("message", response) if isinstance(response, dict) else response.message

    if isinstance(message, dict):
        content = message.get("content", "")
        raw_tool_calls = message.get("tool_calls", [])
    else:
        content = message.content or ""
        raw_tool_calls = getattr(message, "tool_calls", []) or []

    tool_calls: list[ToolCall] = []
    for tc in raw_tool_calls:
        if isinstance(tc, dict):
            func = tc.get("function", {})
            name = func.get("name", "")
            arguments = func.get("arguments", {})
        else:
            name = tc.function.name
            arguments = tc.function.arguments
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}
        tool_calls.append(ToolCall(name=name, arguments=arguments))

    # Ollama doesn't always return detailed usage
    eval_count = 0
    prompt_eval_count = 0
    if isinstance(response, dict):
        eval_count = response.get("eval_count", 0)
        prompt_eval_count = response.get("prompt_eval_count", 0)

    usage = TokenUsage(
        input_tokens=prompt_eval_count,
        output_tokens=eval_count,
        model=model,
        provider="ollama",
    )

    return CompletionResponse(
        content=content,
        tool_calls=tool_calls,
        usage=usage,
        stop_reason="stop",
    )
