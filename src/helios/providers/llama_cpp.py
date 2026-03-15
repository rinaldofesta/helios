"""llama-cpp-python provider for direct GGUF model loading."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from helios.core.models import CompletionResponse, Message, Role, TokenUsage, ToolCall
from helios.providers.base import ToolDefinition

logger = logging.getLogger(__name__)


class LlamaCppProvider:
    """Provider that loads GGUF models directly via llama-cpp-python."""

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        **kwargs: Any,
    ) -> None:
        self._model_path = str(model_path)
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._kwargs = kwargs
        self._llm: Any = None

    def _ensure_loaded(self) -> Any:
        """Lazily load the model on first use."""
        if self._llm is None:
            try:
                from llama_cpp import Llama
            except ImportError:
                raise ImportError(
                    "llama-cpp-python is required for GGUF models. "
                    "Install with: pip install helios[local]"
                )
            logger.info("Loading GGUF model from %s", self._model_path)
            self._llm = Llama(
                model_path=self._model_path,
                n_ctx=self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                verbose=False,
                **self._kwargs,
            )
        return self._llm

    @property
    def model_name(self) -> str:
        return Path(self._model_path).stem

    def supports_tools(self) -> bool:
        return False  # GGUF models don't natively support tool calls

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> CompletionResponse:
        api_messages = _to_chat_messages(messages, system)

        # If tools are provided, inject schema into system prompt for JSON output
        if tools:
            tool_instructions = _build_tool_instructions(tools)
            if api_messages and api_messages[0]["role"] == "system":
                api_messages[0]["content"] += "\n\n" + tool_instructions
            else:
                api_messages.insert(0, {"role": "system", "content": tool_instructions})

        # Run synchronous llama-cpp-python in a thread
        response = await asyncio.to_thread(self._complete_sync, api_messages, max_tokens)
        return response

    def _complete_sync(self, messages: list[dict], max_tokens: int) -> CompletionResponse:
        llm = self._ensure_loaded()
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
        )

        choice = response["choices"][0]
        content = choice["message"].get("content", "")
        usage_data = response.get("usage", {})

        # Try to parse tool calls from JSON output
        tool_calls = _extract_tool_calls(content)

        usage = TokenUsage(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=usage_data.get("completion_tokens", 0),
            model=self.model_name,
            provider="llama_cpp",
        )

        return CompletionResponse(
            content=content if not tool_calls else "",
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=choice.get("finish_reason", "stop"),
        )


def _to_chat_messages(messages: list[Message], system: str | None = None) -> list[dict[str, str]]:
    """Convert Helios Messages to simple chat format."""
    result: list[dict[str, str]] = []
    if system:
        result.append({"role": "system", "content": system})
    for msg in messages:
        if msg.role == Role.SYSTEM:
            result.append({"role": "system", "content": msg.content})
        elif msg.role == Role.ASSISTANT:
            result.append({"role": "assistant", "content": msg.content})
        elif msg.role == Role.TOOL_RESULT:
            content = "\n".join(tr.content for tr in msg.tool_results)
            result.append({"role": "user", "content": f"Tool results:\n{content}"})
        else:
            result.append({"role": "user", "content": msg.content})
    return result


def _build_tool_instructions(tools: list[ToolDefinition]) -> str:
    """Build system prompt instructions for JSON-based tool use."""
    tool_descriptions = []
    for t in tools:
        tool_descriptions.append(
            f"- {t.name}: {t.description}\n  Parameters: {json.dumps(t.input_schema)}"
        )

    return (
        "You have access to the following tools. To use a tool, respond with ONLY "
        "a JSON object in this format:\n"
        '{"tool": "<tool_name>", "arguments": {<parameters>}}\n\n'
        "Available tools:\n" + "\n".join(tool_descriptions)
    )


def _extract_tool_calls(content: str) -> list[ToolCall]:
    """Try to extract tool calls from JSON in the model output."""
    content = content.strip()
    if not content.startswith("{"):
        return []
    try:
        data = json.loads(content)
        if "tool" in data and "arguments" in data:
            return [ToolCall(name=data["tool"], arguments=data["arguments"])]
    except json.JSONDecodeError:
        pass
    return []
