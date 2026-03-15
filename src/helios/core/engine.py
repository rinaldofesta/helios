"""Helios orchestration engine — the core loop."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from helios.config.schema import HeliosSettings
from helios.core.models import (
    CompletionResponse,
    Message,
    Role,
    Session,
    SessionStatus,
    SubTask,
    SubTaskStatus,
    TaskExchange,
    ToolCall,
    ToolResult,
)
from helios.providers.base import Provider, ToolDefinition
from helios.tools.builtin.code_file import CreateCodeFileTool
from helios.tools.builtin.complete import CompleteObjectiveTool
from helios.tools.builtin.decompose import CreateSubtaskTool
from helios.tools.builtin.project_structure import DefineProjectStructureTool
from helios.tools.builtin.read_file import ReadFileTool
from helios.tools.builtin.search import WebSearchTool
from helios.tools.builtin.synthesis import OutputSynthesisTool
from helios.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Callback type for engine events (simple pre-event-bus approach)
EventCallback = Callable[[str, dict[str, Any]], None]


# --------------------------------------------------------------------------- #
#  System prompts
# --------------------------------------------------------------------------- #

ORCHESTRATOR_SYSTEM = """\
You are the Orchestrator in a multi-agent system. Your role is to break down complex \
objectives into focused, actionable sub-tasks.

For each iteration, analyze the objective and all previous sub-task results, then either:
1. Call the `create_subtask` tool to define the next sub-task for the sub-agent to execute.
2. Call the `complete_objective` tool when the objective has been fully achieved.

Guidelines:
- Create one sub-task at a time. Each should be specific, actionable, and self-contained.
- When dealing with code tasks, check for errors and include fixes in subsequent sub-tasks.
- Assess previous results carefully before creating new tasks — avoid redundant work.
- Only call `complete_objective` when ALL aspects of the objective are addressed.
- If a sub-task depends on web search, include a search_query in the create_subtask call.\
"""

SUB_AGENT_SYSTEM = """\
You are a Sub-Agent executing a specific task. Provide a thorough, detailed response \
that fully addresses the task. Be comprehensive but focused on the task at hand.

If you have previous task context, use it to inform your response but focus on the \
current task.\
"""

REFINER_SYSTEM = """\
You are the Refiner. Review all sub-task results and synthesize them into a cohesive \
final output.

You have three tools available:
- `define_project_structure`: Use ONLY for code projects to define the folder structure.
- `create_code_file`: Use ONLY for code projects to create individual code files.
- `output_synthesis`: Use to output the final refined synthesis.

For code projects:
1. First call `define_project_structure` with the project name and folder structure.
2. Then call `create_code_file` for each source file.
3. Finally call `output_synthesis` with a summary.

For non-code objectives:
- Call `output_synthesis` directly with the refined content.\
"""


# --------------------------------------------------------------------------- #
#  Engine
# --------------------------------------------------------------------------- #


class HeliosEngine:
    """Core orchestration engine implementing the three-tier agent pattern."""

    def __init__(
        self,
        orchestrator: Provider,
        sub_agent: Provider,
        refiner: Provider,
        settings: HeliosSettings | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._sub_agent = sub_agent
        self._refiner = refiner
        self._settings = settings or HeliosSettings()
        self._on_event = on_event or (lambda *_: None)

        # Initialize tools
        self._create_subtask = CreateSubtaskTool()
        self._complete_objective = CompleteObjectiveTool()
        self._project_structure = DefineProjectStructureTool()
        self._code_file = CreateCodeFileTool()
        self._synthesis = OutputSynthesisTool()
        self._web_search = WebSearchTool(
            api_key=self._settings.tools.tavily_api_key
        )
        self._read_file = ReadFileTool()

        # Build registry
        self._registry = ToolRegistry()
        for tool in [
            self._create_subtask,
            self._complete_objective,
            self._project_structure,
            self._code_file,
            self._synthesis,
            self._web_search,
            self._read_file,
        ]:
            self._registry.register(tool)

    async def run(self, objective: str, file_content: str | None = None) -> Session:
        """Execute the full orchestration loop for an objective."""
        session = Session(objective=objective, status=SessionStatus.RUNNING, file_content=file_content)
        self._emit("session_started", {"session_id": session.id, "objective": objective})

        try:
            # Phase 1: Orchestrator loop
            await self._orchestrator_loop(session)

            # Phase 2: Refiner
            await self._refiner_phase(session)

            session.status = SessionStatus.COMPLETED
            self._emit("session_completed", {"session_id": session.id, "total_cost": session.total_cost})

        except Exception as e:
            session.status = SessionStatus.FAILED
            self._emit("error", {"session_id": session.id, "error": str(e)})
            raise

        return session

    # ----- Orchestrator loop ----- #

    async def _orchestrator_loop(self, session: Session) -> None:
        """Run the orchestrator → sub-agent loop until completion."""
        max_subtasks = self._settings.general.max_subtasks
        tools = self._registry.get_for_role("orchestrator")

        while len(session.exchanges) < max_subtasks:
            # Build orchestrator messages
            messages = self._build_orchestrator_messages(session)

            # Call orchestrator
            response = await self._orchestrator.complete(
                messages=messages,
                tools=tools,
                system=ORCHESTRATOR_SYSTEM,
                max_tokens=self._settings.orchestrator.max_tokens,
            )
            self._record_cost(session, response, "orchestrator")

            # Process tool calls (with fallback for text-based JSON)
            tool_calls = response.tool_calls
            if not tool_calls and response.content:
                extracted = self._extract_tool_calls_from_text(response.content)
                if extracted:
                    logger.info("Orchestrator returned tool calls as text, extracted %d call(s)", len(extracted))
                    tool_calls = extracted

            if tool_calls:
                # Execute the extracted/native tool calls
                for tc in tool_calls:
                    tool = self._registry.get(tc.name)
                    if tool:
                        await tool.execute(tc.arguments)
                    else:
                        logger.warning("Unknown tool: %s", tc.name)

                # Check if objective was completed
                if self._complete_objective.completed:
                    self._emit("objective_completed", {
                        "session_id": session.id,
                        "summary": self._complete_objective.summary,
                    })
                    break

                # If a subtask was created, execute it
                if self._create_subtask.last_subtask:
                    subtask = self._create_subtask.last_subtask
                    self._create_subtask.last_subtask = None
                    self._emit("subtask_created", {
                        "session_id": session.id,
                        "subtask": subtask.description,
                    })

                    # Execute web search if requested
                    search_context = ""
                    if subtask.search_query and self._settings.tools.web_search_enabled:
                        search_result = await self._web_search.execute({"query": subtask.search_query})
                        search_context = f"\n\nSearch Results:\n{search_result}"

                    # Execute sub-agent
                    result = await self._execute_sub_agent(session, subtask, search_context)

                    exchange = TaskExchange(subtask=subtask, prompt=subtask.description, result=result)
                    session.exchanges.append(exchange)

                    self._emit("subtask_completed", {
                        "session_id": session.id,
                        "subtask_id": subtask.id,
                        "result_preview": result[:200],
                    })
            else:
                # Model returned text without tool calls — treat as completion signal
                if response.content:
                    self._complete_objective.completed = True
                    self._complete_objective.summary = response.content
                break

        if not self._complete_objective.completed:
            logger.warning("Orchestrator loop ended without calling complete_objective (max_subtasks=%d)", max_subtasks)

    async def _process_tool_calls(
        self,
        response: CompletionResponse,
        tools: list[ToolDefinition],
        messages: list[Message],
    ) -> None:
        """Execute tool calls and manage the tool-result loop."""
        for tc in response.tool_calls:
            tool = self._registry.get(tc.name)
            if tool:
                result = await tool.execute(tc.arguments)
            else:
                result = f"Unknown tool: {tc.name}"
            logger.debug("Tool %s -> %s", tc.name, result[:100])

    # ----- Sub-agent execution ----- #

    async def _execute_sub_agent(
        self,
        session: Session,
        subtask: SubTask,
        search_context: str = "",
    ) -> str:
        """Execute a sub-task with the sub-agent provider."""
        # Build context from previous tasks
        prev_context = ""
        if session.exchanges:
            parts = []
            for ex in session.exchanges:
                parts.append(f"Task: {ex.subtask.description}\nResult: {ex.result}")
            prev_context = "Previous tasks:\n" + "\n\n".join(parts)

        prompt = subtask.description
        if session.file_content and not session.exchanges:
            prompt += f"\n\nFile content:\n{session.file_content}"
        if search_context:
            prompt += search_context

        messages = [Message(role=Role.USER, content=prompt)]
        system = prev_context if prev_context else SUB_AGENT_SYSTEM

        response = await self._sub_agent.complete(
            messages=messages,
            system=system,
            max_tokens=self._settings.sub_agent.max_tokens,
        )
        self._record_cost(session, response, "sub_agent")

        result = response.content

        # Continuation handling: if output is near max tokens, request continuation
        if response.usage.output_tokens >= self._settings.sub_agent.max_tokens - 100:
            logger.info("Sub-agent output may be truncated, requesting continuation")
            cont_messages = [
                Message(role=Role.USER, content=prompt),
                Message(role=Role.ASSISTANT, content=result),
                Message(role=Role.USER, content="Please continue from where you left off."),
            ]
            cont_response = await self._sub_agent.complete(
                messages=cont_messages,
                system=system,
                max_tokens=self._settings.sub_agent.max_tokens,
            )
            self._record_cost(session, cont_response, "sub_agent")
            result += cont_response.content

        subtask.status = SubTaskStatus.COMPLETED
        subtask.result = result
        return result

    # ----- Refiner phase ----- #

    async def _refiner_phase(self, session: Session) -> None:
        """Run the refiner to synthesize all results."""
        if not session.exchanges:
            return

        tools = self._registry.get_for_role("refiner")

        # Build refiner prompt
        results_text = "\n\n".join(
            f"Sub-task {i}: {ex.subtask.description}\nResult: {ex.result}"
            for i, ex in enumerate(session.exchanges, 1)
        )
        prompt = f"Objective: {session.objective}\n\nSub-task results:\n{results_text}"

        messages = [Message(role=Role.USER, content=prompt)]

        # Refiner tool-call loop
        max_rounds = 20  # Safety limit for tool call rounds
        for _ in range(max_rounds):
            response = await self._refiner.complete(
                messages=messages,
                tools=tools,
                system=REFINER_SYSTEM,
                max_tokens=self._settings.refiner.max_tokens,
            )
            self._record_cost(session, response, "refiner")

            tool_calls_to_process = response.tool_calls

            # Fallback: if model output JSON text instead of using tools, extract them
            if not tool_calls_to_process and response.content:
                extracted = self._extract_tool_calls_from_text(response.content)
                if extracted:
                    logger.info("Refiner returned tool calls as text, extracted %d call(s)", len(extracted))
                    tool_calls_to_process = extracted

            if not tool_calls_to_process:
                # No tool calls at all — refiner is done
                if response.content and not self._synthesis.content:
                    session.synthesis = response.content
                break

            # Process tool calls
            tool_results: list[ToolResult] = []
            for tc in tool_calls_to_process:
                tool = self._registry.get(tc.name)
                if tool:
                    result_text = await tool.execute(tc.arguments)
                else:
                    result_text = f"Unknown tool: {tc.name}"
                tool_results.append(ToolResult(tool_call_id=tc.id, content=result_text))

            # Append assistant message with tool calls + tool results for next round
            messages.append(Message(
                role=Role.ASSISTANT,
                content=response.content,
                tool_calls=tool_calls_to_process,
            ))
            messages.append(Message(
                role=Role.TOOL_RESULT,
                tool_results=tool_results,
            ))

        # Collect refiner outputs
        if self._project_structure.project_name:
            session.project_name = self._project_structure.project_name
            session.project_structure = self._project_structure.structure
        if self._code_file.files:
            session.code_files = self._code_file.files
        if self._synthesis.content:
            session.synthesis = self._synthesis.content

        self._emit("output_generated", {
            "session_id": session.id,
            "is_code_project": self._synthesis.is_code_project,
        })

    # ----- Helpers ----- #

    def _build_orchestrator_messages(self, session: Session) -> list[Message]:
        """Build the message list for the orchestrator."""
        previous_results = ""
        if session.exchanges:
            parts = [ex.result for ex in session.exchanges]
            previous_results = "\n\n".join(parts)

        content = (
            f"Objective: {session.objective}\n\n"
            f"Previous sub-task results:\n{previous_results if previous_results else 'None'}"
        )
        if session.file_content and not session.exchanges:
            content += f"\n\nFile content:\n{session.file_content}"

        return [Message(role=Role.USER, content=content)]

    def _record_cost(self, session: Session, response: CompletionResponse, phase: str) -> None:
        """Record token usage from a provider response.

        Local models are free so cost is always 0 — we still track tokens
        for observability.
        """
        usage = response.usage
        self._emit("cost_incurred", {
            "session_id": session.id,
            "phase": phase,
            "model": usage.model,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost": 0.0,
        })

    def _extract_tool_calls_from_text(self, text: str) -> list[ToolCall]:
        """Try to extract tool calls from text when models output JSON instead of using tools.

        Small local models often output tool calls as JSON text like:
          {"name": "output_synthesis", "parameters": {"content": "..."}}
        instead of using the proper tool calling mechanism.
        """
        extracted: list[ToolCall] = []
        # Find JSON objects in the text
        for match in re.finditer(r'\{[^{}]*"name"\s*:\s*"(\w+)"[^{}]*"parameters"\s*:\s*(\{.*?\})\s*\}', text, re.DOTALL):
            try:
                name = match.group(1)
                params = json.loads(match.group(2))
                extracted.append(ToolCall(name=name, arguments=params))
                logger.debug("Extracted text-based tool call: %s", name)
            except (json.JSONDecodeError, IndexError):
                continue

        # Also try parsing the entire text as a single JSON tool call
        if not extracted:
            try:
                data = json.loads(text.strip())
                if isinstance(data, dict) and "name" in data:
                    params = data.get("parameters", data.get("arguments", {}))
                    extracted.append(ToolCall(name=data["name"], arguments=params))
                    logger.debug("Extracted full-text tool call: %s", data["name"])
            except (json.JSONDecodeError, KeyError):
                pass

        return extracted

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit an event via the callback."""
        try:
            self._on_event(event_type, data)
        except Exception:
            logger.debug("Event callback failed for %s", event_type, exc_info=True)
