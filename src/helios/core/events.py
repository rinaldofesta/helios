"""Event types and event bus for the Helios engine."""

from __future__ import annotations

import inspect
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Coroutine

from pydantic import BaseModel, Field

from helios.core.models import CostRecord, SubTask, TokenUsage

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Base
# --------------------------------------------------------------------------- #


class Event(BaseModel):
    """Base class for all Helios events."""

    timestamp: datetime = Field(default_factory=datetime.now)


# --------------------------------------------------------------------------- #
#  Session events
# --------------------------------------------------------------------------- #


class SessionStarted(Event):
    """Emitted when a new session begins."""

    session_id: str
    objective: str


class SessionCompleted(Event):
    """Emitted when a session finishes successfully."""

    session_id: str
    total_cost: float = 0.0


# --------------------------------------------------------------------------- #
#  Subtask events
# --------------------------------------------------------------------------- #


class SubtaskCreated(Event):
    """Emitted when the orchestrator creates a new sub-task."""

    session_id: str
    subtask: SubTask


class SubtaskCompleted(Event):
    """Emitted when a sub-agent finishes executing a sub-task."""

    session_id: str
    subtask_id: str
    result: str


# --------------------------------------------------------------------------- #
#  Usage events
# --------------------------------------------------------------------------- #


class TokensUsed(Event):
    """Emitted after each provider call with token counts."""

    session_id: str
    usage: TokenUsage


class CostIncurred(Event):
    """Emitted after each provider call with cost breakdown."""

    session_id: str
    cost: CostRecord


# --------------------------------------------------------------------------- #
#  Output events
# --------------------------------------------------------------------------- #


class StreamChunk(Event):
    """Emitted during streaming for each text delta."""

    session_id: str
    delta: str
    role: str = "assistant"


class OutputGenerated(Event):
    """Emitted when the refiner produces final output."""

    session_id: str
    content: str
    is_code_project: bool = False


# --------------------------------------------------------------------------- #
#  Error events
# --------------------------------------------------------------------------- #


class ErrorOccurred(Event):
    """Emitted when an error occurs during processing."""

    session_id: str
    error: str
    phase: str = ""  # "orchestrator", "sub_agent", "refiner"


# --------------------------------------------------------------------------- #
#  Event Bus
# --------------------------------------------------------------------------- #

# Callback types
SyncCallback = Callable[[Event], None]
AsyncCallback = Callable[[Event], Coroutine[Any, Any, None]]
Callback = SyncCallback | AsyncCallback


class EventBus:
    """Pub/sub event bus for decoupling engine from presentation.

    Supports both sync and async callbacks. Preserves emission order.
    """

    def __init__(self) -> None:
        self._subscribers: dict[type[Event], list[Callback]] = defaultdict(list)
        self._global_subscribers: list[Callback] = []

    def subscribe(self, event_type: type[Event], callback: Callback) -> None:
        """Subscribe to a specific event type."""
        self._subscribers[event_type].append(callback)

    def subscribe_all(self, callback: Callback) -> None:
        """Subscribe to all event types."""
        self._global_subscribers.append(callback)

    async def emit(self, event: Event) -> None:
        """Emit an event to all subscribers, preserving order."""
        callbacks = list(self._global_subscribers) + list(self._subscribers.get(type(event), []))
        for cb in callbacks:
            try:
                if inspect.iscoroutinefunction(cb):
                    await cb(event)
                else:
                    cb(event)
            except Exception:
                logger.debug("Event callback failed for %s", type(event).__name__, exc_info=True)

    def emit_sync(self, event: Event) -> None:
        """Emit an event synchronously (for use outside async context)."""
        callbacks = list(self._global_subscribers) + list(self._subscribers.get(type(event), []))
        for cb in callbacks:
            try:
                if not inspect.iscoroutinefunction(cb):
                    cb(event)
            except Exception:
                logger.debug("Sync event callback failed for %s", type(event).__name__, exc_info=True)
