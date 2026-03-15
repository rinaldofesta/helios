"""Tests for the event bus."""

from __future__ import annotations

import pytest

from helios.core.events import (
    ErrorOccurred,
    Event,
    EventBus,
    SessionCompleted,
    SessionStarted,
    SubtaskCreated,
)
from helios.core.models import SubTask


class TestEventBus:
    @pytest.mark.asyncio
    async def test_subscribe_and_emit(self):
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(SessionStarted, lambda e: received.append(e))

        await bus.emit(SessionStarted(session_id="s1", objective="Test"))
        assert len(received) == 1
        assert isinstance(received[0], SessionStarted)
        assert received[0].session_id == "s1"

    @pytest.mark.asyncio
    async def test_subscribe_specific_type(self):
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(SessionStarted, lambda e: received.append(e))

        # Emit a different event type — should NOT be received
        await bus.emit(SessionCompleted(session_id="s1", total_cost=0.5))
        assert len(received) == 0

        # Emit the subscribed type
        await bus.emit(SessionStarted(session_id="s1", objective="Test"))
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_subscribe_all(self):
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe_all(lambda e: received.append(e))

        await bus.emit(SessionStarted(session_id="s1", objective="Test"))
        await bus.emit(SessionCompleted(session_id="s1", total_cost=0.5))
        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_async_callback(self):
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe(SessionStarted, handler)
        await bus.emit(SessionStarted(session_id="s1", objective="Test"))
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self):
        bus = EventBus()
        received_a: list[Event] = []
        received_b: list[Event] = []

        bus.subscribe(SessionStarted, lambda e: received_a.append(e))
        bus.subscribe(SessionStarted, lambda e: received_b.append(e))

        await bus.emit(SessionStarted(session_id="s1", objective="Test"))
        assert len(received_a) == 1
        assert len(received_b) == 1

    @pytest.mark.asyncio
    async def test_emission_order_preserved(self):
        bus = EventBus()
        order: list[str] = []

        bus.subscribe_all(lambda e: order.append("global"))
        bus.subscribe(SessionStarted, lambda e: order.append("specific"))

        await bus.emit(SessionStarted(session_id="s1", objective="Test"))
        assert order == ["global", "specific"]

    @pytest.mark.asyncio
    async def test_callback_error_does_not_propagate(self):
        bus = EventBus()
        received: list[Event] = []

        def bad_handler(event: Event) -> None:
            raise ValueError("oops")

        bus.subscribe(SessionStarted, bad_handler)
        bus.subscribe(SessionStarted, lambda e: received.append(e))

        # Should not raise
        await bus.emit(SessionStarted(session_id="s1", objective="Test"))
        # Second handler should still be called
        assert len(received) == 1

    def test_emit_sync(self):
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(SessionStarted, lambda e: received.append(e))

        bus.emit_sync(SessionStarted(session_id="s1", objective="Test"))
        assert len(received) == 1
