"""Tests for the SQLite session store."""

from __future__ import annotations

from pathlib import Path

import pytest

from helios.core.models import (
    CostRecord,
    Session,
    SessionStatus,
    SubTask,
    TaskExchange,
    TokenUsage,
)
from helios.memory.store import SessionStore


@pytest.fixture
async def store(tmp_path: Path):
    s = SessionStore(data_dir=tmp_path)
    await s.initialize()
    yield s
    await s.close()


def _make_session(objective: str = "Build an API") -> Session:
    subtask = SubTask(description="Setup project")
    exchange = TaskExchange(
        subtask=subtask,
        prompt="Initialize Flask project",
        result="Flask project initialized successfully",
        cost=CostRecord(
            usage=TokenUsage(input_tokens=100, output_tokens=50, model="test", provider="mock"),
            input_cost=0.001,
            output_cost=0.002,
            total_cost=0.003,
        ),
    )
    return Session(
        objective=objective,
        status=SessionStatus.COMPLETED,
        exchanges=[exchange],
        total_cost=0.003,
        synthesis="Final output",
    )


class TestSessionStore:
    @pytest.mark.asyncio
    async def test_save_and_get(self, store: SessionStore):
        session = _make_session()
        await store.save_session(session)

        loaded = await store.get_session(session.id)
        assert loaded is not None
        assert loaded.objective == session.objective
        assert loaded.status == SessionStatus.COMPLETED
        assert len(loaded.exchanges) == 1
        assert loaded.exchanges[0].subtask.description == "Setup project"
        assert loaded.total_cost == 0.003

    @pytest.mark.asyncio
    async def test_list_sessions(self, store: SessionStore):
        for i in range(3):
            await store.save_session(_make_session(f"Task {i}"))

        sessions = await store.list_sessions()
        assert len(sessions) == 3

    @pytest.mark.asyncio
    async def test_list_sessions_with_limit(self, store: SessionStore):
        for i in range(5):
            await store.save_session(_make_session(f"Task {i}"))

        sessions = await store.list_sessions(limit=2)
        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_delete_session(self, store: SessionStore):
        session = _make_session()
        await store.save_session(session)

        deleted = await store.delete_session(session.id)
        assert deleted is True

        loaded = await store.get_session(session.id)
        assert loaded is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store: SessionStore):
        deleted = await store.delete_session("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_get_latest_incomplete(self, store: SessionStore):
        # Save a completed session
        completed = _make_session()
        await store.save_session(completed)

        # Save a running session
        running = Session(objective="In progress", status=SessionStatus.RUNNING)
        await store.save_session(running)

        latest = await store.get_latest_incomplete()
        assert latest is not None
        assert latest.id == running.id

    @pytest.mark.asyncio
    async def test_get_latest_incomplete_none(self, store: SessionStore):
        await store.save_session(_make_session())
        latest = await store.get_latest_incomplete()
        assert latest is None

    @pytest.mark.asyncio
    async def test_search(self, store: SessionStore):
        await store.save_session(_make_session("Build a Flask API"))
        await store.save_session(_make_session("Create a React app"))

        results = await store.search("Flask")
        assert len(results) >= 1
        assert any("Flask" in s.objective for s in results)

    @pytest.mark.asyncio
    async def test_incremental_save_exchange(self, store: SessionStore):
        session = Session(objective="Test", status=SessionStatus.RUNNING)
        await store.save_session(session)

        exchange = TaskExchange(
            subtask=SubTask(description="Step 1"),
            prompt="Do step 1",
            result="Step 1 done",
        )
        await store.save_exchange(session.id, exchange, sequence=0)

        loaded = await store.get_session(session.id)
        assert loaded is not None
        assert len(loaded.exchanges) == 1

    @pytest.mark.asyncio
    async def test_json_snapshot_created(self, store: SessionStore, tmp_path: Path):
        session = _make_session()
        await store.save_session(session)

        json_path = tmp_path / "sessions" / f"{session.id}.json"
        assert json_path.exists()

    @pytest.mark.asyncio
    async def test_code_files_persistence(self, store: SessionStore):
        session = _make_session()
        session.code_files = [
            {"filename": "main.py", "content": "print('hi')", "language": "python"},
            {"filename": "test.py", "content": "assert True", "language": "python"},
        ]
        session.project_name = "myproject"
        session.project_structure = {"src": {"main.py": None}}
        await store.save_session(session)

        loaded = await store.get_session(session.id)
        assert loaded is not None
        assert len(loaded.code_files) == 2
        assert loaded.project_name == "myproject"
        assert loaded.project_structure == {"src": {"main.py": None}}
