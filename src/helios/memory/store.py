"""SQLite session store with JSON dual-write."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

from helios.core.models import (
    Session,
    SessionStatus,
    SubTask,
    TaskExchange,
)

logger = logging.getLogger(__name__)

SCHEMA = """\
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    objective TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    file_content TEXT,
    project_name TEXT,
    project_structure TEXT,
    synthesis TEXT,
    total_cost REAL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS task_exchanges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    subtask_json TEXT NOT NULL,
    prompt TEXT NOT NULL,
    result TEXT NOT NULL DEFAULT '',
    cost_json TEXT,
    sequence_order INTEGER NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS code_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    content TEXT NOT NULL,
    language TEXT DEFAULT '',
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
    session_id UNINDEXED,
    objective
);

CREATE VIRTUAL TABLE IF NOT EXISTS exchanges_fts USING fts5(
    session_id UNINDEXED,
    prompt,
    result
);
"""


class SessionStore:
    """Async SQLite session store with FTS5 search and JSON snapshots."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir or Path.home() / ".helios"
        self._db_path = self._data_dir / "helios.db"
        self._sessions_dir = self._data_dir / "sessions"
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create database and directories."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        await self._db.executescript(SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def _ensure_db(self) -> aiosqlite.Connection:
        if not self._db:
            await self.initialize()
        assert self._db is not None
        return self._db

    # ----- CRUD ----- #

    async def save_session(self, session: Session) -> None:
        """Save or update a complete session."""
        db = await self._ensure_db()

        await db.execute(
            """INSERT OR REPLACE INTO sessions
               (id, objective, status, file_content, project_name, project_structure,
                synthesis, total_cost, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session.id,
                session.objective,
                session.status.value,
                session.file_content,
                session.project_name,
                json.dumps(session.project_structure) if session.project_structure else None,
                session.synthesis,
                session.total_cost,
                session.created_at.isoformat(),
                datetime.now().isoformat(),
            ),
        )

        # Save exchanges
        await db.execute("DELETE FROM task_exchanges WHERE session_id = ?", (session.id,))
        for i, ex in enumerate(session.exchanges):
            await db.execute(
                """INSERT INTO task_exchanges
                   (session_id, subtask_json, prompt, result, cost_json, sequence_order)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    session.id,
                    ex.subtask.model_dump_json(),
                    ex.prompt,
                    ex.result,
                    ex.cost.model_dump_json() if ex.cost else None,
                    i,
                ),
            )

        # Save code files
        await db.execute("DELETE FROM code_files WHERE session_id = ?", (session.id,))
        for cf in session.code_files:
            await db.execute(
                "INSERT INTO code_files (session_id, filename, content, language) VALUES (?, ?, ?, ?)",
                (session.id, cf["filename"], cf["content"], cf.get("language", "")),
            )

        # Update FTS
        await db.execute(
            "INSERT OR REPLACE INTO sessions_fts (session_id, objective) VALUES (?, ?)",
            (session.id, session.objective),
        )

        await db.commit()

        # JSON snapshot
        self._write_json_snapshot(session)

    async def save_exchange(self, session_id: str, exchange: TaskExchange, sequence: int) -> None:
        """Incrementally save a single exchange (for mid-session persistence)."""
        db = await self._ensure_db()
        await db.execute(
            """INSERT INTO task_exchanges
               (session_id, subtask_json, prompt, result, cost_json, sequence_order)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                exchange.subtask.model_dump_json(),
                exchange.prompt,
                exchange.result,
                exchange.cost.model_dump_json() if exchange.cost else None,
                sequence,
            ),
        )

        # Update FTS
        await db.execute(
            "INSERT INTO exchanges_fts (session_id, prompt, result) VALUES (?, ?, ?)",
            (session_id, exchange.prompt, exchange.result),
        )

        await db.commit()

    async def get_session(self, session_id: str) -> Session | None:
        """Load a session by ID."""
        db = await self._ensure_db()
        async with db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return await self._row_to_session(db, row)

    async def list_sessions(self, limit: int = 20, offset: int = 0) -> list[Session]:
        """List sessions ordered by creation time (newest first)."""
        db = await self._ensure_db()
        sessions = []
        async with db.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ) as cursor:
            async for row in cursor:
                sessions.append(await self._row_to_session(db, row))
        return sessions

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all related data."""
        db = await self._ensure_db()
        await db.execute("DELETE FROM task_exchanges WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM code_files WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM sessions_fts WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM exchanges_fts WHERE session_id = ?", (session_id,))
        cursor = await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await db.commit()

        # Remove JSON snapshot
        json_path = self._sessions_dir / f"{session_id}.json"
        if json_path.exists():
            json_path.unlink()

        return cursor.rowcount > 0

    async def get_latest_incomplete(self) -> Session | None:
        """Get the most recent running/interrupted session."""
        db = await self._ensure_db()
        async with db.execute(
            "SELECT * FROM sessions WHERE status IN ('running', 'interrupted') ORDER BY updated_at DESC LIMIT 1",
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return await self._row_to_session(db, row)

    async def search(self, query: str, limit: int = 10) -> list[Session]:
        """Full-text search across sessions and exchanges."""
        db = await self._ensure_db()
        session_ids: set[str] = set()

        # Search sessions
        async with db.execute(
            "SELECT session_id FROM sessions_fts WHERE sessions_fts MATCH ? LIMIT ?",
            (query, limit),
        ) as cursor:
            async for row in cursor:
                session_ids.add(row[0])

        # Search exchanges
        async with db.execute(
            "SELECT session_id FROM exchanges_fts WHERE exchanges_fts MATCH ? LIMIT ?",
            (query, limit),
        ) as cursor:
            async for row in cursor:
                session_ids.add(row[0])

        # Load matching sessions
        sessions = []
        for sid in session_ids:
            session = await self.get_session(sid)
            if session:
                sessions.append(session)

        return sorted(sessions, key=lambda s: s.created_at, reverse=True)[:limit]

    # ----- Helpers ----- #

    async def _row_to_session(self, db: aiosqlite.Connection, row: Any) -> Session:
        """Convert a database row to a Session model."""
        session_id = row[0]

        # Load exchanges
        exchanges = []
        async with db.execute(
            "SELECT * FROM task_exchanges WHERE session_id = ? ORDER BY sequence_order",
            (session_id,),
        ) as cursor:
            async for ex_row in cursor:
                subtask = SubTask.model_validate_json(ex_row[2])
                from helios.core.models import CostRecord
                cost = CostRecord.model_validate_json(ex_row[5]) if ex_row[5] else None
                exchanges.append(TaskExchange(
                    subtask=subtask,
                    prompt=ex_row[3],
                    result=ex_row[4],
                    cost=cost,
                ))

        # Load code files
        code_files = []
        async with db.execute(
            "SELECT filename, content, language FROM code_files WHERE session_id = ?",
            (session_id,),
        ) as cursor:
            async for cf_row in cursor:
                code_files.append({
                    "filename": cf_row[0],
                    "content": cf_row[1],
                    "language": cf_row[2],
                })

        return Session(
            id=session_id,
            objective=row[1],
            status=SessionStatus(row[2]),
            file_content=row[3],
            project_name=row[4],
            project_structure=json.loads(row[5]) if row[5] else None,
            synthesis=row[6],
            total_cost=row[7],
            exchanges=exchanges,
            code_files=code_files,
            created_at=datetime.fromisoformat(row[8]),
            updated_at=datetime.fromisoformat(row[9]),
        )

    def _write_json_snapshot(self, session: Session) -> None:
        """Write a JSON snapshot of the session for portability."""
        json_path = self._sessions_dir / f"{session.id}.json"
        json_path.write_text(
            session.model_dump_json(indent=2),
            encoding="utf-8",
        )
