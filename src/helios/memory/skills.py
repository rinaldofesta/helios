"""Skills learning loop — extract and recall reusable procedures."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import aiosqlite

from helios.core.models import Session, Skill

logger = logging.getLogger(__name__)

SKILLS_SCHEMA = """\
CREATE TABLE IF NOT EXISTS skills (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    trigger_keywords TEXT DEFAULT '[]',
    procedure_steps TEXT DEFAULT '[]',
    usage_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(
    skill_id,
    name,
    trigger_keywords,
    description
);
"""


class SkillManager:
    """Manages skill extraction, storage, and recall."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir or Path.home() / ".helios"
        self._db_path = self._data_dir / "helios.db"
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Initialize the skills tables."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        await self._db.executescript(SKILLS_SCHEMA)
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

    async def save_skill(self, skill: Skill) -> None:
        """Save a skill to the database."""
        db = await self._ensure_db()
        await db.execute(
            """INSERT OR REPLACE INTO skills
               (id, name, description, trigger_keywords, procedure_steps,
                usage_count, success_count, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                skill.id,
                skill.name,
                skill.description,
                json.dumps(skill.trigger_keywords),
                json.dumps(skill.procedure_steps),
                skill.usage_count,
                skill.success_count,
                skill.created_at.isoformat(),
            ),
        )
        await db.execute(
            "INSERT OR REPLACE INTO skills_fts (skill_id, name, trigger_keywords, description) VALUES (?, ?, ?, ?)",
            (skill.id, skill.name, " ".join(skill.trigger_keywords), skill.description),
        )
        await db.commit()

    async def recall_skills(self, objective: str, limit: int = 3) -> list[Skill]:
        """Recall relevant skills via FTS5 matching against the objective."""
        db = await self._ensure_db()
        skills: list[Skill] = []

        # Simple word-based query — match any word from the objective
        words = [w.strip(".,!?;:'\"") for w in objective.split() if len(w) > 3]
        if not words:
            return []

        fts_query = " OR ".join(words[:10])

        try:
            async with db.execute(
                "SELECT skill_id FROM skills_fts WHERE skills_fts MATCH ? LIMIT ?",
                (fts_query, limit),
            ) as cursor:
                async for row in cursor:
                    skill = await self._get_skill(db, row[0])
                    if skill:
                        skills.append(skill)
        except Exception:
            logger.debug("FTS5 search failed", exc_info=True)

        return skills

    async def increment_usage(self, skill_id: str, success: bool = True) -> None:
        """Increment usage counter for a skill."""
        db = await self._ensure_db()
        if success:
            await db.execute(
                "UPDATE skills SET usage_count = usage_count + 1, success_count = success_count + 1 WHERE id = ?",
                (skill_id,),
            )
        else:
            await db.execute(
                "UPDATE skills SET usage_count = usage_count + 1 WHERE id = ?",
                (skill_id,),
            )
        await db.commit()

    async def list_skills(self, limit: int = 20) -> list[Skill]:
        """List all skills ordered by usage count."""
        db = await self._ensure_db()
        skills = []
        async with db.execute(
            "SELECT * FROM skills ORDER BY usage_count DESC LIMIT ?", (limit,)
        ) as cursor:
            async for row in cursor:
                skills.append(self._row_to_skill(row))
        return skills

    def should_extract(self, session: Session) -> bool:
        """Heuristic: only attempt skill extraction if session has >= 3 exchanges."""
        return len(session.exchanges) >= 3 and session.status.value == "completed"

    async def _get_skill(self, db: aiosqlite.Connection, skill_id: str) -> Skill | None:
        async with db.execute("SELECT * FROM skills WHERE id = ?", (skill_id,)) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return self._row_to_skill(row)

    def _row_to_skill(self, row: tuple) -> Skill:
        return Skill(
            id=row[0],
            name=row[1],
            description=row[2],
            trigger_keywords=json.loads(row[3]),
            procedure_steps=json.loads(row[4]),
            usage_count=row[5],
            success_count=row[6],
            created_at=datetime.fromisoformat(row[7]),
        )
