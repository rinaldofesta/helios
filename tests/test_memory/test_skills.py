"""Tests for the skills learning loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from helios.core.models import Session, SessionStatus, Skill, SubTask, TaskExchange
from helios.memory.skills import SkillManager


@pytest.fixture
async def manager(tmp_path: Path):
    m = SkillManager(data_dir=tmp_path)
    await m.initialize()
    yield m
    await m.close()


class TestSkillManager:
    @pytest.mark.asyncio
    async def test_save_and_list(self, manager: SkillManager):
        skill = Skill(
            name="Flask API Design",
            description="How to design RESTful APIs with Flask",
            trigger_keywords=["flask", "api", "rest"],
            procedure_steps=["Create app", "Define routes", "Add error handling"],
        )
        await manager.save_skill(skill)

        skills = await manager.list_skills()
        assert len(skills) == 1
        assert skills[0].name == "Flask API Design"

    @pytest.mark.asyncio
    async def test_recall_skills(self, manager: SkillManager):
        skill = Skill(
            name="Flask API Design",
            description="How to design RESTful APIs with Flask",
            trigger_keywords=["flask", "api", "rest", "endpoint"],
        )
        await manager.save_skill(skill)

        results = await manager.recall_skills("Build a Flask REST API")
        assert len(results) >= 1
        assert results[0].name == "Flask API Design"

    @pytest.mark.asyncio
    async def test_recall_no_match(self, manager: SkillManager):
        skill = Skill(
            name="Flask API Design",
            trigger_keywords=["flask", "api"],
        )
        await manager.save_skill(skill)

        results = await manager.recall_skills("xyz")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_increment_usage(self, manager: SkillManager):
        skill = Skill(name="Test Skill", trigger_keywords=["test"])
        await manager.save_skill(skill)

        await manager.increment_usage(skill.id, success=True)
        await manager.increment_usage(skill.id, success=False)

        skills = await manager.list_skills()
        assert skills[0].usage_count == 2
        assert skills[0].success_count == 1

    def test_should_extract_heuristic(self, manager: SkillManager):
        # Too few exchanges
        short_session = Session(
            objective="Test",
            status=SessionStatus.COMPLETED,
            exchanges=[
                TaskExchange(subtask=SubTask(description="t1"), prompt="p1", result="r1"),
            ],
        )
        assert manager.should_extract(short_session) is False

        # Enough exchanges
        long_session = Session(
            objective="Test",
            status=SessionStatus.COMPLETED,
            exchanges=[
                TaskExchange(subtask=SubTask(description=f"t{i}"), prompt=f"p{i}", result=f"r{i}")
                for i in range(3)
            ],
        )
        assert manager.should_extract(long_session) is True

        # Not completed
        running_session = Session(
            objective="Test",
            status=SessionStatus.RUNNING,
            exchanges=[
                TaskExchange(subtask=SubTask(description=f"t{i}"), prompt=f"p{i}", result=f"r{i}")
                for i in range(3)
            ],
        )
        assert manager.should_extract(running_session) is False
