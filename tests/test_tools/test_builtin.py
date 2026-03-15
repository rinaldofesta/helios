"""Tests for built-in tools."""

from pathlib import Path

import pytest

from helios.tools.builtin.code_file import CreateCodeFileTool
from helios.tools.builtin.complete import CompleteObjectiveTool
from helios.tools.builtin.decompose import CreateSubtaskTool
from helios.tools.builtin.project_structure import DefineProjectStructureTool
from helios.tools.builtin.read_file import ReadFileTool
from helios.tools.builtin.search import WebSearchTool
from helios.tools.builtin.synthesis import OutputSynthesisTool
from helios.tools.registry import ToolRegistry


class TestCreateSubtaskTool:
    @pytest.mark.asyncio
    async def test_execute(self):
        tool = CreateSubtaskTool()
        result = await tool.execute({"description": "Write unit tests", "priority": 1})
        assert "Write unit tests" in result
        assert tool.last_subtask is not None
        assert tool.last_subtask.priority == 1

    @pytest.mark.asyncio
    async def test_default_priority(self):
        tool = CreateSubtaskTool()
        await tool.execute({"description": "Setup project"})
        assert tool.last_subtask.priority == 3

    def test_definition(self):
        tool = CreateSubtaskTool()
        defn = tool.definition
        assert defn.name == "create_subtask"
        assert "description" in defn.input_schema["required"]


class TestCompleteObjectiveTool:
    @pytest.mark.asyncio
    async def test_execute(self):
        tool = CompleteObjectiveTool()
        assert tool.completed is False
        result = await tool.execute({"summary": "All tasks done"})
        assert tool.completed is True
        assert tool.summary == "All tasks done"
        assert "All tasks done" in result


class TestDefineProjectStructureTool:
    @pytest.mark.asyncio
    async def test_execute(self):
        tool = DefineProjectStructureTool()
        structure = {"src": {"main.py": None}, "tests": {"test_main.py": None}}
        await tool.execute({"project_name": "MyProject", "structure": structure})
        assert tool.project_name == "MyProject"
        assert tool.structure == structure


class TestCreateCodeFileTool:
    @pytest.mark.asyncio
    async def test_execute(self):
        tool = CreateCodeFileTool()
        await tool.execute({"filename": "main.py", "content": "print('hi')", "language": "python"})
        await tool.execute({"filename": "test.py", "content": "assert True"})
        assert len(tool.files) == 2
        assert tool.files[0]["filename"] == "main.py"
        assert tool.files[1]["language"] == ""


class TestOutputSynthesisTool:
    @pytest.mark.asyncio
    async def test_execute(self):
        tool = OutputSynthesisTool()
        await tool.execute({"content": "Final summary", "is_code_project": True})
        assert tool.content == "Final summary"
        assert tool.is_code_project is True


class TestWebSearchTool:
    @pytest.mark.asyncio
    async def test_no_api_key(self):
        tool = WebSearchTool(api_key=None)
        result = await tool.execute({"query": "python async"})
        assert "unavailable" in result


class TestReadFileTool:
    @pytest.mark.asyncio
    async def test_read_existing_file(self, tmp_path: Path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        tool = ReadFileTool()
        result = await tool.execute({"file_path": str(test_file)})
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self):
        tool = ReadFileTool()
        result = await tool.execute({"file_path": "/nonexistent/file.txt"})
        assert "not found" in result.lower()


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = CreateSubtaskTool()
        registry.register(tool)
        assert registry.get("create_subtask") is tool
        assert registry.get("nonexistent") is None

    def test_list_all(self):
        registry = ToolRegistry()
        registry.register(CreateSubtaskTool())
        registry.register(CompleteObjectiveTool())
        assert len(registry.list_all()) == 2

    def test_get_for_role(self):
        registry = ToolRegistry()
        registry.register(CreateSubtaskTool())
        registry.register(CompleteObjectiveTool())
        registry.register(DefineProjectStructureTool())
        registry.register(CreateCodeFileTool())
        registry.register(OutputSynthesisTool())

        orch_tools = registry.get_for_role("orchestrator")
        assert len(orch_tools) == 2
        names = {t.name for t in orch_tools}
        assert names == {"create_subtask", "complete_objective"}

        refiner_tools = registry.get_for_role("refiner")
        assert len(refiner_tools) == 3

        sub_tools = registry.get_for_role("sub_agent")
        assert len(sub_tools) == 0
