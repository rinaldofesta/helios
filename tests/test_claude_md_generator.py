"""Tests for CLAUDE.md generation."""

from helios.indexing.api_extractor import ApiEndpoint, ApiSurface
from helios.indexing.claude_md_generator import (
    HELIOS_END,
    HELIOS_START,
    generate_claude_md,
    write_claude_md,
)


def _make_surface(endpoints=None, frameworks=None):
    return ApiSurface(
        endpoints=endpoints or [],
        frameworks=frameworks or [],
    )


def _sample_endpoints():
    return [
        ApiEndpoint(
            method="GET", path="/api/users", handler="findAll",
            file_path="src/users/users.controller.ts", line_number=10,
            framework="nestjs",
        ),
        ApiEndpoint(
            method="POST", path="/api/users", handler="create",
            file_path="src/users/users.controller.ts", line_number=20,
            request_type="CreateUserDto", framework="nestjs",
        ),
    ]


class TestGenerateClaudeMd:
    def test_contains_markers(self):
        content = generate_claude_md("myproject", "/path", _make_surface(), [], 10, 0)
        assert HELIOS_START in content
        assert HELIOS_END in content

    def test_contains_project_name(self):
        content = generate_claude_md("myproject", "/path", _make_surface(), [], 10, 0)
        assert "myproject" in content

    def test_contains_api_table(self):
        surface = _make_surface(_sample_endpoints(), ["nestjs"])
        content = generate_claude_md("myproject", "/path", surface, [], 10, 0)
        assert "| GET |" in content
        assert "| POST |" in content
        assert "/api/users" in content
        assert "CreateUserDto" in content

    def test_no_endpoints_message(self):
        content = generate_claude_md("myproject", "/path", _make_surface(), [], 10, 0)
        assert "No API endpoints were automatically detected" in content

    def test_contains_fe_rules(self):
        content = generate_claude_md("myproject", "/path", _make_surface(), [], 10, 0)
        assert "Frontend Development Rules" in content
        assert "Do not invent endpoints" in content

    def test_contains_frameworks(self):
        surface = _make_surface(frameworks=["fastapi", "nestjs"])
        content = generate_claude_md("myproject", "/path", surface, [], 10, 0)
        assert "fastapi" in content
        assert "nestjs" in content

    def test_contains_ecosystems(self):
        content = generate_claude_md("myproject", "/path", _make_surface(), ["python", "javascript"], 10, 5)
        assert "python" in content
        assert "javascript" in content

    def test_contains_dep_count(self):
        content = generate_claude_md("myproject", "/path", _make_surface(), [], 10, 5)
        assert "5" in content


class TestWriteClaudeMd:
    def test_creates_new_file(self, tmp_path):
        helios_section = generate_claude_md("test", str(tmp_path), _make_surface(), [], 0, 0)
        path = write_claude_md(str(tmp_path), helios_section)
        content = (tmp_path / "CLAUDE.md").read_text()
        assert HELIOS_START in content
        assert HELIOS_END in content
        assert path == str(tmp_path / "CLAUDE.md")

    def test_appends_to_existing(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("# My Project\n\nExisting content.\n")
        helios_section = generate_claude_md("test", str(tmp_path), _make_surface(), [], 0, 0)
        write_claude_md(str(tmp_path), helios_section)
        content = (tmp_path / "CLAUDE.md").read_text()
        assert "Existing content." in content
        assert HELIOS_START in content

    def test_replaces_existing_section(self, tmp_path):
        initial = f"# My Project\n\n{HELIOS_START}\nold content\n{HELIOS_END}\n\nFooter.\n"
        (tmp_path / "CLAUDE.md").write_text(initial)
        surface = _make_surface(_sample_endpoints(), ["nestjs"])
        helios_section = generate_claude_md("test", str(tmp_path), surface, [], 10, 0)
        write_claude_md(str(tmp_path), helios_section)
        content = (tmp_path / "CLAUDE.md").read_text()
        assert "old content" not in content
        assert "/api/users" in content
        assert "Footer." in content
        # Exactly one pair of markers
        assert content.count(HELIOS_START) == 1
        assert content.count(HELIOS_END) == 1
