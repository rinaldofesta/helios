"""Tests for the content indexing and search system."""

from pathlib import Path

import pytest

from helios.indexing.chunker import chunk_document
from helios.indexing.dependencies import (
    detect_dependencies,
    _parse_requirement_name,
    _find_python_site_packages,
)
from helios.indexing.embeddings import cosine_similarity, serialize_f32, deserialize_f32
from helios.indexing.scanner import detect_language, scan_directory
from helios.indexing.store import ContentStore


@pytest.fixture
async def store(tmp_path):
    """Create a ContentStore with a temporary database."""
    s = ContentStore(data_dir=tmp_path)
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
def sample_project(tmp_path):
    """Create a sample project directory for indexing."""
    project = tmp_path / "myproject"
    project.mkdir()

    # Python files
    (project / "main.py").write_text(
        "def hello():\n    print('Hello, world!')\n\nif __name__ == '__main__':\n    hello()\n"
    )
    (project / "utils.py").write_text(
        "import os\n\ndef read_config(path):\n    with open(path) as f:\n        return f.read()\n"
    )

    # Subdirectory
    sub = project / "lib"
    sub.mkdir()
    (sub / "parser.py").write_text(
        "class TokenParser:\n    def parse(self, text):\n        return text.split()\n"
    )
    (sub / "models.py").write_text(
        "from dataclasses import dataclass\n\n@dataclass\nclass User:\n    name: str\n    email: str\n"
    )

    # Markdown docs
    (project / "README.md").write_text(
        "# MyProject\n\nA sample project for testing the indexing system.\n"
    )

    # Config
    (project / "config.toml").write_text('[server]\nhost = "localhost"\nport = 8080\n')

    # Should be ignored
    git = project / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/main\n")

    pycache = project / "__pycache__"
    pycache.mkdir()
    (pycache / "main.cpython-311.pyc").write_bytes(b"\x00\x00\x00\x00")

    # Binary file (should be skipped)
    (project / "image.bin").write_bytes(b"\x00\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    return project


# --------------------------------------------------------------------------- #
#  Language detection
# --------------------------------------------------------------------------- #


class TestDetectLanguage:
    def test_python(self):
        assert detect_language(Path("foo.py")) == "python"

    def test_typescript(self):
        assert detect_language(Path("app.tsx")) == "typescript"

    def test_markdown(self):
        assert detect_language(Path("README.md")) == "markdown"

    def test_unknown(self):
        assert detect_language(Path("data.xyz")) == ""

    def test_makefile(self):
        assert detect_language(Path("Makefile")) == "makefile"

    def test_dockerfile(self):
        assert detect_language(Path("Dockerfile")) == "dockerfile"


# --------------------------------------------------------------------------- #
#  ContentStore
# --------------------------------------------------------------------------- #


class TestContentStore:
    async def test_add_source(self, store):
        sid = await store.add_source("test", "/tmp/test")
        assert sid
        sources = await store.list_sources()
        assert len(sources) == 1
        assert sources[0].name == "test"

    async def test_add_source_idempotent(self, store):
        sid1 = await store.add_source("test", "/tmp/test")
        sid2 = await store.add_source("test", "/tmp/test2")
        assert sid1 == sid2
        sources = await store.list_sources()
        assert len(sources) == 1
        assert sources[0].path == "/tmp/test2"

    async def test_upsert_document_insert(self, store):
        sid = await store.add_source("test", "/tmp/test")
        changed = await store.upsert_document(
            sid, "/tmp/test/a.py", "a.py", "print('hello')", "python", 15,
        )
        assert changed is True

    async def test_upsert_document_no_change(self, store):
        sid = await store.add_source("test", "/tmp/test")
        await store.upsert_document(sid, "/tmp/test/a.py", "a.py", "print('hello')", "python", 15)
        changed = await store.upsert_document(
            sid, "/tmp/test/a.py", "a.py", "print('hello')", "python", 15,
        )
        assert changed is False

    async def test_upsert_document_update(self, store):
        sid = await store.add_source("test", "/tmp/test")
        await store.upsert_document(sid, "/tmp/test/a.py", "a.py", "v1", "python", 2)
        changed = await store.upsert_document(
            sid, "/tmp/test/a.py", "a.py", "v2", "python", 2,
        )
        assert changed is True

    async def test_remove_source(self, store):
        sid = await store.add_source("test", "/tmp/test")
        await store.upsert_document(sid, "/tmp/test/a.py", "a.py", "content", "python", 7)
        removed = await store.remove_source("test")
        assert removed is True
        assert await store.list_sources() == []

    async def test_remove_source_not_found(self, store):
        assert await store.remove_source("nope") is False

    async def test_get_document_by_path(self, store):
        sid = await store.add_source("test", "/tmp/test")
        await store.upsert_document(sid, "/tmp/test/a.py", "a.py", "content", "python", 7)
        doc = await store.get_document_by_path("/tmp/test/a.py")
        assert doc is not None
        assert doc.content == "content"
        assert doc.language == "python"

    async def test_find_document_by_relative_path(self, store):
        sid = await store.add_source("test", "/tmp/test")
        await store.upsert_document(sid, "/tmp/test/a.py", "a.py", "content", "python", 7)
        doc = await store.find_document_by_relative_path("a.py", source_name="test")
        assert doc is not None
        assert doc.relative_path == "a.py"

    async def test_search(self, store):
        sid = await store.add_source("test", "/tmp/test")
        await store.upsert_document(sid, "/tmp/test/a.py", "a.py", "def hello(): pass", "python", 18)
        await store.upsert_document(sid, "/tmp/test/b.py", "b.py", "def goodbye(): pass", "python", 20)
        results = await store.search("hello")
        assert len(results) >= 1
        assert results[0].relative_path == "a.py"

    async def test_search_with_source_filter(self, store):
        s1 = await store.add_source("proj1", "/tmp/proj1")
        s2 = await store.add_source("proj2", "/tmp/proj2")
        await store.upsert_document(s1, "/tmp/proj1/a.py", "a.py", "hello world", "python", 11)
        await store.upsert_document(s2, "/tmp/proj2/b.py", "b.py", "hello earth", "python", 11)
        results = await store.search("hello", source_name="proj1")
        assert len(results) == 1
        assert results[0].source_name == "proj1"

    async def test_search_no_results(self, store):
        results = await store.search("nonexistent_xyz_123")
        assert results == []

    async def test_remove_stale_documents(self, store):
        sid = await store.add_source("test", "/tmp/test")
        await store.upsert_document(sid, "/tmp/test/a.py", "a.py", "keep", "python", 4)
        await store.upsert_document(sid, "/tmp/test/b.py", "b.py", "remove", "python", 6)
        removed = await store.remove_stale_documents(sid, {"/tmp/test/a.py"})
        assert removed == 1
        assert await store.get_document_by_path("/tmp/test/b.py") is None

    async def test_get_source_files(self, store):
        sid = await store.add_source("test", "/tmp/test")
        await store.upsert_document(sid, "/tmp/test/a.py", "a.py", "c", "python", 1)
        await store.upsert_document(sid, "/tmp/test/b.ts", "b.ts", "c", "typescript", 1)
        files = await store.get_source_files("test")
        assert set(files) == {"a.py", "b.ts"}

    async def test_get_source_files_with_pattern(self, store):
        sid = await store.add_source("test", "/tmp/test")
        await store.upsert_document(sid, "/tmp/test/a.py", "a.py", "c", "python", 1)
        await store.upsert_document(sid, "/tmp/test/b.ts", "b.ts", "c", "typescript", 1)
        files = await store.get_source_files("test", pattern="*.py")
        assert files == ["a.py"]

    async def test_update_source_stats(self, store):
        sid = await store.add_source("test", "/tmp/test")
        await store.upsert_document(sid, "/tmp/test/a.py", "a.py", "hello", "python", 100)
        await store.upsert_document(sid, "/tmp/test/b.py", "b.py", "world", "python", 200)
        await store.update_source_stats(sid)
        sources = await store.list_sources()
        assert sources[0].file_count == 2
        assert sources[0].total_size == 300


# --------------------------------------------------------------------------- #
#  Scanner
# --------------------------------------------------------------------------- #


class TestScanner:
    async def test_scan_directory(self, store, sample_project):
        sid = await store.add_source("myproject", str(sample_project))
        stats = await scan_directory(sample_project, sid, store)

        # Should index: main.py, utils.py, lib/parser.py, lib/models.py, README.md, config.toml
        assert stats.files_indexed == 6
        assert stats.files_updated == 6
        assert stats.total_size > 0
        assert len(stats.errors) == 0

        # Verify source stats
        sources = await store.list_sources()
        assert sources[0].file_count == 6

    async def test_scan_ignores_git_and_pycache(self, store, sample_project):
        sid = await store.add_source("myproject", str(sample_project))
        await scan_directory(sample_project, sid, store)

        # .git/HEAD and __pycache__/*.pyc should not be indexed
        files = await store.get_source_files("myproject")
        assert not any(".git" in f for f in files)
        assert not any("__pycache__" in f for f in files)

    async def test_scan_skips_binary(self, store, sample_project):
        sid = await store.add_source("myproject", str(sample_project))
        await scan_directory(sample_project, sid, store)

        files = await store.get_source_files("myproject")
        assert not any("image.bin" in f for f in files)

    async def test_scan_incremental(self, store, sample_project):
        sid = await store.add_source("myproject", str(sample_project))

        # First scan
        stats1 = await scan_directory(sample_project, sid, store)
        assert stats1.files_updated == 6

        # Second scan — no changes
        stats2 = await scan_directory(sample_project, sid, store)
        assert stats2.files_updated == 0
        assert stats2.files_indexed == 6

    async def test_scan_detects_changes(self, store, sample_project):
        sid = await store.add_source("myproject", str(sample_project))
        await scan_directory(sample_project, sid, store)

        # Modify a file
        (sample_project / "main.py").write_text("def hello():\n    print('Updated!')\n")

        stats = await scan_directory(sample_project, sid, store)
        assert stats.files_updated == 1

    async def test_scan_removes_deleted_files(self, store, sample_project):
        sid = await store.add_source("myproject", str(sample_project))
        await scan_directory(sample_project, sid, store)

        # Delete a file
        (sample_project / "utils.py").unlink()

        stats = await scan_directory(sample_project, sid, store)
        assert stats.files_removed == 1
        assert await store.get_document_by_path(str(sample_project / "utils.py")) is None

    async def test_scan_invalid_directory(self, store):
        sid = await store.add_source("test", "/tmp/nonexistent")
        with pytest.raises(ValueError, match="Not a directory"):
            await scan_directory("/tmp/nonexistent_xyz_123", sid, store)

    async def test_search_after_indexing(self, store, sample_project):
        sid = await store.add_source("myproject", str(sample_project))
        await scan_directory(sample_project, sid, store)

        results = await store.search("TokenParser")
        assert len(results) >= 1
        assert any("parser.py" in r.relative_path for r in results)

        results = await store.search("dataclass")
        assert len(results) >= 1
        assert any("models.py" in r.relative_path for r in results)


# --------------------------------------------------------------------------- #
#  Chunker
# --------------------------------------------------------------------------- #


class TestChunker:
    def test_chunk_python_code(self):
        code = (
            "import os\n\n"
            "def hello():\n    print('hello')\n\n"
            "def goodbye():\n    print('bye')\n\n"
            "class Greeter:\n    def greet(self):\n        pass\n"
        )
        chunks = chunk_document(code, language="python")
        assert len(chunks) >= 2  # Should split at def/class boundaries

    def test_chunk_markdown(self):
        md = (
            "# Title\n\nIntro paragraph.\n\n"
            "## Section 1\n\nContent for section 1.\n\n"
            "## Section 2\n\nContent for section 2.\n"
        )
        chunks = chunk_document(md, language="markdown")
        assert len(chunks) >= 2  # Should split at headings

    def test_chunk_empty(self):
        assert chunk_document("") == []
        assert chunk_document("   \n  \n  ") == []

    def test_chunk_small_file(self):
        chunks = chunk_document("x = 1", language="python")
        assert len(chunks) == 1
        assert chunks[0].content == "x = 1"

    def test_chunk_respects_max_chars(self):
        # Create a file with a very long function
        long_body = "\n".join(f"    line_{i} = {i}" for i in range(200))
        code = f"def big_function():\n{long_body}\n"
        chunks = chunk_document(code, language="python", max_chars=500)
        assert all(len(c.content) <= 600 for c in chunks)  # Allow some slack

    def test_chunk_merges_tiny(self):
        code = "x = 1\n\ny = 2\n\nz = 3\n"
        chunks = chunk_document(code, language="python", min_chars=20)
        # Tiny chunks should be merged
        assert len(chunks) <= 2

    def test_chunk_line_numbers(self):
        code = "def a():\n    pass\n\ndef b():\n    pass\n"
        chunks = chunk_document(code, language="python", min_chars=1)
        assert chunks[0].start_line == 0
        for c in chunks:
            assert c.end_line >= c.start_line

    def test_chunk_generic_text(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.\n"
        chunks = chunk_document(text, language="toml")
        assert len(chunks) >= 1


# --------------------------------------------------------------------------- #
#  Embeddings helpers
# --------------------------------------------------------------------------- #


class TestEmbeddingHelpers:
    def test_serialize_roundtrip(self):
        vec = [0.1, 0.2, 0.3, -0.5, 1.0]
        blob = serialize_f32(vec)
        result = deserialize_f32(blob)
        assert len(result) == len(vec)
        for a, b in zip(vec, result):
            assert abs(a - b) < 1e-6

    def test_cosine_identical(self):
        v = [1.0, 0.0, 1.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_cosine_opposite(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_cosine_zero_vector(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# --------------------------------------------------------------------------- #
#  Chunks in store
# --------------------------------------------------------------------------- #


class TestChunkStore:
    async def test_sync_chunks(self, store):
        sid = await store.add_source("test", "/tmp/test")
        await store.upsert_document(sid, "/tmp/test/a.py", "a.py", "content", "python", 7)
        doc = await store.get_document_by_path("/tmp/test/a.py")

        created, removed = await store.sync_document_chunks(
            doc.id, sid, [("chunk one", 0, 5), ("chunk two", 6, 10)],
        )
        assert created == 2
        assert removed == 0
        assert await store.get_chunk_count(sid) == 2

    async def test_sync_chunks_idempotent(self, store):
        sid = await store.add_source("test", "/tmp/test")
        await store.upsert_document(sid, "/tmp/test/a.py", "a.py", "content", "python", 7)
        doc = await store.get_document_by_path("/tmp/test/a.py")

        chunks = [("chunk one", 0, 5), ("chunk two", 6, 10)]
        await store.sync_document_chunks(doc.id, sid, chunks)
        created, removed = await store.sync_document_chunks(doc.id, sid, chunks)
        assert created == 0
        assert removed == 0

    async def test_sync_chunks_preserves_embeddings(self, store):
        sid = await store.add_source("test", "/tmp/test")
        await store.upsert_document(sid, "/tmp/test/a.py", "a.py", "content", "python", 7)
        doc = await store.get_document_by_path("/tmp/test/a.py")

        await store.sync_document_chunks(doc.id, sid, [("chunk one", 0, 5)])

        # Store an embedding
        to_embed = await store.get_chunks_without_embeddings(sid)
        assert len(to_embed) == 1
        fake_embedding = [0.1] * 10
        await store.store_embeddings([(to_embed[0][0], fake_embedding)])

        # Re-sync with same content — embedding should be preserved
        await store.sync_document_chunks(doc.id, sid, [("chunk one", 0, 5)])
        embed_count = await store.get_embedding_count(sid)
        assert embed_count == 1
        no_embed = await store.get_chunks_without_embeddings(sid)
        assert len(no_embed) == 0

    async def test_sync_chunks_removes_stale(self, store):
        sid = await store.add_source("test", "/tmp/test")
        await store.upsert_document(sid, "/tmp/test/a.py", "a.py", "content", "python", 7)
        doc = await store.get_document_by_path("/tmp/test/a.py")

        await store.sync_document_chunks(doc.id, sid, [("old chunk", 0, 5)])
        created, removed = await store.sync_document_chunks(doc.id, sid, [("new chunk", 0, 5)])
        assert created == 1
        assert removed == 1

    async def test_fts_chunk_search(self, store):
        sid = await store.add_source("test", "/tmp/test")
        await store.upsert_document(
            sid, "/tmp/test/a.py", "a.py",
            "def authenticate():\n    check_password()", "python", 40,
        )
        doc = await store.get_document_by_path("/tmp/test/a.py")
        await store.sync_document_chunks(
            doc.id, sid,
            [("def authenticate():\n    check_password()", 0, 1)],
        )

        results = await store.fts_chunk_search("authenticate")
        assert len(results) >= 1
        assert "authenticate" in results[0].content

    async def test_vector_search(self, store):
        sid = await store.add_source("test", "/tmp/test")
        await store.upsert_document(sid, "/tmp/test/a.py", "a.py", "c", "python", 1)
        doc = await store.get_document_by_path("/tmp/test/a.py")

        await store.sync_document_chunks(
            doc.id, sid, [("auth code", 0, 0), ("unrelated", 1, 1)],
        )

        to_embed = await store.get_chunks_without_embeddings(sid)
        # Fake embeddings: first chunk points in direction [1,0], second in [0,1]
        await store.store_embeddings([
            (to_embed[0][0], [1.0, 0.0, 0.0]),
            (to_embed[1][0], [0.0, 1.0, 0.0]),
        ])

        # Query pointing in [1,0] direction should match first chunk
        results = await store.vector_search([1.0, 0.0, 0.0], limit=1)
        assert len(results) == 1
        assert results[0].content == "auth code"

    async def test_hybrid_search(self, store):
        sid = await store.add_source("test", "/tmp/test")
        await store.upsert_document(sid, "/tmp/test/a.py", "a.py", "c", "python", 1)
        doc = await store.get_document_by_path("/tmp/test/a.py")

        await store.sync_document_chunks(
            doc.id, sid, [("auth middleware", 0, 0), ("logging handler", 1, 1)],
        )

        to_embed = await store.get_chunks_without_embeddings(sid)
        await store.store_embeddings([
            (to_embed[0][0], [1.0, 0.0]),
            (to_embed[1][0], [0.0, 1.0]),
        ])

        # Hybrid: FTS matches "auth", vector matches [1,0]
        results = await store.hybrid_search(
            "auth", query_embedding=[1.0, 0.0], limit=2,
        )
        assert len(results) >= 1
        assert results[0].content == "auth middleware"

    async def test_hybrid_search_fts_fallback(self, store):
        """Hybrid search works without embeddings (FTS only)."""
        sid = await store.add_source("test", "/tmp/test")
        await store.upsert_document(sid, "/tmp/test/a.py", "a.py", "c", "python", 1)
        doc = await store.get_document_by_path("/tmp/test/a.py")

        await store.sync_document_chunks(
            doc.id, sid, [("authentication flow", 0, 0)],
        )

        results = await store.hybrid_search("authentication", query_embedding=None)
        assert len(results) >= 1

    async def test_remove_source_cleans_chunks(self, store):
        sid = await store.add_source("test", "/tmp/test")
        await store.upsert_document(sid, "/tmp/test/a.py", "a.py", "c", "python", 1)
        doc = await store.get_document_by_path("/tmp/test/a.py")
        await store.sync_document_chunks(doc.id, sid, [("chunk", 0, 0)])
        assert await store.get_chunk_count(sid) == 1

        await store.remove_source("test")
        # Chunks should be gone (source_id FK)
        assert not await store.has_chunks()


# --------------------------------------------------------------------------- #
#  Dependency detection
# --------------------------------------------------------------------------- #


class TestDependencyDetection:
    def test_parse_requirement_name(self):
        assert _parse_requirement_name("pydantic>=2.0.0") == "pydantic"
        assert _parse_requirement_name("pydantic-settings>=2.0.0") == "pydantic-settings"
        assert _parse_requirement_name("ollama") == "ollama"
        assert _parse_requirement_name("PyYAML==6.0") == "PyYAML"
        assert _parse_requirement_name("") is None

    def test_detect_pyproject(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            '[project]\ndependencies = ["requests>=2.0", "click"]\n'
            '[project.optional-dependencies]\ndev = ["pytest"]\n'
        )
        result = detect_dependencies(tmp_path, include_dev=True)
        assert "python" in result.ecosystems
        names = [d.name for d in result.dependencies]
        assert "requests" in names
        assert "click" in names
        assert "pytest" in names

    def test_detect_pyproject_no_dev(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            '[project]\ndependencies = ["requests"]\n'
            '[project.optional-dependencies]\ndev = ["pytest"]\n'
        )
        result = detect_dependencies(tmp_path, include_dev=False)
        names = [d.name for d in result.dependencies]
        assert "requests" in names
        assert "pytest" not in names

    def test_detect_requirements_txt(self, tmp_path):
        (tmp_path / "requirements.txt").write_text(
            "flask==2.3.0\nredis>=4.0\n# comment\n-r other.txt\n"
        )
        result = detect_dependencies(tmp_path)
        names = [d.name for d in result.dependencies]
        assert "flask" in names
        assert "redis" in names
        assert len(names) == 2

    def test_detect_package_json(self, tmp_path):
        (tmp_path / "package.json").write_text(
            '{"dependencies": {"react": "^18.0"}, "devDependencies": {"jest": "^29.0"}}'
        )
        result = detect_dependencies(tmp_path, include_dev=True)
        assert "javascript" in result.ecosystems
        names = [d.name for d in result.dependencies]
        assert "react" in names
        assert "jest" in names

    def test_detect_no_deps(self, tmp_path):
        result = detect_dependencies(tmp_path)
        assert result.dependencies == []
        assert result.ecosystems == []

    def test_find_site_packages(self, tmp_path):
        # Create a fake venv structure
        sp = tmp_path / ".venv" / "lib" / "python3.11" / "site-packages"
        sp.mkdir(parents=True)
        found = _find_python_site_packages(tmp_path)
        assert found == sp

    def test_find_site_packages_missing(self, tmp_path):
        assert _find_python_site_packages(tmp_path) is None

    def test_resolve_real_project(self):
        """Test against the actual Helios project."""
        result = detect_dependencies(Path.cwd())
        assert "python" in result.ecosystems
        assert result.resolved > 0
        # At least pydantic should be resolved
        pydantic = next((d for d in result.dependencies if d.name == "pydantic"), None)
        assert pydantic is not None
        assert pydantic.source_path is not None
        assert pydantic.source_path.is_dir()


# --------------------------------------------------------------------------- #
#  File watcher
# --------------------------------------------------------------------------- #


class TestFileWatcher:
    async def test_watcher_creates(self):
        from helios.indexing.watcher import FileWatcher

        changes: list[str] = []

        async def on_change(sid: str) -> None:
            changes.append(sid)

        watcher = FileWatcher(on_change=on_change, debounce=0.5)
        assert watcher._running is False

    async def test_watcher_watch_nonexistent(self):
        from helios.indexing.watcher import FileWatcher

        watcher = FileWatcher(on_change=lambda sid: None)
        watcher.watch("s1", "/nonexistent/path/xyz")
        assert len(watcher._watched) == 0

    async def test_watcher_watch_valid(self, tmp_path):
        from helios.indexing.watcher import FileWatcher

        watcher = FileWatcher(on_change=lambda sid: None)
        watcher.watch("s1", str(tmp_path))
        assert str(tmp_path) in watcher._watched


# --------------------------------------------------------------------------- #
#  Crawler
# --------------------------------------------------------------------------- #


class TestCrawler:
    def test_extract_text_basic(self):
        from helios.indexing.crawler import extract_text

        html = "<h1>Title</h1><p>Hello <b>world</b></p>"
        title, text = extract_text(html)
        assert "Hello" in text
        assert "world" in text

    def test_extract_text_skips_scripts(self):
        from helios.indexing.crawler import extract_text

        html = "<p>Keep</p><script>var x = 1;</script><p>Also keep</p>"
        _, text = extract_text(html)
        assert "Keep" in text
        assert "var x" not in text

    def test_extract_text_skips_nav_footer(self):
        from helios.indexing.crawler import extract_text

        html = "<nav>Navigation</nav><main>Content</main><footer>Foot</footer>"
        _, text = extract_text(html)
        assert "Content" in text
        assert "Navigation" not in text
        assert "Foot" not in text

    def test_extract_text_preserves_code(self):
        from helios.indexing.crawler import extract_text

        html = "<pre><code>def hello():\n    pass</code></pre>"
        _, text = extract_text(html)
        assert "def hello():" in text

    def test_extract_title(self):
        from helios.indexing.crawler import extract_text

        html = "<html><head><title>My Page</title></head><body>Content</body></html>"
        title, _ = extract_text(html)
        assert title == "My Page"

    def test_extract_links(self):
        from helios.indexing.crawler import extract_links

        html = '<a href="/docs">Docs</a><a href="https://other.com/x">Other</a>'
        links = extract_links(html, "https://example.com/")
        assert "https://example.com/docs" in links
        assert "https://other.com/x" in links

    def test_extract_links_skips_special(self):
        from helios.indexing.crawler import extract_links

        html = '<a href="javascript:void(0)">JS</a><a href="mailto:x@y.com">Mail</a><a href="/ok">OK</a>'
        links = extract_links(html, "https://example.com/")
        assert len(links) == 1
        assert links[0] == "https://example.com/ok"

    def test_normalize_url(self):
        from helios.indexing.crawler import _normalize_url

        assert _normalize_url("https://x.com/docs/") == "https://x.com/docs"
        assert _normalize_url("https://x.com/docs#section") == "https://x.com/docs"
        assert _normalize_url("https://x.com/") == "https://x.com/"
        assert _normalize_url("https://x.com/a?q=1") == "https://x.com/a?q=1"

    def test_extract_text_empty(self):
        from helios.indexing.crawler import extract_text

        title, text = extract_text("")
        assert title == ""
        assert text == ""
