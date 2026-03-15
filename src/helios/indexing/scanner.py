"""Directory scanner for content indexing."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from helios.indexing.store import ContentStore

logger = logging.getLogger(__name__)

# Extension -> language mapping
LANGUAGE_MAP: dict[str, str] = {
    # Code
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".mts": "typescript",
    ".jsx": "javascript", ".tsx": "typescript",
    ".go": "go", ".rs": "rust", ".java": "java",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
    ".rb": "ruby", ".php": "php", ".swift": "swift",
    ".kt": "kotlin", ".scala": "scala",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell",
    ".r": "r", ".jl": "julia", ".lua": "lua",
    ".ex": "elixir", ".exs": "elixir",
    ".hs": "haskell", ".dart": "dart", ".zig": "zig",
    ".cs": "csharp", ".fs": "fsharp",
    # Docs
    ".md": "markdown", ".mdx": "markdown",
    ".rst": "restructuredtext", ".txt": "text",
    ".adoc": "asciidoc",
    # Config
    ".toml": "toml", ".yaml": "yaml", ".yml": "yaml",
    ".json": "json", ".jsonc": "json",
    ".xml": "xml", ".ini": "ini", ".cfg": "ini",
    # Web
    ".html": "html", ".htm": "html",
    ".css": "css", ".scss": "scss", ".less": "less",
    ".vue": "vue", ".svelte": "svelte",
    # Other
    ".sql": "sql", ".graphql": "graphql", ".gql": "graphql",
    ".proto": "protobuf", ".tf": "terraform",
}

# Filename -> language (for extensionless files)
FILENAME_MAP: dict[str, str] = {
    "makefile": "makefile",
    "dockerfile": "dockerfile",
    "rakefile": "ruby",
    "gemfile": "ruby",
    "procfile": "yaml",
    "justfile": "makefile",
    "vagrantfile": "ruby",
    "cmakelists.txt": "cmake",
}

IGNORE_DIRS = {
    ".git", ".svn", ".hg",
    "node_modules", "bower_components",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "env",
    ".tox", ".nox",
    "dist", "build", "_build",
    ".next", ".nuxt",
    "target",
    ".idea", ".vscode",
    "coverage", ".coverage",
    ".terraform",
}

IGNORE_FILES = {
    ".DS_Store", "Thumbs.db", ".gitkeep",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "Cargo.lock", "Gemfile.lock", "composer.lock",
}

MAX_FILE_SIZE = 1_000_000  # 1 MB


@dataclass
class ScanStats:
    files_scanned: int = 0
    files_indexed: int = 0
    files_updated: int = 0
    files_skipped: int = 0
    files_removed: int = 0
    total_size: int = 0
    errors: list[str] = field(default_factory=list)


def detect_language(path: Path) -> str:
    """Detect language from file extension or name."""
    # Check extension first
    lang = LANGUAGE_MAP.get(path.suffix.lower(), "")
    if lang:
        return lang
    # Check filename
    return FILENAME_MAP.get(path.name.lower(), "")


def _is_binary(content: bytes, sample_size: int = 8192) -> bool:
    """Check if content appears to be binary."""
    return b"\x00" in content[:sample_size]


async def scan_directory(
    directory: str | Path,
    source_id: str,
    store: ContentStore,
) -> ScanStats:
    """Scan a directory and index all supported files.

    Handles incremental updates: only changed files are re-indexed,
    and deleted files are removed from the index.
    """
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    stats = ScanStats()
    indexed_paths: set[str] = set()

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune ignored directories (modifies in-place to skip subtrees)
        dirnames[:] = [
            d for d in dirnames
            if d not in IGNORE_DIRS and not d.startswith(".")
        ]

        for filename in filenames:
            if filename in IGNORE_FILES:
                stats.files_skipped += 1
                continue

            filepath = Path(dirpath) / filename
            lang = detect_language(filepath)

            # Skip files with unrecognized extensions
            if not lang:
                stats.files_skipped += 1
                continue

            stats.files_scanned += 1

            # Check size
            try:
                file_size = filepath.stat().st_size
            except OSError:
                stats.files_skipped += 1
                continue

            if file_size > MAX_FILE_SIZE or file_size == 0:
                stats.files_skipped += 1
                continue

            # Read content
            try:
                raw = filepath.read_bytes()
            except OSError as e:
                stats.errors.append(f"{filepath}: {e}")
                continue

            if _is_binary(raw):
                stats.files_skipped += 1
                continue

            try:
                content = raw.decode("utf-8", errors="replace")
            except Exception:
                stats.files_skipped += 1
                continue

            # Index
            abs_path = str(filepath)
            rel_path = str(filepath.relative_to(root))
            indexed_paths.add(abs_path)

            changed = await store.upsert_document(
                source_id=source_id,
                path=abs_path,
                relative_path=rel_path,
                content=content,
                language=lang,
                size=file_size,
            )

            if changed:
                stats.files_updated += 1
            stats.files_indexed += 1
            stats.total_size += file_size

    # Remove stale documents
    stats.files_removed = await store.remove_stale_documents(source_id, indexed_paths)

    # Update source stats
    await store.update_source_stats(source_id)

    return stats
