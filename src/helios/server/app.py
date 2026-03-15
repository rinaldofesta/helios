"""Helios MCP server — exposes context intelligence tools to AI coding agents."""

from __future__ import annotations

import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from helios.indexing.chunker import chunk_document
from helios.indexing.embeddings import (
    generate_embeddings,
    get_query_embedding,
    is_ollama_available,
    ensure_embed_model,
    DEFAULT_EMBED_MODEL,
)
from helios.indexing.scanner import scan_directory
from helios.indexing.store import ContentStore

logger = logging.getLogger(__name__)

# Module-level store singleton
_store: ContentStore | None = None


async def get_store() -> ContentStore:
    global _store
    if _store is None:
        _store = ContentStore()
        await _store.initialize()
    return _store


# --------------------------------------------------------------------------- #
#  MCP Server
# --------------------------------------------------------------------------- #

mcp = FastMCP(
    "helios",
    instructions=(
        "Helios is a local-first context intelligence engine. "
        "It indexes codebases and documents on the user's machine and makes them "
        "searchable via hybrid keyword + semantic search. All data stays local.\n\n"
        "Workflow: use helios_index to index a codebase (with optional embeddings "
        "via Ollama), helios_search to find relevant code/docs, helios_read to get "
        "full file contents, and helios_explore to understand project structure."
    ),
)


# --------------------------------------------------------------------------- #
#  Tools
# --------------------------------------------------------------------------- #


@mcp.tool()
async def helios_search(
    query: str,
    source: str | None = None,
    language: str | None = None,
    limit: int = 10,
    mode: str = "auto",
) -> str:
    """Search across all indexed codebases and documents.

    Supports three search modes:
    - "auto" (default): hybrid keyword + semantic search if embeddings exist, keyword-only otherwise
    - "keyword": FTS5 full-text search only (fast, exact matches)
    - "semantic": vector similarity only (finds conceptually similar content)

    Args:
        query: Search query — keywords, function names, concepts, error messages, etc.
        source: Limit search to a specific indexed source by name.
        language: Filter by language (e.g. "python", "typescript", "markdown").
        limit: Maximum number of results (default 10).
        mode: Search mode — "auto", "keyword", or "semantic".
    """
    store = await get_store()

    # Check if chunks exist (Phase 2 indexing done)
    has_chunks = await store.has_chunks()

    if not has_chunks or mode == "keyword":
        # Fall back to document-level FTS search
        results = await store.search(
            query, source_name=source, language=language, limit=limit,
        )
        if not results:
            return f"No results found for: {query}"

        lines = [f"Found {len(results)} result(s) for: {query} [keyword search]\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"--- Result {i} ---")
            lines.append(f"Source: {r.source_name}")
            lines.append(f"File: {r.relative_path}")
            lines.append(f"Language: {r.language}")
            lines.append(f"Snippet:\n{r.snippet}")
            lines.append("")
        return "\n".join(lines)

    # Chunk-level search (Phase 2)
    query_embedding = None
    if mode in ("auto", "semantic"):
        query_embedding = await get_query_embedding(query)

    if mode == "semantic" and not query_embedding:
        return "Semantic search unavailable: could not generate query embedding (is Ollama running?)"

    chunk_results = await store.hybrid_search(
        query,
        query_embedding=query_embedding,
        source_name=source,
        language=language,
        limit=limit,
    )

    if not chunk_results:
        return f"No results found for: {query}"

    search_type = "hybrid" if query_embedding else "keyword"
    lines = [f"Found {len(chunk_results)} result(s) for: {query} [{search_type} search]\n"]
    for i, r in enumerate(chunk_results, 1):
        lines.append(f"--- Result {i} ---")
        lines.append(f"Source: {r.source_name}")
        lines.append(f"File: {r.relative_path} (lines {r.start_line + 1}-{r.end_line + 1})")
        lines.append(f"Language: {r.language}")
        # Truncate long chunks for display
        content = r.content
        if len(content) > 800:
            content = content[:800] + "\n... (truncated)"
        lines.append(f"Content:\n{content}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
async def helios_index(
    path: str,
    name: str | None = None,
    embed: bool = True,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> str:
    """Index a directory with file scanning, chunking, and optional semantic embeddings.

    Three-phase pipeline:
    1. Scan files and store content (fast)
    2. Chunk documents into searchable segments (fast)
    3. Generate embeddings via Ollama for semantic search (optional, requires Ollama)

    Re-indexing the same path is incremental — only changed files are re-processed,
    and embeddings for unchanged chunks are preserved.

    Args:
        path: Absolute path to the directory to index.
        name: Friendly name for this source (defaults to directory name).
        embed: Generate embeddings for semantic search (default True, requires Ollama).
        embed_model: Ollama embedding model (default "nomic-embed-text").
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        return f"Error: '{path}' is not a directory"

    source_name = name or resolved.name
    store = await get_store()
    source_id = await store.add_source(source_name, str(resolved))

    # Phase 1: Scan files
    stats = await scan_directory(resolved, source_id, store)

    lines = [
        f"Indexed: {source_name} ({resolved})",
        f"  Files scanned: {stats.files_scanned}",
        f"  Files indexed: {stats.files_indexed}",
        f"  Files updated: {stats.files_updated}",
        f"  Files skipped: {stats.files_skipped}",
        f"  Files removed: {stats.files_removed}",
        f"  Total size: {stats.total_size:,} bytes",
    ]

    # Phase 2: Chunk documents
    docs = await store.get_documents_for_source(source_id)
    chunks_created = 0
    chunks_removed = 0
    for doc in docs:
        chunks = chunk_document(doc.content, doc.language)
        created, removed = await store.sync_document_chunks(
            doc.id, source_id,
            [(c.content, c.start_line, c.end_line) for c in chunks],
        )
        chunks_created += created
        chunks_removed += removed

    total_chunks = await store.get_chunk_count(source_id)
    lines.append(f"  Chunks: {total_chunks} ({chunks_created} new, {chunks_removed} removed)")

    # Phase 3: Generate embeddings
    if embed:
        to_embed = await store.get_chunks_without_embeddings(source_id)
        if to_embed:
            ollama_ok = await is_ollama_available()
            if ollama_ok:
                model_ok = await ensure_embed_model(embed_model)
                if model_ok:
                    texts = [content for _, content in to_embed]
                    embeddings = await generate_embeddings(texts, model=embed_model)
                    if embeddings:
                        pairs = list(zip([cid for cid, _ in to_embed], embeddings))
                        await store.store_embeddings(pairs)
                        lines.append(f"  Embeddings: {len(embeddings)} generated ({embed_model})")
                    else:
                        lines.append("  Embeddings: generation failed")
                else:
                    lines.append(f"  Embeddings: skipped (could not pull {embed_model})")
            else:
                lines.append("  Embeddings: skipped (Ollama not available)")
        else:
            existing = await store.get_embedding_count(source_id)
            lines.append(f"  Embeddings: {existing} (all up to date)")
    else:
        lines.append("  Embeddings: disabled")

    if stats.errors:
        lines.append(f"  Errors: {len(stats.errors)}")
        for err in stats.errors[:5]:
            lines.append(f"    - {err}")

    return "\n".join(lines)


@mcp.tool()
async def helios_status() -> str:
    """Show all indexed sources and their statistics.

    Lists every indexed codebase/directory with file counts,
    chunk counts, embedding status, and when it was last indexed.
    """
    store = await get_store()
    sources = await store.list_sources()

    if not sources:
        return "No sources indexed yet. Use helios_index to add a codebase."

    lines = [f"Indexed sources ({len(sources)}):\n"]
    for s in sources:
        size_mb = s.total_size / (1024 * 1024)
        chunks = await store.get_chunk_count(s.id)
        embeds = await store.get_embedding_count(s.id)
        lines.append(f"  {s.name}")
        lines.append(f"    Path: {s.path}")
        lines.append(f"    Files: {s.file_count}")
        lines.append(f"    Chunks: {chunks}")
        lines.append(f"    Embeddings: {embeds}/{chunks}")
        lines.append(f"    Size: {size_mb:.1f} MB")
        lines.append(f"    Last indexed: {s.updated_at.strftime('%Y-%m-%d %H:%M')}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
async def helios_read(
    path: str,
    source: str | None = None,
) -> str:
    """Read the full contents of an indexed file.

    Use after helios_search to get the complete content of a file
    that appeared in search results.

    Args:
        path: File path — absolute path or relative path within a source.
        source: Source name to search in (optional, searches all if omitted).
    """
    store = await get_store()

    # Try absolute path first
    doc = await store.get_document_by_path(path)
    if doc:
        return f"# {doc.relative_path} ({doc.language})\n\n{doc.content}"

    # Try as relative path
    doc = await store.find_document_by_relative_path(path, source_name=source)
    if doc:
        return f"# {doc.relative_path} ({doc.language})\n\n{doc.content}"

    return f"File not found: {path}"


@mcp.tool()
async def helios_explore(
    source: str,
    pattern: str | None = None,
) -> str:
    """Explore the file structure of an indexed codebase.

    Lists all files in a source, optionally filtered by glob pattern.
    Useful for understanding project layout before diving into search.

    Args:
        source: Name of the indexed source.
        pattern: Glob pattern to filter files (e.g. "*.py", "src/**/*.ts").
    """
    store = await get_store()
    files = await store.get_source_files(source, pattern=pattern)

    if not files:
        sources = await store.list_sources()
        source_names = [s.name for s in sources]
        if source not in source_names:
            avail = ", ".join(source_names) if source_names else "(none)"
            return f"Source '{source}' not found. Available sources: {avail}"
        return f"No files match pattern '{pattern}' in source '{source}'"

    header = f"Files in '{source}'"
    if pattern:
        header += f" matching '{pattern}'"
    header += f" ({len(files)}):"

    lines = [header, ""]
    for f in files:
        lines.append(f"  {f}")

    return "\n".join(lines)


@mcp.tool()
async def helios_remove(source: str) -> str:
    """Remove an indexed source and all its documents, chunks, and embeddings.

    Args:
        source: Name of the source to remove.
    """
    store = await get_store()
    removed = await store.remove_source(source)
    if removed:
        return f"Removed source '{source}' and all its indexed data."
    return f"Source not found: {source}"
