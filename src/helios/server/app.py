"""Helios MCP server — exposes context intelligence tools to AI coding agents."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP

from helios.indexing.api_extractor import extract_api_surface
from helios.indexing.chunker import chunk_document
from helios.indexing.claude_md_generator import generate_claude_md, write_claude_md
from helios.indexing.stack_analyzer import analyze_stack
from helios.indexing.crawler import crawl_and_index
from helios.indexing.dependencies import (
    detect_dependencies,
    estimate_package_size,
    MAX_PACKAGE_FILES,
)
from helios.indexing.embeddings import (
    DEFAULT_EMBED_MODEL,
    ensure_embed_model,
    generate_embeddings,
    get_query_embedding,
    is_ollama_available,
)
from helios.indexing.scanner import scan_directory
from helios.indexing.store import ContentStore

logger = logging.getLogger(__name__)

# Module-level singletons
_store: ContentStore | None = None
_watcher = None  # FileWatcher | None


async def get_store() -> ContentStore:
    global _store
    if _store is None:
        _store = ContentStore()
        await _store.initialize()
        await _start_watcher(_store)
    return _store


# --------------------------------------------------------------------------- #
#  File watcher
# --------------------------------------------------------------------------- #


async def _start_watcher(store: ContentStore) -> None:
    global _watcher
    try:
        from helios.indexing.watcher import FileWatcher
    except ImportError:
        return

    async def on_change(source_id: str) -> None:
        await _reindex_source(source_id, store)

    _watcher = FileWatcher(on_change=on_change)
    sources = await store.list_sources()
    for s in sources:
        if s.source_type != "web" and Path(s.path).is_dir():
            _watcher.watch(s.id, s.path)
    if sources:
        await _watcher.start()


async def _watch_source(source_id: str, path: str) -> None:
    global _watcher
    if _watcher is None:
        return
    _watcher.watch(source_id, path)
    if not _watcher._running:
        await _watcher.start()


async def _reindex_source(source_id: str, store: ContentStore) -> None:
    sources = await store.list_sources()
    source = next((s for s in sources if s.id == source_id), None)
    if not source or not Path(source.path).is_dir():
        return
    await scan_directory(source.path, source_id, store)
    await _chunk_source(source_id, store)
    await _embed_source(source_id, store)
    # Re-extract API surface if this source was previously onboarded
    if await store.has_api_endpoints(source_id):
        surface = await extract_api_surface(source_id, store)
        endpoints = [ep.__dict__ for ep in surface.endpoints]
        await store.store_api_endpoints(source_id, endpoints)


# --------------------------------------------------------------------------- #
#  Shared pipeline helpers
# --------------------------------------------------------------------------- #


async def _chunk_source(source_id: str, store: ContentStore) -> tuple[int, int]:
    """Chunk all documents for a source. Returns (created, removed)."""
    docs = await store.get_documents_for_source(source_id)
    total_created = total_removed = 0
    for doc in docs:
        chunks = chunk_document(doc.content, doc.language)
        created, removed = await store.sync_document_chunks(
            doc.id, source_id,
            [(c.content, c.start_line, c.end_line) for c in chunks],
        )
        total_created += created
        total_removed += removed
    return total_created, total_removed


async def _embed_source(
    source_id: str,
    store: ContentStore,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> tuple[int, str]:
    """Generate embeddings for chunks without them. Returns (count, status_msg)."""
    to_embed = await store.get_chunks_without_embeddings(source_id)
    if not to_embed:
        existing = await store.get_embedding_count(source_id)
        return existing, f"{existing} (all up to date)"

    if not await is_ollama_available():
        return 0, "Ollama not available"
    if not await ensure_embed_model(embed_model):
        return 0, f"could not pull {embed_model}"

    texts = [content for _, content in to_embed]
    embeddings = await generate_embeddings(texts, model=embed_model)
    if not embeddings:
        return 0, "generation failed"

    pairs = list(zip([cid for cid, _ in to_embed], embeddings))
    await store.store_embeddings(pairs)
    return len(embeddings), f"{len(embeddings)} generated ({embed_model})"


async def _run_index_pipeline(
    path: Path,
    source_name: str,
    store: ContentStore,
    source_type: str = "directory",
    embed: bool = True,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> dict:
    """Full directory indexing pipeline: scan → chunk → embed."""
    source_id = await store.add_source(source_name, str(path), source_type)
    scan_stats = await scan_directory(path, source_id, store)
    chunks_created, chunks_removed = await _chunk_source(source_id, store)
    total_chunks = await store.get_chunk_count(source_id)

    embed_status = "disabled"
    embed_count = 0
    if embed:
        embed_count, embed_status = await _embed_source(source_id, store, embed_model)

    if source_type != "web":
        await _watch_source(source_id, str(path))

    return {
        "source_id": source_id,
        "source_name": source_name,
        "scan": scan_stats,
        "chunks_created": chunks_created,
        "chunks_removed": chunks_removed,
        "chunks_total": total_chunks,
        "embed_count": embed_count,
        "embed_status": embed_status,
    }


# --------------------------------------------------------------------------- #
#  MCP Server
# --------------------------------------------------------------------------- #

mcp = FastMCP(
    "helios",
    instructions=(
        "Helios is a local-first context intelligence engine. "
        "It indexes codebases, project dependencies, and documentation "
        "on the user's machine and makes them searchable via hybrid "
        "keyword + semantic search. All data stays local.\n\n"
        "Workflow:\n"
        "1. helios_onboard — one-shot project setup: index codebase, "
        "analyze dependencies, extract API surface, generate CLAUDE.md "
        "with API contract and FE development rules\n"
        "2. helios_search — find relevant code/docs across all sources\n"
        "3. helios_context — assemble multi-source context for a task\n"
        "4. helios_api — query extracted API endpoints\n"
        "5. helios_read — get full file contents\n"
        "6. helios_explore — browse project structure\n\n"
        "Individual tools (if you need finer control):\n"
        "- helios_index — index a codebase or directory\n"
        "- helios_deps — auto-detect and index project dependencies\n"
        "- helios_web — index documentation sites or web pages"
    ),
)


# --------------------------------------------------------------------------- #
#  MCP Resources
# --------------------------------------------------------------------------- #


@mcp.resource("helios://status")
async def resource_status() -> str:
    """Current Helios index status — sources, file counts, chunk counts, embeddings."""
    store = await get_store()
    sources = await store.list_sources()
    if not sources:
        return "No sources indexed."
    lines: list[str] = []
    for s in sources:
        chunks = await store.get_chunk_count(s.id)
        embeds = await store.get_embedding_count(s.id)
        lines.append(f"{s.name} ({s.source_type}): {s.file_count} files, {chunks} chunks, {embeds} embeddings")
    return "\n".join(lines)


@mcp.resource("helios://sources")
async def resource_sources() -> str:
    """List of all indexed source names for use with helios_search source parameter."""
    store = await get_store()
    sources = await store.list_sources()
    return "\n".join(s.name for s in sources) if sources else "No sources indexed."


# --------------------------------------------------------------------------- #
#  Tools
# --------------------------------------------------------------------------- #


@mcp.tool()
async def helios_onboard(
    path: str,
    name: str | None = None,
    embed: bool = True,
    embed_model: str = DEFAULT_EMBED_MODEL,
    include_dev_deps: bool = False,
    write_claude: bool = True,
) -> str:
    """One-shot project onboarding: index codebase, analyze dependencies, extract API surface, and generate CLAUDE.md.

    This is the recommended first step when starting work on a project.
    It indexes all source files, detects and indexes dependencies,
    extracts the backend API surface (endpoints, methods, types),
    and generates a CLAUDE.md with API contract and frontend development rules.

    Supported frameworks: Next.js App Router, FastAPI, NestJS, Express.

    Args:
        path: Absolute path to the project directory.
        name: Friendly name for this source (defaults to directory name).
        embed: Generate embeddings for semantic search (default True).
        embed_model: Ollama embedding model (default "nomic-embed-text").
        include_dev_deps: Include dev/test dependencies (default False).
        write_claude: Write CLAUDE.md to project root (default True).
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        return f"Error: '{path}' is not a directory"

    source_name = name or resolved.name
    store = await get_store()

    # 1. Index the codebase
    result = await _run_index_pipeline(
        resolved, source_name, store, embed=embed, embed_model=embed_model,
    )
    stats = result["scan"]

    # 2. Detect and index dependencies
    deps_result = detect_dependencies(resolved, include_dev=include_dev_deps)
    dep_count = 0
    ecosystems = deps_result.ecosystems if deps_result.dependencies else []

    for dep in deps_result.dependencies:
        if not dep.source_path:
            continue
        file_count = estimate_package_size(dep.source_path)
        if file_count > MAX_PACKAGE_FILES:
            continue
        await _run_index_pipeline(
            dep.source_path, f"dep:{dep.name}", store,
            source_type="dependency", embed=embed, embed_model=embed_model,
        )
        dep_count += 1

    # 3. Extract API surface
    surface = await extract_api_surface(result["source_id"], store)
    endpoints = [ep.__dict__ for ep in surface.endpoints]
    await store.store_api_endpoints(result["source_id"], endpoints)

    # 4. Analyze project stack (FE libraries, types, API clients, patterns)
    stack = await analyze_stack(result["source_id"], resolved, store)

    # 5. Generate and write CLAUDE.md
    claude_md_path = ""
    helios_section = generate_claude_md(
        project_name=source_name,
        project_path=str(resolved),
        api_surface=surface,
        ecosystems=ecosystems,
        file_count=stats.files_indexed,
        dep_count=dep_count,
        stack=stack,
    )

    if write_claude:
        claude_md_path = write_claude_md(str(resolved), helios_section)

    # 6. Build summary
    method_counts: dict[str, int] = {}
    for ep in surface.endpoints:
        method_counts[ep.method] = method_counts.get(ep.method, 0) + 1
    method_summary = ", ".join(f"{c} {m}" for m, c in sorted(method_counts.items()))

    lines = [
        f"Onboarded: {source_name} ({resolved})",
        "",
        "  Codebase:",
        f"    Files indexed: {stats.files_indexed}",
        f"    Chunks: {result['chunks_total']} ({result['chunks_created']} new)",
        f"    Embeddings: {result['embed_status']}",
        "    Live watch: enabled",
        "",
        "  Dependencies:",
        f"    Ecosystems: {', '.join(ecosystems) if ecosystems else 'none detected'}",
        f"    Found: {len(deps_result.dependencies)} dependencies",
        f"    Indexed: {dep_count} packages",
        "",
        "  API Surface:",
        f"    Frameworks: {', '.join(surface.frameworks) if surface.frameworks else 'none detected'}",
        f"    Endpoints: {len(surface.endpoints)}" + (f" ({method_summary})" if method_summary else ""),
    ]

    # Stack analysis summary
    if stack:
        lines.append("")
        lines.append("  Stack Analysis:")
        if stack.libraries:
            lib_cats = set(lib.category.replace("_", " ").title() for lib in stack.libraries)
            lines.append(f"    FE libraries: {len(stack.libraries)} ({', '.join(sorted(lib_cats))})")
        if stack.api_clients:
            lines.append(f"    API clients: {len(stack.api_clients)} found")
        if stack.type_defs:
            lines.append(f"    Types/schemas: {len(stack.type_defs)} extracted")
        if stack.patterns:
            p = stack.patterns
            if p.naming_convention != "not detected":
                lines.append(f"    Naming: {p.naming_convention}")
            if p.key_directories:
                lines.append(f"    Structure: {', '.join(p.key_directories[:8])}")

    if write_claude and claude_md_path:
        lines.extend([
            "",
            "  CLAUDE.md:",
            f"    Written to: {claude_md_path}",
            "    Contains: API contract, tech stack, types, conventions, FE rules",
        ])

    return "\n".join(lines)


@mcp.tool()
async def helios_api(
    source: str | None = None,
    method: str | None = None,
    path_contains: str | None = None,
) -> str:
    """Query extracted API endpoints from onboarded projects.

    Returns the stored API surface extracted by helios_onboard.
    Use this to check if a specific endpoint exists before writing frontend code.

    Args:
        source: Filter by source name (e.g. "myproject").
        method: Filter by HTTP method (e.g. "GET", "POST").
        path_contains: Filter endpoints whose path contains this string.
    """
    store = await get_store()

    # Find source_id if source name given
    source_id = None
    if source:
        sources = await store.list_sources()
        match = next((s for s in sources if s.name == source), None)
        if match:
            source_id = match.id

    endpoints = await store.get_api_endpoints(source_id=source_id)

    if not endpoints:
        if source:
            return f"No API endpoints found for source '{source}'. Run helios_onboard first."
        return "No API endpoints stored. Run helios_onboard on a project first."

    # Apply filters
    if method:
        endpoints = [e for e in endpoints if e["method"] == method.upper()]
    if path_contains:
        endpoints = [e for e in endpoints if path_contains in e["path"]]

    if not endpoints:
        return "No endpoints match the given filters."

    lines = [f"API Endpoints ({len(endpoints)}):", ""]
    lines.append("| Method | Path | Handler | File | Request | Response | Framework |")
    lines.append("|--------|------|---------|------|---------|----------|-----------|")

    for ep in endpoints:
        req = ep["request_type"] or "-"
        res = ep["response_type"] or "-"
        lines.append(
            f"| {ep['method']} | {ep['path']} | {ep['handler']} "
            f"| {ep['file_path']} | {req} | {res} | {ep['framework']} |"
        )

    return "\n".join(lines)


@mcp.tool()
async def helios_search(
    query: str,
    source: str | None = None,
    language: str | None = None,
    limit: int = 10,
    mode: str = "auto",
) -> str:
    """Search across all indexed codebases, dependencies, and documentation.

    Supports three search modes:
    - "auto" (default): hybrid keyword + semantic if embeddings exist
    - "keyword": FTS5 full-text search only (fast, exact matches)
    - "semantic": vector similarity only (conceptually similar content)

    Args:
        query: Search query — keywords, function names, concepts, etc.
        source: Limit to a specific source (e.g. "myproject", "dep:fastapi", "web:docs.fastapi.tiangolo.com").
        language: Filter by language (e.g. "python", "typescript").
        limit: Maximum results (default 10).
        mode: Search mode — "auto", "keyword", or "semantic".
    """
    store = await get_store()
    has_chunks = await store.has_chunks()

    if not has_chunks or mode == "keyword":
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

    query_embedding = None
    if mode in ("auto", "semantic"):
        query_embedding = await get_query_embedding(query)
    if mode == "semantic" and not query_embedding:
        return "Semantic search unavailable (is Ollama running?)"

    chunk_results = await store.hybrid_search(
        query, query_embedding=query_embedding,
        source_name=source, language=language, limit=limit,
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
        content = r.content[:800] + "\n... (truncated)" if len(r.content) > 800 else r.content
        lines.append(f"Content:\n{content}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
async def helios_context(
    task: str,
    limit: int = 15,
) -> str:
    """Assemble comprehensive, multi-source context for a task in one call.

    Unlike helios_search (flat list of results), this tool searches ALL indexed
    sources — project code, dependencies, and documentation — then organizes
    results by source type. Ideal for complex tasks that span multiple sources.

    Example: "Add OAuth2 login to the FastAPI app" returns your project's
    existing auth code, FastAPI's OAuth2 implementation from dep source, and
    any indexed documentation — all grouped and ready to use.

    Args:
        task: Description of what you're trying to do.
        limit: Maximum total context items (default 15).
    """
    store = await get_store()
    has_chunks = await store.has_chunks()

    if not has_chunks:
        return "No indexed content. Use helios_index, helios_deps, or helios_web first."

    query_embedding = await get_query_embedding(task)

    results = await store.hybrid_search(
        task, query_embedding=query_embedding, limit=limit * 2,
    )

    if not results:
        return f"No relevant context found for: {task}"

    # Group by source
    grouped: dict[str, list] = defaultdict(list)
    for r in results:
        grouped[r.source_name].append(r)

    # Categorize sources
    sources = await store.list_sources()
    source_types = {s.name: s.source_type for s in sources}

    project_sources: dict[str, list] = {}
    dep_sources: dict[str, list] = {}
    web_sources: dict[str, list] = {}

    for name, items in grouped.items():
        stype = source_types.get(name, "directory")
        if stype == "dependency":
            dep_sources[name] = items
        elif stype == "web":
            web_sources[name] = items
        else:
            project_sources[name] = items

    lines = [f"Context for: {task}\n"]
    total = 0

    if project_sources:
        lines.append("=== Your Project ===\n")
        for name, items in project_sources.items():
            for r in items:
                if total >= limit:
                    break
                lines.append(f"[{name}] {r.relative_path}:{r.start_line + 1}-{r.end_line + 1}")
                content = r.content[:600] + "\n..." if len(r.content) > 600 else r.content
                lines.append(content)
                lines.append("")
                total += 1

    if dep_sources and total < limit:
        lines.append("=== Dependencies ===\n")
        for name, items in dep_sources.items():
            for r in items:
                if total >= limit:
                    break
                lines.append(f"[{name}] {r.relative_path}:{r.start_line + 1}-{r.end_line + 1}")
                content = r.content[:600] + "\n..." if len(r.content) > 600 else r.content
                lines.append(content)
                lines.append("")
                total += 1

    if web_sources and total < limit:
        lines.append("=== Documentation ===\n")
        for name, items in web_sources.items():
            for r in items:
                if total >= limit:
                    break
                lines.append(f"[{name}] {r.relative_path}")
                content = r.content[:600] + "\n..." if len(r.content) > 600 else r.content
                lines.append(content)
                lines.append("")
                total += 1

    return "\n".join(lines)


@mcp.tool()
async def helios_index(
    path: str,
    name: str | None = None,
    embed: bool = True,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> str:
    """Index a directory with file scanning, chunking, and optional embeddings.

    After indexing, the directory is watched for changes and auto-reindexed.

    Args:
        path: Absolute path to the directory to index.
        name: Friendly name for this source (defaults to directory name).
        embed: Generate embeddings for semantic search (default True).
        embed_model: Ollama embedding model (default "nomic-embed-text").
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        return f"Error: '{path}' is not a directory"

    source_name = name or resolved.name
    store = await get_store()
    result = await _run_index_pipeline(
        resolved, source_name, store, embed=embed, embed_model=embed_model,
    )

    stats = result["scan"]
    return "\n".join([
        f"Indexed: {source_name} ({resolved})",
        f"  Files scanned: {stats.files_scanned}",
        f"  Files indexed: {stats.files_indexed}",
        f"  Files updated: {stats.files_updated}",
        f"  Files skipped: {stats.files_skipped}",
        f"  Files removed: {stats.files_removed}",
        f"  Total size: {stats.total_size:,} bytes",
        f"  Chunks: {result['chunks_total']} ({result['chunks_created']} new, {result['chunks_removed']} removed)",
        f"  Embeddings: {result['embed_status']}",
        "  Live watch: enabled",
    ] + ([f"  Errors: {len(stats.errors)}"] + [f"    - {e}" for e in stats.errors[:5]] if stats.errors else []))


@mcp.tool()
async def helios_web(
    url: str,
    name: str | None = None,
    max_pages: int = 50,
    max_depth: int = 3,
    embed: bool = True,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> str:
    """Index a documentation site or web page for searching.

    Crawls the URL and linked pages (same domain), extracts text content,
    and indexes it. Great for API docs, guides, tutorials, and references.
    Crawled content is chunked and embedded just like local code.

    Args:
        url: Starting URL to crawl (e.g. "https://docs.pydantic.dev").
        name: Source name (defaults to "web:<domain>").
        max_pages: Maximum pages to crawl (default 50).
        max_depth: Maximum link depth from start URL (default 3).
        embed: Generate embeddings (default True).
        embed_model: Embedding model (default "nomic-embed-text").
    """
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return f"Error: invalid URL '{url}'"

    source_name = name or f"web:{parsed.netloc}"
    store = await get_store()
    source_id = await store.add_source(source_name, url, "web")

    # Crawl
    crawl_stats = await crawl_and_index(url, source_id, store, max_pages, max_depth)

    # Chunk and embed
    chunks_created, chunks_removed = await _chunk_source(source_id, store)
    total_chunks = await store.get_chunk_count(source_id)

    embed_status = "disabled"
    if embed:
        _, embed_status = await _embed_source(source_id, store, embed_model)

    lines = [
        f"Indexed: {source_name} ({url})",
        f"  Pages crawled: {crawl_stats.pages_crawled}",
        f"  Pages skipped: {crawl_stats.pages_skipped}",
        f"  Total text: {crawl_stats.total_chars:,} chars",
        f"  Chunks: {total_chunks} ({chunks_created} new, {chunks_removed} removed)",
        f"  Embeddings: {embed_status}",
    ]
    if crawl_stats.errors:
        lines.append(f"  Errors: {len(crawl_stats.errors)}")
        for err in crawl_stats.errors[:5]:
            lines.append(f"    - {err}")

    return "\n".join(lines)


@mcp.tool()
async def helios_deps(
    path: str,
    include_dev: bool = False,
    embed: bool = True,
    embed_model: str = DEFAULT_EMBED_MODEL,
    max_files: int = MAX_PACKAGE_FILES,
) -> str:
    """Detect and index all project dependencies for version-correct search.

    Parses dependency files (pyproject.toml, package.json, requirements.txt),
    finds the installed source code, and indexes each dependency.
    Each dependency is indexed as "dep:<package>".

    Args:
        path: Project directory containing dependency/manifest files.
        include_dev: Include dev/test dependencies (default False).
        embed: Generate embeddings (default True).
        embed_model: Embedding model (default "nomic-embed-text").
        max_files: Skip packages larger than this (default 500).
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        return f"Error: '{path}' is not a directory"

    store = await get_store()
    deps_result = detect_dependencies(resolved, include_dev=include_dev)

    if not deps_result.dependencies:
        return f"No dependency files found in {resolved}"

    lines = [
        f"Dependencies for: {resolved.name}",
        f"  Ecosystems: {', '.join(deps_result.ecosystems)}",
        f"  Found: {len(deps_result.dependencies)} dependencies",
        f"  Resolved: {deps_result.resolved} with source paths",
        "",
    ]

    indexed = 0
    skipped_large: list[str] = []

    for dep in deps_result.dependencies:
        if not dep.source_path:
            continue
        file_count = estimate_package_size(dep.source_path)
        if file_count > max_files:
            skipped_large.append(f"{dep.name} ({file_count} files)")
            continue

        result = await _run_index_pipeline(
            dep.source_path, f"dep:{dep.name}", store,
            source_type="dependency", embed=embed, embed_model=embed_model,
        )
        lines.append(
            f"  dep:{dep.name} — {result['scan'].files_indexed} files, "
            f"{result['chunks_total']} chunks, embeds: {result['embed_status']}"
        )
        indexed += 1

    lines.insert(4, f"  Indexed: {indexed} packages")

    if skipped_large:
        lines.append(f"\n  Skipped (>{max_files} files): {', '.join(skipped_large)}")
    if deps_result.unresolved:
        short = deps_result.unresolved[:10]
        more = f" (+{len(deps_result.unresolved) - 10} more)" if len(deps_result.unresolved) > 10 else ""
        lines.append(f"  Not found: {', '.join(short)}{more}")

    # Report discovered documentation URLs
    docs_deps = [d for d in deps_result.dependencies if d.docs_url]
    if docs_deps:
        lines.append(f"\n  Documentation URLs discovered ({len(docs_deps)}):")
        for d in docs_deps[:10]:
            lines.append(f"    {d.name}: {d.docs_url}")
        lines.append("  Use helios_web(url=...) to index any of these.")

    return "\n".join(lines)


@mcp.tool()
async def helios_status() -> str:
    """Show all indexed sources — projects, dependencies, and documentation."""
    store = await get_store()
    sources = await store.list_sources()

    if not sources:
        return "No sources indexed yet. Use helios_index, helios_deps, or helios_web."

    projects = [s for s in sources if s.source_type not in ("dependency", "web")]
    deps = [s for s in sources if s.source_type == "dependency"]
    webs = [s for s in sources if s.source_type == "web"]

    lines: list[str] = []

    if projects:
        lines.append(f"Projects ({len(projects)}):\n")
        for s in projects:
            chunks = await store.get_chunk_count(s.id)
            embeds = await store.get_embedding_count(s.id)
            lines.append(f"  {s.name}")
            lines.append(f"    Path: {s.path}")
            lines.append(f"    Files: {s.file_count} | Chunks: {chunks} | Embeddings: {embeds}/{chunks}")
            lines.append(f"    Size: {s.total_size / 1048576:.1f} MB | Updated: {s.updated_at.strftime('%Y-%m-%d %H:%M')}")
            lines.append("")

    if deps:
        tf = sum(s.file_count for s in deps)
        tc = sum([await store.get_chunk_count(s.id) for s in deps])
        te = sum([await store.get_embedding_count(s.id) for s in deps])
        lines.append(f"Dependencies ({len(deps)}):")
        lines.append(f"  Total: {tf} files, {tc} chunks, {te} embeddings\n")
        for s in deps:
            c = await store.get_chunk_count(s.id)
            lines.append(f"  {s.name} — {s.file_count} files, {c} chunks")
        lines.append("")

    if webs:
        lines.append(f"Documentation ({len(webs)}):\n")
        for s in webs:
            chunks = await store.get_chunk_count(s.id)
            embeds = await store.get_embedding_count(s.id)
            lines.append(f"  {s.name}")
            lines.append(f"    URL: {s.path}")
            lines.append(f"    Pages: {s.file_count} | Chunks: {chunks} | Embeddings: {embeds}/{chunks}")
            lines.append("")

    return "\n".join(lines)


@mcp.tool()
async def helios_read(path: str, source: str | None = None) -> str:
    """Read the full contents of an indexed file or web page.

    Args:
        path: File path (absolute or relative) or URL.
        source: Source name (optional).
    """
    store = await get_store()
    doc = await store.get_document_by_path(path)
    if doc:
        return f"# {doc.relative_path} ({doc.language})\n\n{doc.content}"
    doc = await store.find_document_by_relative_path(path, source_name=source)
    if doc:
        return f"# {doc.relative_path} ({doc.language})\n\n{doc.content}"
    return f"File not found: {path}"


@mcp.tool()
async def helios_explore(source: str, pattern: str | None = None) -> str:
    """Explore the file structure of an indexed source.

    Args:
        source: Name of the indexed source.
        pattern: Glob pattern to filter files (e.g. "*.py").
    """
    store = await get_store()
    files = await store.get_source_files(source, pattern=pattern)
    if not files:
        sources = await store.list_sources()
        names = [s.name for s in sources]
        if source not in names:
            return f"Source '{source}' not found. Available: {', '.join(names) or '(none)'}"
        return f"No files match '{pattern}' in '{source}'"
    header = f"Files in '{source}'" + (f" matching '{pattern}'" if pattern else "") + f" ({len(files)}):"
    return "\n".join([header, ""] + [f"  {f}" for f in files])


@mcp.tool()
async def helios_remove(source: str) -> str:
    """Remove an indexed source and all its data.

    Args:
        source: Name of the source to remove.
    """
    store = await get_store()
    if await store.remove_source(source):
        return f"Removed '{source}' and all its indexed data."
    return f"Source not found: {source}"
