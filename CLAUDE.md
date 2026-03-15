# Helios — Claude Code Project Instructions

## Project Overview

Helios is an open-source, local-first context intelligence engine for AI coding agents. It indexes codebases, project dependencies, and documentation sites — then exposes everything as searchable tools via MCP (Model Context Protocol). Hybrid search combines FTS5 keyword matching with Ollama vector embeddings, fused via Reciprocal Rank Fusion.

Runs exclusively on open-weight models and local infrastructure. No cloud APIs required.

## Architecture

```
helios_index  ──→  Scan files    ──→  Chunk  ──→  Embed  ──→  SQLite + FTS5
helios_deps   ──→  Scan deps     ──→  Chunk  ──→  Embed  ──→  SQLite + FTS5
helios_web    ──→  Crawl URLs    ──→  Chunk  ──→  Embed  ──→  SQLite + FTS5
                                                                    ↑
helios_search ──→  Hybrid (FTS5 + vector + RRF)  ───────────────────┘
helios_context──→  Multi-source search + group by type  ────────────┘

File Watcher (watchdog) ──→ Auto-reindex on changes
```

## Package Structure

```
src/helios/
  indexing/          — Content intelligence pipeline
    store.py           ContentStore: SQLite + FTS5 for documents, chunks, embeddings
    scanner.py         Directory scanner (60+ languages, ignore patterns, binary detection)
    chunker.py         Language-aware chunking (code boundaries, markdown headings, paragraphs)
    embeddings.py      Ollama embedding generation, cosine similarity, f32 serialization
    crawler.py         URL crawler with HTML text extraction (stdlib html.parser + httpx)
    dependencies.py    Dependency detection (pyproject.toml, package.json, requirements.txt)
    watcher.py         File watcher with debounced auto-reindexing (watchdog)
  server/            — MCP server
    app.py             FastMCP server with 9 tools + shared pipeline helpers
    __main__.py        Entry point: python -m helios.server
  core/              — Task orchestration engine (legacy, retained for future use)
    engine.py          Three-tier orchestration loop (Orchestrator/Sub-agent/Refiner)
    models.py          Pydantic v2 data models (Session, SubTask, Message, etc.)
    session.py         In-memory session manager
    events.py          Type-safe event bus (pub/sub, sync + async callbacks)
  providers/         — LLM provider abstractions
    base.py            Provider protocol (structural subtyping, not ABC)
    ollama.py          Local models via Ollama
    groq.py            Fast cloud inference for open-weight models
    openai_compatible.py  LM Studio, vLLM, LocalAI, etc.
    llama_cpp.py       Direct GGUF file loading
    registry.py        Factory function for provider creation
  memory/            — Session persistence
    store.py           SessionStore: SQLite + JSON dual-write for sessions
    skills.py          Skill extraction and FTS5 recall
    search.py          FTS5 query builder
  cli/               — Typer CLI
    app.py             Root CLI app with command registration
    commands/          run, resume, serve, sessions, models, config, web
  web/               — FastAPI dashboard
    api.py             REST + WebSocket endpoints
    frontend/          HTML/CSS/JS dashboard
  config/            — Configuration
    loader.py          TOML + env var + CLI flag layered loading
    schema.py          Pydantic settings models
  tools/             — Orchestration tools
    base.py            Tool protocol
    registry.py        Role-based tool registry
    builtin/           7 built-in tools (decompose, complete, synthesis, etc.)
  models/            — HuggingFace Hub integration
    hub.py             Model search + GGUF download
```

## Tech Stack

- **Python 3.11+** with async-first design
- **SQLite + FTS5** for documents, chunks, embeddings, sessions, skills
- **MCP SDK** (`mcp` package with FastMCP) for agent integration
- **Ollama** for embeddings (`nomic-embed-text`) and LLM inference
- **Pydantic v2** for all data models and settings
- **aiosqlite** for async database access
- **watchdog** for file system monitoring
- **httpx** (transitive from mcp) for URL crawling
- **Typer** for CLI, **Rich** for console output
- **FastAPI + WebSocket** for web dashboard

## Code Conventions

- **Async-first**: All store/provider/embedding calls use `async/await`.
- **Protocol pattern**: Providers use a `Protocol` (not ABC) for structural subtyping.
- **Pydantic models**: All data structures use Pydantic v2. Config uses `pydantic-settings`.
- **Content-addressed chunking**: Chunks are keyed by content hash to preserve embeddings on re-index.
- **Graceful degradation**: If Ollama isn't running, search falls back to FTS5-only. No hard failures.
- **No hardcoded secrets**: API keys from env vars only.
- **Open-weight only**: No proprietary cloud providers (Anthropic, OpenAI removed). Ollama, Groq, OpenAI-compatible, llama.cpp.

## MCP Tools (9 total)

| Tool | Description |
|------|-------------|
| `helios_index` | Index a directory (scan → chunk → embed). Auto-watches for changes. |
| `helios_deps` | Auto-detect and index project dependencies from manifest files. |
| `helios_web` | Crawl and index a documentation site or web page. |
| `helios_search` | Hybrid keyword + semantic search across all sources. |
| `helios_context` | Multi-source context assembly grouped by type (project/deps/docs). |
| `helios_status` | Show all indexed sources with stats. |
| `helios_read` | Read full contents of an indexed file or page. |
| `helios_explore` | Browse file structure of a source. |
| `helios_remove` | Remove a source and all its data. |

## Key Design Decisions

1. **Local-first** — All data in SQLite on the user's machine. No cloud services.
2. **MCP-native** — Distributes via Model Context Protocol to any compatible agent.
3. **Hybrid search** — FTS5 + vector embeddings + Reciprocal Rank Fusion.
4. **Content-addressed chunks** — Re-indexing preserves embeddings for unchanged code.
5. **Dependency intelligence** — Indexes actual installed source code from site-packages/node_modules.
6. **URL crawling** — Indexes documentation sites using only stdlib + httpx (no heavy deps).
7. **Live watching** — watchdog-based file monitoring with debounced auto-reindexing.
8. **Graceful degradation** — Works without Ollama (keyword-only search), without watchdog (no auto-watch).
9. **Shared pipeline** — Directory, dependency, and URL sources all share chunk + embed helpers.

## Database Schema

Two SQLite databases in `~/.helios/`:

**`index.db`** (content intelligence):
- `sources` — indexed sources (name, path, type, stats)
- `documents` — file/page content with content hash
- `documents_fts` — FTS5 index on documents
- `chunks` — document chunks with optional embedding blobs
- `chunks_fts` — FTS5 index on chunks

**`helios.db`** (sessions/skills — legacy orchestration):
- `sessions`, `task_exchanges`, `code_files`
- `sessions_fts`, `exchanges_fts`

## Running & Testing

```bash
# Install
pip install -e ".[dev]"

# Run MCP server (stdio)
helios serve

# Or directly
python -m helios.server

# Run tests
pytest                    # 152 tests
pytest tests/test_indexing.py -v  # indexing-specific tests
```

## Configuration

Config layers (each overrides the previous):
1. Built-in defaults
2. Config file (`helios.toml` or `~/.config/helios/config.toml`)
3. Environment variables (`GROQ_API_KEY`, etc.)
4. CLI flags

## Dependencies

Core: `ollama`, `groq`, `openai` (for openai-compatible), `huggingface-hub`, `rich`, `typer`, `pydantic`, `pydantic-settings`, `aiosqlite`, `tavily-python`, `mcp`, `watchdog`

Optional: `llama-cpp-python` (local GGUF), `fastapi`+`uvicorn`+`websockets` (web dashboard)
