# Helios

Open-source, local-first context intelligence engine for AI coding agents. Indexes your codebases, project dependencies, and documentation — makes everything searchable via hybrid keyword + semantic search through MCP. All data stays on your machine.

Think of it as a self-hosted, free alternative to [Nia](https://trynia.ai) — with the added ability to index your actual installed dependency source code for version-correct search.

## What It Does

```
Your Project Code  ──→  helios_index   ──┐
Installed Packages ──→  helios_deps    ──┤──→  Chunk  ──→  Embed  ──→  SQLite + FTS5
Documentation URLs ──→  helios_web     ──┘         (Ollama)              ↑
                                                                         │
Any AI Agent  ←──  MCP  ←──  helios_search / helios_context  ───────────┘
```

Helios runs as an MCP server. Any agent that supports MCP (Claude Code, Cursor, Windsurf, Continue.dev, Cline, etc.) connects to it and gets access to:

- **Your project code** — indexed, chunked, and searchable
- **Every library you use** — the actual installed source code from `site-packages` / `node_modules`, not stale training data
- **Documentation sites** — crawled and indexed locally
- **Hybrid search** — FTS5 keyword matching + vector semantic similarity via Ollama embeddings, fused with Reciprocal Rank Fusion
- **Live file watching** — auto-reindexes when your code changes

## Quick Start

```bash
# Clone and install
git clone https://github.com/your-org/helios.git
cd helios
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Start the MCP server (for testing)
helios serve
```

### Connect to Claude Code

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "helios": {
      "type": "stdio",
      "command": "/path/to/helios/.venv/bin/python",
      "args": ["-m", "helios.server"]
    }
  }
}
```

Or for any MCP-compatible agent, point it at `python -m helios.server` via stdio transport.

### Index Your Project

Once connected, the agent can use these tools:

```
# Index a codebase (auto-watches for changes)
helios_index(path="/path/to/myproject")

# Index all project dependencies (finds site-packages/node_modules automatically)
helios_deps(path="/path/to/myproject")

# Index a documentation site
helios_web(url="https://docs.pydantic.dev")

# Search across everything
helios_search(query="how to validate nested models")

# Assemble multi-source context for a task
helios_context(task="Add OAuth2 login to the FastAPI app")
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `helios_index` | Index a directory with scan + chunk + embed pipeline. Auto-watches for changes. |
| `helios_deps` | Auto-detect and index all project dependencies from manifest files. |
| `helios_web` | Crawl and index a documentation site or web page. |
| `helios_search` | Hybrid keyword + semantic search across all indexed sources. |
| `helios_context` | Multi-source context assembly — searches project, deps, and docs, returns results grouped by source type. |
| `helios_status` | Show all indexed sources with stats (files, chunks, embeddings). |
| `helios_read` | Read the full contents of an indexed file or web page. |
| `helios_explore` | Browse the file structure of an indexed source. |
| `helios_remove` | Remove an indexed source and all its data. |

## How Search Works

Helios uses a three-layer search architecture:

1. **FTS5 keyword search** — SQLite full-text search with BM25 ranking. Fast, exact matches.
2. **Vector semantic search** — Ollama embeddings (`nomic-embed-text`, 768 dims) with cosine similarity. Finds conceptually similar content even with different wording.
3. **Reciprocal Rank Fusion** — Merges keyword and semantic rankings into a single result set. Gets the best of both.

Search modes via `helios_search`:
- `"auto"` (default) — hybrid if embeddings exist, keyword-only otherwise
- `"keyword"` — FTS5 only
- `"semantic"` — vector only

## Dependency Intelligence

`helios_deps` is the killer feature. It:

1. Parses your dependency files (`pyproject.toml`, `requirements.txt`, `package.json`)
2. Finds the installed source code in your venv's `site-packages` or `node_modules`
3. Indexes each package with the full pipeline (scan → chunk → embed)
4. Names them `dep:<package>` so you can search within specific libraries

This means your AI agent has access to the **actual installed version** of every library — not stale training data. No more hallucinated APIs.

```
# Search within a specific dependency
helios_search(query="OAuth2PasswordBearer", source="dep:fastapi")

# Search across all dependencies
helios_search(query="connection pool configuration")
```

## Live File Watching

After indexing, Helios watches directories for changes using `watchdog`. When files change:

1. Debounces changes (2-second window)
2. Re-scans only modified files (content-hash based)
3. Re-chunks only changed documents
4. Generates embeddings only for new chunks

The watcher runs as a background task inside the MCP server process.

## Architecture

```
src/helios/
  indexing/        Content intelligence pipeline
    store.py       SQLite + FTS5 for documents, chunks, and embeddings
    scanner.py     Directory scanner with 60+ language mappings
    chunker.py     Language-aware document chunking
    embeddings.py  Ollama embedding generation + cosine similarity
    crawler.py     URL crawler with HTML text extraction
    dependencies.py Dependency detection and source path resolution
    watcher.py     File watcher with debounced auto-reindexing
  server/          MCP server
    app.py         FastMCP server with 9 tools
    __main__.py    Entry point (python -m helios.server)
  core/            Task orchestration engine (Orchestrator/Sub-agent/Refiner)
  providers/       LLM provider abstractions (Ollama, Groq, OpenAI-compatible, llama.cpp)
  memory/          Session persistence (SQLite + JSON), skills system
  cli/             Typer CLI (run, resume, serve, sessions, models, config, web)
  web/             FastAPI dashboard with WebSocket streaming
  config/          TOML + env var + CLI flag config loading
  tools/           Tool protocol and 7 built-in orchestration tools
  models/          HuggingFace Hub integration (search + GGUF download)
```

## Tech Stack

- **Python 3.11+** with async-first design
- **SQLite + FTS5** for all persistence and full-text search
- **MCP SDK** (`mcp` package) for agent integration
- **Ollama** for local embeddings (`nomic-embed-text`) and LLM inference
- **Pydantic v2** for all data models
- **aiosqlite** for async database access
- **watchdog** for file system monitoring
- **httpx** for URL crawling (transitive dep from mcp)
- **Typer** for CLI, **Rich** for console output
- **FastAPI** for web dashboard (optional)

## Configuration

Helios uses layered configuration (each overrides the previous):

1. Built-in defaults
2. Config file (`helios.toml` or `~/.config/helios/config.toml`)
3. Environment variables (`GROQ_API_KEY`, etc.)
4. CLI flags

See `helios.toml.example` for all options.

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

152 tests covering indexing, chunking, search, embeddings, dependencies, crawler, and file watcher.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.ai) (for embeddings and local LLM inference) — optional but recommended
- No cloud APIs required — everything runs locally

## License

MIT
