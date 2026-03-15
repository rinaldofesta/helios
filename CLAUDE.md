# Helios v1.0 — Claude Code Project Instructions

## Project Overview

Helios is an open-source, local-first AI task orchestration framework that breaks complex objectives into sub-tasks using a three-tier agent pattern: **Orchestrator** (decomposes), **Sub-agent** (executes), **Refiner** (synthesizes). It runs exclusively on open-weight models — locally via Ollama/llama.cpp/GGUF or through open-model inference providers like Groq.

## Architecture

```
User Objective
      |
      v
Orchestrator (tool: create_subtask / complete_objective)
      |
      v  (loop until complete_objective is called)
Sub-Agent (executes each sub-task, returns free text)
      |
      v  (after loop)
Refiner (tools: define_project_structure, create_code_file, output_synthesis)
      |
      v
Output: files, exchange log, session stored in SQLite
```

## Tech Stack

- **Python 3.11+** with async-first design
- **Pydantic** for all data models and settings
- **Typer** for CLI
- **Rich** for console output (via event bus, not direct calls)
- **FastAPI + WebSocket** for web UI
- **SQLite + FTS5** for session/skills persistence
- **aiosqlite** for async DB access
- **huggingface-hub** for model search and GGUF downloads
- Providers: `ollama`, `groq`, `openai-compatible` (LM Studio/vLLM), `llama-cpp-python`

## Package Structure

```
src/helios/
  core/          — engine.py (orchestration loop), session.py, models.py, events.py
  providers/     — base.py (Protocol), ollama.py, groq.py,
                   openai_compatible.py, llama_cpp.py
  models/        — hub.py (HuggingFace search + GGUF download)
  tools/         — base.py, adapter.py, registry.py, builtin/ (7 tools)
  memory/        — store.py (SQLite+JSON), search.py (FTS5), skills.py
  cli/           — app.py, commands/ (run, resume, sessions, models, config)
  web/           — api.py (FastAPI), websocket.py, frontend/
  output/        — renderer.py, project.py, markdown.py
  config/        — loader.py, schema.py
```

## Code Conventions

- **Async-first**: All provider calls use `async/await`. Sync APIs wrapped with `asyncio.to_thread()`.
- **Protocol pattern**: Providers implement a `Provider` protocol (not ABC). Type checking via structural subtyping.
- **Pydantic models**: All data structures use Pydantic v2 models. Config uses `pydantic-settings`.
- **Event-driven rendering**: Engine emits events; CLI renderer and WebSocket subscribe. Never call `console.print()` from engine code.
- **Tool use over regex**: All structured data extracted via native tool calls. `LegacyParsingAdapter` only for models without tool support.
- **Open-weight only**: No proprietary API providers (Anthropic, OpenAI). Only open models.
- **No hardcoded secrets**: API keys (e.g. Groq) from env vars only.

## Key Design Decisions

1. **Open-weight only** — No proprietary cloud providers; Ollama, Groq, OpenAI-compatible, llama.cpp
2. **HuggingFace Hub integration** — Search and download GGUF models directly from the web UI (like LM Studio)
3. **Native tool use replaces regex parsing** — Structured tool calls for all data extraction
4. **Provider abstraction** — Any open model from any provider for any role
5. **GGUF direct loading** — `llama-cpp-python` loads GGUF files directly, no server needed
6. **Event bus** — Decouples engine from presentation (CLI, WebSocket, future UIs)
7. **Skills learning loop** — Post-session skill extraction, FTS5 recall for future sessions
8. **SQLite + JSON dual-write** — SQLite for queries/search, JSON for portability

## Available Providers

| Provider | Description | Tool Support |
|----------|-------------|-------------|
| `ollama` | Local models via Ollama (default) | Yes |
| `groq` | Fast cloud inference for open-weight models | Yes |
| `openai_compatible` | LM Studio, vLLM, LocalAI, etc. | Yes |
| `llama_cpp` | Direct GGUF file loading | No (JSON fallback) |

## Running & Testing

```bash
# Install
pip install -e ".[dev,local,web]"

# Run with Ollama (default)
helios run "Build a Flask TODO API"

# Run with Groq
helios run --provider groq --model llama-3.3-70b-versatile "Explain recursion"

# Run with local GGUF model
helios run --provider llama_cpp --model-path ~/models/model.gguf "Explain recursion"

# Start web UI (includes HuggingFace model browser)
helios web

# Run tests
pytest
```

## Configuration

Config layers (each overrides the previous):
1. Built-in defaults (Ollama with llama3:instruct)
2. Config file (`helios.toml` or `~/.config/helios/config.toml`)
3. Environment variables (`GROQ_API_KEY`, `HELIOS_ORCHESTRATOR_MODEL`, etc.)
4. CLI flags (`--provider`, `--model`, etc.)

## Legacy Code Reference

Original scripts preserved in `legacy/` for reference during porting:
- `legacy/anthropic-helios.py` — orchestration loop, cost tracking, Tavily search
- `legacy/anthropic_openai-helios.py` — dual-provider switching
- `legacy/helios-ollama.py` — session persistence, resume, Ollama model management
