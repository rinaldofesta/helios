# Helios

AI-powered task orchestration framework that breaks complex objectives into manageable sub-tasks, executes them with specialized agents, and synthesizes cohesive results. Supports cloud APIs and local models — bring your own GGUF.

## How It Works

Helios uses a three-tier agent pattern:

```
Objective: "Build a REST API for task management"
                    |
        +-----------v-----------+
        |     ORCHESTRATOR      |  Breaks down the objective
        |  (Claude/GPT-4/Local) |  into sub-tasks using tool calls
        +-----------+-----------+
                    |
          create_subtask() x N
                    |
        +-----------v-----------+
        |      SUB-AGENT        |  Executes each sub-task
        |  (Haiku/Llama/Local)  |  with full context of previous work
        +-----------+-----------+
                    |
          complete_objective()
                    |
        +-----------v-----------+
        |       REFINER         |  Synthesizes all results into
        |  (Sonnet/GPT-4/Local) |  final output + code files
        +-----------+-----------+
                    |
                    v
            Project files, exchange log, stored session
```

## Quick Start

```bash
# Install
pip install -e .

# With local model support
pip install -e ".[local]"

# With web dashboard
pip install -e ".[web]"

# Everything
pip install -e ".[local,web,dev]"
```

Set your API keys (only needed for cloud providers):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export TAVILY_API_KEY="tvly-..."
```

Run your first objective:

```bash
helios run "Build a simple Flask TODO API with SQLite"
```

## Providers

Helios supports 6 provider backends. Any provider can be assigned to any role.

| Provider | Setup | Best For |
|---|---|---|
| **Anthropic** | Set `ANTHROPIC_API_KEY` | Claude models (Opus, Sonnet, Haiku) |
| **OpenAI** | Set `OPENAI_API_KEY` | GPT-4o, o1 |
| **Groq** | Set `GROQ_API_KEY` | Fast cloud inference (Llama, Mixtral) |
| **Ollama** | Install [Ollama](https://ollama.ai), pull models | Local models via Ollama |
| **GGUF (llama.cpp)** | Download `.gguf` files | Direct local model loading, no server |
| **OpenAI-Compatible** | Run any compatible server | LM Studio, LocalAI, vLLM |

## Configuration

Create a `helios.toml` in your project or `~/.config/helios/config.toml` globally:

```toml
[general]
output_dir = "./output"
streaming = true
max_subtasks = 20

# --- Cloud setup ---
[orchestrator]
provider = "anthropic"
model = "claude-opus-4-6-20250527"

[sub_agent]
provider = "anthropic"
model = "claude-haiku-4-5-20251001"

[refiner]
provider = "anthropic"
model = "claude-sonnet-4-6-20250514"

[tools.web_search]
enabled = true
provider = "tavily"
```

### Fully Local Setup (GGUF)

```toml
[orchestrator]
provider = "llama_cpp"
model_path = "~/models/qwen2.5-72b-instruct-q4_k_m.gguf"
n_gpu_layers = -1
n_ctx = 32768

[sub_agent]
provider = "llama_cpp"
model_path = "~/models/mistral-7b-instruct-v0.3-q5_k_m.gguf"
n_gpu_layers = -1

[refiner]
provider = "llama_cpp"
model_path = "~/models/deepseek-coder-v2-q4_k_m.gguf"
n_gpu_layers = -1
```

### Hybrid Setup (Cloud + Local)

```toml
[orchestrator]
provider = "anthropic"
model = "claude-opus-4-6-20250527"

[sub_agent]
provider = "ollama"
model = "llama3:70b-instruct"

[refiner]
provider = "openai_compatible"
base_url = "http://localhost:1234/v1"
model = "local-model"
```

## CLI Commands

```bash
# Run a new objective
helios run "Your objective here"
helios run "Analyze this code" --file ./main.py
helios run --provider llama_cpp --model-path ~/models/model.gguf "Objective"

# Session management
helios resume                      # Resume most recent interrupted session
helios resume <session_id>         # Resume specific session
helios sessions list               # List all sessions
helios sessions show <id>          # Show session details and cost
helios sessions search "REST API"  # Full-text search across sessions
helios sessions delete <id>        # Delete a session

# Model management
helios models list                 # Show configured models per role
helios models scan ~/models/       # Discover GGUF files
helios models test                 # Test provider connectivity
helios models set orchestrator llama_cpp ~/models/qwen.gguf

# Configuration
helios config show                 # Display resolved config
helios config validate             # Validate config file

# Web UI
helios web                         # Start web dashboard (default: http://localhost:8000)
helios web --port 3000             # Custom port
```

## Skills & Learning

Helios learns from completed sessions. After a successful run:

1. The system analyzes the task breakdown pattern
2. If it detects a reusable procedure, it extracts a **skill**
3. Future sessions with similar objectives automatically recall relevant skills
4. The orchestrator uses these skills as guidance for better task decomposition

Skills improve over time — frequently successful skills rank higher.

```bash
helios sessions search "API"       # Find past sessions
helios skills list                 # Browse extracted skills
helios skills show <id>            # View skill details
```

## Web Dashboard

Start with `helios web`. Features:

- **Session list** — Browse all sessions with status, cost, and timestamps
- **Live run view** — Watch agent output stream in real-time
- **Cost analytics** — Track spending across providers and sessions
- **Skills browser** — View and manage extracted skills

## Architecture

```
src/helios/
  core/          Engine, session management, data models, event bus
  providers/     Anthropic, OpenAI, Ollama, Groq, llama.cpp, OpenAI-compatible
  tools/         Tool protocol, adapters, 7 built-in tools
  memory/        SQLite store, FTS5 search, skills system
  cli/           Typer CLI with run/resume/sessions/models/config commands
  web/           FastAPI backend, WebSocket streaming, dashboard frontend
  output/        Console renderer, project file creator, markdown logger
  config/        TOML + env var config loading, Pydantic settings
```

## License

MIT
