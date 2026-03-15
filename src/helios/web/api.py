"""FastAPI application for Helios web UI."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from helios.config.loader import load_settings
from helios.core.engine import HeliosEngine
from helios.memory.skills import SkillManager
from helios.memory.store import SessionStore
from helios.models.hub import ModelHub
from helios.providers.registry import create_provider

app = FastAPI(title="Helios", version="1.0.0a1")

# State
_store: SessionStore | None = None
_skills: SkillManager | None = None
_hub: ModelHub | None = None


async def get_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
        await _store.initialize()
    return _store


async def get_skills() -> SkillManager:
    global _skills
    if _skills is None:
        _skills = SkillManager()
        await _skills.initialize()
    return _skills


def get_hub() -> ModelHub:
    global _hub
    if _hub is None:
        settings = load_settings()
        _hub = ModelHub(models_dir=settings.models.models_dir)
    return _hub


# ----- Request/Response models ----- #


class RoleOverride(BaseModel):
    provider: str | None = None
    model: str | None = None
    model_path: str | None = None


class RunRequest(BaseModel):
    objective: str
    file_content: str | None = None
    provider: str | None = None
    model: str | None = None
    orchestrator: RoleOverride | None = None
    sub_agent: RoleOverride | None = None
    refiner: RoleOverride | None = None


class SessionSummary(BaseModel):
    id: str
    objective: str
    status: str
    exchange_count: int
    total_cost: float
    created_at: str


class ModelDownloadRequest(BaseModel):
    repo_id: str
    filename: str


# ----- Startup/Shutdown ----- #


@app.on_event("startup")
async def startup():
    await get_store()
    await get_skills()


@app.on_event("shutdown")
async def shutdown():
    if _store:
        await _store.close()
    if _skills:
        await _skills.close()


# ----- API Routes ----- #


@app.get("/api/sessions")
async def list_sessions(limit: int = 20, offset: int = 0):
    store = await get_store()
    sessions = await store.list_sessions(limit=limit, offset=offset)
    return [
        SessionSummary(
            id=s.id,
            objective=s.objective,
            status=s.status.value,
            exchange_count=len(s.exchanges),
            total_cost=s.total_cost,
            created_at=s.created_at.isoformat(),
        )
        for s in sessions
    ]


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    store = await get_store()
    session = await store.get_session(session_id)
    if not session:
        return {"error": "Session not found"}
    return session.model_dump(mode="json")


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    store = await get_store()
    deleted = await store.delete_session(session_id)
    return {"deleted": deleted}


@app.get("/api/sessions/search/{query}")
async def search_sessions(query: str, limit: int = 10):
    store = await get_store()
    results = await store.search(query, limit=limit)
    return [
        SessionSummary(
            id=s.id,
            objective=s.objective,
            status=s.status.value,
            exchange_count=len(s.exchanges),
            total_cost=s.total_cost,
            created_at=s.created_at.isoformat(),
        )
        for s in results
    ]


@app.get("/api/skills")
async def list_skills(limit: int = 20):
    skills = await get_skills()
    return [s.model_dump(mode="json") for s in await skills.list_skills(limit=limit)]


@app.get("/api/config")
async def get_config():
    settings = load_settings()
    return settings.model_dump(mode="json")


@app.get("/api/models")
async def get_models():
    settings = load_settings()
    return {
        "orchestrator": {"provider": settings.orchestrator.provider, "model": settings.orchestrator.model},
        "sub_agent": {"provider": settings.sub_agent.provider, "model": settings.sub_agent.model},
        "refiner": {"provider": settings.refiner.provider, "model": settings.refiner.model},
    }


@app.get("/api/providers")
async def get_providers():
    """Return available providers and their known models."""
    return {
        "providers": [
            {
                "id": "ollama",
                "name": "Ollama",
                "models": [],
                "description": "Run open-weight models locally via Ollama",
                "needs_model_name": True,
            },
            {
                "id": "groq",
                "name": "Groq",
                "models": [
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant",
                    "llama-3.1-70b-versatile",
                    "mixtral-8x7b-32768",
                    "gemma2-9b-it",
                ],
                "description": "Fast cloud inference for open-weight models",
                "needs_model_name": False,
            },
            {
                "id": "openai_compatible",
                "name": "OpenAI Compatible",
                "models": [],
                "description": "Any OpenAI-compatible server (LM Studio, vLLM, LocalAI)",
                "needs_model_name": True,
            },
            {
                "id": "llama_cpp",
                "name": "llama.cpp (GGUF)",
                "models": [],
                "description": "Load GGUF files directly — download from HuggingFace below",
                "needs_model_name": False,
                "needs_model_path": True,
            },
        ],
    }


# ----- HuggingFace Hub Model Endpoints ----- #


@app.get("/api/hub/trending")
async def hub_trending(limit: int = 15):
    """Return trending/popular GGUF models from HuggingFace Hub."""
    hub = get_hub()
    try:
        results = await hub.search("gguf", limit=limit, sort="trending_score")
        return {
            "models": [
                {
                    "repo_id": m.repo_id,
                    "name": m.name,
                    "author": m.author,
                    "downloads": m.downloads,
                    "likes": m.likes,
                    "tags": m.tags[:10],
                }
                for m in results
            ]
        }
    except ImportError:
        return {"error": "huggingface-hub not installed. Run: pip install huggingface-hub"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/hub/search")
async def hub_search(q: str, limit: int = 20):
    """Search HuggingFace Hub for GGUF models."""
    hub = get_hub()
    try:
        results = await hub.search(q, limit=limit)
        return {
            "models": [
                {
                    "repo_id": m.repo_id,
                    "name": m.name,
                    "author": m.author,
                    "downloads": m.downloads,
                    "likes": m.likes,
                    "tags": m.tags[:10],
                }
                for m in results
            ]
        }
    except ImportError:
        return {"error": "huggingface-hub not installed. Run: pip install huggingface-hub"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/hub/files/{repo_id:path}")
async def hub_files(repo_id: str):
    """List GGUF files available in a HuggingFace repo."""
    hub = get_hub()
    try:
        files = await hub.list_gguf_files(repo_id)
        return {
            "repo_id": repo_id,
            "files": [
                {
                    "filename": f.filename,
                    "size_bytes": f.size_bytes,
                    "size_display": f.size_display,
                    "quantisation": f.quantisation,
                }
                for f in files
            ],
        }
    except ImportError:
        return {"error": "huggingface-hub not installed. Run: pip install huggingface-hub"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/hub/download")
async def hub_download(request: ModelDownloadRequest):
    """Download a GGUF model from HuggingFace Hub."""
    hub = get_hub()
    try:
        local_path = await hub.download(request.repo_id, request.filename)
        return {
            "status": "complete",
            "path": str(local_path),
            "filename": request.filename,
            "repo_id": request.repo_id,
        }
    except ImportError:
        return {"error": "huggingface-hub not installed. Run: pip install huggingface-hub"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/hub/local")
async def hub_local_models():
    """List locally downloaded GGUF models."""
    hub = get_hub()
    models = hub.list_local_models()
    return {
        "models": [
            {
                "path": str(m.path),
                "filename": m.filename,
                "repo_id": m.repo_id,
                "size_bytes": m.size_bytes,
                "size_display": m.size_display,
            }
            for m in models
        ],
        "models_dir": str(hub.models_dir),
    }


@app.delete("/api/hub/local")
async def hub_delete_local(path: str):
    """Delete a locally downloaded GGUF model."""
    hub = get_hub()
    deleted = hub.delete_local_model(path)
    return {"deleted": deleted}


# ----- Ollama model listing ----- #


@app.get("/api/ollama/models")
async def ollama_models():
    """List models available in the local Ollama instance."""
    try:
        import ollama as ollama_lib
        client = ollama_lib.Client()
        response = client.list()
        models = []
        raw_models = response.get("models", response) if isinstance(response, dict) else getattr(response, "models", [])
        for m in raw_models:
            if isinstance(m, dict):
                name = m.get("model", m.get("name", ""))
                size = m.get("size", 0)
            else:
                name = getattr(m, "model", None) or getattr(m, "name", str(m))
                size = getattr(m, "size", 0)
            if name:
                models.append({"name": name, "size": size})
        return {"models": models}
    except Exception as e:
        return {"models": [], "error": str(e)}


# ----- Run ----- #


def _apply_role_override(cfg, override: RoleOverride | None) -> None:
    """Apply a per-role override to a ProviderConfig."""
    if override:
        if override.provider:
            cfg.provider = override.provider
        if override.model:
            cfg.model = override.model
        if override.model_path:
            cfg.model_path = override.model_path


@app.post("/api/run")
async def start_run(request: RunRequest):
    """Start a new Helios run (non-streaming, returns when complete)."""
    import logging as _log
    _logger = _log.getLogger("helios.web.run")

    settings = load_settings()

    # Global overrides
    if request.provider:
        for cfg in [settings.orchestrator, settings.sub_agent, settings.refiner]:
            cfg.provider = request.provider
    if request.model:
        for cfg in [settings.orchestrator, settings.sub_agent, settings.refiner]:
            cfg.model = request.model

    # Per-role overrides take precedence
    _apply_role_override(settings.orchestrator, request.orchestrator)
    _apply_role_override(settings.sub_agent, request.sub_agent)
    _apply_role_override(settings.refiner, request.refiner)

    _logger.info(
        "Starting run: objective=%r orch=%s/%s sub=%s/%s ref=%s/%s",
        request.objective[:60],
        settings.orchestrator.provider, settings.orchestrator.model,
        settings.sub_agent.provider, settings.sub_agent.model,
        settings.refiner.provider, settings.refiner.model,
    )

    try:
        orchestrator = create_provider(settings.orchestrator)
        sub_agent = create_provider(settings.sub_agent)
        refiner = create_provider(settings.refiner)
    except Exception as e:
        _logger.error("Provider creation failed: %s", e)
        return {"error": f"Failed to create provider: {e}"}

    engine = HeliosEngine(
        orchestrator=orchestrator,
        sub_agent=sub_agent,
        refiner=refiner,
        settings=settings,
    )

    try:
        session = await engine.run(request.objective, file_content=request.file_content)
    except Exception as e:
        _logger.error("Engine run failed: %s", e, exc_info=True)
        return {"error": f"Run failed: {e}"}

    # Save to store
    store = await get_store()
    await store.save_session(session)

    _logger.info("Run complete: session=%s exchanges=%d", session.id, len(session.exchanges))
    return session.model_dump(mode="json")


# ----- WebSocket for streaming ----- #


@app.websocket("/ws/run")
async def websocket_run(websocket: WebSocket):
    """WebSocket endpoint for streaming run events."""
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        objective = data.get("objective", "")
        file_content = data.get("file_content")

        if not objective:
            await websocket.send_json({"type": "error", "data": {"error": "No objective provided"}})
            return

        settings = load_settings()

        # Apply per-role overrides from WebSocket message
        for role_key in ("orchestrator", "sub_agent", "refiner"):
            role_data = data.get(role_key)
            if role_data and isinstance(role_data, dict):
                cfg = getattr(settings, role_key)
                if role_data.get("provider"):
                    cfg.provider = role_data["provider"]
                if role_data.get("model"):
                    cfg.model = role_data["model"]
                if role_data.get("model_path"):
                    cfg.model_path = role_data["model_path"]

        orchestrator = create_provider(settings.orchestrator)
        sub_agent = create_provider(settings.sub_agent)
        refiner = create_provider(settings.refiner)

        async def send_event(event_type: str, event_data: dict[str, Any]) -> None:
            try:
                await websocket.send_json({"type": event_type, "data": event_data})
            except Exception:
                pass

        # Wrap sync callback for engine
        def on_event(event_type: str, event_data: dict[str, Any]) -> None:
            asyncio.ensure_future(send_event(event_type, event_data))

        engine = HeliosEngine(
            orchestrator=orchestrator,
            sub_agent=sub_agent,
            refiner=refiner,
            settings=settings,
            on_event=on_event,
        )

        session = await engine.run(objective, file_content=file_content)

        store = await get_store()
        await store.save_session(session)

        await websocket.send_json({
            "type": "complete",
            "data": session.model_dump(mode="json"),
        })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "data": {"error": str(e)}})
        except Exception:
            pass


# ----- Static frontend ----- #

FRONTEND_DIR = Path(__file__).parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
