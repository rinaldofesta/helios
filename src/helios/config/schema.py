"""Configuration schema for Helios."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ProviderConfig(BaseModel):
    """Configuration for a single provider instance."""

    provider: str = "ollama"
    model: str = ""
    model_path: Path | None = None
    base_url: str | None = None
    api_key: str | None = None
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    max_tokens: int = 4096
    extra: dict[str, Any] = Field(default_factory=dict)


class ToolsConfig(BaseModel):
    """Configuration for built-in tools."""

    web_search_enabled: bool = False
    tavily_api_key: str | None = None
    read_file_enabled: bool = True


class ModelStorageConfig(BaseModel):
    """Configuration for local model storage."""

    models_dir: Path = Path.home() / ".helios" / "models"


class GeneralSettings(BaseModel):
    """General application settings."""

    output_dir: Path = Path(".")
    data_dir: Path = Path.home() / ".helios"
    streaming: bool = False
    max_subtasks: int = 20


class HeliosSettings(BaseSettings):
    """Root configuration for Helios."""

    general: GeneralSettings = Field(default_factory=GeneralSettings)
    models: ModelStorageConfig = Field(default_factory=ModelStorageConfig)

    orchestrator: ProviderConfig = Field(
        default_factory=lambda: ProviderConfig(
            provider="ollama",
            model="llama3:instruct",
        )
    )
    sub_agent: ProviderConfig = Field(
        default_factory=lambda: ProviderConfig(
            provider="ollama",
            model="llama3:instruct",
        )
    )
    refiner: ProviderConfig = Field(
        default_factory=lambda: ProviderConfig(
            provider="ollama",
            model="llama3:instruct",
        )
    )

    tools: ToolsConfig = Field(default_factory=ToolsConfig)

    model_config = {
        "env_prefix": "HELIOS_",
        "env_nested_delimiter": "__",
    }
