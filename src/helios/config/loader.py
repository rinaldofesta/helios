"""Configuration loader with TOML + env vars + CLI overrides."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from helios.config.schema import HeliosSettings


def _find_config_file() -> Path | None:
    """Search for config file in standard locations."""
    candidates = [
        Path("helios.toml"),
        Path.home() / ".config" / "helios" / "config.toml",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML file."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_settings(
    config_path: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> HeliosSettings:
    """Load settings from TOML file, env vars, and CLI overrides.

    Precedence (highest to lowest):
    1. CLI overrides
    2. Environment variables
    3. TOML config file
    4. Built-in defaults
    """
    toml_data: dict[str, Any] = {}

    # Load TOML
    path = config_path or _find_config_file()
    if path and path.exists():
        toml_data = _load_toml(path)

    # Create settings from TOML (env vars applied automatically by pydantic-settings)
    settings = HeliosSettings(**toml_data)

    # Apply CLI overrides
    if cli_overrides:
        _apply_cli_overrides(settings, cli_overrides)

    return settings


def _apply_cli_overrides(settings: HeliosSettings, overrides: dict[str, Any]) -> None:
    """Apply CLI flag overrides to settings."""
    if "provider" in overrides and overrides["provider"]:
        provider = overrides["provider"]
        for role_config in [settings.orchestrator, settings.sub_agent, settings.refiner]:
            role_config.provider = provider

    if "model" in overrides and overrides["model"]:
        model = overrides["model"]
        for role_config in [settings.orchestrator, settings.sub_agent, settings.refiner]:
            role_config.model = model

    if "model_path" in overrides and overrides["model_path"]:
        for role_config in [settings.orchestrator, settings.sub_agent, settings.refiner]:
            role_config.model_path = overrides["model_path"]

    if "search" in overrides and overrides["search"] is not None:
        settings.tools.web_search_enabled = overrides["search"]
