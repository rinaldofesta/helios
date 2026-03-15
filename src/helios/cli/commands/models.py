"""helios models commands."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from helios.config.loader import load_settings

console = Console()
models_app = typer.Typer(name="models", help="Manage model configurations.")


@models_app.command("list")
def models_list() -> None:
    """Show configured models for each role."""
    settings = load_settings()

    table = Table(title="Model Configuration")
    table.add_column("Role", style="bold")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Max Tokens", justify="right")

    for role_name, config in [
        ("Orchestrator", settings.orchestrator),
        ("Sub-Agent", settings.sub_agent),
        ("Refiner", settings.refiner),
    ]:
        model_display = config.model
        if config.model_path:
            model_display = str(config.model_path)
        table.add_row(role_name, config.provider, model_display, str(config.max_tokens))

    console.print(table)
