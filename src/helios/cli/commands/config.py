"""helios config commands."""

from __future__ import annotations

import typer
from rich.console import Console

from helios.config.loader import load_settings

console = Console()
config_app = typer.Typer(name="config", help="View and validate configuration.")


@config_app.command("show")
def config_show() -> None:
    """Show the resolved configuration."""
    settings = load_settings()
    console.print_json(settings.model_dump_json(indent=2))


@config_app.command("validate")
def config_validate() -> None:
    """Validate the current configuration."""
    try:
        settings = load_settings()
        console.print("[bold green]Configuration is valid.[/bold green]")

        # Check for potential issues
        warnings = []
        if not settings.orchestrator.api_key and settings.orchestrator.provider == "groq":
            warnings.append("No API key configured for Groq (will check GROQ_API_KEY env var)")
        if settings.orchestrator.provider == "llama_cpp" and not settings.orchestrator.model_path:
            warnings.append("llama_cpp provider requires model_path")

        for w in warnings:
            console.print(f"  [yellow]Warning:[/yellow] {w}")

    except Exception as e:
        console.print(f"[bold red]Configuration error:[/bold red] {e}")
        raise typer.Exit(1)
