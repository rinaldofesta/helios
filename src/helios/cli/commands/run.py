"""helios run command."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from helios.config.loader import load_settings
from helios.core.engine import HeliosEngine
from helios.output.markdown import ExchangeLogWriter
from helios.output.project import ProjectCreator
from helios.output.renderer import ConsoleRenderer
from helios.providers.registry import create_provider

console = Console()


def run_command(
    objective: str = typer.Argument(..., help="The objective to accomplish."),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="File to include as context."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider to use."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name."),
    model_path: Optional[Path] = typer.Option(None, "--model-path", help="Path to GGUF model file."),
    search: Optional[bool] = typer.Option(None, "--search/--no-search", help="Enable web search."),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file path."),
) -> None:
    """Run Helios with the given objective."""
    # Load settings
    cli_overrides = {
        "provider": provider,
        "model": model,
        "model_path": model_path,
        "search": search,
    }
    settings = load_settings(config_path=config_file, cli_overrides=cli_overrides)

    # Read file content if provided
    file_content = None
    if file:
        if not file.exists():
            console.print(f"[bold red]File not found: {file}[/bold red]")
            raise typer.Exit(1)
        file_content = file.read_text(encoding="utf-8")

    # Setup renderer
    renderer = ConsoleRenderer(console)

    # Create providers
    orchestrator = create_provider(settings.orchestrator)
    sub_agent = create_provider(settings.sub_agent)
    refiner = create_provider(settings.refiner)

    # Create engine
    engine = HeliosEngine(
        orchestrator=orchestrator,
        sub_agent=sub_agent,
        refiner=refiner,
        settings=settings,
        on_event=renderer.handle_event,
    )

    # Run
    try:
        session = asyncio.run(engine.run(objective, file_content=file_content))
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Session interrupted.[/bold yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)

    # Write output
    output_dir = settings.general.output_dir

    # Write exchange log
    log_writer = ExchangeLogWriter(output_dir=output_dir)
    log_path = log_writer.write(session)
    console.print(f"\nExchange log saved to: [bold]{log_path}[/bold]")

    # Create project files if applicable
    if session.project_name and session.project_structure:
        creator = ProjectCreator(output_dir=output_dir, console=console)
        project_path = creator.create(
            session.project_name,
            session.project_structure,
            session.code_files,
        )
        console.print(f"Project created at: [bold]{project_path}[/bold]")

    # Final synthesis
    if session.synthesis:
        console.print(f"\n[bold]Refined Output:[/bold]\n{session.synthesis}")
