"""helios web command."""

from __future__ import annotations

import typer
from rich.console import Console

console = Console()


def web_command(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to."),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on."),
) -> None:
    """Start the Helios web dashboard."""
    try:
        import uvicorn
    except ImportError:
        console.print("[bold red]uvicorn not installed. Run: pip install helios[web][/bold red]")
        raise typer.Exit(1)

    console.print(f"[bold cyan]Starting Helios Web UI at http://{host}:{port}[/bold cyan]")
    uvicorn.run("helios.web.api:app", host=host, port=port, reload=False)
