"""Helios CLI application."""

from __future__ import annotations

import typer

from helios import __version__

app = typer.Typer(
    name="helios",
    help="AI-powered task orchestration framework.",
    no_args_is_help=True,
)


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", "-v", help="Show version."),
) -> None:
    if version:
        typer.echo(f"Helios v{__version__}")
        raise typer.Exit()


# Import and register commands
from helios.cli.commands.run import run_command  # noqa: E402
from helios.cli.commands.resume import resume_command  # noqa: E402
from helios.cli.commands.sessions import sessions_app  # noqa: E402
from helios.cli.commands.models import models_app  # noqa: E402
from helios.cli.commands.config import config_app  # noqa: E402
from helios.cli.commands.web import web_command  # noqa: E402
from helios.cli.commands.serve import serve_command  # noqa: E402

app.command(name="run")(run_command)
app.command(name="resume")(resume_command)
app.command(name="web")(web_command)
app.command(name="serve")(serve_command)
app.add_typer(sessions_app)
app.add_typer(models_app)
app.add_typer(config_app)
