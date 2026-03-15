"""helios resume command."""

from __future__ import annotations

import asyncio
from typing import Optional

import typer
from rich.console import Console

from helios.config.loader import load_settings
from helios.core.engine import HeliosEngine
from helios.memory.store import SessionStore
from helios.output.renderer import ConsoleRenderer
from helios.providers.registry import create_provider

console = Console()


def resume_command(
    session_id: Optional[str] = typer.Argument(None, help="Session ID to resume (latest if omitted)."),
) -> None:
    """Resume an interrupted session."""

    async def _resume() -> None:
        store = SessionStore()
        await store.initialize()

        try:
            if session_id:
                session = await store.get_session(session_id)
            else:
                session = await store.get_latest_incomplete()

            if not session:
                console.print("[yellow]No incomplete sessions found.[/yellow]")
                return

            console.print(f"[bold]Resuming session:[/bold] {session.id[:12]}")
            console.print(f"[bold]Objective:[/bold] {session.objective}")
            console.print(f"[bold]Completed exchanges:[/bold] {len(session.exchanges)}")

            settings = load_settings()
            renderer = ConsoleRenderer(console)

            orchestrator = create_provider(settings.orchestrator)
            sub_agent = create_provider(settings.sub_agent)
            refiner = create_provider(settings.refiner)

            engine = HeliosEngine(
                orchestrator=orchestrator,
                sub_agent=sub_agent,
                refiner=refiner,
                settings=settings,
                on_event=renderer.handle_event,
            )

            # Resume by re-running with existing exchanges as context
            resumed = await engine.run(session.objective, file_content=session.file_content)

            # Save the completed session
            await store.save_session(resumed)
            console.print("\n[bold green]Session completed and saved.[/bold green]")
        finally:
            await store.close()

    asyncio.run(_resume())
