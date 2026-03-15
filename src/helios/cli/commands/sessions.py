"""helios sessions commands."""

from __future__ import annotations

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from helios.memory.store import SessionStore

console = Console()
sessions_app = typer.Typer(name="sessions", help="Manage session history.")


def _get_store() -> SessionStore:
    return SessionStore()


@sessions_app.command("list")
def sessions_list(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of sessions to show."),
) -> None:
    """List past sessions."""

    async def _list() -> None:
        store = _get_store()
        await store.initialize()
        try:
            sessions = await store.list_sessions(limit=limit)
            if not sessions:
                console.print("[dim]No sessions found.[/dim]")
                return

            table = Table(title="Sessions")
            table.add_column("ID", style="dim", max_width=12)
            table.add_column("Objective", max_width=50)
            table.add_column("Status", style="bold")
            table.add_column("Tasks", justify="right")
            table.add_column("Cost", justify="right")
            table.add_column("Created", style="dim")

            for s in sessions:
                table.add_row(
                    s.id[:12],
                    s.objective[:50],
                    s.status.value,
                    str(len(s.exchanges)),
                    f"${s.total_cost:.4f}",
                    s.created_at.strftime("%Y-%m-%d %H:%M"),
                )

            console.print(table)
        finally:
            await store.close()

    asyncio.run(_list())


@sessions_app.command("show")
def sessions_show(
    session_id: str = typer.Argument(..., help="Session ID (or prefix)."),
) -> None:
    """Show details of a session."""

    async def _show() -> None:
        store = _get_store()
        await store.initialize()
        try:
            session = await store.get_session(session_id)
            if not session:
                # Try prefix match
                all_sessions = await store.list_sessions(limit=100)
                matches = [s for s in all_sessions if s.id.startswith(session_id)]
                if len(matches) == 1:
                    session = matches[0]
                elif len(matches) > 1:
                    console.print(f"[yellow]Multiple matches for '{session_id}'. Be more specific.[/yellow]")
                    return
                else:
                    console.print(f"[red]Session not found: {session_id}[/red]")
                    return

            console.print(f"\n[bold]Session:[/bold] {session.id}")
            console.print(f"[bold]Objective:[/bold] {session.objective}")
            console.print(f"[bold]Status:[/bold] {session.status.value}")
            console.print(f"[bold]Total Cost:[/bold] ${session.total_cost:.4f}")
            console.print(f"[bold]Created:[/bold] {session.created_at}")
            console.print(f"[bold]Exchanges:[/bold] {len(session.exchanges)}")

            for i, ex in enumerate(session.exchanges, 1):
                console.print(f"\n  [bold]Task {i}:[/bold] {ex.subtask.description}")
                console.print(f"  [dim]Result:[/dim] {ex.result[:200]}...")

            if session.synthesis:
                console.print(f"\n[bold]Synthesis:[/bold]\n{session.synthesis[:500]}")
        finally:
            await store.close()

    asyncio.run(_show())


@sessions_app.command("delete")
def sessions_delete(
    session_id: str = typer.Argument(..., help="Session ID to delete."),
) -> None:
    """Delete a session."""

    async def _delete() -> None:
        store = _get_store()
        await store.initialize()
        try:
            deleted = await store.delete_session(session_id)
            if deleted:
                console.print(f"[green]Session {session_id} deleted.[/green]")
            else:
                console.print(f"[red]Session not found: {session_id}[/red]")
        finally:
            await store.close()

    asyncio.run(_delete())


@sessions_app.command("search")
def sessions_search(
    query: str = typer.Argument(..., help="Search query."),
    limit: int = typer.Option(10, "--limit", "-n"),
) -> None:
    """Search sessions by content."""

    async def _search() -> None:
        store = _get_store()
        await store.initialize()
        try:
            results = await store.search(query, limit=limit)
            if not results:
                console.print(f"[dim]No results for '{query}'.[/dim]")
                return

            for s in results:
                console.print(f"  [{s.status.value}] {s.id[:12]}  {s.objective[:60]}")
        finally:
            await store.close()

    asyncio.run(_search())
