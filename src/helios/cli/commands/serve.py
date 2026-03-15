"""Start the Helios MCP server."""

from __future__ import annotations

import typer


def serve_command(
    transport: str = typer.Option(
        "stdio", help="Transport protocol: stdio, sse, or streamable-http.",
    ),
    host: str = typer.Option("127.0.0.1", help="Host for HTTP transports."),
    port: int = typer.Option(8080, help="Port for HTTP transports."),
) -> None:
    """Start the Helios MCP server for AI coding agents."""
    from helios.server.app import mcp

    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=transport, host=host, port=port)
