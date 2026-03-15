"""Entry point: python -m helios.server"""

from helios.server.app import mcp

mcp.run(transport="stdio")
