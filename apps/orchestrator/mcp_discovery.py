"""Discover ML model tools from the in-process FastMCP `ml-models` server.

Returns a list of `MCPTool` instances that BeeAI can register on the agent.
The orchestrator never imports any specific ML model — it sees them only
through the MCP tool interface, which is exactly the indirection that lets
new models drop in without touching the orchestrator.

For the thin slice we boot the server in-process. Once we promote to a
hosted setup (next slice), this function will accept a connection to a
remote MCP server instead.
"""

from typing import TYPE_CHECKING

from fastmcp import Client

from services.ml_mcp_server.server import build_server

if TYPE_CHECKING:
    from beeai_framework.tools.mcp import MCPTool


async def discover_ml_mcp_tools() -> list["MCPTool"]:
    """Boot the ml-models FastMCP server in-process and return BeeAI MCPTools."""
    from beeai_framework.tools.mcp import MCPTool

    server = build_server()
    async with Client(server) as client:
        return await MCPTool.from_client(client.session)
