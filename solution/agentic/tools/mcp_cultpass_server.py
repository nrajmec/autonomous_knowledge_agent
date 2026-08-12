"""FastMCP server exposing CultPass (external database) tools.

This is the thin transport layer over `cultpass_tools.py` -- all the actual
business logic (validation, DB access, structured {ok, ...} returns) lives
there and is unit-tested independently in `tests/test_cultpass_tools.py`.
This module just registers those already-typed, already-documented
functions as MCP tools; FastMCP derives each tool's name, parameters, and
description straight from the function signature and docstring.

Run standalone (manual smoke test / MCP inspector, or for use by any other
MCP-speaking client):
    python -m agentic.tools.mcp_cultpass_server

Note: the LangGraph resolver nodes in this project do NOT go through this
server -- they call the functions in `cultpass_tools.py` directly (see
`_TOOL_BUILDERS` in `agentic/agents/resolver.py`), since a resolver already
runs in-process and a stdio MCP round-trip would add nothing but latency.
This server exists so the same tools are available to any other
MCP-speaking client/agent, and is exercised directly (via `fastmcp.Client`,
no subprocess needed) in `tests/test_mcp_servers.py`.
"""
from fastmcp import FastMCP

from agentic.tools.cultpass_tools import (
    get_customer_profile,
    get_subscription_status,
    list_reservations,
    manage_reservation,
    manage_subscription,
    search_experiences,
)

mcp = FastMCP("cultpass")

mcp.tool(get_customer_profile)
mcp.tool(get_subscription_status)
mcp.tool(manage_subscription)
mcp.tool(list_reservations)
mcp.tool(manage_reservation)
mcp.tool(search_experiences)


if __name__ == "__main__":
    mcp.run()
