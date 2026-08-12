"""FastMCP server exposing UDA-Hub's own core-database tools: knowledge-base
search (RAG), ticket history/updates, and long-term customer memory.

Same split as `mcp_cultpass_server.py`: business logic lives in
`knowledge_tools.py` / `udahub_tools.py` / `memory_tools.py` and is
unit-tested there; this module only registers those functions as MCP tools.

Run standalone (manual smoke test / MCP inspector):
    python -m agentic.tools.mcp_udahub_server

Agents connect to this over stdio via
`langchain_mcp_adapters.client.MultiServerMCPClient` (wired up in
`agentic/workflow.py`), which spawns this file as a subprocess per session.
"""
from fastmcp import FastMCP

from agentic.tools.knowledge_tools import search_knowledge_base
from agentic.tools.memory_tools import recall_customer_memory, save_customer_memory
from agentic.tools.udahub_tools import get_internal_user_id, get_ticket_history, update_ticket_record

mcp = FastMCP("udahub")

mcp.tool(search_knowledge_base)
mcp.tool(get_ticket_history)
mcp.tool(update_ticket_record)
mcp.tool(get_internal_user_id)
mcp.tool(recall_customer_memory)
mcp.tool(save_customer_memory)


if __name__ == "__main__":
    mcp.run()
