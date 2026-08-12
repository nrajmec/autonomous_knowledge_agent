# UDA-Hub — Universal Decision Agent

UDA-Hub is a LangGraph-powered multi-agent customer support system built for **CultPass**, a
subscription-based cultural experiences platform. It reads incoming support tickets, classifies
them, routes them to a specialized resolver agent, retrieves grounding context from a knowledge
base (RAG) and the customer's own account/history, resolves the ticket or escalates it to a
human, and persists the outcome — maintaining both short-term (session) and long-term
(cross-session) memory throughout.

This is the capstone project for Udacity's AI Engineer nanodegree.

**Everything required to run, test, and grade this project lives in the [`solution/`](solution/)
folder** — all source code, notebooks, seeded databases, tests, and the architecture design
document. The [`starter/`](starter/) folder is the original, unmodified project scaffold kept
for reference only and is not part of the graded deliverable.

## Getting Started

Clone this repository, then follow the setup steps below from inside `solution/` — see
[`solution/README.md`](solution/README.md) for full details (folder structure, exact run steps,
and installed package versions).

### Dependencies

Python 3.13.12

```
langgraph
langchain
langchain-core
langchain-openai
langchain-mcp-adapters
fastmcp
sqlalchemy
python-dotenv
ipykernel
pytest
```

Exact pinned/installed versions are listed in [`solution/requirements.txt`](solution/requirements.txt)
and in the table at the bottom of [`solution/README.md`](solution/README.md).

### Installation

1. Create and activate a Python virtual environment.
2. Install dependencies:
   ```
   pip install -r solution/requirements.txt
   ```
3. Create a `solution/.env` file with your OpenAI-compatible API credentials:
   ```
   OPENAI_API_KEY=your-key-here
   ```
   (See `solution/.env.example` for the expected variable names. This file is never committed.)
4. From the `solution/` folder, run the setup notebooks in order to seed the databases:
   - `01_external_db_setup.ipynb` — seeds `data/external/cultpass.db`
   - `02_core_db_setup.ipynb` — seeds `data/core/udahub.db`, including a 14-article knowledge base
5. Run the entrypoint notebook, `03_agentic_app.ipynb`, which imports the compiled LangGraph
   orchestrator (`agentic/workflow.py`) and drives it via an interactive `chat_interface()`.

## Testing

From the `solution/` folder:

```
pytest
```

This runs the full test suite (92+ tests) covering every tool module, every agent node, both
MCP servers, and two full end-to-end runs of the compiled orchestrator graph against throwaway
temporary databases — the real seeded databases are never touched by tests.

### Break Down Tests

- **Tool tests** (`test_cultpass_tools.py`, `test_udahub_tools.py`, `test_knowledge_tools.py`,
  `test_memory_tools.py`) — verify each database-abstraction tool's validation, error handling,
  and structured `{ok, ...}` return contract against isolated temp SQLite databases.
- **MCP server tests** (`test_mcp_servers.py`) — verify both FastMCP servers expose the correct
  tool set and respond correctly via an in-process `fastmcp.Client`.
- **Agent node tests** (`test_classifier.py`, `test_supervisor.py`, `test_resolver.py`,
  `test_escalation.py`, `test_context_loader.py`, `test_finalize.py`) — verify each LangGraph
  node's decision logic in isolation, using a `FakeChatModel` test double so no OpenAI API key
  is required to run the suite.
- **Integration tests** (`test_workflow_integration.py`) — run the actual compiled
  `orchestrator` graph end-to-end through both a resolution and an escalation scenario, with
  only the LLM calls faked (real tool logic, real DB writes, real routing).

## Project Instructions

All student deliverables are inside [`solution/`](solution/):

- `agentic/agents/` — the 8 LangGraph agent nodes (Classifier, Supervisor, 5 category
  Resolvers, Escalation) plus the Context Loader and Finalize system nodes
- `agentic/tools/` — database-abstraction tools for both CultPass and UDA-Hub's own database,
  plus two FastMCP servers exposing them
- `agentic/workflow.py` — the hand-built LangGraph `StateGraph` orchestrating all agents (no
  prebuilt `create_react_agent` / `langgraph_supervisor` helpers)
- `agentic/design/architecture.md` — the full architecture design document (pattern, diagram,
  agent roster, RAG mechanics, memory design, routing logic)
- `data/` — SQLAlchemy models and the seeded SQLite databases for both CultPass and UDA-Hub
- `tests/` — the full pytest suite
- `01_external_db_setup.ipynb`, `02_core_db_setup.ipynb`, `03_agentic_app.ipynb` — the database
  setup notebooks and the application entrypoint, including saved outputs from successful runs
- `README.md` — detailed setup, run, and testing instructions with the full package-version table

## Built With

* [LangGraph](https://github.com/langchain-ai/langgraph) - orchestration graph/state machine for the multi-agent workflow
* [LangChain](https://github.com/langchain-ai/langchain) / [langchain-openai](https://github.com/langchain-ai/langchain) - LLM and structured-output tooling
* [FastMCP](https://github.com/jlowin/fastmcp) - MCP servers exposing the support-operation tools
* [langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters) - MCP tool interop
* [SQLAlchemy](https://www.sqlalchemy.org/) - ORM/database abstraction for both the CultPass and UDA-Hub databases
* [OpenAI](https://platform.openai.com/) (`gpt-4o-mini`, `text-embedding-3-small`) - classification, resolution, escalation, and knowledge-base embeddings
* [pytest](https://pytest.org/) - test suite

## License

[License](LICENSE)
