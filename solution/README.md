# UDA-Hub — Multi-Agent Customer Support System

A LangGraph multi-agent system that classifies, routes, resolves, or
escalates CultPass support tickets. Ticket resolution is grounded in a
knowledge base (RAG) and uses both short-term (session) and long-term
(cross-session) memory. Full architecture writeup and diagram:
[`agentic/design/architecture.md`](agentic/design/architecture.md).

Everything needed to run and grade this project lives in this folder
(`solution/`) — nothing outside it is imported or required.

## Folder structure

```
solution/
  01_external_db_setup.ipynb   # seeds data/external/cultpass.db
  02_core_db_setup.ipynb       # seeds data/core/udahub.db
  03_agentic_app.ipynb         # entrypoint — run the compiled agent graph
  utils.py                     # chat_interface(), DB session/reset helpers
  requirements.txt
  agentic/
    agents/                    # one module per node (classifier, supervisor,
                                # the 5 category resolvers, escalation,
                                # context_loader, finalize)
    tools/                     # plain, independently-testable tool functions
                                # + db.py (path/engine helpers) + two FastMCP
                                # servers (mcp_cultpass_server.py,
                                # mcp_udahub_server.py) wrapping them
    design/architecture.md     # architecture doc + Mermaid diagram
    state.py                   # TicketState (StateGraph schema)
    tracing.py                 # structured JSONL logging (logs/)
    workflow.py                # hand-built StateGraph — the orchestrator
  data/
    core/, external/, models/  # SQLAlchemy models + seeded SQLite DBs
  tests/                       # pytest suite (92 tests)
```

## Setup

1. Install dependencies (see [Installed package versions](#installed-package-versions)
   below for exact versions this was built/tested against):
   ```
   pip install -r requirements.txt
   ```
2. Create a `.env` file in this folder (`solution/.env`) with:
   ```
   OPENAI_API_KEY=sk-...
   ```
   This file is intentionally not included in the submission — never commit
   API keys.

## Running

Run notebooks in order from this folder:

1. `01_external_db_setup.ipynb` — seeds `data/external/cultpass.db`
   (CultPass users, subscriptions, experiences, reservations).
2. `02_core_db_setup.ipynb` — seeds `data/core/udahub.db` (accounts, users,
   tickets, ticket messages/metadata, and the knowledge base — expanded to
   14 articles across technical/billing/account/booking/general categories
   in `data/external/cultpass_articles.jsonl`).
3. `03_agentic_app.ipynb` — the entrypoint. Imports `orchestrator` from
   `agentic/workflow.py` and drives it via `utils.chat_interface()`, an
   interactive REPL. Type `quit` / `exit` / `q` to end a session. The demo
   cell uses a customer already seeded by step 2 (`account_id="cultpass"`,
   `external_user_id="a4ab87"`); swap in any other seeded
   `external_user_id` to try a different customer.

`chat_interface()` seeds `ticket_id`, `account_id`, `external_user_id`,
`channel`, and `reported_urgency` into graph state each turn — the loaded
customer identity a real support channel would supply.
`ticket_id` doubles as the LangGraph `thread_id`: reusing the same id
resumes the prior session's short-term memory; a new id starts a fresh
ticket.

## Testing

```
pytest
```

92 tests, covering every tool module, every agent node, the two MCP
servers, and two full end-to-end runs of the *compiled* orchestrator graph
(`tests/test_workflow_integration.py`) — one resolution path, one
escalation path — against throwaway temp SQLite databases (the real seeded
databases under `data/` are never touched by tests). `pyproject.toml` sets
`pythonpath = ["."]` so imports resolve regardless of invocation directory.

LLM calls (`ChatOpenAI`) and OpenAI embedding calls are behind injectable
parameters/functions everywhere they're used, defaulting to the real
client only when actually invoked with no override — so the full test
suite runs without an `OPENAI_API_KEY`, using deterministic fakes instead.

## Architecture summary

- **Pattern**: Supervisor (hub-and-spoke). `agentic/workflow.py` hand-builds
  a `StateGraph` — no `create_react_agent` / `langgraph_supervisor` prebuilt
  helpers anywhere in the orchestration.
- **8 agent nodes**: Classifier, Supervisor (rule-based router), 5
  category-specific Resolvers (Technical, Billing & Subscription, Account
  Management, Booking & Reservations, General — sharing one factory
  function), Escalation. Plus two deterministic system nodes: Context
  Loader (entry) and Finalize/Persist (exit).
- **Tools** are exposed via two FastMCP servers (`agentic/tools/mcp_*.py`),
  wrapping plain, independently-testable Python functions.
- **RAG**: OpenAI embeddings + in-memory cosine similarity over the
  `Knowledge` table, cached per account.
- **Memory**: short-term via LangGraph's `MemorySaver` checkpointer
  (`thread_id = ticket_id`); long-term via a custom `customer_memory` SQLite
  table, keyed by internal `user_id`, read by the Context Loader and
  resolvers and written by Finalize.

See [`agentic/design/architecture.md`](agentic/design/architecture.md) for
the full diagram, state schema, routing rules, and design rationale.

## Installed package versions

`requirements.txt` mirrors the project's original pinned minimums. Exact
versions this solution was built and tested against (some packages/versions
newer than the pins, plus `pytest` which is test-only and not in the
original list):

| Package | Version |
|---|---|
| fastmcp | 3.4.6 |
| httpx | 0.28.1 |
| ipykernel | 7.3.0 |
| langchain | 1.3.14 |
| langchain-core | 1.5.3 |
| langchain-mcp-adapters | 0.3.2 |
| langchain-openai | 1.4.2 |
| langgraph | 1.2.10 |
| langgraph-checkpoint | 4.2.0 |
| langgraph-prebuilt | 1.1.0 |
| langgraph-sdk | 0.4.2 |
| langgraph-supervisor | 0.0.31 (installed, **not used** — see Architecture) |
| python-dotenv | 1.2.2 |
| sqlalchemy | 2.0.51 |
| pytest | 9.1.1 (test-only) |
