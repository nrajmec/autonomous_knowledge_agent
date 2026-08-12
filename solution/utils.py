# reset_udahub.py
import os
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph


Base = declarative_base()

def reset_db(db_path: str, echo: bool = True):
    """Drops the existing udahub.db file and recreates all tables."""

    # Remove the file if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"✅ Removed existing {db_path}")

    # Create a new engine and recreate tables
    engine = create_engine(f"sqlite:///{db_path}", echo=echo)
    Base.metadata.create_all(engine)
    print(f"✅ Recreated {db_path} with fresh schema")


@contextmanager
def get_session(engine: Engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


def model_to_dict(instance):
    """Convert a SQLAlchemy model instance to a dictionary."""
    return {
        column.name: getattr(instance, column.name)
        for column in instance.__table__.columns
    }

def chat_interface(
    agent: CompiledStateGraph,
    ticket_id: str,
    account_id: str,
    external_user_id: str,
    channel: str = "chat",
    reported_urgency: str | None = None,
):
    """Simple REPL for exercising a compiled UDA-Hub graph.

    UDA-Hub's graph resolves-or-escalates a ticket in one straight-line pass
    per invocation (classify -> route -> resolve/escalate -> finalize), so
    each line typed here is treated as the ticket's current text and run
    through the whole pipeline again, rather than as a mid-resolution
    clarification a single resolver is waiting on. `thread_id=ticket_id`
    still ties every turn to the same LangGraph session, so short-term
    memory (message history, already-loaded context) persists across turns
    via the graph's checkpointer.

    Args:
        agent: A compiled LangGraph graph (see `agentic.workflow.orchestrator`).
        ticket_id: Used as both the UDA-Hub ticket id and the LangGraph
            `thread_id` for this session.
        account_id: UDA-Hub `Account.account_id`, e.g. "cultpass".
        external_user_id: The customer's id in the external system (CultPass
            `user_id`) -- who is chatting.
        channel: Ticket channel/metadata, e.g. "chat", "email".
        reported_urgency: Optional customer-reported urgency.
    """
    config = {"configurable": {"thread_id": ticket_id}}

    while True:
        user_input = input("User: ")
        print("User:", user_input)
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Assistant: Goodbye!")
            break

        trigger = {
            "messages": [HumanMessage(content=user_input)],
            "ticket_id": ticket_id,
            "account_id": account_id,
            "external_user_id": external_user_id,
            "channel": channel,
            "reported_urgency": reported_urgency,
            "ticket_text": user_input,
        }

        result = agent.invoke(input=trigger, config=config)
        print("Assistant:", result["messages"][-1].content)
