"""Tests for the FastMCP server wrappers (mcp_cultpass_server / mcp_udahub_server).

Verifies both servers expose every underlying tool, and that a call
actually round-trips through the MCP protocol layer into the real business
logic -- using an in-process `fastmcp.Client` (no subprocess needed) and the
same isolated temp-DB pattern as the other tool tests, so no real
solution/data/**/*.db file is touched.
"""
import asyncio

import pytest
from fastmcp import Client
from sqlalchemy import create_engine

import agentic.tools.cultpass_tools as cultpass_tools
import agentic.tools.udahub_tools as udahub_tools
from agentic.tools.mcp_cultpass_server import mcp as cultpass_mcp
from agentic.tools.mcp_udahub_server import mcp as udahub_mcp
from data.models import cultpass, udahub
from utils import get_session


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _patch_cultpass_engine(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test_cultpass.db'}")
    cultpass.Base.metadata.create_all(engine)
    monkeypatch.setattr(cultpass_tools, "get_cultpass_engine", lambda: engine)
    with get_session(engine) as session:
        session.add(
            cultpass.User(user_id="u1", full_name="Test User", email="t@example.com", is_blocked=False)
        )


@pytest.fixture(autouse=True)
def _patch_udahub_engine(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test_udahub.db'}")
    udahub.Base.metadata.create_all(engine)
    monkeypatch.setattr(udahub_tools, "get_udahub_engine", lambda: engine)
    with get_session(engine) as session:
        session.add(udahub.Account(account_id="acc1", account_name="Test Account"))
        session.add(
            udahub.User(
                user_id="user1", account_id="acc1", external_user_id="u1", user_name="Test User"
            )
        )


def test_cultpass_server_lists_all_tools():
    async def run():
        async with Client(cultpass_mcp) as client:
            return {t.name for t in await client.list_tools()}

    names = _run(run())

    assert names == {
        "get_customer_profile",
        "get_subscription_status",
        "manage_subscription",
        "list_reservations",
        "manage_reservation",
        "search_experiences",
    }


def test_cultpass_server_call_round_trips_through_real_logic():
    async def run():
        async with Client(cultpass_mcp) as client:
            return await client.call_tool("get_customer_profile", {"external_user_id": "u1"})

    result = _run(run())

    assert result.data == {
        "ok": True,
        "data": {
            "user_id": "u1",
            "full_name": "Test User",
            "email": "t@example.com",
            "is_blocked": False,
        },
    }


def test_cultpass_server_call_surfaces_tool_level_errors():
    async def run():
        async with Client(cultpass_mcp) as client:
            return await client.call_tool("get_customer_profile", {"external_user_id": "no-such-user"})

    result = _run(run())

    assert result.data == {"ok": False, "error": "No CultPass user found for id 'no-such-user'"}


def test_udahub_server_lists_all_tools():
    async def run():
        async with Client(udahub_mcp) as client:
            return {t.name for t in await client.list_tools()}

    names = _run(run())

    assert names == {
        "search_knowledge_base",
        "get_ticket_history",
        "update_ticket_record",
        "get_internal_user_id",
        "recall_customer_memory",
        "save_customer_memory",
    }


def test_udahub_server_call_round_trips_through_real_logic():
    async def run():
        async with Client(udahub_mcp) as client:
            return await client.call_tool(
                "get_ticket_history", {"account_id": "acc1", "external_user_id": "u1"}
            )

    result = _run(run())

    assert result.data == {"ok": True, "data": []}
