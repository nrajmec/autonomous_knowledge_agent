"""Tests for agentic/tools/udahub_tools.py.

Runs against a throwaway SQLite DB (via `temp_engine`), never the real
solution/data/core/udahub.db.
"""
import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine

import agentic.tools.udahub_tools as udahub_tools
from data.models import udahub
from utils import get_session


@pytest.fixture
def temp_engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test_udahub.db'}")
    udahub.Base.metadata.create_all(engine)
    return engine


@pytest.fixture(autouse=True)
def _patch_engine(monkeypatch, temp_engine):
    monkeypatch.setattr(udahub_tools, "get_udahub_engine", lambda: temp_engine)


def _make_ticket(session, *, ticket_id, user_id, account_id, created_at, status, tags, message_count=1):
    ticket = udahub.Ticket(ticket_id=ticket_id, account_id=account_id, user_id=user_id, channel="chat")
    ticket.created_at = created_at
    session.add(ticket)
    session.add(
        udahub.TicketMetadata(ticket_id=ticket_id, status=status, main_issue_type=None, tags=tags)
    )
    for _ in range(message_count):
        session.add(
            udahub.TicketMessage(
                message_id=str(uuid.uuid4()),
                ticket_id=ticket_id,
                role=udahub.RoleEnum.user,
                content="Sample message",
            )
        )


@pytest.fixture
def seeded_data(temp_engine):
    with get_session(temp_engine) as session:
        session.add(udahub.Account(account_id="cultpass", account_name="CultPass Card"))
        session.add(
            udahub.User(
                user_id="user1", account_id="cultpass", external_user_id="ext1", user_name="Alice"
            )
        )
        _make_ticket(
            session,
            ticket_id="ticket-old",
            user_id="user1",
            account_id="cultpass",
            created_at=datetime.now() - timedelta(days=5),
            status="resolved",
            tags="billing",
        )
        _make_ticket(
            session,
            ticket_id="ticket-new",
            user_id="user1",
            account_id="cultpass",
            created_at=datetime.now(),
            status="open",
            tags="login, access",
        )
    return {"account_id": "cultpass", "external_user_id": "ext1"}


# -- get_ticket_history ----------------------------------------------------


def test_get_ticket_history_newest_first(seeded_data):
    result = udahub_tools.get_ticket_history(
        seeded_data["account_id"], seeded_data["external_user_id"]
    )

    assert result["ok"] is True
    assert [t["ticket_id"] for t in result["data"]] == ["ticket-new", "ticket-old"]


def test_get_ticket_history_excludes_given_ticket(seeded_data):
    result = udahub_tools.get_ticket_history(
        seeded_data["account_id"], seeded_data["external_user_id"], exclude_ticket_id="ticket-new"
    )

    assert [t["ticket_id"] for t in result["data"]] == ["ticket-old"]


def test_get_ticket_history_unknown_customer_returns_empty_not_error(seeded_data):
    result = udahub_tools.get_ticket_history(seeded_data["account_id"], "no-such-customer")

    assert result == {"ok": True, "data": []}


def test_get_ticket_history_requires_account_and_user_id():
    result = udahub_tools.get_ticket_history("", "")

    assert result == {"ok": False, "error": "account_id and external_user_id are required"}


# -- get_internal_user_id ----------------------------------------------------


def test_get_internal_user_id_known_customer(seeded_data):
    result = udahub_tools.get_internal_user_id(
        seeded_data["account_id"], seeded_data["external_user_id"]
    )

    assert result == {"ok": True, "data": "user1"}


def test_get_internal_user_id_unknown_customer_is_not_an_error(seeded_data):
    result = udahub_tools.get_internal_user_id(seeded_data["account_id"], "no-such-customer")

    assert result == {"ok": True, "data": None}


def test_get_internal_user_id_requires_both_ids():
    result = udahub_tools.get_internal_user_id("", "")

    assert result == {"ok": False, "error": "account_id and external_user_id are required"}


# -- update_ticket_record ----------------------------------------------------


def test_update_ticket_record_appends_message(seeded_data):
    result = udahub_tools.update_ticket_record(
        "ticket-new", message_role="ai", message_content="Have you tried resetting your password?"
    )

    assert result["ok"] is True
    assert result["data"]["message_id"] is not None
    assert result["data"]["status"] == "open"  # unchanged


def test_update_ticket_record_updates_status_and_tags(seeded_data):
    result = udahub_tools.update_ticket_record("ticket-new", status="resolved", tags="login, resolved")

    assert result["ok"] is True
    assert result["data"]["status"] == "resolved"
    assert result["data"]["tags"] == "login, resolved"


def test_update_ticket_record_reflects_in_history(seeded_data):
    udahub_tools.update_ticket_record("ticket-new", status="resolved")

    history = udahub_tools.get_ticket_history(
        seeded_data["account_id"], seeded_data["external_user_id"]
    )
    updated = next(t for t in history["data"] if t["ticket_id"] == "ticket-new")
    assert updated["status"] == "resolved"


def test_update_ticket_record_requires_ticket_id():
    result = udahub_tools.update_ticket_record("")

    assert result == {"ok": False, "error": "ticket_id is required"}


def test_update_ticket_record_requires_something_to_update(seeded_data):
    result = udahub_tools.update_ticket_record("ticket-new")

    assert result["ok"] is False
    assert "Provide at least" in result["error"]


def test_update_ticket_record_rejects_invalid_role(seeded_data):
    result = udahub_tools.update_ticket_record("ticket-new", message_role="bogus", message_content="x")

    assert result["ok"] is False
    assert "message_role must be one of" in result["error"]


def test_update_ticket_record_unknown_ticket(seeded_data):
    result = udahub_tools.update_ticket_record("no-such-ticket", status="resolved")

    assert result == {"ok": False, "error": "No ticket found for id 'no-such-ticket'"}
