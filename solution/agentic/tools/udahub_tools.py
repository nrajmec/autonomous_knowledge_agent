"""Tools that abstract read/write access to UDA-Hub's own core database.

Where `cultpass_tools.py` talks to the customer's external system, this
module talks to UDA-Hub's own `udahub.db`: ticket history and writing back
the outcome of a ticket (status changes, new messages). Same conventions as
`cultpass_tools.py`: plain functions, structured
``{"ok": True, "data": ...}`` / ``{"ok": False, "error": ...}`` returns,
no exceptions raised for expected failure cases.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from data.models import udahub
from utils import get_session

from agentic.tools.db import get_udahub_engine


def get_ticket_history(
    account_id: str, external_user_id: str, exclude_ticket_id: str | None = None
) -> dict[str, Any]:
    """Summarize a customer's past UDA-Hub tickets for this account, newest first.

    Used to build personalized context (e.g. "this is a repeat login issue")
    before classification/resolution runs.

    Args:
        account_id: UDA-Hub `Account.account_id` (e.g. "cultpass").
        external_user_id: The customer's id in the external system (CultPass
            `user_id`), i.e. `User.external_user_id`.
        exclude_ticket_id: Optional ticket id to leave out (typically the
            ticket currently being handled).

    Returns:
        {"ok": True, "data": [{"ticket_id", "channel", "created_at", "status",
        "main_issue_type", "tags", "message_count"}, ...]}. An unknown
        customer is not an error -- it just means no history yet, so
        {"ok": True, "data": []} is returned.
    """
    if not account_id or not external_user_id:
        return {"ok": False, "error": "account_id and external_user_id are required"}

    with get_session(get_udahub_engine()) as session:
        user = (
            session.query(udahub.User)
            .filter_by(account_id=account_id, external_user_id=external_user_id)
            .one_or_none()
        )
        if user is None:
            return {"ok": True, "data": []}

        tickets = [t for t in user.tickets if t.ticket_id != exclude_ticket_id]
        tickets.sort(key=lambda t: t.created_at or datetime.min, reverse=True)

        history = []
        for t in tickets:
            meta = t.ticket_metadata
            history.append(
                {
                    "ticket_id": t.ticket_id,
                    "channel": t.channel,
                    "created_at": t.created_at.isoformat() if t.created_at else None,
                    "status": meta.status if meta else None,
                    "main_issue_type": meta.main_issue_type if meta else None,
                    "tags": meta.tags if meta else None,
                    "message_count": len(t.messages),
                }
            )

        return {"ok": True, "data": history}


def get_internal_user_id(account_id: str, external_user_id: str) -> dict[str, Any]:
    """Resolve UDA-Hub's internal `User.user_id` for a customer.

    Needed anywhere long-term memory is read/written, since `CustomerMemory`
    is keyed by the internal id, not the external (CultPass) one.

    Args:
        account_id: UDA-Hub `Account.account_id`.
        external_user_id: The customer's id in the external system.

    Returns:
        {"ok": True, "data": "<user_id>"} if the customer has a UDA-Hub
        `User` record, or {"ok": True, "data": None} if not (e.g. a
        brand-new customer with no tickets yet -- not an error).
        {"ok": False, "error": "<reason>"} on missing arguments.
    """
    if not account_id or not external_user_id:
        return {"ok": False, "error": "account_id and external_user_id are required"}

    with get_session(get_udahub_engine()) as session:
        user = (
            session.query(udahub.User)
            .filter_by(account_id=account_id, external_user_id=external_user_id)
            .one_or_none()
        )
        return {"ok": True, "data": user.user_id if user else None}


_VALID_ROLES = {r.value for r in udahub.RoleEnum}


def update_ticket_record(
    ticket_id: str,
    status: str | None = None,
    tags: str | None = None,
    message_role: str | None = None,
    message_content: str | None = None,
) -> dict[str, Any]:
    """Persist an agent's outcome for a ticket: status/tags and/or a new message.

    This is the write side of the durable interaction history -- every
    resolution or escalation goes through here so it lands in `TicketMetadata`
    / `TicketMessage`, independent of (and in addition to) LangGraph's own
    session checkpointing.

    Args:
        ticket_id: The ticket to update.
        status: New `TicketMetadata.status` (e.g. "resolved", "escalated").
            Omit to leave unchanged.
        tags: New `TicketMetadata.tags`. Omit to leave unchanged.
        message_role: Role for a new `TicketMessage` ("user", "agent", "ai",
            "system"). Required together with message_content to append one.
        message_content: Content for the new message.

    Returns:
        {"ok": True, "data": {"ticket_id", "status", "tags", "message_id"}}
        on success, or {"ok": False, "error": "<reason>"} on validation
        failure or an unknown ticket_id. At least one of `status` or
        (`message_role` + `message_content`) must be provided.
    """
    if not ticket_id:
        return {"ok": False, "error": "ticket_id is required"}
    if status is None and tags is None and not (message_role and message_content):
        return {
            "ok": False,
            "error": "Provide at least status/tags, or both message_role and message_content",
        }
    if message_role is not None and message_role not in _VALID_ROLES:
        return {
            "ok": False,
            "error": f"message_role must be one of {sorted(_VALID_ROLES)}, got '{message_role}'",
        }

    with get_session(get_udahub_engine()) as session:
        ticket = session.get(udahub.Ticket, ticket_id)
        if ticket is None:
            return {"ok": False, "error": f"No ticket found for id '{ticket_id}'"}

        if status is not None or tags is not None:
            meta = ticket.ticket_metadata
            if meta is None:
                return {"ok": False, "error": f"Ticket '{ticket_id}' has no metadata row to update"}
            if status is not None:
                meta.status = status
            if tags is not None:
                meta.tags = tags

        new_message = None
        if message_role and message_content:
            new_message = udahub.TicketMessage(
                message_id=str(uuid.uuid4()),
                ticket_id=ticket_id,
                role=udahub.RoleEnum(message_role),
                content=message_content,
            )
            session.add(new_message)

        session.flush()

        return {
            "ok": True,
            "data": {
                "ticket_id": ticket_id,
                "status": ticket.ticket_metadata.status if ticket.ticket_metadata else None,
                "tags": ticket.ticket_metadata.tags if ticket.ticket_metadata else None,
                "message_id": new_message.message_id if new_message else None,
            },
        }
