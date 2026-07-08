from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from typing import Any
from uuid import UUID, uuid4

import csv
import json
import pandas as pd
from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.models import User, UserRole
from app.models.chat import (
    ChatConversation,
    ChatConversationStatus,
    ChatDataset,
    ChatDatasetRow,
    ChatExecution,
    ChatExecutionPhase,
    ChatExecutionStatus,
    ChatIntentRevision,
    ChatMessage,
    ChatMessageRole,
    ChatMessageType,
)
from app.schemas.chat import (
    ChatChartAxisRead,
    ChatChartRead,
    ChatChartSeriesRead,
    ChatClarificationOptionRead,
    ChatClarificationRead,
    ChatConversationCreateRequest,
    ChatConversationDetailRead,
    ChatConversationSummaryRead,
    ChatDatasetColumnRead,
    ChatDatasetRead,
    ChatExecutionRead,
    ChatExecutionStatusRead,
    ChatMessageRead,
    ChatResultColumnRead,
    ChatResultRead,
    ChatSendMessageRequest,
    ChatSendMessageResponse,
)
from app.services.chat_authorization import AuthorizationDecision, authorize_chat_intent
from app.services.chat_catalog import (
    build_dataset_index,
    dataset_catalog_payload,
    dataset_to_frontend_payload,
    ensure_submission_datasets,
    list_chat_datasets,
    load_dataset_frame,
    resolve_dataset,
)
from app.services.chat_grounding import GroundingOutcome, ground_chat_intent
from app.services.chat_intent import (
    ChatCanonicalIntent,
    ChatConversationContext,
    ChatDatasetReference,
    ExtractionOutcome,
    fallback_chat_intent,
)
from app.services.json_safety import make_json_safe
from app.services.websocket_manager import ws_manager

CHAT_WORKFLOW_NAME = "employee-chat"
CHAT_WORKER_SECRET_HEADER = "x-agent-service-secret"


@dataclass(slots=True)
class ChatMessageOutcome:
    response: ChatSendMessageResponse
    assistant_message: ChatMessage | None = None
    execution: ChatExecution | None = None
    clarification: dict[str, Any] | None = None
    denied: bool = False


def utc_now() -> datetime:
    return datetime.now(UTC)


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _short_preview(text: str, length: int = 140) -> str:
    cleaned = _clean_text(text)
    if len(cleaned) <= length:
        return cleaned
    return f"{cleaned[: length - 3]}..."


def _message_type_for_result(result: dict[str, Any] | None) -> str:
    if not isinstance(result, dict):
        return ChatMessageType.result.value
    result_type = str(result.get("result_type", "")).strip().lower()
    if result_type in {item.value for item in ChatMessageType}:
        return result_type
    return ChatMessageType.result.value


def _message_type_to_result_type(message_type: str | None) -> str:
    normalized = _clean_text(message_type).lower()
    if normalized in {"table", "chart", "file", "result", "error", "permission_denied", "clarification", "status"}:
        return normalized
    return "result"


def _result_read(result_json: dict[str, Any] | None) -> ChatResultRead | None:
    if not isinstance(result_json, dict):
        return None
    try:
        return ChatResultRead.model_validate(result_json)
    except Exception:
        return None


def _clarification_read(clarification_json: dict[str, Any] | None) -> ChatClarificationRead | None:
    if not isinstance(clarification_json, dict):
        return None
    options = clarification_json.get("options")
    if isinstance(options, list):
        normalized_options = [
            ChatClarificationOptionRead(
                id=_clean_text(option.get("id")),
                label=_clean_text(option.get("label")),
                description=_clean_text(option.get("description")) or None,
            )
            for option in options
            if isinstance(option, dict)
        ]
    else:
        normalized_options = []
    return ChatClarificationRead(
        question=_clean_text(clarification_json.get("question")),
        options=normalized_options,
        allow_free_text=bool(clarification_json.get("allow_free_text", True)),
        pending_intent_revision_id=UUID(str(clarification_json["pending_intent_revision_id"]))
        if clarification_json.get("pending_intent_revision_id")
        else None,
        reason_code=_clean_text(clarification_json.get("reason_code")) or None,
    )


def _serialize_dataset(dataset: ChatDataset) -> ChatDatasetRead:
    columns = [
        ChatDatasetColumnRead(
            physical_column=_clean_text(column.get("physical_column")),
            semantic_name=_clean_text(column.get("semantic_name")) or None,
            description=_clean_text(column.get("description")) or None,
            data_type=_clean_text(column.get("data_type")) or None,
            sample_values=[_clean_text(value) for value in column.get("sample_values", []) if _clean_text(value)],
            null_percentage=column.get("null_percentage"),
            distinct_count=column.get("distinct_count"),
            synonyms=[_clean_text(value) for value in column.get("synonyms", []) if _clean_text(value)],
            sensitivity=_clean_text(column.get("sensitivity")) or None,
        )
        for column in dataset.columns_json or []
        if isinstance(column, dict)
    ]
    return ChatDatasetRead(
        dataset_id=dataset.id,
        dataset_key=dataset.dataset_key,
        source_submission_id=dataset.source_submission_id,
        display_name=dataset.display_name,
        description=dataset.description,
        domain=dataset.domain,
        storage_location=dataset.storage_location,
        schema_version=dataset.schema_version,
        aliases=[_clean_text(alias) for alias in dataset.aliases or [] if _clean_text(alias)],
        columns=columns,
        sample_values=dataset.sample_values_json or {},
        owner=dataset.owner,
        sensitivity=dataset.sensitivity,
        allowed_roles=[_clean_text(role) for role in dataset.allowed_roles or [] if _clean_text(role)],
        chat_enabled=bool(dataset.chat_enabled),
        last_updated_at=dataset.last_updated_at,
    )


def _serialize_message(message: ChatMessage) -> ChatMessageRead:
    return ChatMessageRead(
        id=message.id,
        conversation_id=message.conversation_id,
        role=message.role.value if hasattr(message.role, "value") else str(message.role),
        message_type=message.message_type.value if hasattr(message.message_type, "value") else str(message.message_type),
        content=message.content,
        status=message.status,
        canonical_intent=make_json_safe(message.canonical_intent_json) if isinstance(message.canonical_intent_json, dict) else None,
        grounded_intent=make_json_safe(message.grounded_intent_json) if isinstance(message.grounded_intent_json, dict) else None,
        result=_result_read(message.result_json),
        clarification=_clarification_read(message.clarification_json),
        execution_id=message.execution_id,
        reply_to_message_id=message.reply_to_message_id,
        error_code=message.error_code,
        created_at=message.created_at,
        updated_at=message.updated_at,
    )


def _serialize_execution(execution: ChatExecution) -> ChatExecutionRead:
    return ChatExecutionRead(
        id=execution.id,
        conversation_id=execution.conversation_id,
        message_id=execution.message_id,
        dataset_id=execution.dataset_id,
        status=execution.status.value if hasattr(execution.status, "value") else str(execution.status),
        phase=execution.phase.value if hasattr(execution.phase, "value") else str(execution.phase),
        canonical_intent=make_json_safe(execution.canonical_intent_json) if isinstance(execution.canonical_intent_json, dict) else None,
        grounded_intent=make_json_safe(execution.grounded_intent_json) if isinstance(execution.grounded_intent_json, dict) else None,
        result=_result_read(execution.result_json),
        error_code=execution.error_code,
        error_message=execution.error_message,
        client_message_id=execution.client_message_id,
        started_at=execution.started_at,
        completed_at=execution.completed_at,
        cancelled_at=execution.cancelled_at,
        updated_at=execution.updated_at,
    )


def _serialize_conversation_summary(
    conversation: ChatConversation,
    *,
    last_message_preview: str | None = None,
) -> ChatConversationSummaryRead:
    return ChatConversationSummaryRead(
        id=conversation.id,
        title=conversation.title,
        status=conversation.status.value if hasattr(conversation.status, "value") else str(conversation.status),
        employee_id=conversation.employee_id,
        active_dataset_id=conversation.active_dataset_id,
        last_message_preview=last_message_preview,
        last_message_at=conversation.last_message_at,
        updated_at=conversation.updated_at,
        created_at=conversation.created_at,
    )


def _serialize_conversation_detail(
    conversation: ChatConversation,
    *,
    messages: list[ChatMessage],
    executions: list[ChatExecution],
) -> ChatConversationDetailRead:
    last_message_preview = messages[-1].content if messages else None
    return ChatConversationDetailRead(
        **_serialize_conversation_summary(conversation, last_message_preview=last_message_preview).model_dump(mode="json"),
        context=make_json_safe(conversation.context_json or {}),
        pending_clarification=_clarification_read(conversation.pending_clarification_json),
        last_successful_intent=make_json_safe(conversation.last_successful_intent_json) if isinstance(conversation.last_successful_intent_json, dict) else None,
        last_result=_result_read(conversation.last_result_json),
        messages=[_serialize_message(message) for message in messages],
        executions=[_serialize_execution(execution) for execution in executions],
    )


def _normalize_role(role: UserRole | str) -> str:
    return role.value if hasattr(role, "value") else str(role)


def _conversation_access_allowed(conversation: ChatConversation, user: User) -> bool:
    if user.role == UserRole.admin:
        return True
    if user.role == UserRole.manager:
        return True
    return conversation.employee_id == user.id


def _assistant_status_message(
    *,
    conversation_id: UUID,
    execution_id: UUID | None,
    content: str,
    status_text: str,
    message_type: str = ChatMessageType.status.value,
) -> ChatMessage:
    return ChatMessage(
        conversation_id=conversation_id,
        execution_id=execution_id,
        role=ChatMessageRole.assistant,
        message_type=ChatMessageType(message_type) if message_type in {item.value for item in ChatMessageType} else ChatMessageType.status,
        content=content,
        status=status_text,
    )


def _intent_context(conversation: ChatConversation) -> dict[str, Any]:
    context = conversation.context_json if isinstance(conversation.context_json, dict) else {}
    context = dict(context)
    context.setdefault("conversation_id", str(conversation.id))
    if conversation.active_dataset_id:
        context.setdefault("active_dataset_id", str(conversation.active_dataset_id))
    return context


def _clarification_payload(
    *,
    question: str,
    options: Iterable[dict[str, Any]] = (),
    allow_free_text: bool = True,
    reason_code: str | None = None,
    pending_intent_revision_id: UUID | None = None,
) -> dict[str, Any]:
    return {
        "question": question,
        "options": [make_json_safe(option) for option in options],
        "allow_free_text": allow_free_text,
        "reason_code": reason_code,
        "pending_intent_revision_id": str(pending_intent_revision_id) if pending_intent_revision_id else None,
    }


async def ensure_default_chat_catalog(db: AsyncSession) -> None:
    return


async def list_chat_conversations(db: AsyncSession, user: User) -> list[ChatConversationSummaryRead]:
    stmt = (
        select(ChatConversation)
        .options(selectinload(ChatConversation.messages))
        .order_by(ChatConversation.updated_at.desc(), ChatConversation.created_at.desc())
    )
    if user.role != UserRole.admin:
        stmt = stmt.where(ChatConversation.employee_id == user.id)
    conversations = (await db.execute(stmt)).scalars().all()
    return [
        _serialize_conversation_summary(
            conversation,
            last_message_preview=conversation.messages[-1].content if conversation.messages else None,
        )
        for conversation in conversations
    ]


async def create_chat_conversation(
    db: AsyncSession,
    *,
    user: User,
    payload: ChatConversationCreateRequest | None = None,
) -> ChatConversationDetailRead:
    payload = payload or ChatConversationCreateRequest()
    dataset = None
    if payload.dataset_id is not None:
        dataset = await db.get(ChatDataset, payload.dataset_id)
        if dataset is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    title = _clean_text(payload.title) or "New conversation"
    conversation = ChatConversation(
        employee_id=user.id,
        title=title,
        status=ChatConversationStatus.active,
        active_dataset_id=dataset.id if dataset else None,
        context_json={},
        allowed_follow_up_operations=[],
        last_message_at=None,
    )
    db.add(conversation)
    await db.flush()
    await db.refresh(conversation)
    await db.commit()
    return await get_chat_conversation_detail(db, conversation.id, user=user)


async def get_chat_conversation_detail(
    db: AsyncSession,
    conversation_id: UUID,
    *,
    user: User,
) -> ChatConversationDetailRead:
    stmt = (
        select(ChatConversation)
        .options(
            selectinload(ChatConversation.messages),
            selectinload(ChatConversation.executions),
        )
        .where(ChatConversation.id == conversation_id)
    )
    conversation = (await db.execute(stmt)).scalars().first()
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    if not _conversation_access_allowed(conversation, user):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    messages = sorted(conversation.messages, key=lambda item: item.created_at or datetime.min.replace(tzinfo=UTC))
    executions = sorted(conversation.executions, key=lambda item: item.created_at or datetime.min.replace(tzinfo=UTC))
    return _serialize_conversation_detail(conversation, messages=messages, executions=executions)


async def list_chat_datasets_for_user(db: AsyncSession, user: User) -> list[ChatDatasetRead]:
    datasets = await list_chat_datasets(db, user)
    return [_serialize_dataset(dataset) for dataset in datasets]


async def list_chat_dataset_payloads(db: AsyncSession, user: User) -> list[dict[str, Any]]:
    datasets = await list_chat_datasets(db, user)
    return [dataset_catalog_payload(dataset) for dataset in datasets]


def _latest_revision_number(conversation: ChatConversation) -> int:
    revisions = conversation.intent_revisions or []
    if not revisions:
        return 0
    return max(int(revision.revision_number or 0) for revision in revisions)


async def _create_intent_revision(
    db: AsyncSession,
    *,
    conversation: ChatConversation,
    message: ChatMessage,
    canonical_intent: dict[str, Any],
    grounded_intent: dict[str, Any] | None = None,
    clarification_answer_json: dict[str, Any] | None = None,
) -> ChatIntentRevision:
    next_revision = (
        await db.scalar(
            select(func.coalesce(func.max(ChatIntentRevision.revision_number), 0)).where(
                ChatIntentRevision.conversation_id == conversation.id
            )
        )
    ) or 0
    revision = ChatIntentRevision(
        conversation_id=conversation.id,
        message_id=message.id,
        revision_number=int(next_revision) + 1,
        intent_version=str(canonical_intent.get("version", "1.0")),
        parent_revision_id=conversation.intent_revisions[-1].id if conversation.intent_revisions else None,
        original_message=message.content,
        canonical_intent_json=make_json_safe(canonical_intent),
        grounded_intent_json=make_json_safe(grounded_intent) if grounded_intent else None,
        clarification_answer_json=make_json_safe(clarification_answer_json) if clarification_answer_json else None,
    )
    db.add(revision)
    await db.flush()
    await db.refresh(revision)
    return revision


def _response_payload(
    *,
    conversation_id: UUID,
    message_id: UUID,
    status_text: str,
    execution_id: UUID | None = None,
    message_type: str | None = None,
) -> ChatSendMessageResponse:
    return ChatSendMessageResponse(
        message_id=message_id,
        execution_id=execution_id,
        status=status_text,
        message_type=message_type,
        conversation_id=conversation_id,
    )


async def submit_chat_message(
    db: AsyncSession,
    *,
    user: User,
    conversation_id: UUID,
    payload: ChatSendMessageRequest,
) -> ChatMessageOutcome:
    stmt = (
        select(ChatConversation)
        .options(
            selectinload(ChatConversation.messages),
            selectinload(ChatConversation.executions),
            selectinload(ChatConversation.intent_revisions),
        )
        .where(ChatConversation.id == conversation_id)
    )
    conversation = (await db.execute(stmt)).scalars().first()
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    if not _conversation_access_allowed(conversation, user):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    if payload.client_message_id:
        existing = (
            await db.execute(
                select(ChatMessage).where(
                    ChatMessage.conversation_id == conversation.id,
                    ChatMessage.client_message_id == payload.client_message_id,
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            execution = None
            if existing.execution_id:
                execution = await db.get(ChatExecution, existing.execution_id)
            return ChatMessageOutcome(
                response=_response_payload(
                    conversation_id=conversation.id,
                    message_id=existing.id,
                    execution_id=execution.id if execution else None,
                    status_text=execution.status.value if execution else existing.status,
                    message_type=existing.message_type.value if hasattr(existing.message_type, "value") else str(existing.message_type),
                ),
                assistant_message=existing,
                execution=execution,
            )

    user_message = ChatMessage(
        conversation_id=conversation.id,
        role=ChatMessageRole.user,
        message_type=ChatMessageType.text,
        content=_clean_text(payload.message),
        status="received",
        reply_to_message_id=payload.reply_to_message_id,
        client_message_id=payload.client_message_id,
    )
    db.add(user_message)
    await db.flush()

    if _clean_text(conversation.title) in {"New conversation", ""}:
        conversation.title = _short_preview(payload.message, 56) or "Employee chat"

    datasets = await list_chat_datasets(db, user)
    dataset_payloads = [dataset_catalog_payload(dataset) for dataset in datasets]
    previous_intent = conversation.last_successful_intent_json if isinstance(conversation.last_successful_intent_json, dict) else None
    extraction: ExtractionOutcome = fallback_chat_intent(
        payload.message,
        dataset_catalog=dataset_payloads,
        conversation_context=_intent_context(conversation),
        previous_intent=previous_intent,
    )

    canonical_intent = extraction.intent.model_dump(mode="json")
    dataset_reference = extraction.intent.dataset_reference
    dataset_resolution = await _resolve_dataset_for_message(
        db,
        user=user,
        conversation=conversation,
        dataset_reference=dataset_reference,
        dataset_catalog=dataset_payloads,
    )

    revision = await _create_intent_revision(
        db,
        conversation=conversation,
        message=user_message,
        canonical_intent=canonical_intent,
        grounded_intent=None,
    )

    if extraction.clarification_needed:
        clarification = dict(extraction.clarification or {})
        clarification["pending_intent_revision_id"] = str(revision.id)
        conversation.pending_clarification_json = clarification
        conversation.context_json = {
            **_intent_context(conversation),
            "last_intent_revision_id": str(revision.id),
            "last_user_message_id": str(user_message.id),
        }
        assistant_message = ChatMessage(
            conversation_id=conversation.id,
            role=ChatMessageRole.assistant,
            message_type=ChatMessageType.clarification,
            content=_clean_text(clarification.get("question")) or "I need one more detail.",
            status=ChatExecutionStatus.awaiting_clarification.value,
            clarification_json=clarification,
            reply_to_message_id=user_message.id,
        )
        db.add(assistant_message)
        conversation.last_message_at = utc_now()
        await db.commit()
        await ws_manager.broadcast(
            CHAT_WORKFLOW_NAME,
            "chat_message_update",
            {
                "conversation_id": str(conversation.id),
                "message_id": str(assistant_message.id),
                "message_type": assistant_message.message_type.value,
                "status": assistant_message.status,
                "clarification": clarification,
            },
        )
        return ChatMessageOutcome(
            response=_response_payload(
                conversation_id=conversation.id,
                message_id=user_message.id,
                status_text=ChatExecutionStatus.awaiting_clarification.value,
                message_type=ChatMessageType.clarification.value,
            ),
            assistant_message=assistant_message,
            clarification=clarification,
        )

    if dataset_resolution.clarification is not None and dataset_resolution.dataset is None:
        clarification = dict(dataset_resolution.clarification)
        clarification["pending_intent_revision_id"] = str(revision.id)
        conversation.pending_clarification_json = clarification
        conversation.context_json = {
            **_intent_context(conversation),
            "last_intent_revision_id": str(revision.id),
            "last_user_message_id": str(user_message.id),
        }
        assistant_message = ChatMessage(
            conversation_id=conversation.id,
            role=ChatMessageRole.assistant,
            message_type=ChatMessageType.clarification,
            content=_clean_text(clarification.get("question")) or "I need one more detail.",
            status=ChatExecutionStatus.awaiting_clarification.value,
            clarification_json=clarification,
            reply_to_message_id=user_message.id,
        )
        db.add(assistant_message)
        conversation.last_message_at = utc_now()
        await db.commit()
        await ws_manager.broadcast(
            CHAT_WORKFLOW_NAME,
            "chat_message_update",
            {
                "conversation_id": str(conversation.id),
                "message_id": str(assistant_message.id),
                "message_type": assistant_message.message_type.value,
                "status": assistant_message.status,
                "clarification": clarification,
            },
        )
        return ChatMessageOutcome(
            response=_response_payload(
                conversation_id=conversation.id,
                message_id=user_message.id,
                status_text=ChatExecutionStatus.awaiting_clarification.value,
                message_type=ChatMessageType.clarification.value,
            ),
            assistant_message=assistant_message,
            clarification=clarification,
        )

    dataset = dataset_resolution.dataset
    if dataset is None:
        if conversation.active_dataset_id is not None:
            dataset = await db.get(ChatDataset, conversation.active_dataset_id)
        elif len(datasets) == 1:
            dataset = datasets[0]
    if dataset is None:
        clarification = _clarification_payload(
            question="Which dataset should I use?",
            options=[{"id": str(item["dataset_id"]), "label": item["display_name"], "description": item["description"]} for item in dataset_payloads],
            allow_free_text=False,
            reason_code="DATASET_AMBIGUOUS",
            pending_intent_revision_id=revision.id,
        )
        conversation.pending_clarification_json = clarification
        assistant_message = ChatMessage(
            conversation_id=conversation.id,
            role=ChatMessageRole.assistant,
            message_type=ChatMessageType.clarification,
            content="Which dataset should I use?",
            status=ChatExecutionStatus.awaiting_clarification.value,
            clarification_json=clarification,
            reply_to_message_id=user_message.id,
        )
        db.add(assistant_message)
        conversation.last_message_at = utc_now()
        await db.commit()
        return ChatMessageOutcome(
            response=_response_payload(
                conversation_id=conversation.id,
                message_id=user_message.id,
                status_text=ChatExecutionStatus.awaiting_clarification.value,
                message_type=ChatMessageType.clarification.value,
            ),
            assistant_message=assistant_message,
            clarification=clarification,
        )

    grounded_outcome: GroundingOutcome = ground_chat_intent(extraction.intent, dataset)
    grounded_intent = grounded_outcome.intent.model_dump(mode="json")
    revision.grounded_intent_json = grounded_intent
    conversation.active_dataset_id = dataset.id
    conversation.pending_clarification_json = None
    conversation.context_json = {
        **_intent_context(conversation),
        "active_dataset_id": str(dataset.id),
        "active_dataset_key": dataset.dataset_key,
        "last_intent_revision_id": str(revision.id),
        "last_user_message_id": str(user_message.id),
    }

    authorization = authorize_chat_intent(user, dataset, grounded_outcome.intent)
    if authorization.decision == "denied":
        denied_message = authorization.message or "You do not have permission to run that request."
        assistant_message = ChatMessage(
            conversation_id=conversation.id,
            role=ChatMessageRole.assistant,
            message_type=ChatMessageType.permission_denied,
            content=denied_message,
            status=ChatExecutionStatus.permission_denied.value,
            grounded_intent_json=grounded_intent,
            result_json={
                "result_type": "message",
                "summary": denied_message,
                "value": None,
                "formatted_value": None,
                "columns": [],
                "rows": [],
                "chart": None,
                "file": None,
            },
            error_code=authorization.reason_code or "PERMISSION_DENIED",
            reply_to_message_id=user_message.id,
        )
        execution = ChatExecution(
            conversation_id=conversation.id,
            message_id=assistant_message.id if assistant_message.id else user_message.id,
            dataset_id=dataset.id,
            status=ChatExecutionStatus.permission_denied,
            phase=ChatExecutionPhase.authorization_check,
            canonical_intent_json=canonical_intent,
            grounded_intent_json=grounded_intent,
            error_code=authorization.reason_code or "PERMISSION_DENIED",
            error_message=denied_message,
            client_message_id=payload.client_message_id,
        )
        db.add(assistant_message)
        await db.flush()
        execution.message_id = assistant_message.id
        db.add(execution)
        await db.flush()
        assistant_message.execution_id = execution.id
        conversation.last_message_at = utc_now()
        conversation.last_result_json = assistant_message.result_json
        await db.commit()
        await ws_manager.broadcast(
            CHAT_WORKFLOW_NAME,
            "chat_execution_update",
            {
                "conversation_id": str(conversation.id),
                "execution_id": str(execution.id),
                "message_id": str(assistant_message.id),
                "status": execution.status.value,
                "phase": execution.phase.value,
                "result": assistant_message.result_json,
                "message_type": assistant_message.message_type.value,
            },
        )
        return ChatMessageOutcome(
            response=_response_payload(
                conversation_id=conversation.id,
                message_id=user_message.id,
                execution_id=execution.id,
                status_text=execution.status.value,
                message_type=assistant_message.message_type.value,
            ),
            assistant_message=assistant_message,
            execution=execution,
            denied=True,
        )

    assistant_message = _assistant_status_message(
        conversation_id=conversation.id,
        execution_id=None,
        content="Working on that now.",
        status_text=ChatExecutionStatus.queued.value,
        message_type=ChatMessageType.status.value,
    )
    db.add(assistant_message)
    await db.flush()
    execution = ChatExecution(
        conversation_id=conversation.id,
        message_id=assistant_message.id,
        dataset_id=dataset.id,
        status=ChatExecutionStatus.queued,
        phase=ChatExecutionPhase.queued,
        canonical_intent_json=canonical_intent,
        grounded_intent_json=grounded_intent,
        client_message_id=payload.client_message_id,
        request_hash=None,
    )
    db.add(execution)
    await db.flush()
    assistant_message.execution_id = execution.id
    execution.message_id = assistant_message.id
    conversation.last_message_at = utc_now()
    await db.commit()

    return ChatMessageOutcome(
        response=_response_payload(
            conversation_id=conversation.id,
            message_id=user_message.id,
            execution_id=execution.id,
            status_text=execution.status.value,
            message_type=assistant_message.message_type.value,
        ),
        assistant_message=assistant_message,
        execution=execution,
    )


async def _resolve_dataset_for_message(
    db: AsyncSession,
    *,
    user: User,
    conversation: ChatConversation,
    dataset_reference: ChatDatasetReference | None,
    dataset_catalog: list[dict[str, Any]],
) -> Any:
    if dataset_reference is None:
        if conversation.active_dataset_id is not None:
            dataset = await db.get(ChatDataset, conversation.active_dataset_id)
            if dataset is not None:
                return type("DatasetResolutionResult", (), {"dataset": dataset, "clarification": None})()
        datasets = await list_chat_datasets(db, user)
        if len(datasets) == 1:
            return type("DatasetResolutionResult", (), {"dataset": datasets[0], "clarification": None})()
        return type(
            "DatasetResolutionResult",
            (),
            {
                "dataset": None,
                "clarification": _clarification_payload(
                    question="Which dataset should I use?",
                    options=[
                        {"id": str(item["dataset_id"]), "label": item["display_name"], "description": item["description"]}
                        for item in dataset_catalog
                    ],
                    allow_free_text=False,
                    reason_code="DATASET_AMBIGUOUS",
                ),
            },
        )()

    raw_reference = _clean_text(dataset_reference.raw_reference) or dataset_reference.resolved_dataset_key or ""
    if dataset_reference.resolved_dataset_id:
        dataset = await db.get(ChatDataset, UUID(str(dataset_reference.resolved_dataset_id)))
        if dataset is not None:
            return type("DatasetResolutionResult", (), {"dataset": dataset, "clarification": None})()

    resolution = await resolve_dataset(db, user=user, raw_reference=raw_reference)
    return type("DatasetResolutionResult", (), {"dataset": resolution.dataset, "clarification": resolution.clarification})()


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _apply_filters(frame: pd.DataFrame, intent: dict[str, Any]) -> pd.DataFrame:
    filters = intent.get("filters", []) if isinstance(intent, dict) else []
    if not isinstance(filters, list) or not filters:
        return frame

    filtered = frame.copy()
    for filter_item in filters:
        if not isinstance(filter_item, dict):
            continue
        field = filter_item.get("field") if isinstance(filter_item.get("field"), dict) else {}
        column = _clean_text(field.get("resolved_column") or field.get("raw_reference"))
        if not column or column not in filtered.columns:
            continue
        operator = _clean_text(filter_item.get("operator")).lower()
        value = filter_item.get("resolved_value", filter_item.get("raw_value"))
        series = filtered[column]

        if operator == "equals":
            filtered = filtered[series.astype(str).str.lower() == _clean_text(value).lower()]
        elif operator == "not_equals":
            filtered = filtered[series.astype(str).str.lower() != _clean_text(value).lower()]
        elif operator == "contains":
            filtered = filtered[series.astype(str).str.contains(_clean_text(value), case=False, na=False)]
        elif operator == "not_contains":
            filtered = filtered[~series.astype(str).str.contains(_clean_text(value), case=False, na=False)]
        elif operator in {"greater_than", "greater_than_or_equal", "less_than", "less_than_or_equal"}:
            numeric = _coerce_numeric(series)
            target = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.isna(target):
                continue
            if operator == "greater_than":
                filtered = filtered[numeric > target]
            elif operator == "greater_than_or_equal":
                filtered = filtered[numeric >= target]
            elif operator == "less_than":
                filtered = filtered[numeric < target]
            else:
                filtered = filtered[numeric <= target]
        elif operator == "between" and isinstance(value, (list, tuple)) and len(value) >= 2:
            numeric = _coerce_numeric(series)
            lower = pd.to_numeric(pd.Series([value[0]]), errors="coerce").iloc[0]
            upper = pd.to_numeric(pd.Series([value[1]]), errors="coerce").iloc[0]
            if pd.isna(lower) or pd.isna(upper):
                continue
            filtered = filtered[(numeric >= lower) & (numeric <= upper)]
        elif operator == "in" and isinstance(value, (list, tuple, set)):
            allowed = {str(item).lower() for item in value}
            filtered = filtered[series.astype(str).str.lower().isin(allowed)]
        elif operator == "not_in" and isinstance(value, (list, tuple, set)):
            blocked = {str(item).lower() for item in value}
            filtered = filtered[~series.astype(str).str.lower().isin(blocked)]
        elif operator == "is_null":
            filtered = filtered[series.isna()]
        elif operator == "is_not_null":
            filtered = filtered[~series.isna()]
    return filtered


def _format_filter_value(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_clean_text(item) for item in value if _clean_text(item))
    return _clean_text(value)


def _describe_filter_clause(filter_item: dict[str, Any]) -> str | None:
    field = filter_item.get("field") if isinstance(filter_item.get("field"), dict) else {}
    column = _clean_text(field.get("resolved_column") or field.get("raw_reference"))
    if not column:
        return None

    operator = _clean_text(filter_item.get("operator")).lower()
    value = filter_item.get("resolved_value", filter_item.get("raw_value"))
    formatted_value = _format_filter_value(value)

    if operator == "equals":
        return f"{column} is {formatted_value}"
    if operator == "not_equals":
        return f"{column} is not {formatted_value}"
    if operator == "contains":
        return f"{column} contains {formatted_value}"
    if operator == "not_contains":
        return f"{column} does not contain {formatted_value}"
    if operator == "greater_than":
        return f"{column} is greater than {formatted_value}"
    if operator == "greater_than_or_equal":
        return f"{column} is at least {formatted_value}"
    if operator == "less_than":
        return f"{column} is less than {formatted_value}"
    if operator == "less_than_or_equal":
        return f"{column} is at most {formatted_value}"
    if operator == "between" and isinstance(value, (list, tuple)) and len(value) >= 2:
        return f"{column} is between {_clean_text(value[0])} and {_clean_text(value[1])}"
    if operator == "in":
        return f"{column} is in {formatted_value}"
    if operator == "not_in":
        return f"{column} is not in {formatted_value}"
    if operator == "is_null":
        return f"{column} is blank"
    if operator == "is_not_null":
        return f"{column} is not blank"
    return f"{column} matches the applied filter"


def _describe_filters(intent: dict[str, Any]) -> str:
    filters = intent.get("filters", []) if isinstance(intent, dict) else []
    if not isinstance(filters, list) or not filters:
        return ""
    clauses = [
        clause
        for filter_item in filters
        if isinstance(filter_item, dict)
        for clause in [_describe_filter_clause(filter_item)]
        if clause
    ]
    if not clauses:
        return ""
    if len(clauses) == 1:
        return clauses[0]
    return "; ".join(clauses)


def _default_metric_column(frame: pd.DataFrame, intent: dict[str, Any]) -> str | None:
    operations = intent.get("operations", []) if isinstance(intent, dict) else []
    for operation in operations:
        if not isinstance(operation, dict):
            continue
        metric = operation.get("metric") if isinstance(operation.get("metric"), dict) else {}
        resolved = _clean_text(metric.get("resolved_column") or metric.get("raw_reference"))
        if resolved and resolved in frame.columns:
            return resolved
    numeric_columns = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    return numeric_columns[0] if numeric_columns else None


def _operation_name(operation: dict[str, Any]) -> str:
    return _clean_text(operation.get("operation")).lower()


def _metric_column(operation: dict[str, Any], frame: pd.DataFrame) -> str | None:
    metric = operation.get("metric") if isinstance(operation.get("metric"), dict) else {}
    resolved = _clean_text(metric.get("resolved_column") or metric.get("raw_reference"))
    if resolved and resolved in frame.columns:
        return resolved
    return _default_metric_column(frame, operation)


def _format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _clean_text(value)
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.2f}"


def _build_scalar_result(
    *,
    title: str,
    summary: str,
    value: Any,
    unit: str | None = None,
    columns: list[ChatResultColumnRead] | None = None,
    rows: list[dict[str, Any]] | None = None,
    chart: ChatChartRead | None = None,
    file_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "result_type": "scalar",
        "title": title,
        "summary": summary,
        "value": make_json_safe(value),
        "formatted_value": _format_number(value) if isinstance(value, (int, float)) else _clean_text(value),
        "unit": unit,
        "columns": [column.model_dump(mode="json") for column in columns or []],
        "rows": [make_json_safe(row) for row in rows or []],
        "chart": chart.model_dump(mode="json") if chart else None,
        "file": file_payload,
    }


def _build_table_result(
    *,
    title: str,
    summary: str,
    columns: list[ChatResultColumnRead],
    rows: list[dict[str, Any]],
    chart: ChatChartRead | None = None,
    file_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "result_type": "table",
        "title": title,
        "summary": summary,
        "value": None,
        "formatted_value": None,
        "unit": None,
        "columns": [column.model_dump(mode="json") for column in columns],
        "rows": [make_json_safe(row) for row in rows],
        "chart": chart.model_dump(mode="json") if chart else None,
        "file": file_payload,
    }


def _build_chart_result(
    *,
    title: str,
    summary: str,
    columns: list[ChatResultColumnRead],
    rows: list[dict[str, Any]],
    chart: ChatChartRead,
    file_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _build_table_result(title=title, summary=summary, columns=columns, rows=rows, chart=chart, file_payload=file_payload)
    payload["result_type"] = "chart"
    return payload


def _chart_from_rows(title: str, rows: list[dict[str, Any]], chart_type: str, x_field: str, y_field: str) -> ChatChartRead:
    if chart_type == "pie":
        return ChatChartRead(
            chart_type="pie",
            title=title,
            x_axis=None,
            y_axis=None,
            series=[],
            data=rows,
        )
    return ChatChartRead(
        chart_type=chart_type,
        title=title,
        x_axis=ChatChartAxisRead(field=x_field, label=x_field.replace("_", " ").title()),
        y_axis=ChatChartAxisRead(field=y_field, label=y_field.replace("_", " ").title()),
        series=[],
        data=rows,
    )


def _apply_operation_to_frame(operation: dict[str, Any], frame: pd.DataFrame) -> tuple[str, Any, str]:
    op_name = _operation_name(operation)
    metric = _metric_column(operation, frame)
    denominator = operation.get("denominator") if isinstance(operation.get("denominator"), dict) else {}
    denominator_column = _clean_text(denominator.get("resolved_column") or denominator.get("raw_reference"))
    if op_name == "count":
        return ("count", int(len(frame.index)), "records")
    if metric is None or metric not in frame.columns:
        return (op_name, None, None)

    series = _coerce_numeric(frame[metric])
    clean_series = frame[metric].dropna()

    if op_name == "count_distinct":
        return ("distinct_count", int(frame[metric].astype(str).nunique(dropna=True)), metric)
    if op_name == "sum":
        return (f"sum_{metric}", float(series.sum(skipna=True)), metric)
    if op_name == "mean":
        return (f"avg_{metric}", float(series.mean(skipna=True)), metric)
    if op_name == "median":
        return (f"median_{metric}", float(series.median(skipna=True)), metric)
    if op_name == "mode":
        mode_value = clean_series.mode().iloc[0] if not clean_series.mode().empty else None
        return (f"mode_{metric}", make_json_safe(mode_value), metric)
    if op_name == "min":
        return (f"min_{metric}", make_json_safe(series.min(skipna=True)), metric)
    if op_name == "max":
        return (f"max_{metric}", make_json_safe(series.max(skipna=True)), metric)
    if op_name == "variance":
        return (f"variance_{metric}", float(series.var(skipna=True)), metric)
    if op_name == "standard_deviation":
        return (f"stdev_{metric}", float(series.std(skipna=True)), metric)
    if op_name == "percentage" and denominator_column and denominator_column in frame.columns:
        numerator = float(series.sum(skipna=True))
        denominator_series = _coerce_numeric(frame[denominator_column])
        denominator_value = float(denominator_series.sum(skipna=True))
        ratio = (numerator / denominator_value * 100.0) if denominator_value else None
        return (f"percentage_{metric}", ratio, f"{metric} / {denominator_column}")
    if op_name == "ratio" and denominator_column and denominator_column in frame.columns:
        numerator = float(series.sum(skipna=True))
        denominator_value = float(_coerce_numeric(frame[denominator_column]).sum(skipna=True))
        ratio = (numerator / denominator_value) if denominator_value else None
        return (f"ratio_{metric}", ratio, f"{metric} / {denominator_column}")
    if op_name == "percentage_change":
        ordered = series.dropna()
        if len(ordered) >= 2:
            first = float(ordered.iloc[0])
            last = float(ordered.iloc[-1])
            if first:
                return (f"pct_change_{metric}", ((last - first) / abs(first)) * 100.0, metric)
        return (f"pct_change_{metric}", None, metric)
    return (op_name, None, metric)


def _grouped_aggregate(frame: pd.DataFrame, intent: dict[str, Any]) -> tuple[list[ChatResultColumnRead], list[dict[str, Any]], str | None]:
    group_fields = []
    for group_item in intent.get("group_by", []) if isinstance(intent, dict) else []:
        if not isinstance(group_item, dict):
            continue
        resolved = _clean_text(group_item.get("resolved_column") or group_item.get("raw_reference"))
        if resolved and resolved in frame.columns:
            group_fields.append(resolved)
    if not group_fields:
        return [], [], None

    operations = intent.get("operations", []) if isinstance(intent, dict) else []
    if not operations:
        operations = [{"operation": "count"}]

    grouped = frame.groupby(group_fields, dropna=False)
    table = grouped.size().reset_index(name="count")
    result_columns = [
        ChatResultColumnRead(key=field, label=field.replace("_", " ").title(), data_type="string")
        for field in group_fields
    ]
    result_columns.append(ChatResultColumnRead(key="count", label="Count", data_type="number"))

    for operation in operations:
        if not isinstance(operation, dict):
            continue
        op_name = _operation_name(operation)
        metric = _metric_column(operation, frame)
        if op_name == "count":
            continue
        if metric is None or metric not in frame.columns:
            continue
        if op_name == "sum":
            table[f"sum_{metric}"] = grouped[metric].sum(numeric_only=True).values
            result_columns.append(ChatResultColumnRead(key=f"sum_{metric}", label=f"Sum {metric.replace('_', ' ').title()}", data_type="number"))
        elif op_name == "mean":
            table[f"avg_{metric}"] = grouped[metric].mean(numeric_only=True).values
            result_columns.append(ChatResultColumnRead(key=f"avg_{metric}", label=f"Average {metric.replace('_', ' ').title()}", data_type="number"))
        elif op_name == "median":
            table[f"median_{metric}"] = grouped[metric].median(numeric_only=True).values
            result_columns.append(ChatResultColumnRead(key=f"median_{metric}", label=f"Median {metric.replace('_', ' ').title()}", data_type="number"))
        elif op_name == "count_distinct":
            table[f"distinct_{metric}"] = grouped[metric].nunique(dropna=True).values
            result_columns.append(ChatResultColumnRead(key=f"distinct_{metric}", label=f"Distinct {metric.replace('_', ' ').title()}", data_type="number"))

    return result_columns, make_json_safe(table.to_dict(orient="records")), group_fields[0]


def _build_file_payload(execution_id: UUID, *, format_name: str, row_count: int) -> dict[str, Any]:
    filename = f"employee-chat-{execution_id}.{format_name}"
    return {
        "filename": filename,
        "format": format_name,
        "row_count": row_count,
        "download_url": f"/api/employee-chat/executions/{execution_id}/download?format={format_name}",
    }


def build_chat_result(
    *,
    execution_id: UUID,
    dataset: ChatDataset,
    intent: dict[str, Any],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    filtered = _apply_filters(frame, intent)
    intent_type = _clean_text(intent.get("intent_type")).lower()
    operations = intent.get("operations", []) if isinstance(intent, dict) else []
    group_fields = [
        _clean_text(group_item.get("resolved_column") or group_item.get("raw_reference"))
        for group_item in intent.get("group_by", []) if isinstance(group_item, dict)
        if _clean_text(group_item.get("resolved_column") or group_item.get("raw_reference"))
    ]
    limit = intent.get("limit")
    chart_type = _clean_text((intent.get("visualization") or {}).get("chart_type")).lower() if isinstance(intent.get("visualization"), dict) else ""

    if filtered.empty:
        summary = "No matching rows were found."
        columns = [ChatResultColumnRead(key=str(column), label=str(column).replace("_", " ").title(), data_type="string") for column in frame.columns]
        return _build_table_result(
            title="No matching rows",
            summary=summary,
            columns=columns,
            rows=[],
            file_payload=_build_file_payload(execution_id, format_name="csv", row_count=0),
        )

    if intent_type == "record_lookup":
        row_limit = int(limit or 10)
        rows = make_json_safe(filtered.head(row_limit).to_dict(orient="records"))
        columns = [
            ChatResultColumnRead(
                key=str(column),
                label=str(column).replace("_", " ").title(),
                data_type="number" if pd.api.types.is_numeric_dtype(filtered[column]) else "string",
            )
            for column in filtered.columns
        ]
        summary = f"Showing {min(row_limit, len(rows))} of {len(filtered)} matching employee records."
        return _build_table_result(
            title="Employee records",
            summary=summary,
            columns=columns,
            rows=rows,
            file_payload=_build_file_payload(execution_id, format_name="csv", row_count=len(rows)),
        )

    if group_fields:
        columns, rows, x_field = _grouped_aggregate(filtered, intent)
        if not rows:
            summary = "I could not build a grouped view from the available columns."
            return _build_table_result(
                title="Grouped result",
                summary=summary,
                columns=[],
                rows=[],
                file_payload=_build_file_payload(execution_id, format_name="csv", row_count=0),
            )
        if chart_type in {"pie", "bar", "line", "scatter", "histogram"}:
            if chart_type == "pie":
                category_key = x_field or group_fields[0]
                value_key = next((column.key for column in columns if column.key == "count" or column.key.startswith("sum_") or column.key.startswith("avg_")), "count")
                chart_rows = rows
                chart_title = _clean_text((intent.get("visualization") or {}).get("title")) if isinstance(intent.get("visualization"), dict) else ""
                if not chart_title:
                    chart_title = f"{dataset.display_name} summary"
                chart = ChatChartRead(
                    chart_type="pie",
                    title=chart_title,
                    x_axis=None,
                    y_axis=None,
                    series=[],
                    data=chart_rows,
                )
                chart_dict = chart.model_dump(mode="json")
                chart_dict["encoding"] = {
                    "category": category_key,
                    "value": value_key,
                    "category_label": category_key.replace("_", " ").title(),
                    "value_label": value_key.replace("_", " ").title(),
                }
                chart_dict["status"] = "ready"
                return _build_chart_result(
                    title=chart.title,
                    summary=f"Grouped by {', '.join(group_fields)}.",
                    columns=columns,
                    rows=rows,
                    chart=ChatChartRead.model_validate(chart_dict),
                    file_payload=_build_file_payload(execution_id, format_name="csv", row_count=len(rows)),
                )
            x_key = x_field or group_fields[0]
            y_key = next((column.key for column in columns if column.key == "count" or column.key.startswith("sum_") or column.key.startswith("avg_")), "count")
            chart = _chart_from_rows(
                title=(intent.get("visualization") or {}).get("title") if isinstance(intent.get("visualization"), dict) else f"{dataset.display_name} by {x_key.replace('_', ' ').title()}",
                rows=rows,
                chart_type=chart_type or "bar",
                x_field=x_key,
                y_field=y_key,
            )
            chart_dict = chart.model_dump(mode="json")
            chart_dict["encoding"] = {
                "x": x_key,
                "y": y_key,
                "x_label": x_key.replace("_", " ").title(),
                "y_label": y_key.replace("_", " ").title(),
            }
            chart_dict["status"] = "ready"
            return _build_chart_result(
                title=chart.title,
                summary=f"Grouped by {', '.join(group_fields)}.",
                columns=columns,
                rows=rows,
                chart=ChatChartRead.model_validate(chart_dict),
                file_payload=_build_file_payload(execution_id, format_name="csv", row_count=len(rows)),
            )
        summary = f"Grouped by {', '.join(group_fields)}."
        return _build_table_result(
            title=f"{dataset.display_name} grouped summary",
            summary=summary,
            columns=columns,
            rows=rows,
            file_payload=_build_file_payload(execution_id, format_name="csv", row_count=len(rows)),
        )

    if len(operations) > 1:
        rows = []
        columns: list[ChatResultColumnRead] = []
        for operation in operations:
            if not isinstance(operation, dict):
                continue
            label, value, unit = _apply_operation_to_frame(operation, filtered)
            rows.append({"metric": label.replace("_", " ").title(), "value": make_json_safe(value), "unit": unit})
        columns = [
            ChatResultColumnRead(key="metric", label="Metric", data_type="string"),
            ChatResultColumnRead(key="value", label="Value", data_type="number"),
            ChatResultColumnRead(key="unit", label="Unit", data_type="string"),
        ]
        return _build_table_result(
            title=f"{dataset.display_name} summary",
            summary="Computed multiple metrics from the filtered dataset.",
            columns=columns,
            rows=rows,
            file_payload=_build_file_payload(execution_id, format_name="csv", row_count=len(rows)),
        )

    operation = operations[0] if operations else {"operation": "count"}
    if not isinstance(operation, dict):
        operation = {"operation": "count"}
    label, value, unit = _apply_operation_to_frame(operation, filtered)
    metric = _metric_column(operation, filtered)
    op_name = _operation_name(operation)

    if op_name == "count":
        filter_summary = _describe_filters(intent)
        summary = f"There are {int(value or 0):,} matching employee records."
        if filter_summary:
            summary = f"{summary} Filter: {filter_summary}."
        return _build_scalar_result(
            title=f"{dataset.display_name} count",
            summary=summary,
            value=value,
            unit=unit,
            file_payload=_build_file_payload(execution_id, format_name="csv", row_count=int(value or 0)),
        )

    if value is None:
        summary = "I could not calculate that metric from the available columns."
        return _build_scalar_result(
            title=f"{dataset.display_name} metric",
            summary=summary,
            value=None,
            unit=None,
            file_payload=_build_file_payload(execution_id, format_name="csv", row_count=0),
        )

    metric_name = metric or label.replace("_", " ")
    filter_summary = _describe_filters(intent)
    summary = f"{label.replace('_', ' ').title()} for the filtered dataset is {_format_number(value)}."
    if filter_summary:
        summary = f"{summary} Filter: {filter_summary}."
    return _build_scalar_result(
        title=f"{dataset.display_name} {metric_name}",
        summary=summary,
        value=value,
        unit=unit,
        file_payload=_build_file_payload(execution_id, format_name="csv", row_count=1),
    )


async def finalize_execution_update(
    db: AsyncSession,
    *,
    execution_id: UUID,
    status_value: str,
    phase_value: str | None = None,
    result_json: dict[str, Any] | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    message_type: str | None = None,
    message_content: str | None = None,
) -> ChatExecutionRead:
    execution = (
        await db.execute(
            select(ChatExecution)
            .options(
                selectinload(ChatExecution.conversation).selectinload(ChatConversation.messages),
                selectinload(ChatExecution.message),
            )
            .where(ChatExecution.id == execution_id)
        )
    ).scalars().first()
    if execution is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Execution not found")

    normalized_status = status_value.lower()
    normalized_phase = phase_value.lower() if phase_value else execution.phase.value if hasattr(execution.phase, "value") else str(execution.phase)

    try:
        execution.status = ChatExecutionStatus(normalized_status)
    except Exception:
        execution.status = ChatExecutionStatus.failed
    try:
        execution.phase = ChatExecutionPhase(normalized_phase)
    except Exception:
        execution.phase = ChatExecutionPhase.executing

    execution.updated_at = utc_now()
    if execution.started_at is None and execution.status in {ChatExecutionStatus.executing, ChatExecutionStatus.queued}:
        execution.started_at = utc_now()
    if execution.status in {ChatExecutionStatus.completed, ChatExecutionStatus.failed, ChatExecutionStatus.permission_denied, ChatExecutionStatus.cancelled, ChatExecutionStatus.unsupported}:
        execution.completed_at = utc_now()
    execution.error_code = error_code
    execution.error_message = error_message
    if result_json is not None:
        execution.result_json = make_json_safe(result_json)
    if execution.message is not None:
        if message_content is not None:
            execution.message.content = message_content
        if message_type is not None:
            try:
                execution.message.message_type = ChatMessageType(message_type)
            except Exception:
                execution.message.message_type = ChatMessageType.result
        execution.message.status = execution.status.value
        execution.message.result_json = make_json_safe(result_json) if result_json is not None else execution.message.result_json
        execution.message.error_code = error_code
        execution.message.updated_at = utc_now()
    if execution.conversation is not None:
        execution.conversation.updated_at = utc_now()
        execution.conversation.last_message_at = utc_now()
        if result_json is not None and execution.status == ChatExecutionStatus.completed:
            execution.conversation.last_result_json = make_json_safe(result_json)
            execution.conversation.last_successful_intent_json = execution.grounded_intent_json or execution.canonical_intent_json
            execution.conversation.pending_clarification_json = None
    await db.commit()
    await db.refresh(execution)
    if execution.message is not None:
        await db.refresh(execution.message)
    if execution.conversation is not None:
        await db.refresh(execution.conversation)
    await ws_manager.broadcast(
        CHAT_WORKFLOW_NAME,
        "chat_execution_update",
        {
            "conversation_id": str(execution.conversation_id),
            "execution_id": str(execution.id),
            "message_id": str(execution.message_id),
            "status": execution.status.value,
            "phase": execution.phase.value,
            "result": make_json_safe(result_json) if result_json is not None else None,
            "error_code": error_code,
            "error_message": error_message,
            "message_type": message_type or (execution.message.message_type.value if execution.message else None),
        },
    )
    return _serialize_execution(execution)


async def cancel_chat_execution(db: AsyncSession, *, execution_id: UUID, user: User) -> ChatExecutionRead:
    execution = (
        await db.execute(
            select(ChatExecution)
            .options(selectinload(ChatExecution.conversation))
            .where(ChatExecution.id == execution_id)
        )
    ).scalars().first()
    if execution is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Execution not found")
    if execution.conversation and not _conversation_access_allowed(execution.conversation, user):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Execution not found")
    if execution.status not in {ChatExecutionStatus.queued, ChatExecutionStatus.executing, ChatExecutionStatus.received}:
        return _serialize_execution(execution)
    execution.status = ChatExecutionStatus.cancelled
    execution.phase = ChatExecutionPhase.completed
    execution.cancelled_at = utc_now()
    execution.updated_at = utc_now()
    if execution.message is not None:
        execution.message.status = ChatExecutionStatus.cancelled.value
        execution.message.message_type = ChatMessageType.status
        execution.message.content = "Execution cancelled."
    if execution.conversation is not None:
        execution.conversation.updated_at = utc_now()
    await db.commit()
    await ws_manager.broadcast(
        CHAT_WORKFLOW_NAME,
        "chat_execution_update",
        {
            "conversation_id": str(execution.conversation_id),
            "execution_id": str(execution.id),
            "message_id": str(execution.message_id),
            "status": execution.status.value,
            "phase": execution.phase.value,
        },
    )
    return _serialize_execution(execution)


async def get_chat_execution(
    db: AsyncSession,
    *,
    execution_id: UUID,
    user: User,
) -> ChatExecutionRead:
    stmt = (
        select(ChatExecution)
        .options(
            selectinload(ChatExecution.conversation),
            selectinload(ChatExecution.message),
        )
        .where(ChatExecution.id == execution_id)
    )
    execution = (await db.execute(stmt)).scalars().first()
    if execution is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Execution not found")
    if execution.conversation and not _conversation_access_allowed(execution.conversation, user):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Execution not found")
    return _serialize_execution(execution)


async def download_chat_execution_result(
    db: AsyncSession,
    *,
    execution_id: UUID,
    user: User,
    format_name: str = "csv",
) -> tuple[str, str, bytes]:
    execution = (
        await db.execute(
            select(ChatExecution)
            .options(selectinload(ChatExecution.conversation), selectinload(ChatExecution.message))
            .where(ChatExecution.id == execution_id)
        )
    ).scalars().first()
    if execution is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Execution not found")
    if execution.conversation and not _conversation_access_allowed(execution.conversation, user):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Execution not found")
    result = execution.result_json or (execution.message.result_json if execution.message else None)
    if not isinstance(result, dict):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Result not available")
    normalized_format = _clean_text(format_name).lower()
    rows = result.get("rows") if isinstance(result.get("rows"), list) else []
    title = _clean_text(result.get("title")) or "employee-chat-result"
    if normalized_format == "json":
        body = json.dumps(make_json_safe(result), indent=2, ensure_ascii=False).encode("utf-8")
        return f"{title}.json", "application/json", body
    buffer = StringIO()
    if rows:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: make_json_safe(value) for key, value in row.items()})
    else:
        writer = csv.DictWriter(buffer, fieldnames=["summary", "value"])
        writer.writeheader()
        writer.writerow({"summary": result.get("summary", ""), "value": result.get("formatted_value", result.get("value", ""))})
    return f"{title}.csv", "text/csv", buffer.getvalue().encode("utf-8")


def conversation_has_pending_work(conversation: ChatConversation) -> bool:
    executions = conversation.executions or []
    return any(
        execution.status in {ChatExecutionStatus.queued, ChatExecutionStatus.executing, ChatExecutionStatus.received}
        for execution in executions
    )
