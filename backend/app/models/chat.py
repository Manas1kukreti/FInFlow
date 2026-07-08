from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models import Base, enum_values


class ChatConversationStatus(str, enum.Enum):
    active = "active"
    archived = "archived"
    closed = "closed"


class ChatMessageRole(str, enum.Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class ChatMessageType(str, enum.Enum):
    text = "text"
    clarification = "clarification"
    result = "result"
    table = "table"
    chart = "chart"
    file = "file"
    error = "error"
    permission_denied = "permission_denied"
    status = "status"


class ChatExecutionStatus(str, enum.Enum):
    received = "received"
    interpreting = "interpreting"
    grounding = "grounding"
    awaiting_clarification = "awaiting_clarification"
    authorization_check = "authorization_check"
    queued = "queued"
    executing = "executing"
    composing_response = "composing_response"
    completed = "completed"
    permission_denied = "permission_denied"
    unsupported = "unsupported"
    failed = "failed"
    cancelled = "cancelled"


class ChatExecutionPhase(str, enum.Enum):
    received = "received"
    interpreting = "interpreting"
    grounding = "grounding"
    authorization_check = "authorization_check"
    queued = "queued"
    executing = "executing"
    composing_response = "composing_response"
    completed = "completed"


class ChatDataset(Base):
    __tablename__ = "chat_datasets"
    __table_args__ = (
        UniqueConstraint("dataset_key", name="uq_chat_datasets_dataset_key"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_key: Mapped[str] = mapped_column(String(120), nullable=False)
    source_submission_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("submissions.id", ondelete="CASCADE"))
    display_name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    domain: Mapped[str] = mapped_column(String(80), nullable=False, default="general")
    physical_source: Mapped[str] = mapped_column(String(255), nullable=False, default="chat_dataset_rows")
    storage_location: Mapped[str] = mapped_column(String(255), nullable=False, default="chat_dataset_rows.payload")
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False, default="1.0")
    columns_json: Mapped[list[dict]] = mapped_column(JSONB, nullable=False, default=list)
    aliases: Mapped[list[str]] = mapped_column(ARRAY(String(120)), nullable=False, default=list)
    sample_values_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    owner: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    sensitivity: Mapped[str] = mapped_column(String(40), nullable=False, default="internal")
    allowed_roles: Mapped[list[str]] = mapped_column(ARRAY(String(32)), nullable=False, default=list)
    row_level_policy_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    column_level_policy_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    chat_enabled: Mapped[bool] = mapped_column(default=True, nullable=False)
    last_updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    rows: Mapped[list["ChatDatasetRow"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")
    conversations: Mapped[list["ChatConversation"]] = relationship(back_populates="active_dataset")


class ChatDatasetRow(Base):
    __tablename__ = "chat_dataset_rows"
    __table_args__ = (
        UniqueConstraint("dataset_id", "row_index", name="uq_chat_dataset_rows_dataset_row_index"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat_datasets.id", ondelete="CASCADE"), nullable=False)
    row_index: Mapped[int] = mapped_column(Integer, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    dataset: Mapped[ChatDataset] = relationship(back_populates="rows")


class ChatConversation(Base):
    __tablename__ = "chat_conversations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(String(240), nullable=False, default="New conversation")
    status: Mapped[ChatConversationStatus] = mapped_column(
        Enum(ChatConversationStatus, name="chat_conversation_status", values_callable=enum_values),
        nullable=False,
        default=ChatConversationStatus.active,
    )
    active_dataset_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("chat_datasets.id"))
    context_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    last_successful_intent_json: Mapped[dict | None] = mapped_column(JSONB)
    last_result_json: Mapped[dict | None] = mapped_column(JSONB)
    pending_clarification_json: Mapped[dict | None] = mapped_column(JSONB)
    allowed_follow_up_operations: Mapped[list[str]] = mapped_column(ARRAY(String(64)), nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_message_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    active_dataset: Mapped[ChatDataset | None] = relationship(back_populates="conversations")
    messages: Mapped[list["ChatMessage"]] = relationship(back_populates="conversation", cascade="all, delete-orphan")
    intent_revisions: Mapped[list["ChatIntentRevision"]] = relationship(back_populates="conversation", cascade="all, delete-orphan")
    executions: Mapped[list["ChatExecution"]] = relationship(back_populates="conversation", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    __table_args__ = (
        UniqueConstraint("conversation_id", "client_message_id", name="uq_chat_messages_conversation_client_message"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat_conversations.id", ondelete="CASCADE"), nullable=False)
    role: Mapped[ChatMessageRole] = mapped_column(
        Enum(ChatMessageRole, name="chat_message_role", values_callable=enum_values),
        nullable=False,
    )
    message_type: Mapped[ChatMessageType] = mapped_column(
        Enum(ChatMessageType, name="chat_message_type", values_callable=enum_values),
        nullable=False,
        default=ChatMessageType.text,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(String(40), nullable=False, default="received")
    canonical_intent_json: Mapped[dict | None] = mapped_column(JSONB)
    grounded_intent_json: Mapped[dict | None] = mapped_column(JSONB)
    result_json: Mapped[dict | None] = mapped_column(JSONB)
    clarification_json: Mapped[dict | None] = mapped_column(JSONB)
    execution_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("chat_executions.id", ondelete="SET NULL"))
    reply_to_message_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("chat_messages.id", ondelete="SET NULL"))
    client_message_id: Mapped[str | None] = mapped_column(String(128))
    error_code: Mapped[str | None] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    conversation: Mapped[ChatConversation] = relationship(back_populates="messages")
    reply_to_message: Mapped["ChatMessage | None"] = relationship(remote_side=[id])
    execution: Mapped["ChatExecution | None"] = relationship(foreign_keys=[execution_id])


class ChatIntentRevision(Base):
    __tablename__ = "chat_intent_revisions"
    __table_args__ = (
        UniqueConstraint("conversation_id", "revision_number", name="uq_chat_intent_revisions_conversation_revision"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat_conversations.id", ondelete="CASCADE"), nullable=False)
    message_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat_messages.id", ondelete="CASCADE"), nullable=False)
    revision_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    intent_version: Mapped[str] = mapped_column(String(32), nullable=False, default="1.0")
    parent_revision_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("chat_intent_revisions.id", ondelete="SET NULL"))
    original_message: Mapped[str] = mapped_column(Text, nullable=False, default="")
    canonical_intent_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    grounded_intent_json: Mapped[dict | None] = mapped_column(JSONB)
    clarification_answer_json: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    conversation: Mapped[ChatConversation] = relationship(back_populates="intent_revisions")
    message: Mapped[ChatMessage] = relationship()
    parent_revision: Mapped["ChatIntentRevision | None"] = relationship(remote_side=[id])


class ChatExecution(Base):
    __tablename__ = "chat_executions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat_conversations.id", ondelete="CASCADE"), nullable=False)
    message_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat_messages.id", ondelete="CASCADE"), nullable=False)
    dataset_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("chat_datasets.id"))
    status: Mapped[ChatExecutionStatus] = mapped_column(
        Enum(ChatExecutionStatus, name="chat_execution_status", values_callable=enum_values),
        nullable=False,
        default=ChatExecutionStatus.received,
    )
    phase: Mapped[ChatExecutionPhase] = mapped_column(
        Enum(ChatExecutionPhase, name="chat_execution_phase", values_callable=enum_values),
        nullable=False,
        default=ChatExecutionPhase.received,
    )
    canonical_intent_json: Mapped[dict | None] = mapped_column(JSONB)
    grounded_intent_json: Mapped[dict | None] = mapped_column(JSONB)
    result_json: Mapped[dict | None] = mapped_column(JSONB)
    error_code: Mapped[str | None] = mapped_column(String(80))
    error_message: Mapped[str | None] = mapped_column(Text)
    client_message_id: Mapped[str | None] = mapped_column(String(128))
    request_hash: Mapped[str | None] = mapped_column(String(64))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    conversation: Mapped[ChatConversation] = relationship(back_populates="executions")
    message: Mapped[ChatMessage] = relationship(foreign_keys=[message_id])
    dataset: Mapped[ChatDataset | None] = relationship()
