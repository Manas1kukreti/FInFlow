"""Add employee chat conversations, datasets, and executions.

Revision ID: 0028_add_employee_chat_tables
Revises: 0027_add_job_visualizations
Create Date: 2026-06-28 00:00:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0028_add_employee_chat_tables"
down_revision = "0027_add_job_visualizations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conversation_status_enum = postgresql.ENUM(
        "active",
        "archived",
        "closed",
        name="chat_conversation_status",
        create_type=False,
    )
    conversation_status_enum.create(op.get_bind(), checkfirst=True)

    message_role_enum = postgresql.ENUM(
        "user",
        "assistant",
        "system",
        name="chat_message_role",
        create_type=False,
    )
    message_role_enum.create(op.get_bind(), checkfirst=True)

    message_type_enum = postgresql.ENUM(
        "text",
        "clarification",
        "result",
        "table",
        "chart",
        "file",
        "error",
        "permission_denied",
        "status",
        name="chat_message_type",
        create_type=False,
    )
    message_type_enum.create(op.get_bind(), checkfirst=True)

    execution_status_enum = postgresql.ENUM(
        "received",
        "interpreting",
        "grounding",
        "awaiting_clarification",
        "authorization_check",
        "queued",
        "executing",
        "composing_response",
        "completed",
        "permission_denied",
        "unsupported",
        "failed",
        "cancelled",
        name="chat_execution_status",
        create_type=False,
    )
    execution_status_enum.create(op.get_bind(), checkfirst=True)

    execution_phase_enum = postgresql.ENUM(
        "received",
        "interpreting",
        "grounding",
        "authorization_check",
        "queued",
        "executing",
        "composing_response",
        "completed",
        name="chat_execution_phase",
        create_type=False,
    )
    execution_phase_enum.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "chat_datasets",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("dataset_key", sa.String(length=120), nullable=False),
        sa.Column("display_name", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("domain", sa.String(length=80), nullable=False, server_default="general"),
        sa.Column("physical_source", sa.String(length=255), nullable=False, server_default="chat_dataset_rows"),
        sa.Column("storage_location", sa.String(length=255), nullable=False, server_default="chat_dataset_rows.payload"),
        sa.Column("schema_version", sa.String(length=32), nullable=False, server_default="1.0"),
        sa.Column("columns_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("aliases", postgresql.ARRAY(sa.String(length=120)), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("sample_values_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("owner", sa.String(length=120), nullable=False, server_default=""),
        sa.Column("sensitivity", sa.String(length=40), nullable=False, server_default="internal"),
        sa.Column("allowed_roles", postgresql.ARRAY(sa.String(length=32)), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("row_level_policy_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("column_level_policy_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("chat_enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("last_updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("dataset_key", name="uq_chat_datasets_dataset_key"),
    )

    op.create_table(
        "chat_dataset_rows",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("dataset_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("chat_datasets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("row_index", sa.Integer(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("dataset_id", "row_index", name="uq_chat_dataset_rows_dataset_row_index"),
    )
    op.create_index("ix_chat_dataset_rows_dataset_id", "chat_dataset_rows", ["dataset_id"])

    op.create_table(
        "chat_conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("employee_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(length=240), nullable=False, server_default="New conversation"),
        sa.Column("status", conversation_status_enum, nullable=False, server_default="active"),
        sa.Column("active_dataset_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("chat_datasets.id"), nullable=True),
        sa.Column("context_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("last_successful_intent_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("last_result_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("pending_clarification_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("allowed_follow_up_operations", postgresql.ARRAY(sa.String(length=64)), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("last_message_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_chat_conversations_employee_id", "chat_conversations", ["employee_id"])
    op.create_index("ix_chat_conversations_status", "chat_conversations", ["status"])
    op.create_index("ix_chat_conversations_active_dataset_id", "chat_conversations", ["active_dataset_id"])

    op.create_table(
        "chat_messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("chat_conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", message_role_enum, nullable=False),
        sa.Column("message_type", message_type_enum, nullable=False, server_default="text"),
        sa.Column("content", sa.Text(), nullable=False, server_default=""),
        sa.Column("status", sa.String(length=40), nullable=False, server_default="received"),
        sa.Column("canonical_intent_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("grounded_intent_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("result_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("clarification_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("execution_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("reply_to_message_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("chat_messages.id", ondelete="SET NULL"), nullable=True),
        sa.Column("client_message_id", sa.String(length=128), nullable=True),
        sa.Column("error_code", sa.String(length=80), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("conversation_id", "client_message_id", name="uq_chat_messages_conversation_client_message"),
    )
    op.create_index("ix_chat_messages_conversation_id", "chat_messages", ["conversation_id"])
    op.create_index("ix_chat_messages_execution_id", "chat_messages", ["execution_id"])

    op.create_table(
        "chat_intent_revisions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("chat_conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("message_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("chat_messages.id", ondelete="CASCADE"), nullable=False),
        sa.Column("revision_number", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("intent_version", sa.String(length=32), nullable=False, server_default="1.0"),
        sa.Column("parent_revision_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("chat_intent_revisions.id", ondelete="SET NULL"), nullable=True),
        sa.Column("original_message", sa.Text(), nullable=False, server_default=""),
        sa.Column("canonical_intent_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("grounded_intent_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("clarification_answer_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("conversation_id", "revision_number", name="uq_chat_intent_revisions_conversation_revision"),
    )
    op.create_index("ix_chat_intent_revisions_conversation_id", "chat_intent_revisions", ["conversation_id"])

    op.create_table(
        "chat_executions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("chat_conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("message_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("chat_messages.id", ondelete="CASCADE"), nullable=False),
        sa.Column("dataset_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("chat_datasets.id"), nullable=True),
        sa.Column("status", execution_status_enum, nullable=False, server_default="received"),
        sa.Column("phase", execution_phase_enum, nullable=False, server_default="received"),
        sa.Column("canonical_intent_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("grounded_intent_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("result_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error_code", sa.String(length=80), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("client_message_id", sa.String(length=128), nullable=True),
        sa.Column("request_hash", sa.String(length=64), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_chat_executions_conversation_id", "chat_executions", ["conversation_id"])
    op.create_index("ix_chat_executions_message_id", "chat_executions", ["message_id"])
    op.create_index("ix_chat_executions_status", "chat_executions", ["status"])
    op.create_index("ix_chat_executions_phase", "chat_executions", ["phase"])

    op.create_foreign_key(
        "fk_chat_messages_execution_id_chat_executions",
        "chat_messages",
        "chat_executions",
        ["execution_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_index("ix_chat_executions_phase", table_name="chat_executions")
    op.drop_index("ix_chat_executions_status", table_name="chat_executions")
    op.drop_index("ix_chat_executions_message_id", table_name="chat_executions")
    op.drop_index("ix_chat_executions_conversation_id", table_name="chat_executions")
    op.drop_table("chat_executions")

    op.drop_constraint("fk_chat_messages_execution_id_chat_executions", "chat_messages", type_="foreignkey")
    op.drop_index("ix_chat_intent_revisions_conversation_id", table_name="chat_intent_revisions")
    op.drop_table("chat_intent_revisions")

    op.drop_index("ix_chat_messages_execution_id", table_name="chat_messages")
    op.drop_index("ix_chat_messages_conversation_id", table_name="chat_messages")
    op.drop_table("chat_messages")

    op.drop_index("ix_chat_conversations_active_dataset_id", table_name="chat_conversations")
    op.drop_index("ix_chat_conversations_status", table_name="chat_conversations")
    op.drop_index("ix_chat_conversations_employee_id", table_name="chat_conversations")
    op.drop_table("chat_conversations")

    op.drop_index("ix_chat_dataset_rows_dataset_id", table_name="chat_dataset_rows")
    op.drop_table("chat_dataset_rows")
    op.drop_table("chat_datasets")

    op.execute("DROP TYPE IF EXISTS chat_execution_phase")
    op.execute("DROP TYPE IF EXISTS chat_execution_status")
    op.execute("DROP TYPE IF EXISTS chat_message_type")
    op.execute("DROP TYPE IF EXISTS chat_message_role")
    op.execute("DROP TYPE IF EXISTS chat_conversation_status")
