"""Add submission linkage to chat datasets.

Revision ID: 0029_chat_dataset_link
Revises: 0028_add_employee_chat_tables
Create Date: 2026-06-28 00:00:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0029_chat_dataset_link"
down_revision = "0028_add_employee_chat_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chat_datasets",
        sa.Column("source_submission_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("submissions.id", ondelete="CASCADE"), nullable=True),
    )
    op.create_index(
        "ix_chat_datasets_source_submission_id",
        "chat_datasets",
        ["source_submission_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_chat_datasets_source_submission_id", table_name="chat_datasets")
    op.drop_column("chat_datasets", "source_submission_id")
