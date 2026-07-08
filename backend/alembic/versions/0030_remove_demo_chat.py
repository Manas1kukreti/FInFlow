"""Remove legacy demo chat datasets and detach stale references.

Revision ID: 0030_remove_demo_chat
Revises: 0029_chat_dataset_link
Create Date: 2026-06-28 00:00:00.000000
"""
from __future__ import annotations

from alembic import op


revision = "0030_remove_demo_chat"
down_revision = "0029_chat_dataset_link"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        DELETE FROM chat_conversations
        WHERE active_dataset_id IN (
            SELECT id FROM chat_datasets
            WHERE source_submission_id IS NULL OR dataset_key = 'employee_directory'
        )
        """,
    )
    op.execute(
        """
        UPDATE chat_executions
        SET dataset_id = NULL
        WHERE dataset_id IN (
            SELECT id FROM chat_datasets
            WHERE source_submission_id IS NULL OR dataset_key = 'employee_directory'
        )
        """,
    )
    op.execute(
        """
        DELETE FROM chat_dataset_rows
        WHERE dataset_id IN (
            SELECT id FROM chat_datasets
            WHERE source_submission_id IS NULL OR dataset_key = 'employee_directory'
        )
        """,
    )
    op.execute(
        """
        DELETE FROM chat_datasets
        WHERE source_submission_id IS NULL OR dataset_key = 'employee_directory'
        """,
    )


def downgrade() -> None:
    # Legacy demo data is intentionally not restored.
    pass
