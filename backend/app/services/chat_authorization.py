from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.models import User, UserRole
from app.models.chat import ChatDataset
from app.services.chat_intent import ChatCanonicalIntent


@dataclass(slots=True)
class AuthorizationDecision:
    decision: str
    reason_code: str | None = None
    restricted_columns: list[str] = None
    allowed_alternative: dict[str, Any] | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if self.restricted_columns is None:
            self.restricted_columns = []


def _restricted_columns(dataset: ChatDataset) -> set[str]:
    restricted: set[str] = set()
    for column in dataset.columns_json or []:
        if not isinstance(column, dict):
            continue
        sensitivity = str(column.get("sensitivity") or "").strip().lower()
        physical = str(column.get("physical_column") or "").strip()
        semantic = str(column.get("semantic_name") or "").strip()
        if sensitivity in {"restricted", "confidential", "pii", "sensitive"}:
            restricted.update({physical, semantic})
    return {column for column in restricted if column}


def _all_referenced_columns(intent: ChatCanonicalIntent) -> set[str]:
    columns: set[str] = set()
    if intent.dataset_reference and intent.dataset_reference.resolved_dataset_key:
        columns.add(intent.dataset_reference.resolved_dataset_key)
    for operation in intent.operations:
        if operation.metric and operation.metric.resolved_column:
            columns.add(operation.metric.resolved_column)
        if operation.denominator and operation.denominator.resolved_column:
            columns.add(operation.denominator.resolved_column)
    for chat_filter in intent.filters:
        if chat_filter.field and chat_filter.field.resolved_column:
            columns.add(chat_filter.field.resolved_column)
    for group in intent.group_by:
        if group.resolved_column:
            columns.add(group.resolved_column)
    for sort in intent.sort:
        if sort.field and sort.field.resolved_column:
            columns.add(sort.field.resolved_column)
    return columns


def authorize_chat_intent(
    user: User,
    dataset: ChatDataset,
    intent: ChatCanonicalIntent,
) -> AuthorizationDecision:
    restricted = _restricted_columns(dataset)
    referenced = _all_referenced_columns(intent)
    sensitive_hits = sorted(column for column in referenced if column in restricted)

    if intent.intent_type == "record_lookup" and sensitive_hits:
        return AuthorizationDecision(
            decision="denied",
            reason_code="COLUMN_ACCESS_RESTRICTED",
            restricted_columns=sensitive_hits,
            allowed_alternative={
                "intent_type": "aggregate_query",
                "operation": "department_level_aggregate",
            },
            message="You do not have permission to view individual restricted records. You can request department-level aggregates instead.",
        )

    if intent.intent_type == "record_lookup" and user.role == UserRole.employee:
        return AuthorizationDecision(
            decision="denied",
            reason_code="ROW_LEVEL_ACCESS_RESTRICTED",
            restricted_columns=sensitive_hits,
            allowed_alternative={
                "intent_type": "aggregate_query",
                "operation": "department_level_aggregate",
            },
            message="You do not have permission to view individual records. You can request an aggregate summary instead.",
        )

    if sensitive_hits and any(
        keyword in intent.intent_type for keyword in ("record_lookup",)
    ):
        return AuthorizationDecision(
            decision="denied",
            reason_code="COLUMN_ACCESS_RESTRICTED",
            restricted_columns=sensitive_hits,
            allowed_alternative={
                "intent_type": "grouped_analysis",
                "group_by": ["department"],
            },
            message="You do not have permission to view individual restricted columns. Try an aggregate or grouped summary instead.",
        )

    return AuthorizationDecision(decision="allowed", restricted_columns=sensitive_hits)

