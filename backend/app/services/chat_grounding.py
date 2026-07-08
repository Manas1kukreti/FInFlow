from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.models.chat import ChatDataset
from app.services.chat_intent import ChatCanonicalIntent, ChatFieldReference, ChatFilter, ChatOperation, ChatSort
from app.services.semantic_schema import normalize_semantic_name


@dataclass(slots=True)
class GroundingOutcome:
    intent: ChatCanonicalIntent
    clarification: dict[str, Any] | None = None
    unresolved_fields: list[str] = field(default_factory=list)
    resolved_columns: list[str] = field(default_factory=list)


def _column_index(dataset: ChatDataset) -> list[dict[str, Any]]:
    columns = dataset.columns_json or []
    return [column for column in columns if isinstance(column, dict)]


def _match_column(reference: str, dataset: ChatDataset) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    normalized = normalize_semantic_name(reference)
    matches: list[dict[str, Any]] = []
    for column in _column_index(dataset):
        physical = normalize_semantic_name(str(column.get("physical_column") or ""))
        semantic = normalize_semantic_name(str(column.get("semantic_name") or ""))
        synonyms = {normalize_semantic_name(value) for value in column.get("synonyms", []) if value}
        tokens = set(physical.split("_")) | set(semantic.split("_")) | synonyms
        if normalized == physical or normalized == semantic:
            return column, [column]
        if normalized in tokens or any(token and token in normalized for token in tokens):
            matches.append(column)
            continue
        if any(token and token in normalized for token in tokens):
            matches.append(column)
    if len(matches) == 1:
        return matches[0], matches
    return None, matches


def _ground_field(field: ChatFieldReference, dataset: ChatDataset) -> tuple[ChatFieldReference, list[str], list[dict[str, Any]]]:
    column, candidates = _match_column(field.raw_reference, dataset)
    if column is None and not candidates:
        # Try sample values for value-like terms that may double as dimensions
        for candidate in _column_index(dataset):
            sample_values = candidate.get("sample_values", [])
            normalized_samples = {normalize_semantic_name(str(value)) for value in sample_values if value is not None}
            if normalize_semantic_name(field.raw_reference) in normalized_samples:
                column = candidate
                candidates = [candidate]
                break

    if column is not None:
        return (
            ChatFieldReference(
                raw_reference=field.raw_reference,
                resolved_column=str(column.get("physical_column") or column.get("semantic_name")),
                resolution_method="deterministic_match",
                candidate_columns=[str(column.get("physical_column") or column.get("semantic_name"))],
                evidence=[column.get("description") or ""],
            ),
            [str(column.get("physical_column") or column.get("semantic_name"))],
            candidates,
        )

    candidate_columns = [
        str(candidate.get("physical_column") or candidate.get("semantic_name"))
        for candidate in candidates
        if candidate.get("physical_column") or candidate.get("semantic_name")
    ]
    return (
        ChatFieldReference(
            raw_reference=field.raw_reference,
            resolved_column=None,
            resolution_method="ambiguous" if candidate_columns else None,
            candidate_columns=candidate_columns,
        ),
        [],
        candidates,
    )


def _ground_value(raw_value: Any, column: dict[str, Any]) -> Any:
    if raw_value is None:
        return None
    sample_values = column.get("sample_values", [])
    normalized_raw = normalize_semantic_name(str(raw_value))
    for sample_value in sample_values:
        if normalize_semantic_name(str(sample_value)) == normalized_raw:
            return sample_value
    return raw_value


def ground_chat_intent(intent: ChatCanonicalIntent, dataset: ChatDataset) -> GroundingOutcome:
    grounded = intent.model_copy(deep=True)
    unresolved: list[str] = []
    resolved_columns: list[str] = []
    clarification: dict[str, Any] | None = None

    grounded.dataset_reference = grounded.dataset_reference.model_copy(deep=True) if grounded.dataset_reference else None
    if grounded.dataset_reference:
        grounded.dataset_reference.resolved_dataset_id = str(dataset.id)
        grounded.dataset_reference.resolved_dataset_key = dataset.dataset_key
        grounded.dataset_reference.resolution_method = "dataset_catalog_match"

    column_lookup = {str(column.get("physical_column") or column.get("semantic_name")): column for column in _column_index(dataset)}

    new_filters: list[ChatFilter] = []
    for filter_item in grounded.filters:
        grounded_field, matched_columns, candidates = _ground_field(filter_item.field, dataset)
        if not grounded_field.resolved_column:
            unresolved.append(filter_item.field.raw_reference)
            if candidates:
                clarification = clarification or {
                    "question": f"Which field should I use for '{filter_item.field.raw_reference}'?",
                    "options": [
                        {
                            "id": str(candidate.get("physical_column") or candidate.get("semantic_name")),
                            "label": str(candidate.get("semantic_name") or candidate.get("physical_column")),
                            "description": candidate.get("description"),
                        }
                        for candidate in candidates
                    ],
                    "allow_free_text": True,
                    "reason_code": "FIELD_AMBIGUOUS",
                }
            continue
        resolved_columns.extend(matched_columns)
        grounded_value = _ground_value(filter_item.raw_value, column_lookup[grounded_field.resolved_column])
        new_filters.append(
            filter_item.model_copy(
                update={
                    "field": grounded_field,
                    "resolved_value": grounded_value,
                }
            )
        )

    grounded.filters = new_filters

    new_group_by: list[ChatFieldReference] = []
    for group_item in grounded.group_by:
        grounded_field, matched_columns, candidates = _ground_field(group_item, dataset)
        if not grounded_field.resolved_column:
            unresolved.append(group_item.raw_reference)
            if candidates and clarification is None:
                clarification = {
                    "question": f"Which field should I group by for '{group_item.raw_reference}'?",
                    "options": [
                        {
                            "id": str(candidate.get("physical_column") or candidate.get("semantic_name")),
                            "label": str(candidate.get("semantic_name") or candidate.get("physical_column")),
                            "description": candidate.get("description"),
                        }
                        for candidate in candidates
                    ],
                    "allow_free_text": True,
                    "reason_code": "FIELD_AMBIGUOUS",
                }
            continue
        resolved_columns.extend(matched_columns)
        new_group_by.append(grounded_field)
    grounded.group_by = new_group_by

    new_sort: list[ChatSort] = []
    for sort_item in grounded.sort:
        grounded_field, matched_columns, candidates = _ground_field(sort_item.field, dataset)
        if not grounded_field.resolved_column:
            unresolved.append(sort_item.field.raw_reference)
            if candidates and clarification is None:
                clarification = {
                    "question": f"Which field should I sort by for '{sort_item.field.raw_reference}'?",
                    "options": [
                        {
                            "id": str(candidate.get("physical_column") or candidate.get("semantic_name")),
                            "label": str(candidate.get("semantic_name") or candidate.get("physical_column")),
                            "description": candidate.get("description"),
                        }
                        for candidate in candidates
                    ],
                    "allow_free_text": True,
                    "reason_code": "FIELD_AMBIGUOUS",
                }
            continue
        resolved_columns.extend(matched_columns)
        new_sort.append(sort_item.model_copy(update={"field": grounded_field}))
    grounded.sort = new_sort

    new_operations: list[ChatOperation] = []
    for operation in grounded.operations:
        if operation.metric is None:
            new_operations.append(operation)
            continue
        grounded_field, matched_columns, candidates = _ground_field(operation.metric, dataset)
        if not grounded_field.resolved_column:
            unresolved.append(operation.metric.raw_reference)
            if candidates and clarification is None:
                clarification = {
                    "question": f"Which field should I use for '{operation.metric.raw_reference}'?",
                    "options": [
                        {
                            "id": str(candidate.get("physical_column") or candidate.get("semantic_name")),
                            "label": str(candidate.get("semantic_name") or candidate.get("physical_column")),
                            "description": candidate.get("description"),
                        }
                        for candidate in candidates
                    ],
                    "allow_free_text": True,
                    "reason_code": "FIELD_AMBIGUOUS",
                }
            continue
        resolved_columns.extend(matched_columns)
        operation_update = {"metric": grounded_field}
        if operation.denominator is not None:
            denom_field, denom_matches, denom_candidates = _ground_field(operation.denominator, dataset)
            if denom_field.resolved_column:
                operation_update["denominator"] = denom_field
                resolved_columns.extend(denom_matches)
            elif denom_candidates and clarification is None:
                clarification = {
                    "question": f"Which denominator field should I use for '{operation.denominator.raw_reference}'?",
                    "options": [
                        {
                            "id": str(candidate.get("physical_column") or candidate.get("semantic_name")),
                            "label": str(candidate.get("semantic_name") or candidate.get("physical_column")),
                            "description": candidate.get("description"),
                        }
                        for candidate in denom_candidates
                    ],
                    "allow_free_text": True,
                    "reason_code": "FIELD_AMBIGUOUS",
                }
                unresolved.append(operation.denominator.raw_reference)
                continue
        new_operations.append(operation.model_copy(update=operation_update))
    grounded.operations = new_operations

    if unresolved and clarification is None:
        clarification = {
            "question": f"I need one more detail about {unresolved[0]}.",
            "options": [],
            "allow_free_text": True,
            "reason_code": "FIELD_AMBIGUOUS",
        }

    return GroundingOutcome(
        intent=grounded,
        clarification=clarification,
        unresolved_fields=unresolved,
        resolved_columns=resolved_columns,
    )

