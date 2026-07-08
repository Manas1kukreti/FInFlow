from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.services.canonical_intent import CanonicalIntent
from app.services.semantic_schema import normalize_semantic_name

logger = logging.getLogger(__name__)


class ChatFieldReference(BaseModel):
    model_config = ConfigDict(extra="ignore")

    raw_reference: str
    resolved_column: str | None = None
    resolution_method: str | None = None
    candidate_columns: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class ChatDatasetReference(BaseModel):
    model_config = ConfigDict(extra="ignore")

    raw_reference: str
    resolved_dataset_id: str | None = None
    resolved_dataset_key: str | None = None
    candidate_dataset_ids: list[str] = Field(default_factory=list)
    candidate_dataset_keys: list[str] = Field(default_factory=list)
    resolution_method: str | None = None


class ChatOperation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    operation: Literal[
        "count",
        "count_distinct",
        "sum",
        "mean",
        "median",
        "mode",
        "min",
        "max",
        "percentage",
        "percentage_change",
        "ratio",
        "variance",
        "standard_deviation",
        "group_count",
        "group_sum",
        "group_mean",
        "group_median",
    ]
    metric: ChatFieldReference | None = None
    alias: str | None = None
    denominator: ChatFieldReference | None = None


class ChatFilter(BaseModel):
    model_config = ConfigDict(extra="ignore")

    field: ChatFieldReference
    operator: Literal[
        "equals",
        "not_equals",
        "greater_than",
        "greater_than_or_equal",
        "less_than",
        "less_than_or_equal",
        "between",
        "in",
        "not_in",
        "contains",
        "not_contains",
        "is_null",
        "is_not_null",
    ]
    raw_value: Any = None
    resolved_value: Any = None


class ChatSort(BaseModel):
    model_config = ConfigDict(extra="ignore")

    field: ChatFieldReference
    direction: Literal["asc", "desc"] = "asc"


class ChatTimeRange(BaseModel):
    model_config = ConfigDict(extra="ignore")

    start: str | None = None
    end: str | None = None
    granularity: Literal["day", "week", "month", "quarter", "year"] | None = None


class ChatVisualization(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chart_type: Literal["bar", "line", "pie", "scatter", "histogram"] | None = None
    title: str | None = None


class ChatResponsePreferences(BaseModel):
    model_config = ConfigDict(extra="ignore")

    format: Literal["natural_language", "structured"] = "natural_language"
    include_table: bool = False
    include_chart: bool = False
    include_explanation: bool = False


class ChatConversationContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    derived_from_previous_intent: bool = False
    parent_intent_revision_id: str | None = None


class ChatCanonicalIntent(CanonicalIntent):
    version: str = "1.0"
    intent_type: Literal[
        "aggregate_query",
        "record_lookup",
        "grouped_analysis",
        "comparison",
        "trend_analysis",
        "variance_analysis",
        "visualization_request",
        "export_request",
        "clarification_answer",
        "unsupported_request",
    ] = "aggregate_query"
    dataset_reference: ChatDatasetReference | None = None
    operations: list[ChatOperation] = Field(default_factory=list)
    filters: list[ChatFilter] = Field(default_factory=list)
    group_by: list[ChatFieldReference] = Field(default_factory=list)
    sort: list[ChatSort] = Field(default_factory=list)
    limit: int | None = None
    time_range: ChatTimeRange | None = None
    visualization: ChatVisualization | None = None
    response_preferences: ChatResponsePreferences = Field(default_factory=ChatResponsePreferences)
    conversation_context: ChatConversationContext = Field(default_factory=ChatConversationContext)


@dataclass(slots=True)
class ExtractionOutcome:
    intent: ChatCanonicalIntent
    confidence: float
    clarification_needed: bool = False
    clarification: dict[str, Any] | None = None
    unresolved_reason: str | None = None


def _strip_code_fences(text: str) -> str:
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _extract_json_fragment(text: str) -> str | None:
    start: int | None = None
    stack: list[str] = []
    in_string = False
    escape = False
    quote = ""
    for index, char in enumerate(text):
        if start is None:
            if char in "{[":
                start = index
                stack = [char]
            continue
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                in_string = False
            continue
        if char in "\"'":
            in_string = True
            quote = char
            continue
        if char in "{[":
            stack.append(char)
            continue
        if char in "}]":
            if not stack:
                return None
            opening = stack.pop()
            if (opening == "{" and char != "}") or (opening == "[" and char != "]"):
                return None
            if not stack and start is not None:
                return text[start : index + 1]
    return None


def _parse_json_like(raw_response: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw_response, dict):
        return raw_response
    text = _strip_code_fences(str(raw_response or "")).strip()
    if not text:
        raise ValueError("empty response")
    candidates = [text]
    fragment = _extract_json_fragment(text)
    if fragment and fragment != text:
        candidates.append(fragment)
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    raise ValueError("unable to parse structured intent")


def parse_chat_intent(raw_response: str | dict[str, Any]) -> ChatCanonicalIntent:
    payload = _parse_json_like(raw_response)
    try:
        return ChatCanonicalIntent.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Chat intent validation failed: {exc}") from exc


def build_chat_extraction_prompt(
    message: str,
    *,
    dataset_catalog: list[dict[str, Any]],
    conversation_context: dict[str, Any] | None = None,
    previous_intent: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    catalog_lines: list[str] = []
    for dataset in dataset_catalog:
        columns = dataset.get("columns", [])
        column_text = ", ".join(
            str(column.get("semantic_name") or column.get("physical_column") or "").strip()
            for column in columns
            if isinstance(column, dict)
        )
        catalog_lines.append(
            f"- {dataset.get('dataset_key')} ({dataset.get('display_name')}): {column_text}"
        )

    context_json = json.dumps(conversation_context or {}, ensure_ascii=False, default=str)
    prior_json = json.dumps(previous_intent or {}, ensure_ascii=False, default=str)
    user_message = (
        f"Employee question:\n{message}\n\n"
        f"Approved datasets:\n" + "\n".join(catalog_lines) + "\n\n"
        f"Conversation context:\n{context_json}\n\n"
        f"Previous grounded intent:\n{prior_json}\n\n"
        "Return only valid JSON matching the chat canonical intent schema."
    )
    system_message = (
        "You are a constrained intent extractor for employee data chat. "
        "Choose only approved datasets and approved columns. "
        "If the request is ambiguous, return a clarification_request object instead of guessing."
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def _make_field_reference(raw_reference: str, resolved_column: str | None = None) -> ChatFieldReference:
    return ChatFieldReference(
        raw_reference=raw_reference,
        resolved_column=resolved_column,
        resolution_method="deterministic" if resolved_column else None,
    )


def _detect_dataset_hint(message: str, dataset_catalog: list[dict[str, Any]]) -> ChatDatasetReference | None:
    normalized = normalize_semantic_name(message)
    matches: list[dict[str, Any]] = []
    for dataset in dataset_catalog:
        dataset_terms = {
            normalize_semantic_name(dataset.get("dataset_key", "")),
            normalize_semantic_name(dataset.get("display_name", "")),
            *(normalize_semantic_name(alias) for alias in dataset.get("aliases", []) if alias),
        }
        if any(term and term in normalized for term in dataset_terms):
            matches.append(dataset)
    if len(matches) == 1:
        match = matches[0]
        return ChatDatasetReference(
            raw_reference=message,
            resolved_dataset_id=str(match["dataset_id"]),
            resolved_dataset_key=str(match["dataset_key"]),
            resolution_method="alias_match",
        )
    if len(matches) > 1:
        return ChatDatasetReference(
            raw_reference=message,
            candidate_dataset_ids=[str(match["dataset_id"]) for match in matches],
            candidate_dataset_keys=[str(match["dataset_key"]) for match in matches],
            resolution_method="ambiguous",
        )
    return None


def _build_deterministic_intent(
    message: str,
    *,
    dataset_catalog: list[dict[str, Any]],
    conversation_context: dict[str, Any] | None = None,
    previous_intent: dict[str, Any] | None = None,
) -> ChatCanonicalIntent:
    text = " ".join(str(message or "").split())
    lowered = text.lower()
    dataset_ref = _detect_dataset_hint(text, dataset_catalog)

    if previous_intent and _looks_like_follow_up(lowered):
        return _patch_follow_up(previous_intent, lowered, dataset_catalog, conversation_context)

    intent_type = "aggregate_query"
    if any(token in lowered for token in ("pie chart", "bar chart", "line chart", "scatter", "histogram", "chart", "graph")):
        intent_type = "visualization_request"
    if any(token in lowered for token in ("compare", "versus", "vs", "difference")):
        intent_type = "comparison"
    if any(token in lowered for token in ("trend", "quarter", "monthly", "month", "year over year", "yoy")):
        intent_type = "trend_analysis"
    if "variance" in lowered or "standard deviation" in lowered:
        intent_type = "variance_analysis"
    if any(token in lowered for token in ("export", "download", "csv")):
        intent_type = "export_request"
    if any(token in lowered for token in ("show me the row", "show me the record", "list all employees", "show every employee", "show names", "with their names")):
        intent_type = "record_lookup"
    if any(token in lowered for token in ("by ", "per ", "group by")) and intent_type in {"aggregate_query", "visualization_request"}:
        intent_type = "grouped_analysis"

    operation = _detect_operation(lowered, intent_type)
    metric_term = _detect_metric_term(lowered, dataset_catalog)
    group_term = _detect_group_term(lowered, dataset_catalog)
    sort_term = _detect_sort_term(lowered, dataset_catalog)
    filters = _detect_filters(lowered, dataset_catalog)
    visualization = _detect_visualization(lowered)

    if intent_type == "visualization_request" and visualization is None:
        visualization = ChatVisualization(chart_type="bar")

    operations: list[ChatOperation] = []
    if operation:
        operations.append(
            ChatOperation(
                operation=operation,
                metric=_make_field_reference(metric_term) if metric_term else None,
                alias=f"{operation}_{normalize_semantic_name(metric_term or 'value')}" if metric_term else operation,
            )
        )

    group_by = [_make_field_reference(group_term)] if group_term else []
    sort = [ChatSort(field=_make_field_reference(sort_term), direction="desc")] if sort_term else []
    time_range = _detect_time_range(lowered)

    response_preferences = ChatResponsePreferences(
        format="natural_language",
        include_table=bool(group_by or intent_type in {"record_lookup", "grouped_analysis"}),
        include_chart=bool(visualization),
        include_explanation=bool(intent_type in {"comparison", "trend_analysis", "variance_analysis"}),
    )

    return ChatCanonicalIntent(
        version="1.0",
        intent_type=intent_type,
        dataset_reference=dataset_ref
        or ChatDatasetReference(raw_reference=_guess_dataset_hint(lowered, dataset_catalog) or text),
        operations=operations,
        filters=filters,
        group_by=group_by,
        sort=sort,
        limit=_detect_limit(lowered),
        time_range=time_range,
        visualization=visualization,
        response_preferences=response_preferences,
        conversation_context=ChatConversationContext(
            derived_from_previous_intent=bool(previous_intent),
            parent_intent_revision_id=str((conversation_context or {}).get("last_intent_revision_id") or "") or None,
        ),
        original_prompt=text,
        normalized_prompt=lowered,
        resolution_status="resolved",
        decision=f"{intent_type} via deterministic parser",
        evidence=["deterministic_chat_parser"],
    )


def _looks_like_follow_up(message_lower: str) -> bool:
    return bool(
        re.match(r"^(only|just|now|also|add|include|exclude|make it|change it|turn it|same as|what about)\b", message_lower)
        or message_lower in {"yes", "same", "that too"}
    )


def _patch_follow_up(
    previous_intent: dict[str, Any],
    message_lower: str,
    dataset_catalog: list[dict[str, Any]],
    conversation_context: dict[str, Any] | None = None,
) -> ChatCanonicalIntent:
    intent = ChatCanonicalIntent.model_validate(previous_intent)
    filters = list(intent.filters)
    visualization = intent.visualization

    if "pie chart" in message_lower:
        visualization = ChatVisualization(chart_type="pie")
        intent.intent_type = "visualization_request"
        intent.response_preferences.include_chart = True
    elif "bar chart" in message_lower:
        visualization = ChatVisualization(chart_type="bar")
        intent.intent_type = "visualization_request"
        intent.response_preferences.include_chart = True
    elif "line chart" in message_lower:
        visualization = ChatVisualization(chart_type="line")
        intent.intent_type = "visualization_request"
        intent.response_preferences.include_chart = True

    filter_field_hint = _detect_group_term(message_lower, dataset_catalog) or _detect_location_term(message_lower, dataset_catalog)
    filter_value_hint = _detect_value_term(message_lower)
    if "only" in message_lower or "for " in message_lower or "include" in message_lower or "exclude" in message_lower:
        if filter_field_hint and filter_value_hint:
            filters.append(
                ChatFilter(
                    field=_make_field_reference(filter_field_hint),
                    operator="equals",
                    raw_value=filter_value_hint,
                    resolved_value=filter_value_hint,
                )
            )

    intent.filters = filters
    intent.visualization = visualization
    intent.conversation_context = ChatConversationContext(
        derived_from_previous_intent=True,
        parent_intent_revision_id=str((conversation_context or {}).get("last_intent_revision_id") or "") or None,
    )
    intent.resolution_status = "resolved"
    intent.decision = "follow-up patch"
    intent.evidence = ["follow_up_patch"]
    return intent


def _detect_operation(message_lower: str, intent_type: str) -> str | None:
    if intent_type == "record_lookup":
        return "count"
    if "median" in message_lower:
        return "median"
    if "average" in message_lower or "mean" in message_lower or "avg" in message_lower:
        return "mean"
    if "count distinct" in message_lower or "unique" in message_lower:
        return "count_distinct"
    if "count" in message_lower or message_lower.startswith("how many"):
        return "count"
    if "sum" in message_lower or "total" in message_lower or "overall" in message_lower:
        return "sum"
    if "mode" in message_lower:
        return "mode"
    if "minimum" in message_lower or "lowest" in message_lower or "smallest" in message_lower or "min " in message_lower:
        return "min"
    if "maximum" in message_lower or "highest" in message_lower or "largest" in message_lower or "max " in message_lower:
        return "max"
    if "variance" in message_lower:
        return "variance"
    if "standard deviation" in message_lower or "std dev" in message_lower or "std" in message_lower:
        return "standard_deviation"
    if "percentage change" in message_lower:
        return "percentage_change"
    if "ratio" in message_lower:
        return "ratio"
    return "sum" if "revenue" in message_lower else None


def _detect_metric_term(message_lower: str, dataset_catalog: list[dict[str, Any]]) -> str | None:
    candidate_terms = [
        "annual salary",
        "salary",
        "claim amount",
        "revenue",
        "amount",
        "premium",
        "employees",
        "employee count",
        "headcount",
        "performance score",
        "tenure",
        "age",
    ]
    for term in candidate_terms:
        if term in message_lower:
            return term
    for dataset in dataset_catalog:
        for column in dataset.get("columns", []):
            if not isinstance(column, dict):
                continue
            semantic = str(column.get("semantic_name") or "").lower()
            physical = str(column.get("physical_column") or "").lower()
            if semantic and semantic in message_lower:
                return str(column.get("physical_column") or column.get("semantic_name"))
            if physical and physical in message_lower:
                return str(column.get("physical_column") or column.get("semantic_name"))
            for synonym in column.get("synonyms", []):
                if str(synonym).lower() in message_lower:
                    return str(column.get("physical_column") or column.get("semantic_name"))
    return None


def _detect_group_term(message_lower: str, dataset_catalog: list[dict[str, Any]]) -> str | None:
    if "by department" in message_lower:
        return "department"
    if "by insurance product" in message_lower:
        return "insurance_product"
    if "by quarter" in message_lower:
        return "quarter"
    if "by location" in message_lower or "by city" in message_lower:
        return "city"
    for dataset in dataset_catalog:
        for column in dataset.get("columns", []):
            if not isinstance(column, dict):
                continue
            semantic = str(column.get("semantic_name") or "").lower()
            if semantic and f"by {semantic.replace('_', ' ')}" in message_lower:
                return str(column.get("physical_column") or column.get("semantic_name"))
    return None


def _detect_sort_term(message_lower: str, dataset_catalog: list[dict[str, Any]]) -> str | None:
    if "order by" in message_lower or "sort by" in message_lower:
        match = re.search(r"(?:order|sort)\s+by\s+([a-z0-9_ ]+)", message_lower)
        if match:
            return normalize_semantic_name(match.group(1))
    return None


def _detect_limit(message_lower: str) -> int | None:
    match = re.search(r"\btop\s+(\d+)\b", message_lower)
    if match:
        return int(match.group(1))
    match = re.search(r"\bfirst\s+(\d+)\b", message_lower)
    if match:
        return int(match.group(1))
    return None


def _detect_time_range(message_lower: str) -> ChatTimeRange | None:
    if "last four quarters" in message_lower:
        return ChatTimeRange(start="last_four_quarters", end="now", granularity="quarter")
    if "this year" in message_lower:
        return ChatTimeRange(start="start_of_year", end="now", granularity="month")
    if "last month" in message_lower:
        return ChatTimeRange(start="last_month", end="now", granularity="day")
    if "last quarter" in message_lower:
        return ChatTimeRange(start="last_quarter", end="now", granularity="quarter")
    return None


def _detect_visualization(message_lower: str) -> ChatVisualization | None:
    if "pie chart" in message_lower:
        return ChatVisualization(chart_type="pie")
    if "bar chart" in message_lower or "histogram" in message_lower:
        return ChatVisualization(chart_type="bar" if "bar chart" in message_lower else "histogram")
    if "line chart" in message_lower or "trend" in message_lower:
        return ChatVisualization(chart_type="line")
    if "scatter" in message_lower:
        return ChatVisualization(chart_type="scatter")
    return None


def _detect_filters(message_lower: str, dataset_catalog: list[dict[str, Any]]) -> list[ChatFilter]:
    filters: list[ChatFilter] = []
    locations = ["chennai", "mumbai", "delhi", "bengaluru", "bangalore", "hyderabad", "pune"]
    departments = ["finance", "operations", "technology", "hr", "sales", "claims"]
    products = ["health", "motor", "life", "travel"]

    value = _detect_value_term(message_lower)
    if value:
        field = _detect_location_term(message_lower, dataset_catalog) or _detect_group_term(message_lower, dataset_catalog) or "city"
        filters.append(ChatFilter(field=_make_field_reference(field), operator="equals", raw_value=value, resolved_value=value))
        return filters

    for location in locations:
        if location in message_lower:
            filters.append(ChatFilter(field=_make_field_reference(_detect_location_term(message_lower, dataset_catalog) or "city"), operator="equals", raw_value=location.title(), resolved_value=location.title()))
            break

    for department in departments:
        if department in message_lower:
            filters.append(ChatFilter(field=_make_field_reference("department"), operator="equals", raw_value=department.title(), resolved_value=department.title()))
            break

    for product in products:
        if product in message_lower and "product" in message_lower:
            filters.append(ChatFilter(field=_make_field_reference("insurance_product"), operator="equals", raw_value=product.title(), resolved_value=product.title()))
            break

    if "finance employees" in message_lower or "finance staff" in message_lower or "finance department" in message_lower:
        filters.append(ChatFilter(field=_make_field_reference("department"), operator="equals", raw_value="Finance", resolved_value="Finance"))
    if "only for chennai" in message_lower:
        filters.append(ChatFilter(field=_make_field_reference("city"), operator="equals", raw_value="Chennai", resolved_value="Chennai"))
    return filters


def _detect_value_term(message_lower: str) -> str | None:
    for token in ["chennai", "mumbai", "delhi", "finance", "operations", "technology", "hr", "health", "motor"]:
        if token in message_lower:
            return token.title()
    return None


def _detect_location_term(message_lower: str, dataset_catalog: list[dict[str, Any]]) -> str | None:
    for column_name in ("city", "location", "office_location", "work_location"):
        if column_name in message_lower:
            return column_name
    for dataset in dataset_catalog:
        for column in dataset.get("columns", []):
            if not isinstance(column, dict):
                continue
            semantic = str(column.get("semantic_name") or "").lower()
            if semantic in {"city", "location", "office_location", "work_location"}:
                return str(column.get("physical_column") or column.get("semantic_name"))
    return None


def _guess_dataset_hint(message_lower: str, dataset_catalog: list[dict[str, Any]]) -> str | None:
    if "employee" in message_lower or "staff" in message_lower or "workforce" in message_lower:
        return "employees"
    if "claim" in message_lower:
        return "claims"
    if "revenue" in message_lower or "quarter" in message_lower:
        return "revenue"
    if dataset_catalog:
        return str(dataset_catalog[0].get("dataset_key", "employees"))
    return "employees"


def detect_ambiguities(intent: ChatCanonicalIntent, dataset_catalog: list[dict[str, Any]]) -> dict[str, Any] | None:
    dataset_ref = intent.dataset_reference
    if dataset_ref and dataset_ref.candidate_dataset_ids and not dataset_ref.resolved_dataset_id:
        return {
            "question": "Which dataset should I use?",
            "options": [
                {
                    "id": candidate_id,
                    "label": candidate_key.replace("_", " ").title(),
                }
                for candidate_id, candidate_key in zip(
                    dataset_ref.candidate_dataset_ids,
                    dataset_ref.candidate_dataset_keys,
                    strict=False,
                )
            ],
            "allow_free_text": False,
            "reason_code": "DATASET_AMBIGUOUS",
        }
    return None


def fallback_chat_intent(
    message: str,
    *,
    dataset_catalog: list[dict[str, Any]],
    conversation_context: dict[str, Any] | None = None,
    previous_intent: dict[str, Any] | None = None,
) -> ExtractionOutcome:
    intent = _build_deterministic_intent(
        message,
        dataset_catalog=dataset_catalog,
        conversation_context=conversation_context,
        previous_intent=previous_intent,
    )
    clarification = detect_ambiguities(intent, dataset_catalog)
    if clarification:
        return ExtractionOutcome(
            intent=intent,
            confidence=0.25,
            clarification_needed=True,
            clarification=clarification,
            unresolved_reason="dataset_ambiguous",
        )
    return ExtractionOutcome(intent=intent, confidence=0.82)
