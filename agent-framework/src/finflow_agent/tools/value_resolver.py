from __future__ import annotations

import re
from typing import Any, Iterable

import pandas as pd
from pydantic import BaseModel, Field


_TEXT_DOMAIN_OPERATORS = {"eq", "contains", "starts_with", "ends_with", "in"}


class ValueResolution(BaseModel):
    """Deterministic evidence for whether a literal value exists in a column."""

    column: str
    operator: str
    requested_value: Any = None
    matched: bool
    match_kind: str = "skipped"
    match_count: int = 0
    sample_matches: list[Any] = Field(default_factory=list)
    reason: str = ""


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _compact_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _iter_requested_values(requested_value: Any, operator: str) -> Iterable[Any]:
    if operator == "in" and isinstance(requested_value, (list, tuple, set)):
        return requested_value
    return [requested_value]


def _matches_text_value(
    requested_value: Any,
    candidate_value: Any,
    operator: str,
    *,
    case_sensitive: bool,
) -> bool:
    if requested_value is None or candidate_value is None:
        return False

    request_raw = str(requested_value).strip()
    candidate_raw = str(candidate_value).strip()
    if not request_raw or not candidate_raw:
        return False

    if case_sensitive:
        request_norm = request_raw
        candidate_norm = candidate_raw
        request_compact = re.sub(r"[^A-Za-z0-9]+", "", request_raw)
        candidate_compact = re.sub(r"[^A-Za-z0-9]+", "", candidate_raw)
    else:
        request_norm = _normalize_text(request_raw)
        candidate_norm = _normalize_text(candidate_raw)
        request_compact = _compact_text(request_raw)
        candidate_compact = _compact_text(candidate_raw)

    if operator == "eq":
        return request_norm == candidate_norm or request_compact == candidate_compact

    if operator == "contains":
        return request_norm in candidate_norm or request_compact in candidate_compact

    if operator == "starts_with":
        return candidate_norm.startswith(request_norm) or candidate_compact.startswith(request_compact)

    if operator == "ends_with":
        return candidate_norm.endswith(request_norm) or candidate_compact.endswith(request_compact)

    if operator == "in":
        return request_norm == candidate_norm or request_compact == candidate_compact

    return False


def resolve_value_domain(
    series: pd.Series,
    requested_value: Any,
    operator: str,
    *,
    case_sensitive: bool = False,
) -> ValueResolution:
    """Return deterministic evidence for the requested literal value.

    Only text-like columns are validated. Numeric and datetime comparisons are
    handled by the execution layer, so they are marked as ``skipped`` here
    rather than being quarantined for lacking exact preview-domain evidence.
    """

    op = str(getattr(operator, "value", operator) or "").strip().lower()
    column_name = str(series.name or "").strip()

    if op not in _TEXT_DOMAIN_OPERATORS:
        return ValueResolution(
            column=column_name,
            operator=op,
            requested_value=requested_value,
            matched=True,
            match_kind="skipped",
            reason=f"operator {op!r} does not require text-domain validation",
        )

    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
        return ValueResolution(
            column=column_name,
            operator=op,
            requested_value=requested_value,
            matched=True,
            match_kind="skipped",
            reason=f"column {column_name!r} is numeric/datetime and is validated by execution rules",
        )

    observed = series.dropna()
    if observed.empty:
        return ValueResolution(
            column=column_name,
            operator=op,
            requested_value=requested_value,
            matched=False,
            match_kind="unresolved",
            reason=f"column {column_name!r} has no non-null values to compare against",
        )

    matches: list[Any] = []
    match_kind = "unresolved"
    for candidate in observed.tolist():
        for value in _iter_requested_values(requested_value, op):
            if _matches_text_value(value, candidate, op, case_sensitive=case_sensitive):
                matches.append(candidate)
                if match_kind == "unresolved":
                    match_kind = "exact" if str(value).strip() == str(candidate).strip() else "normalized"
                break

    if not matches:
        return ValueResolution(
            column=column_name,
            operator=op,
            requested_value=requested_value,
            matched=False,
            match_kind="unresolved",
            reason=(
                f"requested value {requested_value!r} was not observed in column "
                f"{column_name!r}"
            ),
        )

    unique_matches = list(dict.fromkeys(matches))
    return ValueResolution(
        column=column_name,
        operator=op,
        requested_value=requested_value,
        matched=True,
        match_kind=match_kind,
        match_count=len(unique_matches),
        sample_matches=unique_matches[:5],
        reason=(
            f"observed {len(unique_matches)} matching value(s) in column "
            f"{column_name!r}"
        ),
    )


__all__ = ["ValueResolution", "resolve_value_domain"]
