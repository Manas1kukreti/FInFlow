"""DataFrame profiler for the FinFlow Agent Service.

Produces a sanitized, bounded profile of an uploaded dataframe for schema
inspection and downstream semantic grounding. The profile never includes a
full dataframe row. It carries richer per-column evidence than the earlier
3-sample preview so the orchestrator and agent-service can reason about
column intent without sending the whole sheet to the model.
"""

from __future__ import annotations

import hashlib
import math
import re
import json
from collections import Counter
from typing import Any, List, Literal, Tuple

import pandas as pd
from pydantic import BaseModel, Field


class _PydanticJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:  # pragma: no cover - infrastructure
        if isinstance(obj, BaseModel):
            return obj.model_dump(mode="json")
        return super().default(obj)


json._default_encoder = _PydanticJSONEncoder()


# ---------------------------------------------------------------------------
# Public Pydantic models
# ---------------------------------------------------------------------------

SemanticGuess = Literal[
    "date",
    "currency",
    "numeric",
    "categorical",
    "boolean",
    "string",
    "unknown",
]


class ColumnProfile(BaseModel):
    """Sanitized profile of a single dataframe column."""

    column: str
    normalized_name: str
    pandas_dtype: str
    null_count: int = Field(ge=0)
    non_null_count: int = Field(ge=0)
    distinct_count: int = Field(ge=0)
    representative_values: List[Any] = Field(default_factory=list, max_length=5)
    frequent_values: List[Any] = Field(default_factory=list, max_length=5)
    random_distinct_values: List[Any] = Field(default_factory=list, max_length=5)
    rare_values: List[Any] = Field(default_factory=list, max_length=5)
    numeric_min: float | None = None
    numeric_max: float | None = None
    date_min: str | None = None
    date_max: str | None = None
    average_text_length: float | None = None
    semantic_guess: SemanticGuess = "unknown"
    confidence: float = Field(ge=0.0, le=1.0)

    # Backwards-compatible aliases used throughout the repo.
    @property
    def original_name(self) -> str:
        return self.column

    @property
    def dtype(self) -> str:
        return self.pandas_dtype

    @property
    def sample_values(self) -> List[Any]:
        return self.representative_values


class DataFrameProfile(BaseModel):
    """Sanitized, schema-flexible profile of a dataframe."""

    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    columns: List[ColumnProfile]
    duplicate_row_count: int = Field(ge=0)
    warnings: List[str] = Field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    @property
    def null_counts(self) -> dict[str, int]:
        return {column.column: column.null_count for column in self.columns}

    @property
    def likely_date_columns(self) -> List[str]:
        return [column.column for column in self.columns if column.semantic_guess == "date"]

    @property
    def likely_currency_columns(self) -> List[str]:
        return [
            column.column
            for column in self.columns
            if column.semantic_guess == "currency"
        ]

    @property
    def likely_numeric_columns(self) -> List[str]:
        return [
            column.column
            for column in self.columns
            if column.semantic_guess == "numeric"
        ]

    @property
    def likely_categorical_columns(self) -> List[str]:
        return [
            column.column
            for column in self.columns
            if column.semantic_guess == "categorical"
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_SAMPLE_STR_LEN: int = 64
_MAX_SAMPLE_VALUES_PER_BUCKET: int = 5
_MEMORY_WARNING_BYTES: int = 50_000_000  # >50 MB

_DATE_NAME_HINTS: Tuple[str, ...] = (
    "date",
    "time",
    "timestamp",
    "dob",
    "birthday",
    "birth_date",
    "created",
    "modified",
    "updated",
    "_at",
)

_CURRENCY_NAME_HINTS: Tuple[str, ...] = (
    "price",
    "amount",
    "cost",
    "revenue",
    "total",
    "fee",
    "salary",
    "income",
    "balance",
    "currency",
    "usd",
    "eur",
    "gbp",
)

_CATEGORICAL_NAME_HINTS: Tuple[str, ...] = (
    "status",
    "state",
    "category",
    "type",
    "kind",
    "group",
    "class",
    "segment",
    "tier",
    "mode",
)

_SENSITIVE_NAME_HINTS: Tuple[str, ...] = (
    "email",
    "e-mail",
    "phone",
    "mobile",
    "ssn",
    "social",
    "card",
    "credit",
    "debit",
    "account",
    "iban",
    "routing",
    "secret",
    "token",
    "password",
    "passwd",
    "pin",
)

_CURRENCY_SYMBOLS: Tuple[str, ...] = ("$", "€", "£", "¥", "₹")


def _normalize_name(name: Any) -> str:
    """Lowercase, collapse whitespace and dashes, drop other special chars."""
    s = str(name).strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def _stable_seed(*parts: Any) -> int:
    raw = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _looks_sensitive_name(column_name: Any) -> bool:
    normalized = _normalize_name(column_name)
    return any(hint in normalized for hint in _SENSITIVE_NAME_HINTS)


def _mask_sensitive_string(value: str) -> str:
    if "@" in value:
        local, _, domain = value.partition("@")
        if not local:
            return "<redacted>"
        return f"{local[:1]}***@{domain}"
    compact_digits = re.sub(r"\D+", "", value)
    if len(compact_digits) >= 6:
        return f"****{compact_digits[-4:]}"
    if len(value) <= 12:
        return "<redacted>"
    return f"{value[:2]}***{value[-2:]}"


def _mask_sensitive_value(column_name: Any, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value

    text = str(value)
    if _looks_sensitive_name(column_name):
        return _mask_sensitive_string(text)
    return text


def sanitize_value(value: Any, column_name: Any = None) -> Any:
    """Coerce a sample value to a JSON-safe scalar.

    Sensitive columns are masked rather than emitted verbatim.
    """
    masked = _mask_sensitive_value(column_name, value)
    if masked is None:
        return None
    if isinstance(masked, (bool, int, float)):
        return masked
    s = str(masked)
    s = "".join(ch for ch in s if ch in ("\t", "\n") or ord(ch) >= 32)
    if len(s) > _MAX_SAMPLE_STR_LEN:
        s = s[:_MAX_SAMPLE_STR_LEN]
    return s


def infer_semantic_type(
    col_name: Any,
    col_series: pd.Series,
) -> Tuple[SemanticGuess, float]:
    """Classify *col_series* as one of the supported semantic types."""
    name_lower = str(col_name).strip().lower()

    if pd.api.types.is_bool_dtype(col_series):
        return ("boolean", 1.0)

    if pd.api.types.is_datetime64_any_dtype(col_series):
        return ("date", 1.0)

    is_numeric = pd.api.types.is_numeric_dtype(col_series)
    is_string_like = (
        pd.api.types.is_string_dtype(col_series)
        or pd.api.types.is_object_dtype(col_series)
    )

    if any(hint in name_lower for hint in _DATE_NAME_HINTS):
        if is_string_like:
            non_null = col_series.dropna().head(20)
            if len(non_null) > 0:
                try:
                    parsed = pd.to_datetime(non_null, errors="coerce")
                    parse_rate = parsed.notna().sum() / len(non_null)
                    if parse_rate >= 0.6:
                        return ("date", 0.9)
                except Exception:  # pragma: no cover - defensive
                    pass
        return ("date", 0.7)

    if any(hint in name_lower for hint in _CURRENCY_NAME_HINTS):
        if is_numeric:
            return ("currency", 0.9)
        if is_string_like:
            non_null = col_series.dropna().astype(str).head(20)
            if len(non_null) > 0:
                hits = sum(
                    any(sym in v for sym in _CURRENCY_SYMBOLS) for v in non_null
                )
                if hits / len(non_null) >= 0.5:
                    return ("currency", 0.85)
        return ("currency", 0.7)

    if any(hint in name_lower for hint in _CATEGORICAL_NAME_HINTS):
        if is_string_like or not is_numeric:
            return ("categorical", 0.8)

    if is_string_like:
        non_null = col_series.dropna().astype(str).head(20)
        if len(non_null) > 0:
            hits = sum(any(sym in v for sym in _CURRENCY_SYMBOLS) for v in non_null)
            if hits / len(non_null) >= 0.7:
                return ("currency", 0.8)

    if is_numeric:
        return ("numeric", 0.85)

    if is_string_like:
        non_null_count = int(col_series.notna().sum())
        if non_null_count == 0:
            return ("string", 0.5)
        nunique = int(col_series.nunique(dropna=True))
        if (
            nunique <= 20
            and non_null_count >= 5
            and (nunique / non_null_count) <= 0.5
        ):
            return ("categorical", 0.75)
        return ("string", 0.6)

    return ("unknown", 0.0)


def _select_representative_values(
    series: pd.Series,
    *,
    column_name: Any,
    sample_rows: int,
    include_samples: bool,
) -> list[Any]:
    if not include_samples or sample_rows <= 0:
        return []
    distinct = []
    seen: set[str] = set()
    for value in series.dropna().tolist():
        key = repr(value)
        if key in seen:
            continue
        seen.add(key)
        distinct.append(value)
        if len(distinct) >= sample_rows:
            break
    return [sanitize_value(value, column_name) for value in distinct[:_MAX_SAMPLE_VALUES_PER_BUCKET]]


def _select_frequent_values(
    series: pd.Series,
    *,
    column_name: Any,
    sample_rows: int,
    include_samples: bool,
) -> list[Any]:
    if not include_samples or sample_rows <= 0:
        return []
    counts = Counter(repr(value) for value in series.dropna().tolist())
    most_common = [item for item, _ in counts.most_common(sample_rows)]
    lookup: dict[str, Any] = {}
    for value in series.dropna().tolist():
        lookup.setdefault(repr(value), value)
    return [
        sanitize_value(lookup[item], column_name)
        for item in most_common[:_MAX_SAMPLE_VALUES_PER_BUCKET]
    ]


def _select_random_distinct_values(
    series: pd.Series,
    *,
    column_name: Any,
    sample_rows: int,
    include_samples: bool,
) -> list[Any]:
    if not include_samples or sample_rows <= 0:
        return []
    distinct: list[Any] = []
    seen: set[str] = set()
    for value in series.dropna().tolist():
        key = repr(value)
        if key in seen:
            continue
        seen.add(key)
        distinct.append(value)
    if not distinct:
        return []
    if len(distinct) <= sample_rows:
        chosen = distinct
    else:
        import random

        rng = random.Random(_stable_seed(column_name, len(distinct), sample_rows))
        chosen = rng.sample(distinct, k=min(sample_rows, len(distinct)))
    return [
        sanitize_value(value, column_name)
        for value in chosen[:_MAX_SAMPLE_VALUES_PER_BUCKET]
    ]


def _select_rare_values(
    series: pd.Series,
    *,
    column_name: Any,
    sample_rows: int,
    include_samples: bool,
) -> list[Any]:
    if not include_samples or sample_rows <= 0:
        return []
    counts = Counter(repr(value) for value in series.dropna().tolist())
    if not counts:
        return []
    lookup: dict[str, Any] = {}
    for value in series.dropna().tolist():
        lookup.setdefault(repr(value), value)
    rare = sorted(counts.items(), key=lambda item: (item[1], item[0]))
    return [
        sanitize_value(lookup[item], column_name)
        for item, _count in rare[:_MAX_SAMPLE_VALUES_PER_BUCKET]
    ]


def _min_max_numeric(col_series: pd.Series) -> tuple[float | None, float | None]:
    if not pd.api.types.is_numeric_dtype(col_series):
        return (None, None)
    non_null = pd.to_numeric(col_series.dropna(), errors="coerce").dropna()
    if non_null.empty:
        return (None, None)
    return (float(non_null.min()), float(non_null.max()))


def _min_max_date(col_series: pd.Series) -> tuple[str | None, str | None]:
    if not pd.api.types.is_datetime64_any_dtype(col_series):
        return (None, None)
    non_null = pd.to_datetime(col_series.dropna(), errors="coerce").dropna()
    if non_null.empty:
        return (None, None)
    return (non_null.min().isoformat(), non_null.max().isoformat())


def _average_text_length(col_series: pd.Series) -> float | None:
    if not (
        pd.api.types.is_string_dtype(col_series)
        or pd.api.types.is_object_dtype(col_series)
    ):
        return None
    values = [str(v) for v in col_series.dropna().tolist()]
    if not values:
        return None
    return float(sum(len(v) for v in values) / len(values))


def _distinct_count(col_series: pd.Series) -> int:
    seen: set[str] = set()
    for value in col_series.dropna().tolist():
        seen.add(repr(value))
    return len(seen)


# ---------------------------------------------------------------------------
# Public profiler
# ---------------------------------------------------------------------------

def profile_dataframe(
    df: pd.DataFrame,
    sample_rows: int = 3,
    include_samples: bool = False,
) -> DataFrameProfile:
    """Produce a sanitized :class:`DataFrameProfile` of *df*.

    ``include_samples`` gates the representative/frequent/random/rare value
    lists. When it is ``False`` the profile still carries schema statistics
    such as row counts, null counts, dtype, distinct count, and min/max hints.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("profile_dataframe requires a pandas DataFrame")
    sample_rows_int = int(sample_rows)
    if not (0 <= sample_rows_int <= 10):
        raise ValueError("sample_rows must be in [0, 10]")

    column_profiles: List[ColumnProfile] = []
    warnings: List[str] = []

    for col_name in df.columns:
        col_series = df[col_name]
        column_name = str(col_name)
        observed = col_series.dropna()
        semantic_guess, confidence = infer_semantic_type(column_name, col_series)

        representative_values = _select_representative_values(
            col_series,
            column_name=column_name,
            sample_rows=sample_rows_int,
            include_samples=include_samples,
        )
        frequent_values = _select_frequent_values(
            col_series,
            column_name=column_name,
            sample_rows=sample_rows_int,
            include_samples=include_samples,
        )
        random_distinct_values = _select_random_distinct_values(
            col_series,
            column_name=column_name,
            sample_rows=sample_rows_int,
            include_samples=include_samples,
        )
        rare_values = _select_rare_values(
            col_series,
            column_name=column_name,
            sample_rows=sample_rows_int,
            include_samples=include_samples,
        )

        numeric_min, numeric_max = _min_max_numeric(col_series)
        date_min, date_max = _min_max_date(col_series)

        column_profiles.append(
            ColumnProfile(
                column=column_name,
                normalized_name=_normalize_name(column_name),
                pandas_dtype=str(col_series.dtype),
                null_count=int(col_series.isnull().sum()),
                non_null_count=int(observed.shape[0]),
                distinct_count=_distinct_count(col_series),
                representative_values=representative_values,
                frequent_values=frequent_values,
                random_distinct_values=random_distinct_values,
                rare_values=rare_values,
                numeric_min=numeric_min,
                numeric_max=numeric_max,
                date_min=date_min,
                date_max=date_max,
                average_text_length=_average_text_length(col_series),
                semantic_guess=semantic_guess,
                confidence=float(confidence),
            )
        )

    try:
        duplicate_row_count = int(df.duplicated().sum())
    except Exception:  # pragma: no cover - defensive for unhashable cells
        duplicate_row_count = int(df.astype(str).duplicated().sum())
    memory_bytes = int(df.memory_usage(deep=True).sum())

    if memory_bytes > _MEMORY_WARNING_BYTES:
        warnings.append(
            f"DataFrame exceeds 50MB (deep memory: {memory_bytes} bytes). "
            "Consider sampling."
        )

    return DataFrameProfile(
        row_count=int(df.shape[0]),
        column_count=int(df.shape[1]),
        columns=column_profiles,
        duplicate_row_count=duplicate_row_count,
        warnings=warnings,
    )


__all__ = [
    "ColumnProfile",
    "DataFrameProfile",
    "SemanticGuess",
    "infer_semantic_type",
    "sanitize_value",
    "profile_dataframe",
]
