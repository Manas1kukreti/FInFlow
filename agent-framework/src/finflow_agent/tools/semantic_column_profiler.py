"""Semantic column profiling for schema grounding.

This layer converts the rich dataframe profile into a compact semantic view
that downstream predicate grounding can use without sending the full sheet to
an LLM. The output is deterministic and cached per profile fingerprint.
"""

from __future__ import annotations

import copy
import hashlib
import re
from enum import Enum
from typing import Any, Iterable, List

from pydantic import BaseModel, Field

from finflow_agent.tools.dataframe_profile import ColumnProfile, DataFrameProfile


SEMANTIC_PROFILER_VERSION = "1"
_semantic_profile_cache: dict[tuple[str, str], list["SemanticColumnProfile"]] = {}


class BroadSemanticType(str, Enum):
    identifier = "identifier"
    date = "date"
    currency = "currency"
    numeric = "numeric"
    boolean = "boolean"
    categorical = "categorical"
    free_text = "free_text"
    product = "product"
    payment = "payment"
    status = "status"
    unknown = "unknown"


class SemanticColumnProfile(BaseModel):
    column: str
    broad_type: BroadSemanticType
    semantic_description: str
    semantic_tags: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)
    inference_method: str = "deterministic"
    match_score: float = Field(ge=0.0, le=1.0)


_PRODUCT_HINTS = frozenset(
    {
        "product",
        "item",
        "sku",
        "goods",
        "catalog",
        "catalogue",
        "material",
        "article",
        "title",
    }
)

_PAYMENT_HINTS = frozenset(
    {
        "payment",
        "pay",
        "method",
        "mode",
        "tender",
        "merchant",
        "vendor",
        "seller",
        "payee",
        "wallet",
        "card",
        "cash",
        "upi",
        "bank",
    }
)

_STATUS_HINTS = frozenset(
    {
        "status",
        "state",
        "stage",
        "progress",
        "outcome",
        "result",
        "flag",
        "approval",
    }
)

_IDENTIFIER_HINTS = frozenset(
    {
        "id",
        "identifier",
        "uuid",
        "key",
        "code",
        "reference",
        "ref",
        "txn",
        "transaction",
    }
)

_TEXT_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _fingerprint(profile: DataFrameProfile) -> str:
    raw = profile.model_dump_json()
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def reset_semantic_profile_cache() -> None:
    _semantic_profile_cache.clear()


def _tokens(value: Any) -> set[str]:
    return {token for token in _TEXT_TOKEN_RE.findall(str(value).lower()) if token}


def _join_tokens(values: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for value in values:
        out.update(_tokens(value))
    return out


def _choose_broad_type(column: ColumnProfile, tags: set[str]) -> BroadSemanticType:
    if column.semantic_guess == "date" or "date" in tags:
        return BroadSemanticType.date
    if column.semantic_guess == "currency" or "currency" in tags:
        return BroadSemanticType.currency
    if column.semantic_guess == "numeric":
        return BroadSemanticType.numeric
    if column.semantic_guess == "boolean" or "boolean" in tags:
        return BroadSemanticType.boolean
    if tags & _PRODUCT_HINTS:
        return BroadSemanticType.product
    if tags & _PAYMENT_HINTS:
        return BroadSemanticType.payment
    if tags & _STATUS_HINTS:
        return BroadSemanticType.status
    if tags & _IDENTIFIER_HINTS:
        return BroadSemanticType.identifier
    if column.average_text_length is not None and column.average_text_length >= 18:
        return BroadSemanticType.free_text
    if column.distinct_count <= max(10, int(max(column.non_null_count, 1) * 0.25)):
        return BroadSemanticType.categorical
    if column.semantic_guess == "categorical":
        return BroadSemanticType.categorical
    if column.semantic_guess == "string":
        return BroadSemanticType.free_text
    return BroadSemanticType.unknown


def _semantic_description(
    column: ColumnProfile,
    broad_type: BroadSemanticType,
    evidence: list[str],
) -> str:
    parts = [
        f"{column.column} is a {broad_type.value} column",
        f"dtype={column.pandas_dtype}",
        f"distinct={column.distinct_count}",
    ]
    if column.average_text_length is not None:
        parts.append(f"avg_text_len={column.average_text_length:.1f}")
    if column.numeric_min is not None and column.numeric_max is not None:
        parts.append(
            f"numeric_range={column.numeric_min}..{column.numeric_max}"
        )
    if column.date_min is not None and column.date_max is not None:
        parts.append(f"date_range={column.date_min}..{column.date_max}")
    if evidence:
        parts.append("evidence=" + "; ".join(evidence[:4]))
    return ". ".join(parts)


def _score_match(column: ColumnProfile, broad_type: BroadSemanticType, tags: set[str]) -> float:
    score = 0.35
    if broad_type in {BroadSemanticType.date, BroadSemanticType.currency, BroadSemanticType.numeric, BroadSemanticType.boolean}:
        score = 0.95
    elif broad_type in {BroadSemanticType.product, BroadSemanticType.payment, BroadSemanticType.status, BroadSemanticType.identifier}:
        score = 0.85
    elif broad_type == BroadSemanticType.categorical:
        score = 0.75
    elif broad_type == BroadSemanticType.free_text:
        score = 0.7

    if column.confidence >= 0.9:
        score = max(score, 0.9)
    if tags & _PRODUCT_HINTS and broad_type == BroadSemanticType.product:
        score = max(score, 0.92)
    if tags & _PAYMENT_HINTS and broad_type == BroadSemanticType.payment:
        score = max(score, 0.92)
    if tags & _STATUS_HINTS and broad_type == BroadSemanticType.status:
        score = max(score, 0.92)
    if tags & _IDENTIFIER_HINTS and broad_type == BroadSemanticType.identifier:
        score = max(score, 0.92)
    return min(score, 1.0)


def profile_semantic_columns(profile: DataFrameProfile) -> list[SemanticColumnProfile]:
    """Build a semantic summary for each column in *profile*.

    Results are cached per profile fingerprint and profiler version.
    """
    key = (SEMANTIC_PROFILER_VERSION, _fingerprint(profile))
    cached = _semantic_profile_cache.get(key)
    if cached is not None:
        return copy.deepcopy(cached)

    semantic_profiles: list[SemanticColumnProfile] = []
    for column in profile.columns:
        name_tokens = _tokens(column.column)
        sample_tokens = _join_tokens(
            [
                *[str(value) for value in column.representative_values],
                *[str(value) for value in column.frequent_values],
                *[str(value) for value in column.random_distinct_values],
                *[str(value) for value in column.rare_values],
            ]
        )
        tags = set(name_tokens) | sample_tokens
        evidence: list[str] = []
        if name_tokens:
            evidence.append(f"name_tokens={sorted(name_tokens)}")
        if column.representative_values:
            evidence.append(f"representative={column.representative_values[:3]}")
        if column.frequent_values:
            evidence.append(f"frequent={column.frequent_values[:3]}")
        if column.random_distinct_values:
            evidence.append(f"random={column.random_distinct_values[:3]}")
        if column.rare_values:
            evidence.append(f"rare={column.rare_values[:3]}")
        if column.semantic_guess != "unknown":
            evidence.append(f"semantic_guess={column.semantic_guess}")

        broad_type = _choose_broad_type(column, tags)
        match_score = _score_match(column, broad_type, tags)
        semantic_profiles.append(
            SemanticColumnProfile(
                column=column.column,
                broad_type=broad_type,
                semantic_description=_semantic_description(column, broad_type, evidence),
                semantic_tags=sorted(tags),
                evidence=evidence,
                inference_method="deterministic",
                match_score=match_score,
            )
        )

    _semantic_profile_cache[key] = copy.deepcopy(semantic_profiles)
    return semantic_profiles


__all__ = [
    "BroadSemanticType",
    "SemanticColumnProfile",
    "SEMANTIC_PROFILER_VERSION",
    "profile_semantic_columns",
    "reset_semantic_profile_cache",
]
