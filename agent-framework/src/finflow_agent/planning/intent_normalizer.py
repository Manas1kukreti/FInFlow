"""Post-LLM intent normalizer for the FinFlow Agent Service.

Catches common planning gaps where the LLM produced a valid PlanIntent
but missed a user's column-selection or projection intent. Runs AFTER
PlanIntent validation, BEFORE the compiler.

The normalizer is deterministic and never calls the LLM. It pattern-matches
the user instruction against known intent phrases and patches the PlanIntent
only when the LLM clearly missed something.
"""
from __future__ import annotations

import re
from typing import List, Optional

from finflow_agent.planning.intent_schema import PlanIntent
from finflow_agent.operations.schemas import FilterOperationPlan


# Patterns that indicate the user wants column selection (projection).
# These phrases mean "give me only these columns" not "filter rows."
_COLUMN_SELECT_PATTERNS = [
    # "only return X"
    re.compile(r"\b(?:only\s+return|return\s+only)\b\s+(.+)", re.IGNORECASE),
    # "only show X"
    re.compile(r"\b(?:only\s+show|show\s+only)\b\s+(.+)", re.IGNORECASE),
    # "give me only X" / "give me just X"
    re.compile(r"\bgive\s+me\s+(?:only|just)\b\s+(.+)", re.IGNORECASE),
    # "keep only X"
    re.compile(r"\bkeep\s+only\b\s+(.+)", re.IGNORECASE),
    # "select only X"
    re.compile(r"\bselect\s+only\b\s+(.+)", re.IGNORECASE),
    # "just show X" / "just the X"
    re.compile(r"\bjust\s+(?:show|the)\b\s+(.+)", re.IGNORECASE),
    # "extract only X"
    re.compile(r"\bextract\s+only\b\s+(.+)", re.IGNORECASE),
]

# Words to strip from the captured column phrase
_STRIP_WORDS = re.compile(
    r"\b(?:column|columns|field|fields|and\s+clean|then|data)\b",
    re.IGNORECASE,
)


def _extract_column_names(phrase: str) -> List[str]:
    """Extract column name candidates from a captured phrase.

    Handles:
    - "Customer Id" → ["Customer Id"]
    - "Customer Id and Product Name" → ["Customer Id", "Product Name"]
    - "Customer Id, Product Name" → ["Customer Id", "Product Name"]
    """
    # Clean up noise words
    cleaned = _STRIP_WORDS.sub("", phrase).strip()
    # Remove trailing punctuation
    cleaned = cleaned.rstrip(".,;!?")

    # Split on commas and "and"
    parts = re.split(r"\s*,\s*|\s+and\s+", cleaned, flags=re.IGNORECASE)

    columns = []
    for part in parts:
        col = part.strip()
        if col and len(col) > 1:  # Skip single characters
            columns.append(col)

    return columns


def normalize_intent(
    intent: PlanIntent,
    instruction: str,
    available_columns: Optional[List[str]] = None,
) -> PlanIntent:
    """Normalize a PlanIntent to catch column-selection patterns the LLM missed.

    If the instruction contains a column-selection phrase AND the intent
    does NOT already have a filter_plan with select_columns, patches the
    intent to add column projection.

    Parameters
    ----------
    intent:
        The validated PlanIntent from the LLM.
    instruction:
        The original user instruction text.
    available_columns:
        The actual column names from the dataframe profile (used for
        fuzzy matching of user-mentioned names to real column names).
        When None, the normalizer uses the raw extracted names.

    Returns
    -------
    PlanIntent
        The same intent (unmodified) if no normalization is needed, or
        a patched copy with needs_filtering=True and
        filter_plan.select_columns populated.
    """
    # Skip if filtering is already configured with select_columns
    if (
        intent.needs_filtering
        and intent.filter_plan is not None
        and hasattr(intent.filter_plan, "select_columns")
        and intent.filter_plan.select_columns
    ):
        return intent

    # Try to detect a column-selection phrase in the instruction
    extracted_columns = _detect_column_selection(instruction)
    if not extracted_columns:
        return intent

    # Match extracted names to available columns if possible
    if available_columns:
        matched = _match_to_available(extracted_columns, available_columns)
    else:
        matched = extracted_columns

    if not matched:
        return intent

    # Patch the intent: add or extend filter_plan with select_columns
    if intent.filter_plan is not None:
        # Intent already has a filter plan (maybe with conditions) — add select_columns
        updated_plan = intent.filter_plan.model_copy(
            update={"select_columns": matched}
        )
    else:
        # No filter plan at all — create one with just select_columns
        updated_plan = FilterOperationPlan(
            conditions=[],
            select_columns=matched,
            logic="and",
        )

    # Return patched intent
    return intent.model_copy(
        update={
            "needs_filtering": True,
            "filter_plan": updated_plan,
        }
    )


def _detect_column_selection(instruction: str) -> List[str]:
    """Detect column-selection phrases and extract column name candidates."""
    for pattern in _COLUMN_SELECT_PATTERNS:
        match = pattern.search(instruction)
        if match:
            phrase = match.group(1)
            return _extract_column_names(phrase)
    return []


def _match_to_available(
    extracted: List[str],
    available: List[str],
) -> List[str]:
    """Fuzzy-match extracted column names against available columns.

    Uses case-insensitive matching and underscore/space normalization.
    Falls back to the extracted name if no match is found (the filter
    agent's column resolver will handle it downstream).
    """
    available_lower = {col.lower().replace("_", " "): col for col in available}
    # Also map normalized versions
    available_normalized = {
        re.sub(r"[^a-z0-9]+", " ", col.lower()).strip(): col
        for col in available
    }

    matched = []
    for ext in extracted:
        ext_lower = ext.lower().replace("_", " ").strip()
        ext_normalized = re.sub(r"[^a-z0-9]+", " ", ext.lower()).strip()

        # Try exact match (case-insensitive)
        if ext_lower in available_lower:
            matched.append(available_lower[ext_lower])
        elif ext_normalized in available_normalized:
            matched.append(available_normalized[ext_normalized])
        else:
            # Try partial match: "customer id" → "Customer_ID"
            for avail_key, avail_col in available_normalized.items():
                if ext_normalized in avail_key or avail_key in ext_normalized:
                    matched.append(avail_col)
                    break
            else:
                # No match found — pass through raw and let column resolver handle it
                matched.append(ext)

    return matched


__all__ = [
    "normalize_intent",
]
