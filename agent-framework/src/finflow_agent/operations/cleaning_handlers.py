import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List

from finflow_agent.operations.validators import get_string_columns
from finflow_agent.operations.schemas import (
    TrimWhitespaceOperation, NormalizeColumnNamesOperation, DropDuplicatesOperation,
    FillNullsOperation, DropNullsOperation, NormalizeDateOperation, NormalizeCurrencyOperation,
    NormalizeNumberOperation, NormalizeTextCaseOperation, ReplaceValuesOperation,
    StripCurrencySymbolsOperation, RemoveCommasFromNumbersOperation, CoerceColumnTypeOperation,
    RemoveEmptyRowsOperation, RemoveEmptyColumnsOperation, RenameColumnsOperation, ReorderColumnsOperation
)

def apply_trim_whitespace(df: pd.DataFrame, op: TrimWhitespaceOperation) -> Dict[str, Any]:
    cols = get_string_columns(df, op.columns)
    for col in cols:
        # Avoid casting nulls to string
        mask = df[col].notnull()
        df.loc[mask, col] = df.loc[mask, col].astype(str).str.strip()
    return {"columns_affected": cols}

def _to_snake_case(s: str) -> str:
    s = str(s).strip()
    # Replace multiple hyphens, spaces, or underscores with a single underscore
    s = re.sub(r'[\s\-_]+', '_', s)
    # camelCase/PascalCase boundaries
    s = re.sub(r'(?<!^)(?=[A-Z][a-z])', '_', s)
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
    s = re.sub(r'_+', '_', s)
    s = s.strip('_')
    return s.lower()

def _to_camel_case(s: str) -> str:
    s = _to_snake_case(s)
    parts = s.split('_')
    return parts[0] + "".join(word.capitalize() for word in parts[1:])

def _to_pascal_case(s: str) -> str:
    s = _to_snake_case(s)
    parts = s.split('_')
    return "".join(word.capitalize() for word in parts if word)

def apply_normalize_column_names(df: pd.DataFrame, op: NormalizeColumnNamesOperation) -> Dict[str, Any]:
    if op.style == "lowercase":
        df.columns = [str(c).strip().lower() for c in df.columns]
    elif op.style == "snake_case":
        df.columns = [_to_snake_case(c) for c in df.columns]
    elif op.style == "camel_case":
        df.columns = [_to_camel_case(c) for c in df.columns]
    elif op.style == "pascal_case":
        df.columns = [_to_pascal_case(c) for c in df.columns]
    return {}

def apply_drop_duplicates(df: pd.DataFrame, op: DropDuplicatesOperation) -> Dict[str, Any]:
    subset = [c for c in op.subset if c in df.columns] if op.subset else None
    df.drop_duplicates(subset=subset, keep=op.keep, inplace=True)
    return {}

def apply_fill_nulls(df: pd.DataFrame, op: FillNullsOperation) -> Dict[str, Any]:
    cols = [c for c in op.columns if c in df.columns]
    if op.strategy == "zero":
        df[cols] = df[cols].fillna(0)
    elif op.strategy == "empty_string":
        df[cols] = df[cols].fillna("")
    elif op.strategy == "mean":
        df[cols] = df[cols].fillna(df[cols].mean(numeric_only=True))
    elif op.strategy == "median":
        df[cols] = df[cols].fillna(df[cols].median(numeric_only=True))
    elif op.strategy == "mode":
        df[cols] = df[cols].fillna(df[cols].mode().iloc[0])
    elif op.strategy == "constant":
        df[cols] = df[cols].fillna(op.value)
    return {"columns_affected": cols}

def apply_drop_nulls(df: pd.DataFrame, op: DropNullsOperation) -> Dict[str, Any]:
    cols = [c for c in op.columns if c in df.columns] if op.columns else list(df.columns)
    if not cols:
        return {}

    for col in cols:
        series = df[col]
        if series.dtype == object or pd.api.types.is_string_dtype(series):
            blank_mask = series.notna() & series.astype(str).str.strip().eq("")
            if blank_mask.any():
                df.loc[blank_mask, col] = pd.NA

    subset = cols if op.columns else None
    df.dropna(subset=subset, how=op.how, inplace=True)
    return {}

def apply_normalize_date(df: pd.DataFrame, op: NormalizeDateOperation) -> Dict[str, Any]:
    if op.column in df.columns:
        df[op.column] = pd.to_datetime(df[op.column], errors=op.errors, dayfirst=op.dayfirst).dt.strftime(op.target_format)
    return {}

def apply_normalize_currency(df: pd.DataFrame, op: NormalizeCurrencyOperation) -> Dict[str, Any]:
    if op.column in df.columns:
        mask = df[op.column].notnull()
        # Remove currency symbols and commas, preserve nulls
        df.loc[mask, op.column] = df.loc[mask, op.column].astype(str).str.replace(r'[^\d.-]', '', regex=True)
        df[op.column] = pd.to_numeric(df[op.column], errors='coerce')
    return {}

def apply_normalize_number(df: pd.DataFrame, op: NormalizeNumberOperation) -> Dict[str, Any]:
    if op.column in df.columns:
        mask = df[op.column].notnull()
        df.loc[mask, op.column] = df.loc[mask, op.column].astype(str).str.replace(',', '', regex=False)
        df[op.column] = pd.to_numeric(df[op.column], errors='coerce')
    return {}

def apply_normalize_text_case(df: pd.DataFrame, op: NormalizeTextCaseOperation) -> Dict[str, Any]:
    cols = get_string_columns(df, op.columns)
    for col in cols:
        mask = df[col].notnull()
        if op.case == "lower":
            df.loc[mask, col] = df.loc[mask, col].astype(str).str.lower()
        elif op.case == "upper":
            df.loc[mask, col] = df.loc[mask, col].astype(str).str.upper()
        elif op.case == "title":
            df.loc[mask, col] = df.loc[mask, col].astype(str).str.title()
        elif op.case == "capitalize":
            df.loc[mask, col] = df.loc[mask, col].astype(str).str.capitalize()
    return {"columns_affected": cols}

def apply_replace_values(df: pd.DataFrame, op: ReplaceValuesOperation) -> Dict[str, Any]:
    if op.column in df.columns:
        df[op.column] = df[op.column].replace(op.mapping)
    return {}

def apply_strip_currency_symbols(df: pd.DataFrame, op: StripCurrencySymbolsOperation) -> Dict[str, Any]:
    if op.column in df.columns:
        mask = df[op.column].notnull()
        df.loc[mask, op.column] = df.loc[mask, op.column].astype(str).str.replace(r'[$£€¥]', '', regex=True)
    return {}

def apply_remove_commas_from_numbers(df: pd.DataFrame, op: RemoveCommasFromNumbersOperation) -> Dict[str, Any]:
    if op.column in df.columns:
        mask = df[op.column].notnull()
        df.loc[mask, op.column] = df.loc[mask, op.column].astype(str).str.replace(',', '', regex=False)
    return {}

def apply_coerce_column_type(df: pd.DataFrame, op: CoerceColumnTypeOperation) -> Dict[str, Any]:
    col = op.column
    if col in df.columns:
        if op.target_type == "string":
            mask = df[col].notnull()
            df.loc[mask, col] = df.loc[mask, col].astype(str)
        elif op.target_type == "integer":
            df[col] = pd.to_numeric(df[col], errors=op.errors).astype("Int64")
        elif op.target_type == "float" or op.target_type == "decimal":
            df[col] = pd.to_numeric(df[col], errors=op.errors).astype(float)
        elif op.target_type == "boolean":
            df[col] = df[col].astype(bool)
        elif op.target_type == "date":
            df[col] = pd.to_datetime(df[col], errors=op.errors)
    return {}

def apply_remove_empty_rows(df: pd.DataFrame, op: RemoveEmptyRowsOperation) -> Dict[str, Any]:
    df.dropna(how='all', inplace=True)
    return {}

def apply_remove_empty_columns(df: pd.DataFrame, op: RemoveEmptyColumnsOperation) -> Dict[str, Any]:
    df.dropna(axis=1, how='all', inplace=True)
    return {}

def apply_rename_columns(df: pd.DataFrame, op: RenameColumnsOperation) -> Dict[str, Any]:
    df.rename(columns=op.mapping, inplace=True)
    return {}

def apply_reorder_columns(df: pd.DataFrame, op: ReorderColumnsOperation) -> Dict[str, Any]:
    existing_cols = [c for c in op.columns if c in df.columns]
    # Add any columns not specified to the end
    remaining = [c for c in df.columns if c not in existing_cols]
    df_new = df[existing_cols + remaining]
    # To mimic inplace:
    df.drop(df.columns, axis=1, inplace=True)
    for c in df_new.columns:
        df[c] = df_new[c]
    return {}


def apply_absolute_value(df, op):
    """Remove negative signs from numeric columns (convert to absolute values)."""
    columns = getattr(op, "columns", None) or (op.parameters or {}).get("columns")
    if columns == "__all_numeric_columns__" or columns is None:
        numeric_cols = list(df.select_dtypes(include=["number"]).columns)
        for col in numeric_cols:
            df[col] = df[col].abs()
        return {"columns_affected": numeric_cols}
    if isinstance(columns, str):
        columns = [columns]
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].abs()
    return {"columns_affected": columns}


def apply_normalize_categorical_values(df, op):
    """Normalize inconsistent categorical values in text columns.

    Standardizes values like Male/male/M → Male, Bachelor's/Bachelors/Bachelor → Bachelor's.
    Uses title-case as the canonical form and groups abbreviations.
    """
    columns = getattr(op, "columns", None) or (op.parameters or {}).get("columns")
    if columns == "__all_string_columns__" or columns is None:
        columns = [col for col in df.columns if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])]
    elif isinstance(columns, str):
        columns = [columns]

    affected = []
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].copy()
        non_null = series.dropna()
        if non_null.empty:
            continue

        # Get unique values and build normalization map
        unique_vals = non_null.astype(str).str.strip().unique()
        if len(unique_vals) > 50:
            # Skip high-cardinality columns (not categorical)
            continue

        # Group by lowercase stripped form → pick the most common or title-case
        norm_map = {}
        for val in unique_vals:
            canonical = val.strip().title()
            # Handle common abbreviations
            lower = val.lower().strip()
            if lower in ("m", "male", "males"):
                canonical = "Male"
            elif lower in ("f", "female", "females"):
                canonical = "Female"
            elif lower in ("o", "other", "others"):
                canonical = "Other"
            elif lower.rstrip("'s") == lower.rstrip("s"):
                # "Bachelors" / "Bachelor's" / "Bachelor" → Title case with 's
                canonical = val.strip().title()
            norm_map[val] = canonical

        df[col] = series.map(lambda v: norm_map.get(str(v).strip(), v) if pd.notna(v) else v)
        affected.append(col)

    return {"columns_affected": affected}


CLEANING_HANDLERS = {
    "trim_whitespace": apply_trim_whitespace,
    "normalize_column_names": apply_normalize_column_names,
    "drop_duplicates": apply_drop_duplicates,
    "fill_nulls": apply_fill_nulls,
    "drop_nulls": apply_drop_nulls,
    "normalize_date": apply_normalize_date,
    "normalize_currency": apply_normalize_currency,
    "normalize_number": apply_normalize_number,
    "normalize_text_case": apply_normalize_text_case,
    "replace_values": apply_replace_values,
    "strip_currency_symbols": apply_strip_currency_symbols,
    "remove_commas_from_numbers": apply_remove_commas_from_numbers,
    "coerce_column_type": apply_coerce_column_type,
    "remove_empty_rows": apply_remove_empty_rows,
    "remove_empty_columns": apply_remove_empty_columns,
    "rename_columns": apply_rename_columns,
    "reorder_columns": apply_reorder_columns,
    "absolute_value": apply_absolute_value,
    "normalize_categorical_values": apply_normalize_categorical_values,
}


# ---------------------------------------------------------------------------
# Filter-Prep Whitelist (Requirements 2.7, 2.8, 2.9)
# ---------------------------------------------------------------------------
# The Compiler inserts a non-destructive `filter_prep` step (Component 7)
# realized as a `cleaning_agent` invocation whose params restrict execution to
# the seven safe operations enumerated below. The dispatch layer in
# `agents/cleaning_agent.py` (task 5.1) imports `SAFE_FILTER_PREP_OPERATIONS`
# and `assert_safe_for_filter_prep` from this module to enforce the contract.
#
# A `filter_prep` step MUST NOT, per requirement 2.8:
#   - drop rows with partial missing values
#   - impute missing values
#   - remove non-exact duplicates
#   - drop columns containing any null values
#   - rewrite low-confidence values
#   - apply business-specific transformations (renaming, recoding, currency
#     rounding policies, etc.)
#
# Audit of overlap with existing `CLEANING_HANDLERS`:
#   - `trim_whitespace`           : modifies string values in-place using a
#                                   notnull mask. Does not drop rows, impute
#                                   missing values, remove duplicates, drop
#                                   columns, or apply business transforms.
#                                   Non-destructive. No refactor needed.
#   - `normalize_column_names`    : reassigns `df.columns`. Touches no data
#                                   values; never drops rows or columns.
#                                   Non-destructive. No refactor needed.
# The remaining five canonical names
# (`normalize_empty_strings`, `safe_numeric_conversion`,
# `safe_currency_conversion`, `safe_date_detection`,
# `categorical_value_normalization`) have no handler yet; task 5.1 owns the
# `filter_prep` dispatch path and will decide how unmapped names are handled.
SAFE_FILTER_PREP_OPERATIONS: frozenset = frozenset({
    "trim_whitespace",
    "normalize_column_names",
    "normalize_empty_strings",
    "safe_numeric_conversion",
    "safe_currency_conversion",
    "safe_date_detection",
    "categorical_value_normalization",
})


def assert_safe_for_filter_prep(operation: str) -> None:
    """Guard helper: raise when `operation` is not in the filter-prep whitelist.

    The `cleaning_agent` calls this before invoking any handler in
    `filter_prep` mode so that an out-of-whitelist operation is rejected at
    the boundary, before pandas can be touched.

    Parameters
    ----------
    operation:
        The canonical operation name (a string drawn from the params emitted
        by the Compiler for a `filter_prep` step).

    Raises
    ------
    UnsafeFilterPrepOperationError
        If `operation` is not exactly one of the seven names in
        `SAFE_FILTER_PREP_OPERATIONS`. The error message names the offending
        operation and the canonical whitelist so the caller can surface a
        clear `AgentResult(status="failed", ...)` envelope.
    """
    from finflow_agent.operations.errors import UnsafeFilterPrepOperationError

    if not isinstance(operation, str) or operation not in SAFE_FILTER_PREP_OPERATIONS:
        allowed = ", ".join(sorted(SAFE_FILTER_PREP_OPERATIONS))
        raise UnsafeFilterPrepOperationError(
            f"Operation {operation!r} is not safe for filter_prep mode. "
            f"Allowed operations: {{{allowed}}}."
        )
