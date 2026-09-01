import logging
import os
import re
import time
import uuid
import warnings as _warnings
import pandas as pd
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ValidationError, model_validator
from finflow_agent.registry import registry, AgentSpec
from finflow_agent.state import AgentResult
from finflow_agent.operations.schemas import (
    CleaningOperationPlan,
    TrimWhitespaceOperation,
    NormalizeColumnNamesOperation,
)
from finflow_agent.operations.executor import execute_cleaning_plan
from finflow_agent.operations.cleaning_handlers import (
    SAFE_FILTER_PREP_OPERATIONS,
    apply_normalize_column_names,
    apply_trim_whitespace,
    assert_safe_for_filter_prep,
)
from finflow_agent.operations.errors import UnsafeFilterPrepOperationError


logger = logging.getLogger(__name__)


def _summarize_cleaning_plan(plan: Optional[CleaningOperationPlan]) -> str:
    if plan is None or not getattr(plan, "operations", None):
        return "no operations"
    parts: List[str] = []
    for op in list(plan.operations)[:4]:
        op_type = getattr(op, "type", None) or op.__class__.__name__
        column = getattr(op, "column", None)
        columns = getattr(op, "columns", None)
        if column:
            parts.append(f"{op_type}(column={column})")
        elif columns:
            parts.append(f"{op_type}(columns={columns})")
        else:
            parts.append(str(op_type))
    return ", ".join(parts)

# Confidence threshold for "safe" type-detection conversions in filter_prep mode.
# A conversion is only applied to a column when at least this fraction of the
# column's non-null values parse cleanly under the candidate semantic type.
# Below this threshold the column is left untouched (per requirement 2.8: the
# agent MUST NOT rewrite low-confidence values in filter_prep mode).
_FILTER_PREP_CONFIDENCE_THRESHOLD = 0.80

# Strict currency tokens. Anchored to start/end so a currency-shaped value must
# parse as a currency at both endpoints; mid-string symbols alone do not count.
_CURRENCY_PATTERN = re.compile(
    r"^\s*"
    r"(?:[$€£¥]|USD|EUR|GBP|JPY|CAD|AUD|CHF|CNY|INR)?"
    r"\s*[-+]?\d{1,3}(?:[,\s]?\d{3})*(?:\.\d+)?\s*"
    r"(?:[$€£¥]|USD|EUR|GBP|JPY|CAD|AUD|CHF|CNY|INR)?\s*$",
    re.IGNORECASE,
)

# Currency symbols and ISO codes stripped before numeric coercion.
_CURRENCY_STRIP_PATTERN = re.compile(
    r"[$€£¥]|\b(?:USD|EUR|GBP|JPY|CAD|AUD|CHF|CNY|INR)\b",
    re.IGNORECASE,
)


class CleaningAgentParams(BaseModel):
    """Pydantic params model for the Cleaning_Agent.

    `mode == "clean"` runs the full cleaning plan supplied by `plan`.
    `mode == "filter_prep"` is the non-destructive normalization mode the
    Compiler inserts before a `filter_agent` step when no full cleaning was
    requested; in that case `operations` carries the safe whitelist the
    compiler emits (see `SAFE_FILTER_PREP_OPERATIONS`) and `plan` is
    intentionally absent. Dispatch is handled in `execute()` (task 5.1).
    """

    plan: Optional[CleaningOperationPlan] = None
    mode: Literal["clean", "filter_prep"] = "clean"
    operations: Optional[List[str]] = None

    @model_validator(mode="after")
    def _validate_mode_contract(self) -> "CleaningAgentParams":
        # `plan` is required for the default `clean` mode (task 4.2 / Component 4
        # contract); `operations` is required for the `filter_prep` mode emitted
        # by the compiler (task 7.1 / Component 7 contract).
        if self.mode == "clean" and self.plan is None:
            raise ValueError("CleaningAgentParams.plan is required when mode='clean'.")
        if self.mode == "filter_prep" and not self.operations:
            raise ValueError(
                "CleaningAgentParams.operations is required when mode='filter_prep'."
            )
        return self


@registry.register
class CleaningAgent:
    spec = AgentSpec(
        name="cleaning_agent",
        description="Performs structure cleaning and format normalization using execute_cleaning_plan.",
        stage="transform",
        accepts=["dataframe"],
        produces=["dataframe"],
        params_schema={
            "plan": {"type": "object"},
            "mode": {"type": "string"},
            "operations": {"type": "array"}
        }
    )
    # Pydantic params model picked up by the registry so the validator and
    # engine can re-validate `step.params` before this agent is invoked.
    params_model = CleaningAgentParams

    def execute(self, params: dict, input_data: dict) -> AgentResult:
        # Dispatch on `params.mode` at the top of `execute` (task 5.1). The
        # default `"clean"` path is kept verbatim below; only the
        # `"filter_prep"` path is new.
        mode = params.get("mode", "clean") if isinstance(params, dict) else "clean"
        if mode == "filter_prep":
            return self._execute_filter_prep(params, input_data)

        return self._execute_clean(params, input_data)

    # ------------------------------------------------------------------
    # Default (full) cleaning path — preserved verbatim from the previous
    # implementation. This is the `mode == "clean"` branch.
    # ------------------------------------------------------------------
    def _execute_clean(self, params: dict, input_data: dict) -> AgentResult:
        df = input_data.get("input_dataframe")
        if df is None:
            return AgentResult(status="failed", error_message="input_dataframe is required. No input dataframe provided.")

        plan_data = params.get("plan")
        if not plan_data:
            return AgentResult(
                status="failed",
                error_message=(
                    "CleaningAgent requires a typed CleaningOperationPlan; "
                    "canonical execution does not accept raw instructions."
                ),
            )

        try:
            if isinstance(plan_data, CleaningOperationPlan):
                plan = plan_data
            else:
                plan = CleaningOperationPlan.model_validate(plan_data)
        except Exception as e:
            return AgentResult(
                status="failed",
                error_message=f"Invalid cleaning parameters in plan payload {plan_data!r}: {e}",
            )

        # Strict parameter validation
        try:
            CleaningAgentParams.model_validate({"plan": plan})
        except ValidationError as e:
            return AgentResult(
                status="failed",
                error_message=(
                    "Invalid parameter schema for CleaningAgent "
                    f"(plan_operations={_summarize_cleaning_plan(plan)}): {e}"
                ),
            )

        try:
            output = execute_cleaning_plan(df.copy(), plan)
            return AgentResult(
                status="success",
                data=output.data,
                summary=output.summary,
                metrics=output.metrics,
                operations_applied=output.operations_applied,
                warnings=output.warnings,
                artifacts=output.artifacts
            )
        except Exception as e:
            return AgentResult(
                status="failed",
                error_message=(
                    "Failed to execute cleaning plan with operations "
                    f"[{_summarize_cleaning_plan(plan)}] on dataframe columns "
                    f"{list(df.columns)}: {e}"
                ),
            )

    # ------------------------------------------------------------------
    # Non-destructive `filter_prep` path (Component 7 / task 5.1).
    # Runs ONLY the operations whitelisted in
    # `SAFE_FILTER_PREP_OPERATIONS`. Refuses any operation outside the
    # whitelist with an `AgentResult(status="failed", ...)` envelope.
    # Stamps every emitted `operations_applied` entry with
    # `{"origin": "filter_prep"}` so the audit writer can recognise it as
    # an internal preparation step.
    # ------------------------------------------------------------------
    def _execute_filter_prep(self, params: dict, input_data: dict) -> AgentResult:
        df = input_data.get("input_dataframe")
        if df is None:
            return AgentResult(
                status="failed",
                error_message="input_dataframe is required. No input dataframe provided.",
            )

        requested_ops = params.get("operations")
        if requested_ops is None or not isinstance(requested_ops, list):
            return AgentResult(
                status="failed",
                error_message=(
                    "filter_prep mode requires params.operations to be a list of "
                    "operation names from the safe whitelist."
                ),
            )

        # Refuse any operation outside the seven-name whitelist BEFORE touching
        # pandas. The boundary check uses the controlled
        # `assert_safe_for_filter_prep` guard from
        # `operations/cleaning_handlers.py` (task 5.2).
        for op_name in requested_ops:
            try:
                assert_safe_for_filter_prep(op_name)
            except UnsafeFilterPrepOperationError as exc:
                return AgentResult(
                    status="failed",
                    error_message=(
                        f"Operation '{op_name}' is not safe for filter_prep mode: {exc}"
                    ),
                )

        # Copy at the boundary so we never mutate the engine's upstream state.
        df_work = df.copy()
        operations_applied: List[Dict[str, Any]] = []
        warnings: List[str] = []

        # Map each whitelisted operation name to its concrete handler. Each
        # handler is non-destructive: it never drops rows, never imputes
        # missing values, never removes non-exact duplicates, never drops
        # columns containing nulls, and never applies any business-specific
        # transformation (per requirement 2.8).
        handlers = {
            "trim_whitespace": self._fp_trim_whitespace,
            "normalize_column_names": self._fp_normalize_column_names,
            "normalize_empty_strings": self._fp_normalize_empty_strings,
            "safe_numeric_conversion": self._fp_safe_numeric_conversion,
            "safe_currency_conversion": self._fp_safe_currency_conversion,
            "safe_date_detection": self._fp_safe_date_detection,
            "categorical_value_normalization": self._fp_categorical_value_normalization,
        }

        for op_name in requested_ops:
            handler = handlers.get(op_name)
            if handler is None:
                # Defensive: the whitelist guard above should already have
                # rejected unmapped names. This second guard prevents any
                # possible drift between the whitelist constant and the
                # handler dispatch table.
                return AgentResult(
                    status="failed",
                    error_message=(
                        f"Operation '{op_name}' is not safe for filter_prep mode: "
                        f"no handler registered."
                    ),
                )

            started_at = int(time.time() * 1000)
            initial_rows = len(df_work)
            input_cols = list(df_work.columns)

            try:
                metrics = handler(df_work) or {}
            except Exception as exc:
                return AgentResult(
                    status="failed",
                    error_message=(
                        f"filter_prep operation '{op_name}' failed: {exc}"
                    ),
                )

            finished_at = int(time.time() * 1000)
            output_cols = list(df_work.columns)
            cols_modified = list(set(input_cols) ^ set(output_cols))
            cols_affected = metrics.get("columns_affected", []) or []
            for col in cols_affected:
                if col in output_cols and col not in cols_modified:
                    cols_modified.append(col)

            op_warnings = list(metrics.get("warnings", []) or [])
            if op_warnings:
                warnings.extend(op_warnings)

            # Every entry carries the `{"origin": "filter_prep"}` marker
            # required by requirement 5.3 / Component 7 audit hint.
            operations_applied.append({
                "operation_id": f"op_{uuid.uuid4().hex[:8]}",
                "operation_type": op_name,
                "type": op_name,
                "input_row_count": initial_rows,
                "output_row_count": len(df_work),
                "input_columns": input_cols,
                "output_columns": output_cols,
                "columns_modified": cols_modified,
                "warnings": op_warnings,
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_ms": finished_at - started_at,
                "origin": "filter_prep",
            })

        # Contract check: verify filter_prep didn't unexpectedly drop columns
        # that downstream agents need
        intent_package = input_data.get("intent_package")
        if intent_package is not None:
            for rc in intent_package.resolved_columns:
                if rc.resolved_column not in df_work.columns:
                    from finflow_agent.planning.intent_package import ContractViolation
                    violation = ContractViolation(
                        step_id="filter_prep",
                        agent="cleaning_agent",
                        violation_type="column_dropped_by_prep",
                        expected=rc.resolved_column,
                        actual=f"columns_after_prep={list(df_work.columns)}",
                    )
                    intent_package.add_violation(violation)
                    return AgentResult(
                        status="failed",
                        error_message=(
                            f"Contract violation: filter_prep dropped column "
                            f"{rc.resolved_column!r} needed by downstream agents."
                        ),
                    )

        return AgentResult(
            status="success",
            data=df_work,
            summary=(
                f"Applied {len(operations_applied)} non-destructive filter_prep "
                f"normalization operation(s)."
            ),
            operations_applied=operations_applied,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # filter_prep operation handlers. Each operates IN-PLACE on `df` and
    # returns an optional metrics dict (`{"columns_affected": [...],
    # "warnings": [...]}`).
    # ------------------------------------------------------------------
    @staticmethod
    def _fp_trim_whitespace(df: pd.DataFrame) -> Dict[str, Any]:
        # Reuses the existing non-destructive handler. Only string columns are
        # touched and nulls are preserved via the notnull mask inside the
        # handler. No rows or columns are dropped.
        op = TrimWhitespaceOperation(columns="__all_string_columns__")
        return apply_trim_whitespace(df, op)

    @staticmethod
    def _fp_normalize_column_names(df: pd.DataFrame) -> Dict[str, Any]:
        # Renames `df.columns` to `snake_case`. Touches no data values; never
        # drops rows or columns.
        op = NormalizeColumnNamesOperation(style="snake_case")
        return apply_normalize_column_names(df, op)

    @staticmethod
    def _is_string_like(series: pd.Series) -> bool:
        """True for `object` AND `pd.StringDtype` columns.

        Pandas 2.x infers string-column DataFrames as `StringDtype` rather
        than `object`, so the safe filter_prep handlers must accept both.
        """
        return series.dtype == object or pd.api.types.is_string_dtype(series)

    @classmethod
    def _fp_normalize_empty_strings(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Replace pure-empty / pure-whitespace strings with `pd.NA`.

        Only string-like columns (object / `StringDtype`) are scanned and
        only entries that are actually `str` instances are rewritten.
        Non-string dtypes are not coerced — this preserves numeric,
        datetime, and boolean columns.
        """
        affected: List[str] = []
        for col in df.columns:
            if not cls._is_string_like(df[col]):
                continue
            series = df[col]
            mask_str = series.apply(lambda v: isinstance(v, str))
            if not mask_str.any():
                continue
            stripped = series[mask_str].astype(str).str.strip()
            empty_idx = stripped.index[stripped == ""]
            if len(empty_idx) > 0:
                df.loc[empty_idx, col] = pd.NA
                affected.append(col)
        return {"columns_affected": affected}

    @classmethod
    def _fp_safe_numeric_conversion(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Coerce object columns to numeric dtype iff ≥80 % of non-null
        values parse via `pd.to_numeric`.

        Low-confidence columns (<80 % parseable) are left untouched, satisfying
        requirement 2.8's "MUST NOT rewrite low-confidence values" clause.
        Failed parses become NaN; rows are NEVER dropped.
        """
        affected: List[str] = []
        for col in df.columns:
            if not cls._is_string_like(df[col]):
                continue
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue
            as_str = non_null.astype(str).str.strip()
            coerced_sample = pd.to_numeric(as_str, errors="coerce")
            parse_rate = coerced_sample.notnull().sum() / len(non_null)
            if parse_rate >= _FILTER_PREP_CONFIDENCE_THRESHOLD:
                # Convert to numeric, preserving NaN for non-parseable rows.
                coerced = pd.to_numeric(
                    df[col].astype(str).str.strip(), errors="coerce"
                )
                df[col] = coerced
                affected.append(col)
        return {"columns_affected": affected}

    @classmethod
    def _fp_safe_currency_conversion(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Strip currency symbols / ISO codes and coerce to numeric iff
        ≥80 % of non-null values match a currency-shaped pattern.

        Columns that do not predominantly look like currency are left
        untouched. Rows are NEVER dropped on parse failure.
        """
        affected: List[str] = []
        for col in df.columns:
            if not cls._is_string_like(df[col]):
                continue
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue
            as_str = non_null.astype(str)
            match_count = int(as_str.apply(
                lambda v: bool(_CURRENCY_PATTERN.match(v))
            ).sum())
            match_rate = match_count / len(non_null)
            if match_rate >= _FILTER_PREP_CONFIDENCE_THRESHOLD:
                stripped = (
                    df[col]
                    .astype(str)
                    .str.replace(_CURRENCY_STRIP_PATTERN, "", regex=True)
                    .str.replace(",", "", regex=False)
                    .str.strip()
                )
                df[col] = pd.to_numeric(stripped, errors="coerce")
                affected.append(col)
        return {"columns_affected": affected}

    @classmethod
    def _fp_safe_date_detection(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Convert string-like columns to `datetime64[ns]` dtype iff ≥80 %
        of non-null values parse via `pd.to_datetime`.

        Numeric columns are skipped (a prior `safe_numeric_conversion`
        operation will have promoted them out of string dtype already).
        Rows are NEVER dropped on parse failure.
        """
        affected: List[str] = []
        for col in df.columns:
            if not cls._is_string_like(df[col]):
                continue
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue
            try:
                with _warnings.catch_warnings():
                    # `pd.to_datetime` emits a noisy UserWarning when it cannot
                    # infer a single format and falls back to per-element
                    # parsing. The detection step is exploratory by design, so
                    # silence the warning rather than leak it to callers.
                    _warnings.simplefilter("ignore", UserWarning)
                    coerced = pd.to_datetime(non_null, errors="coerce")
            except Exception:
                continue
            parse_rate = coerced.notnull().sum() / len(non_null)
            if parse_rate >= _FILTER_PREP_CONFIDENCE_THRESHOLD:
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore", UserWarning)
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                affected.append(col)
        return {"columns_affected": affected}

    @classmethod
    def _fp_categorical_value_normalization(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Strip + lower-case non-null string values in string-like columns.

        Synonyms are NOT collapsed (per requirement 2.8: no business-specific
        transformations) and low-confidence values are NOT rewritten — the
        operation is a deterministic, value-preserving normalization.
        """
        affected: List[str] = []
        for col in df.columns:
            if not cls._is_string_like(df[col]):
                continue
            mask_str = df[col].apply(lambda v: isinstance(v, str))
            if not mask_str.any():
                continue
            df.loc[mask_str, col] = (
                df.loc[mask_str, col].astype(str).str.strip().str.lower()
            )
            affected.append(col)
        return {"columns_affected": affected}
