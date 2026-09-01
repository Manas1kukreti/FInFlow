import pandas as pd
from typing import Any, Dict, List, Optional
from pathlib import Path
import json

from pydantic import BaseModel, ConfigDict, Field

from finflow_agent.operations.schemas import ReportingOperationPlan
from finflow_agent.tools.path_safety import get_safe_output_path
from finflow_agent.operations.errors import (
    ReportGenerationError,
    UnsafeOutputPathError,
)
from finflow_agent.tools.column_resolver import ColumnResolution

def write_xlsx_report(df: pd.DataFrame, plan: ReportingOperationPlan, output_dir: str, file_prefix: str, **kwargs) -> Dict[str, Any]:
    """Write a tabular ``.xlsx`` report.

    Per requirements 9.4 and 9.5 (visualization is disabled by default in
    this version), this writer only emits a tabular sheet with autofit
    column widths. It MUST NOT generate, render, or embed any chart on
    any sheet. Any caller that passes legacy chart-rendering kwargs (e.g.
    ``chart_configs``) is silently ignored — the kwargs surface is kept
    for back-compat with the executor signature but the writer never
    consumes it.
    """
    safe_path = get_safe_output_path(output_dir, f"{file_prefix}.xlsx")
    try:
        # Using context manager for useful formatting
        with pd.ExcelWriter(safe_path, engine='xlsxwriter') as writer:
            sheet = plan.sheet_name or "Sheet1"
            df.to_excel(writer, sheet_name=sheet, index=False)
            worksheet = writer.sheets[sheet]
            # Useful formatting: autofit columns. No chart is ever rendered
            # on this sheet (requirement 9.5).
            for i, col in enumerate(df.columns):
                max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(i, i, max_len)
    except Exception as e:
        raise ReportGenerationError(
            f"Failed to write XLSX report to path={str(safe_path)!r}: {e}"
        )
    return {"output_file_path": str(safe_path)}

def write_csv_report(df: pd.DataFrame, plan: ReportingOperationPlan, output_dir: str, file_prefix: str, **kwargs) -> Dict[str, Any]:
    safe_path = get_safe_output_path(output_dir, f"{file_prefix}.csv")
    try:
        df.to_csv(safe_path, index=False, encoding='utf-8')
    except Exception as e:
        raise ReportGenerationError(
            f"Failed to write CSV report to path={str(safe_path)!r}: {e}"
        )
    return {"output_file_path": str(safe_path)}

def write_json_report(df: pd.DataFrame, plan: ReportingOperationPlan, output_dir: str, file_prefix: str, **kwargs) -> Dict[str, Any]:
    safe_path = get_safe_output_path(output_dir, f"{file_prefix}.json")
    try:
        records = df.to_dict(orient="records")
        # Ensure datetimes are serialized nicely
        json_str = json.dumps({"metadata": {"title": plan.title or "Report"}, "data": records}, default=str)
        with open(safe_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
    except Exception as e:
        raise ReportGenerationError(
            f"Failed to write JSON report to path={str(safe_path)!r}: {e}"
        )
    return {"output_file_path": str(safe_path)}

def write_txt_report(df: pd.DataFrame, plan: ReportingOperationPlan, output_dir: str, file_prefix: str, **kwargs) -> Dict[str, Any]:
    safe_path = get_safe_output_path(output_dir, f"{file_prefix}.txt")
    try:
        with open(safe_path, 'w', encoding='utf-8') as f:
            if plan.title:
                f.write(f"{plan.title}\n{'=' * len(plan.title)}\n\n")
            f.write(df.to_string(index=False))
    except Exception as e:
        raise ReportGenerationError(
            f"Failed to write TXT report to path={str(safe_path)!r}: {e}"
        )
    return {"output_file_path": str(safe_path)}

REPORTING_HANDLERS = {
    "xlsx": write_xlsx_report,
    "csv": write_csv_report,
    "json": write_json_report,
    "txt": write_txt_report
}


# ---------------------------------------------------------------------------
# Audit Sheet Writer (design Component 5)
# ---------------------------------------------------------------------------

# Marker the Cleaning_Agent stamps on every operation it applies in
# `filter_prep` mode (per design Component 7 / requirement 2.16). The audit
# writer uses it to classify the entry as an internal non-destructive
# normalization step rather than user-requested cleaning.
_FILTER_PREP_ORIGIN_MARKER: Dict[str, str] = {"origin": "filter_prep"}

# Canonical column ordering when the `audit_log` payload is empty. We always
# write headers so a downstream reviewer can see the schema even when no
# operations ran (requirement 8.5).
_AUDIT_LOG_DEFAULT_COLUMNS: List[str] = [
    "step_kind",
    "operation_id",
    "operation_type",
    "summary",
    "input_row_count",
    "output_row_count",
    "started_at",
    "finished_at",
    "duration_ms",
]

# Canonical column ordering when the `column_mapping` payload is empty.
_COLUMN_MAPPING_DEFAULT_COLUMNS: List[str] = [
    "requested_field",
    "matched_column",
    "semantic_type",
    "confidence",
    "reason",
]

# Canonical column for the `warnings` sheet. We keep one column so a reviewer
# can scan the warnings the agents emitted in order.
_WARNINGS_DEFAULT_COLUMNS: List[str] = ["warning"]


class AuditSheetPayload(BaseModel):
    """Payload consumed by :func:`write_excel_with_audit_sheets`.

    The Reporting_Agent assembles this payload from the upstream agent
    envelopes (cleaning, filter, etc.) and hands it to the audit writer. The
    writer never mutates any field; it only serializes the payload to a
    deterministic Excel layout (requirement 8.6).

    Attributes
    ----------
    cleaned_data:
        The dataframe to publish on the always-present ``cleaned_data``
        sheet (requirement 8.2). When the upstream pipeline ran a
        ``filter_prep`` step instead of full cleaning, this is the
        non-destructively normalized dataframe.
    filtered_data:
        The dataframe to publish on the ``filtered_data`` sheet. Set to
        ``None`` when no filter step ran; the writer will then omit the
        sheet entirely (requirements 8.3 and 8.4).
    audit_log:
        The list of ``operations_applied`` dicts collected from the upstream
        agents, in execution order. Each entry tagged with
        ``{"origin": "filter_prep"}`` is recorded as an internal preparation
        step (requirement 8.9).
    warnings:
        Human-readable warning strings produced by the upstream agents.
    column_mapping:
        ``ColumnResolution`` entries produced by the Filter_Agent so the
        reviewer can see which dataframe columns each LLM-requested field
        was matched against.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cleaned_data: pd.DataFrame
    filtered_data: Optional[pd.DataFrame] = None
    audit_log: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    column_mapping: List[ColumnResolution] = Field(default_factory=list)


def _sanitize_cell_value(value: Any) -> Any:
    """Coerce a cell value to something xlsxwriter can serialize.

    pandas/xlsxwriter happily accept scalars (``str``, ``int``, ``float``,
    ``bool``) but lists and dicts (which appear in our ``audit_log``
    entries, e.g. ``input_columns``) are rendered via ``__repr__`` and can
    confuse downstream readers. We round-trip non-scalar values through
    ``json.dumps`` so the cell is human-readable JSON, falling back to
    ``str`` if the value is not JSON-serializable.
    """
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _is_filter_prep_entry(entry: Dict[str, Any]) -> bool:
    """Return ``True`` when an audit entry was produced by a ``filter_prep`` step.

    The Cleaning_Agent stamps every operation it runs in ``filter_prep``
    mode with ``{"origin": "filter_prep"}`` (Component 7 / requirement
    2.16). Entries can carry the marker either as a top-level ``origin``
    field or as a nested dict within the entry; we accept either shape so
    the writer is robust to small upstream changes.
    """
    if not isinstance(entry, dict):
        return False
    if str(entry.get("origin", "")).lower() == "filter_prep":
        return True
    # Defensive: scan one level deep for a dict that matches the marker.
    for v in entry.values():
        if isinstance(v, dict) and v == _FILTER_PREP_ORIGIN_MARKER:
            return True
    return False


def _build_audit_log_dataframe(
    audit_log: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Build the ``audit_log`` sheet dataframe from raw entries.

    Always returns a non-empty schema (``step_kind`` first, then either the
    union of keys from ``audit_log`` or :data:`_AUDIT_LOG_DEFAULT_COLUMNS`)
    so the sheet shows headers even when no operations ran (requirement
    8.5). Entries flagged via :func:`_is_filter_prep_entry` are tagged with
    ``step_kind == "internal_preparation"`` (requirement 8.9); everything
    else is tagged ``"user_requested"``.
    """
    if not audit_log:
        return pd.DataFrame(columns=_AUDIT_LOG_DEFAULT_COLUMNS)

    # Preserve insertion order across the union of keys so the resulting
    # column order is deterministic for any given payload shape.
    seen_keys: List[str] = []
    sanitized_rows: List[Dict[str, Any]] = []
    for entry in audit_log:
        if not isinstance(entry, dict):
            entry = {"summary": entry}
        step_kind = (
            "internal_preparation"
            if _is_filter_prep_entry(entry)
            else "user_requested"
        )
        row: Dict[str, Any] = {"step_kind": step_kind}
        for key, value in entry.items():
            if key == "step_kind":
                # Never let a stray upstream value override the writer's
                # classification; the audit writer is the source of truth
                # for this column.
                continue
            row[key] = _sanitize_cell_value(value)
            if key not in seen_keys:
                seen_keys.append(key)
        sanitized_rows.append(row)

    ordered_columns = ["step_kind"] + seen_keys
    return pd.DataFrame(sanitized_rows, columns=ordered_columns)


def _build_warnings_dataframe(warnings: List[str]) -> pd.DataFrame:
    """Build the ``warnings`` sheet dataframe.

    Always returns a dataframe with the canonical header so the sheet is
    valid even when no warnings exist (requirement 8.5).
    """
    if not warnings:
        return pd.DataFrame(columns=_WARNINGS_DEFAULT_COLUMNS)
    return pd.DataFrame({"warning": [str(w) for w in warnings]})


def _build_column_mapping_dataframe(
    column_mapping: List[ColumnResolution],
) -> pd.DataFrame:
    """Build the ``column_mapping`` sheet dataframe.

    Always returns a dataframe with the canonical headers so the sheet is
    valid even when the Filter_Agent did not run (requirement 8.5).
    """
    if not column_mapping:
        return pd.DataFrame(columns=_COLUMN_MAPPING_DEFAULT_COLUMNS)
    rows = [
        {
            "requested_field": r.requested_field,
            "matched_column": r.matched_column,
            "semantic_type": r.semantic_type,
            "confidence": float(r.confidence),
            "reason": r.reason,
        }
        for r in column_mapping
    ]
    return pd.DataFrame(rows, columns=_COLUMN_MAPPING_DEFAULT_COLUMNS)


def _write_audit_sheet(
    writer: "pd.ExcelWriter",
    sheet_name: str,
    df: pd.DataFrame,
    header_format: Any,
) -> None:
    """Write *df* to *sheet_name* with bold/frozen headers and autofit columns.

    The writer never inserts a chart on any sheet (requirement 9.5); the
    only formatting we apply is a bold header row, a frozen header pane,
    and a per-column width that fits the longest stringified value.
    """
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    worksheet = writer.sheets[sheet_name]

    # Bold the header row and freeze it so it stays visible while scrolling.
    for col_idx, col_name in enumerate(df.columns):
        worksheet.write(0, col_idx, str(col_name), header_format)
    worksheet.freeze_panes(1, 0)

    # Autofit each column to the longest value (header included). When the
    # dataframe is empty we still size the column to its header width so
    # the sheet looks reasonable.
    for col_idx, col_name in enumerate(df.columns):
        header_width = len(str(col_name))
        if len(df) > 0:
            try:
                series_max = df[col_name].astype(str).map(len).max()
            except Exception:
                series_max = 0
            if pd.isna(series_max):
                series_max = 0
            content_width = int(series_max)
        else:
            content_width = 0
        worksheet.set_column(col_idx, col_idx, max(header_width, content_width) + 2)


def write_excel_with_audit_sheets(
    payload: AuditSheetPayload,
    plan: ReportingOperationPlan,
    output_dir: str,
    file_prefix: str,
) -> Dict[str, Any]:
    """Write one Excel file with the deterministic audit sheets.

    Layout (per design Component 5 and acceptance criteria 8.1 - 8.5, 8.7,
    8.8, 8.9):

    1. ``cleaned_data`` (always first; requirement 8.2)
    2. ``filtered_data`` (only when ``payload.filtered_data is not None``;
       requirements 8.3 and 8.4)
    3. ``audit_log`` (always; ``filter_prep`` entries are tagged
       ``internal_preparation`` per requirement 8.9)
    4. ``warnings`` (always)
    5. ``column_mapping`` (always)

    The output path is resolved through
    :func:`finflow_agent.tools.path_safety.get_safe_output_path`, which
    rejects absolute file names and ``..`` traversal segments before any
    file is written (requirement 8.8). No chart is rendered on any sheet
    (requirement 9.5).

    Parameters
    ----------
    payload:
        The :class:`AuditSheetPayload` assembled by the Reporting_Agent.
    plan:
        The validated :class:`ReportingOperationPlan` from the upstream
        engine. The audit writer accepts the plan for signature
        compatibility with the other reporting handlers but does not use
        it to mutate any sheet content.
    output_dir, file_prefix:
        The pre-validated output directory and file prefix. The full
        output path is ``<output_dir>/<file_prefix>.xlsx`` (requirement
        8.1).

    Returns
    -------
    Dict[str, Any]
        ``{"output_file_path": <str>, "sheets_written": [<str>, ...]}``
        per requirement 8.7.

    Raises
    ------
    UnsafeOutputPathError
        When ``file_prefix`` would resolve to an absolute path or escape
        ``output_dir`` via parent-directory traversal.
    ReportGenerationError
        When writing the workbook fails for any reason other than path
        safety.
    """
    # NOTE: The path safety helper raises ``UnsafeOutputPathError`` directly
    # for absolute-path or traversal attempts. We deliberately let that
    # propagate (requirement 8.8) instead of wrapping it in
    # ``ReportGenerationError`` so the engine and tests can branch on it.
    safe_path = get_safe_output_path(output_dir, f"{file_prefix}.xlsx")

    sheets_written: List[str] = []

    try:
        with pd.ExcelWriter(safe_path, engine="xlsxwriter") as writer:
            workbook = writer.book
            header_format = workbook.add_format(
                {
                    "bold": True,
                    "bg_color": "#D9E1F2",
                    "border": 1,
                    "align": "left",
                    "valign": "vcenter",
                }
            )

            # 1. cleaned_data (always first).
            cleaned_df = (
                payload.cleaned_data
                if payload.cleaned_data is not None
                else pd.DataFrame()
            )
            _write_audit_sheet(writer, "cleaned_data", cleaned_df, header_format)
            sheets_written.append("cleaned_data")

            # 2. filtered_data (only when present).
            if payload.filtered_data is not None:
                _write_audit_sheet(
                    writer, "filtered_data", payload.filtered_data, header_format
                )
                sheets_written.append("filtered_data")

            # 3. audit_log (always; filter_prep rows tagged).
            audit_df = _build_audit_log_dataframe(payload.audit_log)
            _write_audit_sheet(writer, "audit_log", audit_df, header_format)
            sheets_written.append("audit_log")

            # 4. warnings (always).
            warnings_df = _build_warnings_dataframe(payload.warnings)
            _write_audit_sheet(writer, "warnings", warnings_df, header_format)
            sheets_written.append("warnings")

            # 5. column_mapping (always).
            mapping_df = _build_column_mapping_dataframe(payload.column_mapping)
            _write_audit_sheet(
                writer, "column_mapping", mapping_df, header_format
            )
            sheets_written.append("column_mapping")
    except UnsafeOutputPathError:
        # Re-raise without wrapping so callers can distinguish path-safety
        # rejection from other write failures.
        raise
    except Exception as e:
        raise ReportGenerationError(
            f"Failed to write audit Excel file to path={str(safe_path)!r}: {e}"
        ) from e

    return {
        "output_file_path": str(safe_path),
        "sheets_written": sheets_written,
    }


__all__ = [
    "REPORTING_HANDLERS",
    "AuditSheetPayload",
    "write_excel_with_audit_sheets",
    "write_csv_report",
    "write_json_report",
    "write_txt_report",
    "write_xlsx_report",
]
