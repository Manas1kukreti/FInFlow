"""Reporting agent for the FinFlow Agent Service.

After the agent-pipeline-hardening spec (task 11.2), this agent is reduced
to a *pure writer*. It MUST NOT modify any dataframe values and MUST NOT
perform cleaning, filtering, calculation, or visualization (req 8.6 / 9.5).

When ``plan.output_format == "xlsx"`` the agent delegates to
:func:`finflow_agent.operations.reporting_handlers.write_excel_with_audit_sheets`,
which produces the deterministic five-sheet audit workbook
(``cleaned_data``, optional ``filtered_data``, ``audit_log``, ``warnings``,
``column_mapping``). Detection of an upstream ``filter_prep`` step inside
``audit_log`` (any entry tagged ``{"origin": "filter_prep"}``) plus an
inferred ``needs_cleaning == False`` upstream switches the user-facing
summary to the canonical ``"Data was normalized for filtering."`` string
required by req 8.10 / 2.17.

For ``csv`` / ``json`` / ``txt`` output formats the agent keeps the
existing back-compatible path through
:func:`finflow_agent.operations.executor.execute_reporting_plan` so older
callers that don't request the audit-sheet workbook continue to work.

The agent's dataframe input contract (req 5.5) is unchanged: the dataframe
is read exclusively from ``input_data["input_dataframe"]``; when the key is
missing or ``None`` the agent returns the canonical ``failed`` envelope.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, List, Optional

import pandas as pd
from pydantic import BaseModel, ValidationError, model_validator

from finflow_agent.operations.executor import execute_reporting_plan
from finflow_agent.operations.reporting_handlers import (
    AuditSheetPayload,
    write_excel_with_audit_sheets,
)
from finflow_agent.operations.schemas import ReportingOperationPlan
from finflow_agent.registry import AgentSpec, registry
from finflow_agent.state import AgentResult
from finflow_agent.tools.column_resolver import ColumnResolution


# Canonical user-facing summary the agent emits whenever the upstream
# pipeline ran a `filter_prep` normalization step in lieu of full cleaning
# (acceptance criteria 8.10 / 2.17). The string is exact and must not be
# altered without coordinating with the spec.
_FILTER_PREP_CANONICAL_SUMMARY = "Data was normalized for filtering."


def _reporting_context(*, plan: ReportingOperationPlan | None = None, output_dir: str | None = None, file_prefix: str | None = None) -> str:
    parts: list[str] = []
    if plan is not None:
        parts.append(f"output_format={plan.output_format!r}")
        if plan.sheet_name:
            parts.append(f"sheet_name={plan.sheet_name!r}")
        if plan.title:
            parts.append(f"title={plan.title!r}")
    if output_dir is not None:
        parts.append(f"output_dir={output_dir!r}")
    if file_prefix is not None:
        parts.append(f"file_prefix={file_prefix!r}")
    return ", ".join(parts)


class ReportingAgentParams(BaseModel):
    """Pydantic params model for the Reporting_Agent.

    Per requirement 10.5, validation MUST fail when ``plan.output_format == "pdf"``.
    The nested :class:`ReportingOperationPlan` already restricts
    ``output_format`` to ``Literal["xlsx", "csv", "json", "txt"]``, which
    transitively rejects ``pdf``. The ``mode="before"`` model validator below
    makes the rejection explicit on this outer model so a misconstructed
    payload (for example, a raw dict with ``plan.output_format == "pdf"``)
    is rejected at the param-model boundary before the engine ever calls
    ``agent.execute``.
    """

    plan: ReportingOperationPlan
    output_dir: Optional[str] = None
    file_prefix: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _reject_pdf_output_format(cls, data: Any) -> Any:
        if isinstance(data, dict):
            plan = data.get("plan")
            output_format: Optional[str] = None
            if isinstance(plan, dict):
                fmt = plan.get("output_format")
                if isinstance(fmt, str):
                    output_format = fmt
            elif isinstance(plan, ReportingOperationPlan):
                output_format = plan.output_format
            elif plan is not None:
                fmt = getattr(plan, "output_format", None)
                if isinstance(fmt, str):
                    output_format = fmt
            if output_format is not None and output_format.lower() == "pdf":
                raise ValueError(
                    "ReportingAgentParams: plan.output_format 'pdf' is not "
                    "supported. Allowed values: xlsx, csv, json, txt."
                )
        return data


def _has_filter_prep_entry(entries: List[Any]) -> bool:
    """Return True iff any entry in *entries* carries the ``filter_prep`` marker.

    The cleaning agent stamps every operation it runs in ``filter_prep`` mode
    with ``{"origin": "filter_prep"}`` (Component 7 / req 5.3). The audit
    writer uses the same detection rule; we duplicate it here only to drive
    the user-facing summary, never to rewrite any payload.
    """
    if not entries:
        return False
    for entry in entries:
        if isinstance(entry, dict) and str(entry.get("origin", "")).lower() == "filter_prep":
            return True
    return False


def _has_non_filter_prep_entry(entries: List[Any]) -> bool:
    """Return True iff any entry in *entries* is *not* a ``filter_prep`` entry.

    Used as the heuristic from the spec: "assume ``needs_cleaning == False``
    whenever the audit_log contains filter_prep entries AND no other
    'clean'-mode entries are present". A non-filter_prep entry is any dict
    in the audit log that does not carry the ``{"origin": "filter_prep"}``
    marker. Non-dict entries are ignored.
    """
    if not entries:
        return False
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("origin", "")).lower() != "filter_prep":
            return True
    return False


def _coerce_column_mapping(value: Any) -> List[ColumnResolution]:
    """Coerce an artifact ``column_mapping`` into a typed list.

    The filter agent stores resolutions as :class:`ColumnResolution`
    ``model_dump`` dicts in ``AgentResult.artifacts["column_mapping"]``
    (task 6.1). The audit writer's :class:`AuditSheetPayload` expects
    typed :class:`ColumnResolution` instances. This helper accepts either
    shape and silently drops any malformed entry — the resolution payload
    is advisory metadata, not a critical-path artifact, so a malformed
    upstream entry must not crash the writer.
    """
    if not value or not isinstance(value, list):
        return []
    out: List[ColumnResolution] = []
    for item in value:
        if isinstance(item, ColumnResolution):
            out.append(item)
        elif isinstance(item, dict):
            try:
                out.append(ColumnResolution.model_validate(item))
            except Exception:
                continue
    return out


def _coerce_list(value: Any) -> List[Any]:
    """Return *value* when it is a list, otherwise an empty list."""
    return list(value) if isinstance(value, list) else []


@registry.register
class ReportingAgent:
    spec = AgentSpec(
        name="reporting_agent",
        description="Assembles the final deliverable and writes it to disk.",
        stage="deliver",
        accepts=["dataframe"],
        produces=["file"],
        params_schema={
            "plan": {"type": "object"},
            "output_format": {
                "type": "string",
                "enum": ["xlsx", "csv", "json", "txt"],
            },
        },
    )
    params_model = ReportingAgentParams

    def execute(self, params: dict, input_data: dict) -> AgentResult:
        # 1. Single-source dataframe contract (req 5.5).
        df = input_data.get("input_dataframe") if input_data else None
        if df is None:
            return AgentResult(
                status="failed",
                error_message=(
                    "input_dataframe is required. No input dataframe provided."
                ),
            )

        params = params or {}
        input_data = input_data or {}

        # 2. Resolve the ReportingOperationPlan. The compiler emits the
        #    structured path (`params["plan"]`) and raw instructions are
        #    no longer accepted.
        plan_or_failure = self._resolve_plan(params)
        if isinstance(plan_or_failure, AgentResult):
            return plan_or_failure
        plan: ReportingOperationPlan = plan_or_failure

        # 3. Path resolution. The audit writer also runs every path through
        #    the path-safety helper, but we ensure the directory exists for
        #    the back-compat (csv / json / txt) handlers as well.
        output_dir = (
            params.get("output_dir")
            or os.environ.get("OUTPUT_DIR", "outputs")
        )
        os.makedirs(output_dir, exist_ok=True)
        file_prefix = (
            params.get("file_prefix") or f"output_{uuid.uuid4().hex}"
        )

        # 4. Defensive params re-validation. The engine already validates
        #    `step.params` against this model, but re-validating here
        #    catches direct callers (tests, scripts) that bypass the engine.
        try:
            ReportingAgentParams.model_validate(
                {
                    "plan": plan,
                    "output_dir": output_dir,
                    "file_prefix": file_prefix,
                }
            )
        except ValidationError as exc:
            return AgentResult(
                status="failed",
                error_message=(
                    f"Invalid parameter schema for ReportingAgent: {exc}"
                ),
            )

        # 5. Aggregate the audit context. Both `params` and `input_data`
        #    are accepted carriers so the engine can route the upstream
        #    `operations_applied`, `warnings`, and `column_mapping`
        #    artifacts in whichever way it ultimately wires (see task 9.1).
        audit_log = (
            _coerce_list(params.get("audit_log"))
            or _coerce_list(input_data.get("audit_log"))
            or _coerce_list(input_data.get("operations_applied"))
        )
        warnings_list = (
            _coerce_list(params.get("warnings"))
            or _coerce_list(input_data.get("warnings"))
        )
        column_mapping = _coerce_column_mapping(
            input_data.get("column_mapping")
            or params.get("column_mapping")
        )

        # `cleaned_dataframe` and `filtered_dataframe` keys, when present,
        # let the engine route the cleaned and filtered outputs separately
        # so both can land on dedicated audit sheets. When absent we fall
        # back to the single `input_dataframe` (req 5.5 path) and leave
        # `filtered_data` unset.
        cleaned_df = input_data.get("cleaned_dataframe", df)
        if not isinstance(cleaned_df, pd.DataFrame):
            cleaned_df = df
        filtered_df = input_data.get("filtered_dataframe")
        if filtered_df is not None and not isinstance(filtered_df, pd.DataFrame):
            filtered_df = None

        # 6. Filter-prep detection drives the canonical summary. The
        #    upstream `intent.needs_cleaning` is inferred from the audit
        #    log: any non-filter_prep entry indicates a clean-mode step
        #    ran, in which case we do NOT emit the canonical summary
        #    (per req 8.10 / 2.17).
        filter_prep_present = _has_filter_prep_entry(audit_log)
        clean_mode_present = _has_non_filter_prep_entry(audit_log)
        use_canonical_summary = filter_prep_present and not clean_mode_present

        # 7. Dispatch on output format.
        if plan.output_format == "xlsx":
            return self._write_audit_workbook(
                plan=plan,
                output_dir=output_dir,
                file_prefix=file_prefix,
                cleaned_df=cleaned_df,
                filtered_df=filtered_df,
                audit_log=audit_log,
                warnings_list=warnings_list,
                column_mapping=column_mapping,
                use_canonical_summary=use_canonical_summary,
            )

        # Non-xlsx back-compat path (csv / json / txt). The agent does not
        # mutate the dataframe before handing it to the deterministic
        # executor; the executor itself only writes the file and never
        # mutates values either.
        try:
            output = execute_reporting_plan(df, plan, output_dir, file_prefix)
        except Exception as exc:
            return AgentResult(
                status="failed",
                error_message=(
                    "Reporting failed while writing deliverable "
                    f"({_reporting_context(plan=plan, output_dir=output_dir, file_prefix=file_prefix)}): {exc}"
                ),
            )

        final_path = (output.artifacts or {}).get("output_file_path")
        artifacts: dict = {
            "primary_output_path": final_path,
            "output_file_path": final_path,
            "output_format": plan.output_format,
        }
        for k, v in (output.artifacts or {}).items():
            if k not in artifacts:
                artifacts[k] = v

        summary = (
            _FILTER_PREP_CANONICAL_SUMMARY
            if use_canonical_summary
            else output.summary
        )

        return AgentResult(
            status="success",
            data=final_path,
            summary=summary,
            metrics=output.metrics,
            operations_applied=output.operations_applied,
            warnings=output.warnings,
            artifacts=artifacts,
        )

    # ------------------------------------------------------------------
    # Plan resolution
    # ------------------------------------------------------------------
    def _resolve_plan(self, params: dict):
        """Resolve ``params`` into a :class:`ReportingOperationPlan`.

        Returns either a :class:`ReportingOperationPlan` or an
        :class:`AgentResult` describing a controlled failure.
        """
        plan_data = params.get("plan")
        if plan_data is not None:
            try:
                if isinstance(plan_data, ReportingOperationPlan):
                    return plan_data
                if isinstance(plan_data, dict) and plan_data:
                    return ReportingOperationPlan.model_validate(plan_data)
            except Exception as exc:
                return AgentResult(
                    status="failed",
                    error_message=(
                        "Failed to build reporting plan from params "
                        f"(plan={plan_data!r}): {exc}"
                    ),
                )

        return AgentResult(
            status="failed",
            error_message=(
                "ReportingAgent requires a typed ReportingOperationPlan; "
                "canonical execution does not accept raw instructions."
            ),
        )

    # ------------------------------------------------------------------
    # XLSX path: delegate to the audit-sheet writer.
    # ------------------------------------------------------------------
    def _write_audit_workbook(
        self,
        *,
        plan: ReportingOperationPlan,
        output_dir: str,
        file_prefix: str,
        cleaned_df: pd.DataFrame,
        filtered_df: Optional[pd.DataFrame],
        audit_log: List[Any],
        warnings_list: List[Any],
        column_mapping: List[ColumnResolution],
        use_canonical_summary: bool,
    ) -> AgentResult:
        try:
            payload = AuditSheetPayload(
                cleaned_data=cleaned_df,
                filtered_data=filtered_df,
                audit_log=list(audit_log),
                warnings=[str(w) for w in warnings_list],
                column_mapping=column_mapping,
            )
            writer_result = write_excel_with_audit_sheets(
                payload, plan, output_dir, file_prefix,
            )
        except Exception as exc:
            return AgentResult(
                status="failed",
                error_message=(
                    "Reporting failed while writing audit workbook "
                    f"({_reporting_context(plan=plan, output_dir=output_dir, file_prefix=file_prefix)}): {exc}"
                ),
            )

        final_path = writer_result.get("output_file_path")
        sheets_written = writer_result.get("sheets_written", [])

        if use_canonical_summary:
            summary = _FILTER_PREP_CANONICAL_SUMMARY
        else:
            summary = (
                f"Successfully exported {plan.output_format} report to "
                f"{final_path}."
            )

        artifacts = {
            "primary_output_path": final_path,
            "output_file_path": final_path,
            "output_format": plan.output_format,
            "sheets_written": sheets_written,
        }

        return AgentResult(
            status="success",
            data=final_path,
            summary=summary,
            metrics={},
            operations_applied=[],
            warnings=[],
            artifacts=artifacts,
        )
logger = logging.getLogger(__name__)
