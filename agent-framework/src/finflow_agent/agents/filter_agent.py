"""Filter agent for the FinFlow Agent Service."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd
from pydantic import BaseModel, ValidationError

from finflow_agent.operations.schemas import FilterCondition, FilterOperationPlan
from finflow_agent.planning.intent_package import (
    ContractViolation,
    IntentPackage,
    PackageStatus,
)
from finflow_agent.planning.package_builder import build_intent_package
from finflow_agent.registry import AgentSpec, registry
from finflow_agent.llm_telemetry import log_runtime_event
from finflow_agent.tools.column_resolver import (
    ColumnResolution,
    enforce_low_confidence_policy,
)
from finflow_agent.tools.dataframe_profile import profile_dataframe
from finflow_agent.tools.value_resolver import resolve_value_domain


logger = logging.getLogger(__name__)


class FilterAgentParams(BaseModel):
    plan: FilterOperationPlan


@registry.register
class FilterAgent:
    spec = AgentSpec(
        name="filter_agent",
        description="Filters rows and selects columns using execute_filter_plan.",
        stage="transform",
        accepts=["dataframe"],
        produces=["dataframe"],
        params_schema={"plan": {"type": "object"}},
    )
    params_model = FilterAgentParams

    def execute(self, params: dict, input_data: dict) -> AgentResult:
        from finflow_agent.state import AgentResult

        df = input_data.get("input_dataframe") if input_data else None
        if df is None:
            return AgentResult(
                status="failed",
                error_message="input_dataframe is required. No input dataframe provided.",
            )

        plan_or_failure = self._extract_or_build_plan(params, df)
        if isinstance(plan_or_failure, AgentResult):
            return plan_or_failure
        plan: FilterOperationPlan = plan_or_failure

        intent_package = input_data.get("intent_package")
        if not isinstance(intent_package, IntentPackage):
            try:
                profile = profile_dataframe(df, include_samples=True, sample_rows=5)
                intent_package = build_intent_package(
                    submission_id=str(
                        input_data.get("submission_id")
                        or input_data.get("job_id")
                        or "legacy-filter-agent"
                    ),
                    filter_plan=plan,
                    profile=profile,
                )
            except Exception as exc:
                return AgentResult(
                    status="failed",
                    error_message=(
                        "intent_package could not be built from the legacy "
                        f"filter-agent inputs: {exc}"
                    ),
                )

        if intent_package.status == PackageStatus.QUARANTINED:
            return AgentResult(
                status="failed",
                error_message=(
                    intent_package.quarantine_reason
                    or "IntentPackage is quarantined."
                ),
                artifacts={
                    "contract_violation": {
                        "status": "quarantined",
                        "reason": intent_package.quarantine_reason,
                        "package_version": intent_package.version_key,
                    }
                },
            )

        review_artifact = None
        if intent_package.status == PackageStatus.NEEDS_REVIEW:
            review_artifact = {
                "status": intent_package.status.value,
                "reason": intent_package.grounding_result.reason
                if intent_package.grounding_result
                else "Filter grounding requires review.",
                "package_version": intent_package.version_key,
                "unresolved_fields": intent_package.unresolved_fields,
                "grounding_result": (
                    intent_package.grounding_result.model_dump(mode="json")
                    if intent_package.grounding_result is not None
                    else None
                ),
            }
            logger.info(
                "event=filter_intent_package_needs_review submission_id=%s reason=%s",
                intent_package.submission_id,
                review_artifact["reason"],
            )

        resolutions, grounding_artifacts = self._build_resolutions(
            plan=plan,
            intent_package=intent_package,
            dataframe=df,
        )
        column_mapping_artifact: List[Dict[str, Any]] = [
            resolution.model_dump(mode="json") for resolution in resolutions
        ]

        warnings: List[str] = []
        skipped: Set[int] = set()

        for idx, resolution in enumerate(resolutions):
            decision, message = enforce_low_confidence_policy(resolution)
            if decision == "allow":
                continue
            if decision == "warn":
                warnings.append(
                    message
                    or (
                        f"Low-confidence column match for "
                        f"{resolution.requested_field!r}; condition skipped."
                    )
                )
                skipped.add(idx)
                continue
            if decision == "fail":
                return AgentResult(
                    status="failed",
                    error_message=message
                    or (
                        f"Low-confidence column match for "
                        f"{resolution.requested_field!r}."
                    ),
                    artifacts={
                        "column_mapping": column_mapping_artifact,
                        "predicate_grounding": grounding_artifacts,
                    },
                    warnings=warnings,
                )
            if decision == "quarantine":
                return AgentResult(
                    status="failed",
                    error_message=message
                    or (
                        f"Low-confidence column match for "
                        f"{resolution.requested_field!r}; quarantined."
                    ),
                    artifacts={
                        "column_mapping": column_mapping_artifact,
                        "predicate_grounding": grounding_artifacts,
                        "quarantine": {
                            "reason": message,
                            "resolution": resolution.model_dump(mode="json"),
                        },
                    },
                    warnings=warnings,
                )
            return AgentResult(
                status="failed",
                error_message=(
                    f"Unknown low-confidence policy decision: {decision!r}."
                ),
                artifacts={
                    "column_mapping": column_mapping_artifact,
                    "predicate_grounding": grounding_artifacts,
                },
                warnings=warnings,
            )

        effective_conditions: List[FilterCondition] = []
        for idx, cond in enumerate(plan.conditions):
            if idx in skipped:
                continue
            resolution = resolutions[idx]
            target_column = resolution.matched_column
            if target_column not in df.columns:
                violation = ContractViolation(
                    step_id="filter",
                    agent="filter_agent",
                    violation_type="column_missing",
                    expected=target_column,
                    actual=f"columns={list(df.columns)}",
                )
                intent_package.add_violation(violation)
                return AgentResult(
                    status="failed",
                    error_message=(
                        f"Contract violation: resolved column {target_column!r} "
                        f"not found in dataframe. Package quarantined."
                    ),
                    artifacts={
                        "contract_violation": violation.model_dump(mode="json"),
                        "column_mapping": column_mapping_artifact,
                        "predicate_grounding": grounding_artifacts,
                    },
                    warnings=warnings,
                )
            effective_conditions.append(
                cond.model_copy(update={"column": target_column})
            )

        effective_plan = plan.model_copy(update={"conditions": effective_conditions})

        try:
            FilterAgentParams.model_validate({"plan": effective_plan})
        except ValidationError as exc:
            return AgentResult(
                status="failed",
                error_message=f"Invalid parameter schema for FilterAgent: {exc}",
                artifacts={
                    "column_mapping": column_mapping_artifact,
                    "predicate_grounding": grounding_artifacts,
                },
                warnings=warnings,
            )

        if effective_plan.select_columns:
            missing_cols = [
                col for col in effective_plan.select_columns if col not in df.columns
            ]
            if missing_cols:
                return AgentResult(
                    status="failed",
                    error_message=(
                        "Missing selected columns in dataframe: "
                        + ", ".join(missing_cols)
                    ),
                    artifacts={
                        "column_mapping": column_mapping_artifact,
                        "predicate_grounding": grounding_artifacts,
                    },
                    warnings=warnings,
                )

        value_resolutions: List[Dict[str, Any]] = []
        for cond in effective_plan.conditions:
            resolved_column = cond.column
            value_resolution = resolve_value_domain(
                df[resolved_column],
                cond.value,
                cond.operator,
                case_sensitive=cond.case_sensitive,
            )
            value_resolutions.append(value_resolution.model_dump(mode="json"))

            if not value_resolution.matched:
                warnings.append(
                    f"Requested value {cond.value!r} was not observed in column "
                    f"{resolved_column!r}; filter may return zero rows."
                )

        try:
            from finflow_agent.operations.executor import execute_filter_plan

            output = execute_filter_plan(df.copy(), effective_plan)
        except Exception as exc:
            return AgentResult(
                status="failed",
                error_message=f"Failed to execute filter plan: {exc}",
                artifacts={
                    "column_mapping": column_mapping_artifact,
                    "predicate_grounding": grounding_artifacts,
                },
                warnings=warnings,
            )

        merged_warnings = list(warnings) + list(output.warnings or [])
        merged_artifacts: Dict[str, Any] = dict(output.artifacts) if output.artifacts else {}
        merged_artifacts["column_mapping"] = column_mapping_artifact
        if grounding_artifacts:
            merged_artifacts["predicate_grounding"] = grounding_artifacts
        if value_resolutions:
            merged_artifacts["value_resolution"] = value_resolutions
        if review_artifact is not None:
            merged_artifacts["needs_review"] = review_artifact

        return AgentResult(
            status="success",
            data=output.data,
            summary=output.summary,
            metrics=output.metrics,
            operations_applied=output.operations_applied,
            warnings=merged_warnings,
            artifacts=merged_artifacts,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_resolutions(
        self,
        *,
        plan: FilterOperationPlan,
        intent_package: IntentPackage,
        dataframe: pd.DataFrame,
    ) -> tuple[List[ColumnResolution], List[Dict[str, Any]]]:
        resolutions: List[ColumnResolution] = []
        grounding_artifacts: List[Dict[str, Any]] = []
        available_columns = [str(column) for column in dataframe.columns]
        available_column_keys = {
            _normalize_column_key(column): str(column) for column in available_columns
        }
        binding_keys = _intent_package_binding_keys(intent_package)

        for cond in plan.conditions:
            requested_field = str(cond.column).strip()
            physical_column = _match_physical_column(requested_field, available_column_keys)

            _log_grounding_decision(
                requested_field=requested_field,
                semantic_field=requested_field,
                physical_column=physical_column,
                resolved_column=physical_column,
                available_dataframe_columns=available_columns,
                intent_package_binding_keys=binding_keys,
                grounding_path_selected=(
                    "canonical_physical_column" if physical_column is not None else "intent_package_lookup"
                ),
                llm_fallback_reason="not_needed" if physical_column is not None else "pending",
                llm_called=False,
            )

            if physical_column is not None:
                resolution = ColumnResolution(
                    requested_field=requested_field,
                    matched_column=physical_column,
                    semantic_type="grounded",
                    confidence=1.0,
                    reason="canonical_physical_column",
                )
                resolutions.append(resolution)
                continue

            grounded_clause = intent_package.get_grounded_clause(requested_field)
            resolved_column = intent_package.get_resolved_column(requested_field)

            if grounded_clause is not None:
                semantic_type = "grounded"
                if grounded_clause.candidate_scores:
                    selected_candidate = next(
                        (
                            candidate
                            for candidate in grounded_clause.candidate_scores
                            if candidate.column == grounded_clause.resolved_column
                        ),
                        None,
                    )
                    if selected_candidate is not None:
                        semantic_type = selected_candidate.broad_type.value
                resolution = ColumnResolution(
                    requested_field=cond.column,
                    matched_column=grounded_clause.resolved_column,
                    semantic_type=semantic_type,
                    confidence=grounded_clause.confidence,
                    reason=(
                        f"semantic grounding ({grounded_clause.grounding_method}): "
                        + "; ".join(grounded_clause.positive_evidence[:3])
                    ),
                )
                grounding_artifacts.append(grounded_clause.model_dump(mode="json"))
            elif resolved_column is not None:
                resolution = ColumnResolution(
                    requested_field=requested_field,
                    matched_column=resolved_column.resolved_column,
                    semantic_type=resolved_column.semantic_type or "unknown",
                    confidence=resolved_column.confidence,
                    reason=f"from_intent_package_v{intent_package.version}",
                )
            else:
                violation = ContractViolation(
                    step_id="filter",
                    agent="filter_agent",
                    violation_type="resolution_missing",
                    expected=requested_field,
                    actual=(
                        "package_fields="
                        f"{[rc.requested_field for rc in intent_package.resolved_columns]}"
                    ),
                )
                intent_package.add_violation(violation)
                raise ValueError(
                    f"Contract violation: resolved column for {requested_field!r} "
                    f"is missing from IntentPackage. Package quarantined."
                )

            resolutions.append(resolution)

        return resolutions, grounding_artifacts

    def _extract_or_build_plan(
        self,
        params: dict,
        df: pd.DataFrame,
    ) -> Union[FilterOperationPlan, AgentResult]:
        from finflow_agent.state import AgentResult

        params = params or {}

        plan_data = params.get("plan")
        if plan_data is not None:
            try:
                if isinstance(plan_data, FilterOperationPlan):
                    return plan_data
                return FilterOperationPlan.model_validate(plan_data)
            except Exception as exc:
                return AgentResult(
                    status="failed",
                    error_message=f"Invalid filter parameters: {exc}",
                )

        return AgentResult(
            status="failed",
            error_message=(
                "FilterAgent requires a typed FilterOperationPlan; "
                "canonical execution does not accept raw instructions."
            ),
        )


def _normalize_column_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _match_physical_column(
    requested_field: str,
    available_column_keys: Dict[str, str],
) -> Optional[str]:
    if not requested_field:
        return None
    normalized = _normalize_column_key(requested_field)
    if normalized in available_column_keys:
        return available_column_keys[normalized]
    requested_lower = requested_field.strip().lower()
    return available_column_keys.get(requested_lower)


def _intent_package_binding_keys(intent_package: IntentPackage) -> List[str]:
    keys: List[str] = []
    keys.extend(
        rc.requested_field
        for rc in intent_package.resolved_columns
        if getattr(rc, "requested_field", None)
    )
    if intent_package.grounding_result is not None:
        keys.extend(
            clause.requested_field
            for clause in intent_package.grounding_result.grounded_clauses
            if getattr(clause, "requested_field", None)
        )
    return sorted({str(key).strip() for key in keys if str(key).strip()})


def _log_grounding_decision(
    *,
    requested_field: str,
    semantic_field: str,
    physical_column: Optional[str],
    resolved_column: Optional[str],
    available_dataframe_columns: List[str],
    intent_package_binding_keys: List[str],
    grounding_path_selected: str,
    llm_fallback_reason: str,
    llm_called: bool,
) -> None:
    try:
        log_runtime_event(
            "predicate_grounding_decision",
            service="agent-service",
            trigger="worker",
            requested_field=requested_field,
            semantic_field=semantic_field,
            physical_column=physical_column,
            resolved_column=resolved_column,
            available_dataframe_columns=available_dataframe_columns,
            intent_package_binding_keys=intent_package_binding_keys,
            grounding_path_selected=grounding_path_selected,
            llm_fallback_reason=llm_fallback_reason,
            llm_called=llm_called,
        )
    except Exception:
        pass
