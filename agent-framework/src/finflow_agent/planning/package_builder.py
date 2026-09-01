"""Build the shared IntentPackage from resolution and grounding results."""

from __future__ import annotations

from typing import List, Optional

from finflow_agent.operations.schemas import FilterOperationPlan
from finflow_agent.planning.intent_package import (
    IntentPackage,
    PackageStatus,
    ResolvedColumn,
    ResolutionMethod,
)
from finflow_agent.tools.column_resolver import (
    CONFIDENCE_THRESHOLD,
    ColumnResolution,
)
from finflow_agent.tools.dataframe_profile import DataFrameProfile
from finflow_agent.tools.predicate_grounder import (
    GroundedFilterClause,
    PredicateGroundingResult,
    UnresolvedFilterClause,
    ground_filter_clauses,
)
from finflow_agent.tools.semantic_column_profiler import (
    SemanticColumnProfile,
    profile_semantic_columns,
)


def _map_resolution_method(reason: str) -> ResolutionMethod:
    reason_lower = reason.lower()
    if "exact" in reason_lower:
        return ResolutionMethod.EXACT
    if "normalized" in reason_lower:
        return ResolutionMethod.NORMALIZED
    if "synonym" in reason_lower:
        return ResolutionMethod.SYNONYM
    if "fuzzy" in reason_lower:
        return ResolutionMethod.FUZZY
    if "llm" in reason_lower:
        return ResolutionMethod.LLM
    if "ground" in reason_lower:
        return ResolutionMethod.GROUNDING
    return ResolutionMethod.UNRESOLVED


def _build_resolved_columns_from_grounding(
    grounding_result: PredicateGroundingResult,
) -> list[ResolvedColumn]:
    resolved_columns: list[ResolvedColumn] = []
    for clause in grounding_result.grounded_clauses:
        selected_candidate = next(
            (
                candidate
                for candidate in clause.candidate_scores
                if candidate.column == clause.resolved_column
            ),
            None,
        )
        resolved_columns.append(
            ResolvedColumn(
                requested_field=clause.requested_field,
                resolved_column=clause.resolved_column,
                semantic_type=(
                    selected_candidate.broad_type.value
                    if selected_candidate is not None
                    else "grounded"
                ),
                confidence=clause.confidence,
                resolution_method=(
                    ResolutionMethod.LLM
                    if clause.grounding_method == "llm"
                    else ResolutionMethod.GROUNDING
                ),
                reason=(
                    f"{clause.grounding_method}: "
                    + "; ".join(clause.positive_evidence[:3])
                ).strip(),
                filter_value=str(clause.value) if clause.value is not None else None,
            )
        )
    return resolved_columns


def _build_resolved_columns_from_resolutions(
    resolutions: List[ColumnResolution],
    filter_values: Optional[List[Optional[str]]] = None,
) -> tuple[list[ResolvedColumn], list[str]]:
    unresolved_fields: list[str] = []
    resolved_columns: list[ResolvedColumn] = []
    values = filter_values or [None] * len(resolutions)
    for resolution, fv in zip(resolutions, values):
        method = _map_resolution_method(resolution.reason)
        resolved_columns.append(
            ResolvedColumn(
                requested_field=resolution.requested_field,
                resolved_column=resolution.matched_column,
                semantic_type=resolution.semantic_type,
                confidence=resolution.confidence,
                resolution_method=method,
                reason=resolution.reason,
                filter_value=fv,
            )
        )
        if resolution.confidence < CONFIDENCE_THRESHOLD:
            unresolved_fields.append(resolution.requested_field)
    return resolved_columns, unresolved_fields


def build_intent_package(
    *,
    submission_id: str,
    resolutions: Optional[List[ColumnResolution]] = None,
    filter_values: Optional[List[Optional[str]]] = None,
    filter_plan: Optional[FilterOperationPlan] = None,
    profile: Optional[DataFrameProfile] = None,
    semantic_profiles: Optional[List[SemanticColumnProfile]] = None,
    version: int = 1,
) -> IntentPackage:
    """Build an IntentPackage from column resolution and grounding results."""

    if filter_values is not None and resolutions is not None:
        if len(filter_values) != len(resolutions):
            raise ValueError(
                "filter_values must be the same length as resolutions when provided"
            )

    if filter_plan is not None and profile is not None:
        semantic_profiles = semantic_profiles or profile_semantic_columns(profile)
        clauses = [
            UnresolvedFilterClause(
                requested_field=condition.column,
                operator=condition.operator,
                value=condition.value,
                value_to=condition.value_to,
                case_sensitive=bool(getattr(condition, "case_sensitive", False)),
            )
            for condition in filter_plan.conditions
        ]
        grounding_result = ground_filter_clauses(
            clauses,
            profile=profile,
            semantic_profiles=semantic_profiles,
        )
        resolved_columns = _build_resolved_columns_from_grounding(grounding_result)
        unresolved_fields = [clause.requested_field for clause in grounding_result.unresolved_clauses]
        status = {
            "grounded": PackageStatus.VALID,
            "needs_review": PackageStatus.NEEDS_REVIEW,
            "quarantined": PackageStatus.QUARANTINED,
        }[grounding_result.status]
        resolution_summary = {
            "total_fields": len(clauses),
            "resolved_count": len(resolved_columns),
            "unresolved_count": len(unresolved_fields),
            "grounding_status": grounding_result.status,
            "grounding_reason": grounding_result.reason,
            "methods_used": sorted(
                {rc.resolution_method.value for rc in resolved_columns}
            ),
        }
        return IntentPackage(
            submission_id=submission_id,
            version=version,
            status=status,
            resolved_columns=resolved_columns,
            unresolved_fields=unresolved_fields,
            semantic_profiles=semantic_profiles,
            grounding_result=grounding_result,
            resolution_summary=resolution_summary,
        )

    if resolutions is None:
        resolutions = []

    resolved_columns, unresolved_fields = _build_resolved_columns_from_resolutions(
        resolutions,
        filter_values=filter_values,
    )
    semantic_profiles = semantic_profiles or []
    status = (
        PackageStatus.VALID
        if not unresolved_fields
        else PackageStatus.NEEDS_REVIEW
    )
    resolution_summary = {
        "total_fields": len(resolutions),
        "resolved_count": len(resolved_columns),
        "unresolved_count": len(unresolved_fields),
        "methods_used": sorted({rc.resolution_method.value for rc in resolved_columns}),
    }
    return IntentPackage(
        submission_id=submission_id,
        version=version,
        status=status,
        resolved_columns=resolved_columns,
        unresolved_fields=unresolved_fields,
        semantic_profiles=semantic_profiles,
        resolution_summary=resolution_summary,
    )
