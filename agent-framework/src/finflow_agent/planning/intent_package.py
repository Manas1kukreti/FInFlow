"""Shared, versioned schema-resolution artifact."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from enum import Enum
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from finflow_agent.tools.predicate_grounder import (
    GroundedFilterClause,
    PredicateGroundingResult,
    UnresolvedFilterClause,
)
from finflow_agent.tools.semantic_column_profiler import SemanticColumnProfile


class ResolutionMethod(str, Enum):
    EXACT = "exact"
    NORMALIZED = "normalized"
    SYNONYM = "synonym"
    FUZZY = "fuzzy"
    LLM = "llm"
    UNRESOLVED = "unresolved"
    GROUNDING = "grounding"


class ResolvedColumn(BaseModel):
    """A single column resolution result stored in the package."""

    requested_field: str
    resolved_column: str
    semantic_type: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    resolution_method: ResolutionMethod
    reason: str
    filter_value: Optional[str] = None


class PackageStatus(str, Enum):
    VALID = "valid"
    QUARANTINED = "quarantined"
    NEEDS_REVIEW = "needs_review"


class ContractViolation(BaseModel):
    """Structured signal emitted when runtime evidence contradicts the schema."""

    step_id: str
    agent: str
    violation_type: str
    expected: str
    actual: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IntentPackage(BaseModel):
    """The shared resolved schema artifact for a job execution."""

    submission_id: str
    version: int = 1
    version_key: str = ""

    status: PackageStatus = PackageStatus.VALID
    resolved_columns: List[ResolvedColumn] = Field(default_factory=list)
    unresolved_fields: List[str] = Field(default_factory=list)

    semantic_profiles: List[SemanticColumnProfile] = Field(default_factory=list)
    grounding_result: Optional[PredicateGroundingResult] = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolution_summary: Dict[str, Any] = Field(default_factory=dict)

    violations: List[ContractViolation] = Field(default_factory=list)
    quarantine_reason: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        if not self.version_key:
            self.version_key = f"{self.submission_id}:v{self.version}"

    def get_resolved_column(self, requested_field: str) -> Optional[ResolvedColumn]:
        requested_key = _normalize_lookup_key(requested_field)
        for rc in self.resolved_columns:
            if _normalize_lookup_key(rc.requested_field) == requested_key:
                return rc
        for rc in self.resolved_columns:
            if _normalize_lookup_key(rc.resolved_column) == requested_key:
                return rc
        return None

    def get_grounded_clause(self, requested_field: str) -> Optional[GroundedFilterClause]:
        requested_key = _normalize_lookup_key(requested_field)
        if self.grounding_result is None:
            return None
        for clause in self.grounding_result.grounded_clauses:
            if _normalize_lookup_key(clause.requested_field) == requested_key:
                return clause
        for clause in self.grounding_result.grounded_clauses:
            if _normalize_lookup_key(clause.resolved_column) == requested_key:
                return clause
        return None

    def has_resolution_for(self, requested_field: str) -> bool:
        return self.get_resolved_column(requested_field) is not None

    def add_violation(self, violation: ContractViolation) -> None:
        self.violations.append(violation)
        self.status = PackageStatus.QUARANTINED
        self.quarantine_reason = (
            f"Runtime contract violation in {violation.agent} "
            f"(step {violation.step_id}): {violation.violation_type} — "
            f"expected {violation.expected!r}, got {violation.actual!r}"
        )

    def patch_column(
        self,
        requested_field: str,
        new_column: str,
        reason: str,
    ) -> "IntentPackage":
        """Create a new version with one column resolution patched."""

        new_resolutions: list[ResolvedColumn] = []
        patched = False
        for rc in self.resolved_columns:
            if rc.requested_field == requested_field:
                new_resolutions.append(
                    ResolvedColumn(
                        requested_field=requested_field,
                        resolved_column=new_column,
                        semantic_type=rc.semantic_type,
                        confidence=1.0,
                        resolution_method=ResolutionMethod.EXACT,
                        reason=f"patched: {reason}",
                        filter_value=rc.filter_value,
                    )
                )
                patched = True
            else:
                new_resolutions.append(rc)

        if not patched:
            new_resolutions.append(
                ResolvedColumn(
                    requested_field=requested_field,
                    resolved_column=new_column,
                    semantic_type="",
                    confidence=1.0,
                    resolution_method=ResolutionMethod.EXACT,
                    reason=f"patched: {reason}",
                )
            )

        new_unresolved = [f for f in self.unresolved_fields if f != requested_field]
        new_status = (
            PackageStatus.VALID if not new_unresolved else PackageStatus.NEEDS_REVIEW
        )

        new_grounding_result = None
        if self.grounding_result is not None:
            new_grounded = []
            for clause in self.grounding_result.grounded_clauses:
                if clause.requested_field == requested_field:
                    new_grounded.append(
                        clause.model_copy(
                            update={
                                "resolved_column": new_column,
                                "confidence": 1.0,
                                "grounding_method": "manual",
                                "positive_evidence": clause.positive_evidence
                                + [f"patched: {reason}"],
                            }
                        )
                    )
                else:
                    new_grounded.append(clause)
            new_unresolved_clauses = [
                clause
                for clause in self.grounding_result.unresolved_clauses
                if clause.requested_field != requested_field
            ]
            grounding_status = "grounded" if not new_unresolved_clauses else "needs_review"
            new_grounding_result = self.grounding_result.model_copy(
                update={
                    "grounded_clauses": new_grounded,
                    "unresolved_clauses": new_unresolved_clauses,
                    "status": grounding_status,
                    "reason": (
                        "Patched grounding from version "
                        f"{self.version} for field {requested_field}"
                    ),
                }
            )

        return IntentPackage(
            submission_id=self.submission_id,
            version=self.version + 1,
            status=new_status,
            resolved_columns=new_resolutions,
            unresolved_fields=new_unresolved,
            semantic_profiles=deepcopy(self.semantic_profiles),
            grounding_result=new_grounding_result,
            resolution_summary={
                **self.resolution_summary,
                "patched_from_version": self.version,
                "patched_field": requested_field,
            },
            violations=[],
            quarantine_reason=None,
        )


def _normalize_lookup_key(value: str | None) -> str:
    """Normalize lookup keys so raw references and physical names can match."""
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
