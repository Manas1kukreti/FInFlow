"""Deterministic tests for the IntentPackage shared resolution layer.

Covers:
1. Package creation from column resolutions via build_intent_package.
2. Version key derivation.
3. Resolved column lookup.
4. Patch-and-increment semantics.
5. Contract violation quarantine.
6. Filter agent integration with intent package.
7. Engine pass-through of intent_package to agents.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Type
from unittest.mock import patch

import pandas as pd
import pytest

from finflow_agent.planning.intent_package import (
    ContractViolation,
    IntentPackage,
    PackageStatus,
    ResolvedColumn,
    ResolutionMethod,
)
from finflow_agent.planning.package_builder import build_intent_package
from finflow_agent.tools.column_resolver import ColumnResolution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_resolution(
    requested_field: str,
    matched_column: str,
    confidence: float,
    reason: str = "exact name match (case-insensitive)",
) -> ColumnResolution:
    return ColumnResolution(
        requested_field=requested_field,
        matched_column=matched_column,
        semantic_type="string",
        confidence=confidence,
        reason=reason,
    )


def _make_package(
    submission_id: str = "sub-001",
    resolved: Optional[List[ResolvedColumn]] = None,
    unresolved: Optional[List[str]] = None,
    version: int = 1,
) -> IntentPackage:
    return IntentPackage(
        submission_id=submission_id,
        version=version,
        resolved_columns=resolved or [],
        unresolved_fields=unresolved or [],
    )


# ---------------------------------------------------------------------------
# 1. IntentPackage creation from resolutions
# ---------------------------------------------------------------------------


def test_intent_package_creation_from_resolutions():
    """build_intent_package produces a valid package with correct fields."""
    resolutions = [
        _make_resolution("gender", "Gender", 1.0),
        _make_resolution("dob", "date_of_birth", 0.95, reason="normalized name match"),
        _make_resolution("amount", "total_amount", 0.85, reason="semantic synonym match (currency)"),
    ]

    pkg = build_intent_package(
        submission_id="test-sub-001",
        resolutions=resolutions,
        filter_values=["female", "2000-01-01", "100"],
    )

    assert pkg.submission_id == "test-sub-001"
    assert pkg.version == 1
    assert pkg.status == PackageStatus.VALID
    assert len(pkg.resolved_columns) == 3
    assert pkg.unresolved_fields == []
    assert pkg.resolution_summary["total_fields"] == 3
    assert pkg.resolution_summary["resolved_count"] == 3
    assert pkg.resolution_summary["unresolved_count"] == 0

    # Check individual resolutions
    gender_rc = pkg.get_resolved_column("gender")
    assert gender_rc is not None
    assert gender_rc.resolved_column == "Gender"
    assert gender_rc.confidence == 1.0
    assert gender_rc.resolution_method == ResolutionMethod.EXACT
    assert gender_rc.filter_value == "female"

    dob_rc = pkg.get_resolved_column("dob")
    assert dob_rc is not None
    assert dob_rc.resolution_method == ResolutionMethod.NORMALIZED

    amount_rc = pkg.get_resolved_column("amount")
    assert amount_rc is not None
    assert amount_rc.resolution_method == ResolutionMethod.SYNONYM


# ---------------------------------------------------------------------------
# 2. Version key derivation
# ---------------------------------------------------------------------------


def test_intent_package_version_key_derivation():
    """version_key is '{submission_id}:v{version}'."""
    pkg = _make_package(submission_id="job-42", version=3)
    assert pkg.version_key == "job-42:v3"

    pkg2 = _make_package(submission_id="abc", version=1)
    assert pkg2.version_key == "abc:v1"


# ---------------------------------------------------------------------------
# 3. Resolved column lookup
# ---------------------------------------------------------------------------


def test_intent_package_get_resolved_column():
    """Lookup works for existing fields and returns None for missing."""
    rc = ResolvedColumn(
        requested_field="gender",
        resolved_column="Gender",
        confidence=1.0,
        resolution_method=ResolutionMethod.EXACT,
        reason="exact match",
    )
    pkg = _make_package(resolved=[rc])

    # Existing field
    found = pkg.get_resolved_column("gender")
    assert found is not None
    assert found.resolved_column == "Gender"

    # Missing field
    assert pkg.get_resolved_column("nonexistent") is None
    assert not pkg.has_resolution_for("nonexistent")
    assert pkg.has_resolution_for("gender")


def test_intent_package_get_resolved_column_matches_physical_name():
    """Lookup should also work when callers ask by the physical column name."""
    rc = ResolvedColumn(
        requested_field="education",
        resolved_column="education_level",
        confidence=1.0,
        resolution_method=ResolutionMethod.EXACT,
        reason="grounded by semantic field",
    )
    pkg = _make_package(resolved=[rc])

    found = pkg.get_resolved_column("education_level")
    assert found is not None
    assert found.requested_field == "education"
    assert found.resolved_column == "education_level"
    assert pkg.has_resolution_for("education_level")


# ---------------------------------------------------------------------------
# 4. Patch increments version
# ---------------------------------------------------------------------------


def test_intent_package_patch_increments_version():
    """patch_column returns v+1 with only the patched field changed."""
    rc = ResolvedColumn(
        requested_field="gender",
        resolved_column="Gender",
        confidence=0.85,
        resolution_method=ResolutionMethod.SYNONYM,
        reason="synonym match",
    )
    pkg = _make_package(submission_id="sub-X", version=2, resolved=[rc])

    patched = pkg.patch_column("gender", "sex", "user correction")

    assert patched.version == 3
    assert patched.submission_id == "sub-X"
    assert patched.version_key == "sub-X:v3"
    assert patched.status == PackageStatus.VALID

    gender_rc = patched.get_resolved_column("gender")
    assert gender_rc is not None
    assert gender_rc.resolved_column == "sex"
    assert gender_rc.confidence == 1.0
    assert gender_rc.resolution_method == ResolutionMethod.EXACT
    assert "patched" in gender_rc.reason


# ---------------------------------------------------------------------------
# 5. Patch preserves other resolutions
# ---------------------------------------------------------------------------


def test_intent_package_patch_preserves_other_resolutions():
    """Other fields are untouched after patch."""
    rc1 = ResolvedColumn(
        requested_field="gender",
        resolved_column="Gender",
        confidence=1.0,
        resolution_method=ResolutionMethod.EXACT,
        reason="exact match",
    )
    rc2 = ResolvedColumn(
        requested_field="age",
        resolved_column="Age",
        confidence=0.95,
        resolution_method=ResolutionMethod.NORMALIZED,
        reason="normalized name match",
    )
    pkg = _make_package(resolved=[rc1, rc2])

    patched = pkg.patch_column("gender", "sex", "correction")

    # age should remain unchanged
    age_rc = patched.get_resolved_column("age")
    assert age_rc is not None
    assert age_rc.resolved_column == "Age"
    assert age_rc.confidence == 0.95
    assert age_rc.resolution_method == ResolutionMethod.NORMALIZED


# ---------------------------------------------------------------------------
# 6. Add violation quarantines
# ---------------------------------------------------------------------------


def test_intent_package_add_violation_quarantines():
    """Adding a violation sets status to QUARANTINED."""
    pkg = _make_package()
    assert pkg.status == PackageStatus.VALID

    violation = ContractViolation(
        step_id="filter",
        agent="filter_agent",
        violation_type="column_missing",
        expected="Gender",
        actual="columns=['age', 'score']",
    )
    pkg.add_violation(violation)

    assert pkg.status == PackageStatus.QUARANTINED
    assert pkg.quarantine_reason is not None
    assert "column_missing" in pkg.quarantine_reason
    assert "filter_agent" in pkg.quarantine_reason
    assert len(pkg.violations) == 1


# ---------------------------------------------------------------------------
# 7. Unresolved fields set NEEDS_REVIEW
# ---------------------------------------------------------------------------


def test_intent_package_unresolved_fields_set_needs_review():
    """When resolutions below threshold, status is NEEDS_REVIEW."""
    resolutions = [
        _make_resolution("gender", "Gender", 1.0),
        _make_resolution("xyz_unknown", "a", 0.3, reason="low-confidence fuzzy match"),
    ]

    pkg = build_intent_package(
        submission_id="sub-review",
        resolutions=resolutions,
    )

    assert pkg.status == PackageStatus.NEEDS_REVIEW
    assert "xyz_unknown" in pkg.unresolved_fields
    assert len(pkg.resolved_columns) == 2
    assert pkg.get_resolved_column("xyz_unknown") is not None
    assert pkg.resolution_summary["unresolved_count"] == 1


# ---------------------------------------------------------------------------
# 8. Filter agent uses intent package instead of re-resolving
# ---------------------------------------------------------------------------


def test_filter_agent_uses_intent_package_instead_of_re_resolving(
    bootstrap_agents,
):
    """When an intent_package is passed with pre-resolved columns, the
    filter agent must NOT call the column resolver again."""
    from finflow_agent.agents.filter_agent import FilterAgent

    df = pd.DataFrame({
        "Gender": ["female", "male", "female", "male"],
        "Age": [25, 30, 35, 40],
    })

    # Pre-resolve "gender" -> "Gender" in the package
    rc = ResolvedColumn(
        requested_field="gender",
        resolved_column="Gender",
        confidence=1.0,
        resolution_method=ResolutionMethod.EXACT,
        reason="exact match from package",
    )
    intent_pkg = IntentPackage(
        submission_id="test-filter-pkg",
        version=1,
        resolved_columns=[rc],
    )

    plan = {
        "conditions": [
            {"column": "gender", "operator": "eq", "value": "female", "case_sensitive": False}
        ],
        "logic": "and",
    }

    with patch(
        "finflow_agent.tools.column_resolver.resolve_columns"
    ) as mock_resolve:
        result = FilterAgent().execute(
            {"plan": plan},
            {"input_dataframe": df, "intent_package": intent_pkg},
        )

    # The resolver should NOT have been called because the field was
    # pre-resolved in the intent package.
    mock_resolve.assert_not_called()

    assert result.status == "success", result.error_message
    assert isinstance(result.data, pd.DataFrame)
    # All rows with Gender == "female" should be returned
    assert len(result.data) == 2
    assert all(result.data["Gender"] == "female")


# ---------------------------------------------------------------------------
# 9. Filter agent runtime violation on missing column
# ---------------------------------------------------------------------------


def test_filter_agent_runtime_violation_on_missing_column(bootstrap_agents):
    """When intent_package has a resolved column that doesn't exist in the
    dataframe, the filter agent should return a failed result with a
    contract_violation artifact."""
    from finflow_agent.agents.filter_agent import FilterAgent

    # The dataframe does NOT have a "Gender" column
    df = pd.DataFrame({
        "age": [25, 30, 35],
        "score": [10, 20, 30],
    })

    # Package says "Gender" exists (but it doesn't in our df)
    rc = ResolvedColumn(
        requested_field="gender",
        resolved_column="Gender",
        confidence=1.0,
        resolution_method=ResolutionMethod.EXACT,
        reason="exact match",
    )
    intent_pkg = IntentPackage(
        submission_id="test-violation",
        version=1,
        resolved_columns=[rc],
    )

    plan = {
        "conditions": [
            {
                "column": "gender",
                "operator": "eq",
                "value": "female",
                "case_sensitive": False,
            }
        ],
        "logic": "and",
    }

    result = FilterAgent().execute(
        {"plan": plan},
        {"input_dataframe": df, "intent_package": intent_pkg},
    )

    assert result.status == "failed"
    assert "Contract violation" in result.error_message
    assert "Gender" in result.error_message
    assert "contract_violation" in result.artifacts

    # Package should be quarantined
    assert intent_pkg.status == PackageStatus.QUARANTINED
    assert len(intent_pkg.violations) == 1


# ---------------------------------------------------------------------------
# 10. Engine passes intent_package to agents
# ---------------------------------------------------------------------------


def test_engine_passes_intent_package_to_agents(bootstrap_agents):
    """Build a plan, pass intent_package to engine.execute(), verify
    agents receive it in input_data."""
    from contextlib import contextmanager

    from finflow_agent.execution.engine import ExecutionEngine
    from finflow_agent.registry import AgentSpec, registry
    from finflow_agent.state import AgentResult, ExecutionPlan, PlanStep

    captured_input_data: List[Dict[str, Any]] = []
    fake_name = f"test_pkg_pass_{uuid.uuid4().hex[:8]}"

    class PackageCapturingAgent:
        spec = AgentSpec(
            name=fake_name,
            description="Captures intent_package from input_data",
            stage="ingest",
            accepts=["file"],
            produces=["dataframe"],
            params_schema={},
        )

        def execute(self, params, input_data):
            captured_input_data.append(dict(input_data))
            return AgentResult(status="success", data=pd.DataFrame({"x": [1]}))

    registry.register(PackageCapturingAgent)
    try:
        plan = ExecutionPlan(
            steps=[
                PlanStep(
                    step_id="ingest",
                    agent=fake_name,
                    params={},
                    depends_on=[],
                    input_from=[],
                    output_key="df_ingested",
                ),
            ]
        )

        rc = ResolvedColumn(
            requested_field="col_a",
            resolved_column="Column_A",
            confidence=1.0,
            resolution_method=ResolutionMethod.EXACT,
            reason="exact match",
        )
        intent_pkg = IntentPackage(
            submission_id="engine-test",
            version=1,
            resolved_columns=[rc],
        )

        result = ExecutionEngine().execute(plan, intent_package=intent_pkg)
    finally:
        registry._agents.pop(fake_name, None)
        registry._specs.pop(fake_name, None)
        registry._param_models.pop(fake_name, None)

    assert result["status"] == "complete", result
    assert len(captured_input_data) == 1

    received = captured_input_data[0]
    assert "intent_package" in received
    assert received["intent_package"] is intent_pkg
    assert received["intent_package"].submission_id == "engine-test"
    assert received["intent_package"].get_resolved_column("col_a") is not None
