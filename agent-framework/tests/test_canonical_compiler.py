import os
import sys

import pytest


from finflow_agent.planning.canonical_intent import (
    CanonicalIntent,
    CleanIntent,
    CleaningIntentOperation,
    ProjectColumnsIntent,
    UnresolvedColumnReference,
)
from finflow_agent.planning.compiler import compile_canonical_intent


def test_compile_canonical_intent_expands_semantic_family_projection():
    intent = CanonicalIntent(
        schema_version="2.0",
        resolution_status="repaired",
        output_format="xlsx",
        dataframe_profile={
            "source_columns": ["age", "gender", "loan_amount", "loan_status", "loan_term_months"],
        },
        actions=[
            ProjectColumnsIntent(
                kind="project_columns",
                requested_fields=[
                    UnresolvedColumnReference(raw_reference="age", resolved_column="age", selection_mode="single"),
                    UnresolvedColumnReference(raw_reference="gender", resolved_column="gender", selection_mode="single"),
                    UnresolvedColumnReference(
                        raw_reference="loans",
                        selection_mode="semantic_family",
                        resolved_columns=["loan_amount", "loan_status", "loan_term_months"],
                        candidate_columns=["loan_amount", "loan_status", "loan_term_months"],
                    ),
                ],
            )
        ],
    )

    plan = compile_canonical_intent(
        intent,
        resolved_file_path="ignored.csv",
        file_type="csv",
        output_dir="outputs",
        artifact_prefix="submission_test",
    )

    assert plan.steps[2].agent == "filter_agent"
    assert plan.steps[2].params["plan"]["select_columns"] == [
        "age",
        "gender",
        "loan_amount",
        "loan_status",
        "loan_term_months",
    ]


def test_compile_canonical_intent_rejects_unresolved_projection():
    intent = CanonicalIntent(
        schema_version="2.0",
        resolution_status="repaired",
        output_format="xlsx",
        dataframe_profile={"source_columns": ["age", "gender"]},
        actions=[
            ProjectColumnsIntent(
                kind="project_columns",
                requested_fields=[
                    UnresolvedColumnReference(raw_reference="age", resolved_column="age", selection_mode="single"),
                    UnresolvedColumnReference(
                        raw_reference="loans",
                        selection_mode="ambiguous",
                        candidate_columns=["loan_amount", "loan_status"],
                    ),
                ],
            )
        ],
    )

    with pytest.raises(ValueError, match="Unresolved canonical column reference"):
        compile_canonical_intent(
            intent,
            resolved_file_path="ignored.csv",
            file_type="csv",
            output_dir="outputs",
            artifact_prefix="submission_test",
        )


def test_compile_canonical_intent_maps_drop_nulls_cleaning_operation():
    intent = CanonicalIntent(
        schema_version="2.0",
        resolution_status="resolved",
        output_format="xlsx",
        dataframe_profile={"source_columns": ["gender", "education_level", "marital_status", "age"]},
        actions=[
            CleanIntent(
                kind="clean",
                mode="explicit",
                operations=[
                    CleaningIntentOperation(
                        name="drop_nulls",
                        parameters={"columns": None, "how": "any"},
                    )
                ],
            )
        ],
    )

    plan = compile_canonical_intent(
        intent,
        resolved_file_path="ignored.csv",
        file_type="csv",
        output_dir="outputs",
        artifact_prefix="submission_test",
    )

    clean_step = next(step for step in plan.steps if step.step_id == "clean")
    operations = clean_step.params["plan"]["operations"]
    assert operations[0]["type"] == "drop_nulls"
    assert operations[0]["columns"] is None
    assert operations[0]["how"] == "any"
