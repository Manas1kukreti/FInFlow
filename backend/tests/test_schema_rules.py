from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from app.services.canonical_intent import build_canonical_intent
from app.services.data_profile import build_data_profile_from_file
from app.services.new_pipeline_bridge import _ground_draft_references, run_new_semantic_pipeline_sync
from app.services.rule_engine import build_validation_warnings
from app.services.new_pipeline_bridge import _convert_filter, _convert_action


def test_forbidden_substring_flags_normalized_variants():
    frame = pd.DataFrame({"payment_method": ["Card", "PayPal", "pay pal"]})

    warnings = build_validation_warnings(
        frame,
        [{"column": "*", "rule": "forbidden_substring", "value": "PayPal", "severity": "warning"}],
    )

    assert warnings == [
        {
            "column": "payment_method",
            "rule": "forbidden_substring",
            "severity": "warning",
            "reason": "",
            "invalid_count": 2,
            "sample_values": ["PayPal", "pay pal"],
        }
    ]


def test_starts_with_flags_invalid_account_prefix():
    frame = pd.DataFrame({"account_number": ["ACC100", "ZCC200"]})

    warnings = build_validation_warnings(
        frame,
        [{"column": "account_number", "rule": "starts_with", "value": "ACC", "severity": "error"}],
    )

    assert warnings[0]["column"] == "account_number"
    assert warnings[0]["rule"] == "starts_with"
    assert warnings[0]["invalid_count"] == 1
    assert warnings[0]["sample_values"] == ["ZCC200"]


def test_date_not_future_handles_naive_dates_without_crashing():
    future_year = pd.Timestamp.now().year + 3
    frame = pd.DataFrame({"invoice_date": ["2024-01-01", f"{future_year}-01-01"]})

    warnings = build_validation_warnings(
        frame,
        [{"column": "invoice_date", "rule": "date_not_future", "severity": "warning"}],
    )

    assert warnings[0]["column"] == "invoice_date"
    assert warnings[0]["rule"] == "date_not_future"
    assert warnings[0]["invalid_count"] == 1


def test_canonical_intent_distinguishes_projection_drop_and_filter():
    columns = ["Customer_ID", "Customer_Name", "Amount"]

    projection = build_canonical_intent(columns, [], "customer id only")
    drop_columns = build_canonical_intent(columns, [], "remove customer id")
    row_filter = build_canonical_intent(columns, [], "show rows where customer id is 1002")

    assert [action["kind"] for action in projection["actions"]] == ["project_columns"]
    assert projection["actions"][0]["requested_fields"][0]["resolved_column"] == "Customer_ID"

    assert [action["kind"] for action in drop_columns["actions"]] == ["drop_columns"]
    assert drop_columns["actions"][0]["requested_fields"][0]["resolved_column"] == "Customer_ID"

    assert [action["kind"] for action in row_filter["actions"]] == ["filter_rows"]
    assert row_filter["actions"][0]["conditions"][0]["field"]["resolved_column"] == "Customer_ID"
    assert row_filter["actions"][0]["conditions"][0]["value"] == 1002


def test_canonical_intent_expands_projection_families_explicitly():
    columns = ["age", "gender", "loan_amount", "loan_status", "loan_term_months"]

    projection = build_canonical_intent(columns, [], "only show age, gender, loans columns")

    assert [action["kind"] for action in projection["actions"]] == ["project_columns"]
    requested_fields = projection["actions"][0]["requested_fields"]
    assert requested_fields[0]["resolved_column"] == "age"
    assert requested_fields[1]["resolved_column"] == "gender"
    assert requested_fields[2]["selection_mode"] == "semantic_family"
    assert requested_fields[2]["resolved_columns"] == ["loan_amount", "loan_status", "loan_term_months"]
    assert projection["resolution_status"] in {"resolved", "repaired"}


def test_canonical_intent_pauses_on_unresolved_projection_family():
    columns = ["age", "gender"]

    projection = build_canonical_intent(columns, [], "only show loans columns")

    assert [action["kind"] for action in projection["actions"]] == ["project_columns"]
    requested_field = projection["actions"][0]["requested_fields"][0]
    assert requested_field["selection_mode"] == "ambiguous"
    assert projection["resolution_status"] == "needs_clarification"


def test_canonical_intent_select_all_projection_is_resolved():
    columns = ["name", "amount", "status"]

    projection = build_canonical_intent(columns, [], "Clean the data and return all columns.")

    assert [action["kind"] for action in projection["actions"]] == ["clean", "project_columns"]
    requested_field = projection["actions"][1]["requested_fields"][0]
    assert requested_field["resolution_method"] == "all_columns"
    assert requested_field["resolved_columns"] == columns
    assert projection["resolution_status"] in {"resolved", "repaired"}


def test_canonical_intent_preserves_ratio_and_pie_semantics():
    columns = ["gender", "age", "name"]

    result = build_canonical_intent(columns, [], "Clean this data and show the male-to-female ratio as a pie chart.")

    action_kinds = [action["kind"] for action in result["actions"]]
    assert "calculate" in action_kinds
    assert "visualize" in action_kinds

    calculate = next(action for action in result["actions"] if action["kind"] == "calculate")
    operation = calculate["operations"][0]
    assert operation["type"] == "group_count"
    assert operation["group_by"][0]["resolved_column"] == "gender"
    assert operation["output_column"] == "gender_count"

    visualize = next(action for action in result["actions"] if action["kind"] == "visualize")
    assert visualize["chart_type"] == "pie"
    assert visualize["fields"][0]["resolved_column"] == "gender"
    assert result["resolution_status"] in {"resolved", "repaired"}


def test_data_profile_build_is_deterministic_and_bounded():
    csv_path = Path(__file__).with_name(".tmp_profile.csv")
    try:
        csv_path.write_text("name,amount,status\nAlice,10,ok\nBob,20,ok\nCharlie,30,pending\n", encoding="utf-8")
        first = build_data_profile_from_file(csv_path, max_preview_rows=2)
        second = build_data_profile_from_file(csv_path, max_preview_rows=2)
    finally:
        csv_path.unlink(missing_ok=True)

    assert first is not None
    assert second is not None
    assert first[0]["file_fingerprint"] == second[0]["file_fingerprint"]
    assert first[0]["profiler_version"] == second[0]["profiler_version"]
    assert first[0]["row_count"] == 3
    assert len(first[0]["preview_rows"]) == 2
    assert len(first[0]["columns"][0]["sample_values"]) <= 3


def test_bridge_semantic_result_repairs_generic_filter_reference_from_preview(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_BRIDGE_API_KEY", "test-bridge-key")

    unresolved_bridge_result = {
        "schema_version": "2.0",
        "intent_id": "intent-1",
        "intent_revision": 1,
        "intent_hash": "hash-1",
        "parent_intent_id": None,
        "original_prompt": "Clean the data and extract rows which contains paypal or cash as field",
        "normalized_prompt": "clean the data and extract rows which contains paypal or cash as field",
        "resolution_status": "needs_clarification",
        "decision": "filter rows (1 condition(s))",
        "evidence": ["new_pipeline_extraction: 1.0"],
        "alternatives_considered": [],
        "actions": [
            {
                "kind": "filter_rows",
                "mode": "keep",
                "conditions": [
                    {
                        "field": {
                            "raw_reference": "field",
                            "resolved_column": None,
                            "resolution_method": "generic_reference",
                            "candidate_columns": [],
                            "evidence": [],
                            "resolved_columns": [],
                        },
                        "operator": "contains",
                        "value": ["paypal", "cash"],
                    }
                ],
                "logic": "and",
            }
        ],
        "output_format": "xlsx",
        "assumptions": [],
        "repair_notes": [],
        "dataframe_profile": {"columns": ["transaction_id", "payment_method", "transaction_status"]},
        "capability_version": "backend.capability.1",
        "capability_snapshot": {},
    }

    def _fake_bridge(*args, **kwargs):
        return unresolved_bridge_result

    monkeypatch.setattr("app.services.new_pipeline_bridge.run_new_semantic_pipeline_sync", _fake_bridge)

    preview_rows = [
        {
            "transaction_id": "T0001",
            "payment_method": "pay pal",
            "transaction_status": "Pending",
        },
        {
            "transaction_id": "T0002",
            "payment_method": "credit card",
            "transaction_status": "Completed",
        },
    ]

    result = build_canonical_intent(
        ["transaction_id", "payment_method", "transaction_status"],
        preview_rows,
        "Clean the data and extract rows which contains paypal or cash as field",
        detected_types={
            "transaction_id": "string",
            "payment_method": "string",
            "transaction_status": "string",
        },
    )

    condition = result["actions"][0]["conditions"][0]
    assert condition["field"]["resolved_column"] == "payment_method"
    assert condition["field"]["resolution_method"] == "profile_value_evidence"
    assert result["resolution_status"] == "repaired"


def test_canonical_intent_grounds_payment_field_from_observed_values(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_BRIDGE_API_KEY", "test-bridge-key")

    unresolved_bridge_result = {
        "schema_version": "2.0",
        "intent_id": "intent-2",
        "intent_revision": 1,
        "intent_hash": "hash-2",
        "parent_intent_id": None,
        "original_prompt": "Clean the data and extract rows which contains paypal or cash as field",
        "normalized_prompt": "clean the data and extract rows which contains paypal or cash as field",
        "resolution_status": "needs_clarification",
        "decision": "filter rows (1 condition(s))",
        "evidence": ["new_pipeline_extraction: 1.0"],
        "alternatives_considered": [],
        "actions": [
            {
                "kind": "filter_rows",
                "mode": "keep",
                "conditions": [
                    {
                        "field": {
                            "raw_reference": "field",
                            "resolved_column": None,
                            "resolution_method": "generic_reference",
                            "candidate_columns": [],
                            "evidence": [],
                            "resolved_columns": [],
                        },
                        "operator": "in",
                        "value": ["paypal", "cash"],
                    }
                ],
                "logic": "and",
            }
        ],
        "output_format": "xlsx",
        "assumptions": [],
        "repair_notes": [],
        "dataframe_profile": {"columns": ["transaction_id", "payment_method", "transaction_status"]},
        "capability_version": "backend.capability.1",
        "capability_snapshot": {},
    }

    def _fake_bridge(*args, **kwargs):
        return unresolved_bridge_result

    monkeypatch.setattr("app.services.new_pipeline_bridge.run_new_semantic_pipeline_sync", _fake_bridge)

    data_profile = {
        "source_columns": ["transaction_id", "payment_method", "transaction_status"],
        "detected_types": {
            "transaction_id": "string",
            "payment_method": "string",
            "transaction_status": "string",
        },
        "preview_rows": [
            {
                "transaction_id": "T0001",
                "payment_method": "Cash",
                "transaction_status": "Pending",
            },
            {
                "transaction_id": "T0002",
                "payment_method": "PayPal",
                "transaction_status": "Completed",
            },
        ],
        "columns": [
            {
                "name": "transaction_id",
                "sample_values": ["T0001", "T0002"],
                "semantic_type_hint": "transaction_id",
                "distinct_count": 2,
            },
            {
                "name": "payment_method",
                "sample_values": ["Cash", "PayPal"],
                "semantic_type_hint": "payment_method",
                "distinct_count": 2,
            },
            {
                "name": "transaction_status",
                "sample_values": ["Pending", "Completed"],
                "semantic_type_hint": "status",
                "distinct_count": 2,
            },
        ],
    }

    result = build_canonical_intent(
        ["transaction_id", "payment_method", "transaction_status"],
        data_profile["preview_rows"],
        "Clean the data and extract rows which contains paypal or cash as field",
        detected_types=data_profile["detected_types"],
        data_profile=data_profile,
    )

    condition = result["actions"][0]["conditions"][0]
    assert condition["field"]["resolved_column"] == "payment_method"
    assert condition["field"]["resolution_method"] == "profile_value_evidence"
    assert condition["operator"] == "in"
    assert condition["value"] == ["paypal", "cash"]
    assert result["resolution_status"] == "repaired"


def test_canonical_intent_grounds_marital_status_from_value_evidence():
    columns = ["gender", "education_level", "marital_status"]
    preview_rows = [
        {
            "gender": "Female",
            "education_level": "PhD",
            "marital_status": "Single",
        },
        {
            "gender": "Male",
            "education_level": "Master",
            "marital_status": "Married",
        },
        {
            "gender": "Female",
            "education_level": "PhD",
            "marital_status": "Divorced",
        },
    ]

    result = build_canonical_intent(
        columns,
        preview_rows,
        "Clean the data and extract rows which contains female as gender, phd as education and single as status",
        detected_types={
            "gender": "string",
            "education_level": "string",
            "marital_status": "string",
        },
    )

    filter_action = next(action for action in result["actions"] if action["kind"] == "filter_rows")
    predicates = filter_action["conditions"]

    resolved = {predicate["field"]["resolved_column"]: predicate["value"] for predicate in predicates}
    assert resolved["gender"] == "female"
    assert resolved["education_level"] == "phd"
    assert resolved["marital_status"] == "single"
    assert result["resolution_status"] in {"resolved", "repaired"}


def test_canonical_intent_treats_null_row_cleanup_as_clean_not_drop_columns():
    columns = ["gender", "education_level", "marital_status", "age"]
    preview_rows = [
        {"gender": "Female", "education_level": "PhD", "marital_status": "Single", "age": 30},
        {"gender": "Male", "education_level": "Master", "marital_status": None, "age": 41},
        {"gender": "Female", "education_level": "PhD", "marital_status": "Single", "age": ""},
    ]

    result = build_canonical_intent(
        columns,
        preview_rows,
        "Clean this data and only return rows which contains female as gender, phd as education and single as status also drops the rows which has any field as empty or null",
        detected_types={
            "gender": "string",
            "education_level": "string",
            "marital_status": "string",
            "age": "number",
        },
    )

    kinds = [action["kind"] for action in result["actions"]]
    assert "drop_columns" not in kinds
    assert kinds[0] == "clean"
    assert kinds[1] == "filter_rows"

    clean_action = result["actions"][0]
    operations = clean_action["operations"]
    assert any(op["name"] == "drop_nulls" for op in operations)
    drop_nulls_op = next(op for op in operations if op["name"] == "drop_nulls")
    assert drop_nulls_op["parameters"] == {"columns": None, "how": "any"}

    filter_action = result["actions"][1]
    resolved = {predicate["field"]["resolved_column"]: predicate["value"] for predicate in filter_action["conditions"]}
    assert resolved["gender"] == "female"
    assert resolved["education_level"] == "phd"
    assert resolved["marital_status"] == "single"
    assert result["resolution_status"] in {"resolved", "repaired"}


def test_canonical_intent_grounds_status_to_marital_status_amid_multiple_status_columns():
    columns = ["gender", "education_level", "employment_status", "marital_status"]
    preview_rows = [
        {
            "gender": "Male",
            "education_level": "Master",
            "employment_status": "Full-time",
            "marital_status": "Married",
        },
        {
            "gender": "Female",
            "education_level": "PhD",
            "employment_status": "Part-time",
            "marital_status": "Single",
        },
    ]

    result = build_canonical_intent(
        columns,
        preview_rows,
        "Clean the data and extract rows which contains female as gender, phd as education and single as status",
        detected_types={
            "gender": "string",
            "education_level": "string",
            "employment_status": "string",
            "marital_status": "string",
        },
    )

    filter_action = next(action for action in result["actions"] if action["kind"] == "filter_rows")
    predicates = filter_action["conditions"]

    resolved = {predicate["field"]["resolved_column"]: predicate["value"] for predicate in predicates}
    assert resolved["gender"] == "female"
    assert resolved["education_level"] == "phd"
    assert resolved["marital_status"] == "single"
    assert "employment_status" not in resolved
    assert result["resolution_status"] in {"resolved", "repaired"}


def test_canonical_intent_marks_status_ambiguous_when_two_columns_match_single():
    columns = ["gender", "education_level", "employment_status", "marital_status"]
    preview_rows = [
        {
            "gender": "Male",
            "education_level": "Master",
            "employment_status": "Single",
            "marital_status": "Single",
        },
        {
            "gender": "Female",
            "education_level": "PhD",
            "employment_status": "Multiple",
            "marital_status": "Married",
        },
    ]

    result = build_canonical_intent(
        columns,
        preview_rows,
        "return rows where status is single",
        detected_types={
            "gender": "string",
            "education_level": "string",
            "employment_status": "string",
            "marital_status": "string",
        },
    )

    filter_action = next(action for action in result["actions"] if action["kind"] == "filter_rows")
    condition = filter_action["conditions"][0]

    assert condition["field"]["selection_mode"] == "ambiguous"
    assert set(condition["field"]["candidate_columns"]) == {"employment_status", "marital_status"}
    assert condition["field"]["resolved_column"] is None
    assert result["resolution_status"] == "needs_clarification"


def test_canonical_intent_does_not_override_ambiguity_with_exact_status_column():
    columns = ["status", "marital_status"]
    preview_rows = [
        {"status": "Single", "marital_status": "Single"},
        {"status": "Multiple", "marital_status": "Married"},
    ]

    result = build_canonical_intent(
        columns,
        preview_rows,
        "return rows where status is single",
        detected_types={
            "status": "string",
            "marital_status": "string",
        },
    )

    filter_action = next(action for action in result["actions"] if action["kind"] == "filter_rows")
    condition = filter_action["conditions"][0]

    assert condition["field"]["selection_mode"] == "ambiguous"
    assert set(condition["field"]["candidate_columns"]) == {"status", "marital_status"}
    assert condition["field"]["resolved_column"] is None
    assert result["resolution_status"] == "needs_clarification"


def test_canonical_intent_returns_needs_clarification_without_value_evidence():
    columns = ["gender", "education_level", "employment_status", "marital_status"]
    preview_rows = [
        {
            "gender": "Male",
            "education_level": "Master",
            "employment_status": "Active",
            "marital_status": "Married",
        },
        {
            "gender": "Female",
            "education_level": "PhD",
            "employment_status": "Inactive",
            "marital_status": "Divorced",
        },
    ]

    result = build_canonical_intent(
        columns,
        preview_rows,
        "return rows where status is single",
        detected_types={
            "gender": "string",
            "education_level": "string",
            "employment_status": "string",
            "marital_status": "string",
        },
    )

    filter_action = next(action for action in result["actions"] if action["kind"] == "filter_rows")
    condition = filter_action["conditions"][0]

    assert condition["field"].get("resolved_column") is None
    assert result["resolution_status"] == "needs_clarification"


def test_canonical_intent_normalizes_status_value_case():
    columns = ["gender", "education_level", "marital_status"]
    preview_rows = [
        {
            "gender": "Female",
            "education_level": "PhD",
            "marital_status": "SINGLE",
        },
        {
            "gender": "Male",
            "education_level": "Master",
            "marital_status": "MARRIED",
        },
    ]

    result = build_canonical_intent(
        columns,
        preview_rows,
        "return rows where status is single",
        detected_types={
            "gender": "string",
            "education_level": "string",
            "marital_status": "string",
        },
    )

    filter_action = next(action for action in result["actions"] if action["kind"] == "filter_rows")
    condition = filter_action["conditions"][0]

    assert condition["field"]["resolved_column"] == "marital_status"
    assert condition["field"]["grounded_value"] == "SINGLE"
    assert result["resolution_status"] in {"resolved", "repaired"}


def test_canonical_intent_full_prompt_ambiguous_status_needs_clarification():
    columns = ["gender", "education_level", "employment_status", "marital_status"]
    preview_rows = [
        {
            "gender": "Female",
            "education_level": "PhD",
            "employment_status": "Single",
            "marital_status": "Single",
        },
        {
            "gender": "Male",
            "education_level": "Master",
            "employment_status": "Multiple",
            "marital_status": "Married",
        },
    ]

    result = build_canonical_intent(
        columns,
        preview_rows,
        "Clean this data and only return rows which contains female as gender, phd as education and single as status",
        detected_types={
            "gender": "string",
            "education_level": "string",
            "employment_status": "string",
            "marital_status": "string",
        },
    )

    filter_action = next(action for action in result["actions"] if action["kind"] == "filter_rows")
    predicates = filter_action["conditions"]

    resolved = [predicate["field"].get("resolved_column") for predicate in predicates]
    ambiguous = [predicate["field"] for predicate in predicates if predicate["field"].get("selection_mode") == "ambiguous"]

    assert "gender" in resolved
    assert "education_level" in resolved
    assert ambiguous
    assert set(ambiguous[0]["candidate_columns"]) == {"employment_status", "marital_status"}
    assert result["resolution_status"] == "needs_clarification"


def test_canonical_intent_normalizes_membership_filter_without_bridge(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_BRIDGE_API_KEY", raising=False)

    data_profile = {
        "source_columns": ["transaction_id", "payment_method", "transaction_status"],
        "detected_types": {
            "transaction_id": "string",
            "payment_method": "string",
            "transaction_status": "string",
        },
        "preview_rows": [
            {
                "transaction_id": "T0001",
                "payment_method": "Cash",
                "transaction_status": "Pending",
            },
            {
                "transaction_id": "T0002",
                "payment_method": "PayPal",
                "transaction_status": "Completed",
            },
        ],
        "columns": [
            {
                "name": "payment_method",
                "sample_values": ["Cash", "PayPal"],
                "semantic_type_hint": "payment_method",
                "distinct_count": 2,
            },
        ],
    }

    result = build_canonical_intent(
        data_profile["source_columns"],
        data_profile["preview_rows"],
        "Clean the data and extract rows which contains paypal or cash as field",
        detected_types=data_profile["detected_types"],
        data_profile=data_profile,
    )

    filter_action = next(action for action in result["actions"] if action["kind"] == "filter_rows")
    condition = filter_action["conditions"][0]
    assert condition["field"]["resolved_column"] == "payment_method"
    assert condition["operator"] == "in"
    assert condition["value"] == ["paypal", "cash"]
    assert result["resolution_status"] == "repaired"


def test_canonical_intent_keeps_education_ambiguous():
    columns = ["education_id", "education_status", "education_duration"]

    result = build_canonical_intent(columns, [], "only show education")

    assert [action["kind"] for action in result["actions"]] == ["project_columns"]
    requested_field = result["actions"][0]["requested_fields"][0]
    assert requested_field["selection_mode"] == "ambiguous"
    assert set(requested_field["candidate_columns"]) == set(columns)
    assert result["resolution_status"] == "needs_clarification"


def test_new_pipeline_bridge_preserves_in_membership():
    field_ref = SimpleNamespace(
        reference_text="payment method",
        resolved_column="payment_method",
        reference_kind=SimpleNamespace(value="semantic_concept"),
    )
    predicate = SimpleNamespace(
        field_ref=field_ref,
        operator="in",
        value=["paypal", "cash"],
    )
    group = SimpleNamespace(operator="and", predicates=[predicate])
    action = SimpleNamespace(logical_groups=[group])

    converted = _convert_filter(action)

    assert converted["logic"] == "and"
    assert len(converted["conditions"]) == 1
    assert converted["conditions"][0]["operator"] == "in"
    assert converted["conditions"][0]["value"] == ["paypal", "cash"]


def test_new_pipeline_bridge_preserves_grouped_bar_semantics():
    from finflow_agent.models.draft import (
        ReferenceKind,
        SemanticColumnReference,
        VisualizeAction,
        VisualizationMeasure,
        VisualizationOptions,
    )
    from finflow_agent.models.provenance import PromptSpanProvenance

    provenance = [PromptSpanProvenance(start_offset=0, end_offset=6, source_text="prompt")]
    x_ref = SemanticColumnReference(
        reference_text="education",
        reference_kind=ReferenceKind.SEMANTIC_CONCEPT,
        resolved_column="education_level",
        confidence=1.0,
        provenance=provenance,
    )
    series_ref = SemanticColumnReference(
        reference_text="gender",
        reference_kind=ReferenceKind.SEMANTIC_CONCEPT,
        resolved_column="gender",
        confidence=1.0,
        provenance=provenance,
    )
    action = VisualizeAction(
        chart_type="bar",
        x=x_ref,
        y=VisualizationMeasure(function="count", output_name="count"),
        series=series_ref,
        options=VisualizationOptions(
            bar_mode="grouped",
            show_legend=True,
            show_data_labels=True,
            title="Education by Gender",
            x_axis_title="Education",
            y_axis_title="Count",
        ),
        original_description="grouped bar by education and gender",
        provenance=provenance,
    )

    converted = _convert_action(action, ["education_level", "gender"])

    assert converted["kind"] == "visualize"
    assert converted["chart_type"] == "bar"
    assert converted["x"]["resolved_column"] == "education_level"
    assert converted["series"]["resolved_column"] == "gender"
    assert [item["resolved_column"] for item in converted["group_by"]] == ["education_level", "gender"]
    assert converted["aggregation"] == "count"
    assert converted["output_field"] == "count"
    assert converted["bar_mode"] == "grouped"
    assert converted["show_legend"] is True
    assert converted["show_data_labels"] is True


def test_ground_draft_references_handles_visualization_count_measure():
    from finflow_agent.models.draft import (
        ReferenceKind,
        SemanticColumnReference,
        VisualizeAction,
        VisualizationMeasure,
        VisualizationOptions,
    )
    from finflow_agent.models.provenance import PromptSpanProvenance

    provenance = [PromptSpanProvenance(start_offset=0, end_offset=6, source_text="prompt")]
    x_ref = SemanticColumnReference(
        reference_text="house ownership",
        reference_kind=ReferenceKind.SEMANTIC_CONCEPT,
        resolved_column=None,
        confidence=None,
        provenance=provenance,
    )
    series_ref = SemanticColumnReference(
        reference_text="gender",
        reference_kind=ReferenceKind.SEMANTIC_CONCEPT,
        resolved_column=None,
        confidence=None,
        provenance=provenance,
    )
    action = VisualizeAction(
        chart_type="bar",
        x=x_ref,
        y=VisualizationMeasure(function="count", output_name="count"),
        series=series_ref,
        options=VisualizationOptions(bar_mode="grouped"),
        original_description="grouped bar by house ownership and gender",
        provenance=provenance,
    )
    draft = SimpleNamespace(actions=[action])

    _ground_draft_references(draft, ["House Ownership", "Gender"])

    assert action.x is not None and action.x.resolved_column == "House Ownership"
    assert action.series is not None and action.series.resolved_column == "Gender"
    assert action.y is not None


def test_canonical_intent_preserves_grouped_bar_visualize_action_from_bridge(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_BRIDGE_API_KEY", "test-bridge-key")

    bridge_result = {
        "schema_version": "2.0",
        "intent_id": "intent-viz-1",
        "intent_revision": 1,
        "intent_hash": "hash-viz-1",
        "parent_intent_id": None,
        "original_prompt": "Create a grouped bar chart showing count of people by education and gender.",
        "normalized_prompt": "create a grouped bar chart showing count of people by education and gender.",
        "resolution_status": "resolved",
        "decision": "visualize",
        "evidence": ["new_pipeline_extraction: 1.0"],
        "alternatives_considered": [],
        "actions": [
            {
                "kind": "visualize",
                "chart_type": "bar",
                "x": {
                    "raw_reference": "education",
                    "resolved_column": "education_level",
                    "resolution_method": "semantic_concept",
                    "resolved_columns": ["education_level"],
                    "candidate_columns": [],
                    "evidence": [],
                },
                "series": {
                    "raw_reference": "gender",
                    "resolved_column": "gender",
                    "resolution_method": "semantic_concept",
                    "resolved_columns": ["gender"],
                    "candidate_columns": [],
                    "evidence": [],
                },
                "group_by": [
                    {
                        "raw_reference": "education",
                        "resolved_column": "education_level",
                        "resolution_method": "semantic_concept",
                        "resolved_columns": ["education_level"],
                        "candidate_columns": [],
                        "evidence": [],
                    },
                    {
                        "raw_reference": "gender",
                        "resolved_column": "gender",
                        "resolution_method": "semantic_concept",
                        "resolved_columns": ["gender"],
                        "candidate_columns": [],
                        "evidence": [],
                    },
                ],
                "aggregation": "count",
                "output_field": "count",
                "bar_mode": "grouped",
                "show_legend": True,
                "show_data_labels": True,
            }
        ],
        "output_format": "xlsx",
        "assumptions": [],
        "repair_notes": [],
        "dataframe_profile": {"columns": ["education_level", "gender"]},
        "capability_version": "backend.capability.1",
        "capability_snapshot": {},
    }

    def _fake_bridge(*args, **kwargs):
        return bridge_result

    monkeypatch.setattr("app.services.new_pipeline_bridge.run_new_semantic_pipeline_sync", _fake_bridge)

    result = build_canonical_intent(
        ["education_level", "gender"],
        [{"education_level": "PhD", "gender": "Female"}],
        "Create a grouped bar chart showing count of people by education and gender.",
        detected_types={"education_level": "string", "gender": "string"},
    )

    visualize = next(action for action in result["actions"] if action["kind"] == "visualize")
    assert visualize["chart_type"] == "bar"
    assert visualize["x"]["resolved_column"] == "education_level"
    assert visualize["series"]["resolved_column"] == "gender"
    assert [item["resolved_column"] for item in visualize["group_by"]] == ["education_level", "gender"]
    assert visualize["bar_mode"] == "grouped"
    assert visualize["show_legend"] is True
    assert visualize["show_data_labels"] is True


def test_new_pipeline_bridge_returns_none_on_rate_limit(monkeypatch):
    from finflow_agent.grounding.llm_adapter import LLMProviderError

    monkeypatch.setenv("GROQ_BRIDGE_API_KEY", "bridge-key")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    def _boom(*args, **kwargs):
        raise LLMProviderError(
            "Groq API returned 429: rate limited",
            error_type="rate_limit",
            call_site="extraction",
        )

    monkeypatch.setattr("app.services.new_pipeline_bridge._run_pipeline", _boom)

    result = run_new_semantic_pipeline_sync(
        "Clean the data and extract rows which contains paypal or cash as field",
        ["payment_method"],
    )

    assert result is None
