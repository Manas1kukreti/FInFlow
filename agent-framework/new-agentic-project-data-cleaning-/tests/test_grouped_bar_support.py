from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from finflow_agent.agents.visualization_agent import VisualizationAgent
from finflow_agent.execution.visualization.executor import VisualizationExecutor
from finflow_agent.models.draft import (
    ReferenceKind,
    ResolutionStatus,
    SemanticColumnReference,
    SemanticIntentDraft,
    VisualizeAction,
    VisualizationMeasure,
    VisualizationOptions,
)
from finflow_agent.models.provenance import PromptSpanProvenance
from finflow_agent.models.snapshot import DataSnapshotRef
from finflow_agent.models.upcasters import upcast_canonical_intent
from finflow_agent.pipeline.orchestrator import _detect_visualization_capability_issue
from finflow_agent.planning.canonical_intent import VisualizeIntent
from finflow_agent.planning.compiler import compile_canonical_intent
from finflow_agent.planning.intent_enricher import enrich_intent_with_visualization


def _prov(text: str = "prompt") -> list[PromptSpanProvenance]:
    return [PromptSpanProvenance(start_offset=0, end_offset=max(1, len(text)), source_text=text)]


def test_intent_enricher_extracts_grouped_bar_semantics():
    intent = {
        "original_prompt": (
            "Create a single grouped bar chart showing education classification by gender, "
            "with education on the x-axis and separate Male and Female bars. "
            "Show the legend and value labels."
        ),
        "actions": [],
    }

    result = enrich_intent_with_visualization(intent)
    viz = next(action for action in result["actions"] if action["kind"] == "visualize")

    assert viz["chart_type"] == "bar"
    assert viz["x"] == "education"
    assert viz["series"] == "gender"
    assert viz["group_by"] == ["education", "gender"]
    assert viz["aggregation"] == "count"
    assert viz["bar_mode"] == "grouped"
    assert viz["show_legend"] is True
    assert viz["show_data_labels"] is True


def test_compiler_preserves_grouped_bar_series_dimension():
    intent = type("LegacyIntent", (), {})  # unused placeholder for typing only
    from finflow_agent.planning.canonical_intent import CanonicalIntent

    canonical_intent = CanonicalIntent(
        schema_version="2.0",
        resolution_status="resolved",
        original_prompt="Create a grouped bar chart showing education by gender.",
        actions=[
            VisualizeIntent(
                kind="visualize",
                chart_type="bar",
                x="education_level",
                series="gender",
                group_by=["education_level", "gender"],
                aggregation="count",
                output_field="count",
                bar_mode="grouped",
                show_legend=True,
            )
        ],
        output_format="xlsx",
    )

    plan = compile_canonical_intent(
        canonical_intent,
        resolved_file_path="/tmp/input.csv",
        file_type="csv",
        output_dir="/tmp/out",
        artifact_prefix="grouped",
    )

    viz_step = next(step for step in plan.steps if step.agent == "visualization_agent")
    chart = viz_step.params["plan"]["charts"][0]
    assert chart["x"] == "education_level"
    assert chart["series"] == "gender"
    assert chart["group_by"] == ["education_level", "gender"]
    assert chart["bar_mode"] == "grouped"


def test_visualization_agent_grouped_aggregation_preserves_long_form_rows():
    df = pd.DataFrame(
        {
            "education_level": ["PhD", "PhD", "PhD", "B.Tech", "B.Tech", "B.Tech"],
            "gender": ["Male", "Female", "Male", "Female", "Female", "Male"],
        }
    )

    result = VisualizationAgent._aggregate_for_chart(
        df,
        {
            "x": "education_level",
            "series": "gender",
            "group_by": ["education_level", "gender"],
            "aggregation": "count",
            "output_field": "count",
        },
    )

    rows = result["rows"]
    assert {"education_level": "PhD", "gender": "Male", "count": 2} in rows
    assert {"education_level": "PhD", "gender": "Female", "count": 1} in rows
    assert {"education_level": "B.Tech", "gender": "Male", "count": 1} in rows
    assert {"education_level": "B.Tech", "gender": "Female", "count": 2} in rows


def test_visualization_agent_does_not_fallback_to_gender_for_numeric_measure_bar_chart():
    df = pd.DataFrame(
        {
            "age": [30, 30, 41, 41, 41, 52],
            "gender": ["Female", "Male", "Female", "Male", "Male", "Female"],
            "loan_amount": [100, 110, 120, 130, 140, 150],
        }
    )

    result = VisualizationAgent._aggregate_for_chart(
        df,
        {
            "type": "bar",
            "measure": "age",
            "aggregation": "count",
            "output_field": "record_count",
            "title": "Bar Chart of age",
        },
    )

    fields = {field["id"]: field for field in result["fields"]}
    rows = result["rows"]

    assert "age" in fields
    assert "record_count" in fields
    assert "gender" not in fields
    assert {"age": 30, "record_count": 2} in rows
    assert {"age": 41, "record_count": 3} in rows
    assert {"age": 52, "record_count": 1} in rows


def test_visualization_executor_preserves_series_encoding_and_options():
    executor = VisualizationExecutor()
    spec = executor.execute(
        operation_result={
            "fields": [
                {"id": "education_level", "label": "Education", "data_type": "string", "role": "category", "unit": None, "aggregation": None},
                {"id": "gender", "label": "Gender", "data_type": "string", "role": "dimension", "unit": None, "aggregation": None},
                {"id": "count", "label": "Count", "data_type": "integer", "role": "measure", "unit": None, "aggregation": "count"},
            ],
            "rows": [
                {"education_level": "PhD", "gender": "Male", "count": 2},
                {"education_level": "PhD", "gender": "Female", "count": 1},
            ],
        },
        chart_type="bar",
        encoding_hints={"x": "education_level", "y": "count", "series": "gender"},
        source_result_id="src-1",
        operation_id="op-1",
    )

    assert spec.status == "ready"
    assert spec.encoding["x"] == "education_level"
    assert spec.encoding["y"] == "count"
    assert spec.encoding["series"] == "gender"
    assert spec.encoding["series_label"] == "Gender"


def test_upcaster_preserves_visualize_actions():
    new_intent = upcast_canonical_intent(
        {
            "schema_version": "2.0",
            "resolution_status": "resolved",
            "actions": [
                {
                    "kind": "visualize",
                    "chart_type": "bar",
                    "group_by": ["education_level", "gender"],
                    "aggregation": "count",
                    "output_field": "count",
                    "series": "gender",
                    "x": "education_level",
                    "bar_mode": "grouped",
                }
            ],
            "dataframe_profile": {"columns": ["education_level", "gender"]},
            "original_prompt": "Create a grouped bar chart by education and gender",
        }
    )

    assert new_intent.actions
    viz = new_intent.actions[0]
    assert viz.type == "visualize"
    assert viz.x == "education_level"
    assert viz.series == "gender"


def test_orchestrator_routes_invalid_combo_to_unsupported_not_clarification():
    draft = SemanticIntentDraft(
        raw_prompt="Create a histogram of age split by gender",
        actions=[
            VisualizeAction(
                chart_type="histogram",
                x=SemanticColumnReference(
                    reference_text="age",
                    reference_kind=ReferenceKind.EXPLICIT_NAME,
                    resolved_column="age",
                    provenance=_prov("age"),
                ),
                y=VisualizationMeasure(function="count", output_name="count"),
                series=SemanticColumnReference(
                    reference_text="gender",
                    reference_kind=ReferenceKind.EXPLICIT_NAME,
                    resolved_column="gender",
                    provenance=_prov("gender"),
                ),
                options=VisualizationOptions(),
                provenance=_prov("histogram"),
            )
        ],
        resolution_status=ResolutionStatus.RESOLVED,
        extraction_provenance=_prov("prompt"),
        data_snapshot_ref=DataSnapshotRef(
            file_id="file-1",
            content_hash="abc",
            byte_size=1,
            storage_version="1",
            profile_id="p1",
            structural_schema_fingerprint="s1",
            profile_fingerprint="p1",
        ),
    )

    issue = _detect_visualization_capability_issue(draft)
    assert issue is not None
    assert issue["code"] == "INVALID_CHART_COMBINATION"
