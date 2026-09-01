"""Integration tests for TriggerDetector with Intent Classifier and Plan Compiler.

Tests verify:
1. TriggerDetector is wired into the intent classification pipeline
2. When trigger detected, VisualizeIntent action is produced in canonical intent
3. Plan Compiler creates calc + viz DAG nodes from VisualizeIntent
4. Jobs without visualization intent produce empty visualizations array

Requirements: 1.1, 1.4, 2.1, 5.2, 13.1, 13.2
"""

from __future__ import annotations

import os
import sys


import pytest
from unittest.mock import patch

from finflow_agent.planning.intent_enricher import (
    enrich_intent_with_visualization,
    should_produce_visualization,
)
from finflow_agent.planning.canonical_intent import (
    CanonicalIntent,
    VisualizeIntent,
)
from finflow_agent.planning.plan_compiler_viz import compile_visualization_plan
from finflow_agent.planning.visualization_intent import (
    classify_visualization_intent,
)


# ---------------------------------------------------------------------------
# Tests for enrich_intent_with_visualization
# ---------------------------------------------------------------------------


class TestEnrichIntentWithVisualization:
    """Tests for the intent enricher integration layer."""

    def test_adds_visualize_action_when_trigger_detected(self):
        """Req 1.1: Trigger language produces VisualizeIntent action."""
        intent = {
            "original_prompt": "Show me a bar chart of revenue by month",
            "actions": [{"kind": "filter_rows", "conditions": []}],
        }
        result = enrich_intent_with_visualization(intent)
        action_kinds = [a["kind"] for a in result["actions"]]
        assert "visualize" in action_kinds

    def test_chart_type_hint_propagated(self):
        """Req 1.1: Chart type hint from phrase is captured."""
        intent = {
            "original_prompt": "Create a pie chart of expenses",
            "actions": [],
        }
        result = enrich_intent_with_visualization(intent)
        viz_action = next(a for a in result["actions"] if a["kind"] == "visualize")
        assert viz_action["chart_type"] == "pie"

    def test_no_trigger_leaves_intent_unchanged(self):
        """Req 13.1, 13.2: No trigger → no visualization action."""
        intent = {
            "original_prompt": "Calculate the total revenue for Q1",
            "actions": [{"kind": "calculate", "operations": []}],
        }
        result = enrich_intent_with_visualization(intent)
        action_kinds = [a["kind"] for a in result["actions"]]
        assert "visualize" not in action_kinds

    def test_analytical_only_terms_no_visualization(self):
        """Req 1.2: Analytical terms alone do not trigger visualization."""
        intent = {
            "original_prompt": "Show me the trend and distribution breakdown",
            "actions": [],
        }
        result = enrich_intent_with_visualization(intent)
        action_kinds = [a["kind"] for a in result["actions"]]
        assert "visualize" not in action_kinds

    def test_does_not_duplicate_existing_visualize_action(self):
        """If VisualizeIntent already exists, no duplication."""
        intent = {
            "original_prompt": "Show me a chart",
            "actions": [{"kind": "visualize", "chart_type": "bar", "fields": []}],
        }
        result = enrich_intent_with_visualization(intent)
        viz_actions = [a for a in result["actions"] if a["kind"] == "visualize"]
        assert len(viz_actions) == 1

    def test_empty_prompt_no_visualization(self):
        """Empty prompt → no visualization."""
        intent = {"original_prompt": "", "actions": []}
        result = enrich_intent_with_visualization(intent)
        assert result["actions"] == []

    def test_none_input_returns_as_is(self):
        """Non-dict input returns unchanged."""
        assert enrich_intent_with_visualization(None) is None

    def test_trigger_with_analytical_terms(self):
        """Req 1.4: Trigger + analytical terms → visualization triggered."""
        intent = {
            "original_prompt": "Show a chart of the trend distribution",
            "actions": [],
        }
        result = enrich_intent_with_visualization(intent)
        action_kinds = [a["kind"] for a in result["actions"]]
        assert "visualize" in action_kinds

    def test_substring_does_not_trigger(self):
        """Req 1.3: 'uncharted' should not trigger visualization."""
        intent = {
            "original_prompt": "The uncharted territory was vast",
            "actions": [],
        }
        result = enrich_intent_with_visualization(intent)
        action_kinds = [a["kind"] for a in result["actions"]]
        assert "visualize" not in action_kinds

    def test_line_chart_hint(self):
        """Line chart phrase → chart_type='line'."""
        intent = {
            "original_prompt": "Show a line chart of growth over time",
            "actions": [],
        }
        result = enrich_intent_with_visualization(intent)
        viz_action = next(a for a in result["actions"] if a["kind"] == "visualize")
        assert viz_action["chart_type"] == "line"

    def test_generic_trigger_no_chart_type(self):
        """Generic 'graph' term → chart_type is None."""
        intent = {
            "original_prompt": "Show me a graph of the data",
            "actions": [],
        }
        result = enrich_intent_with_visualization(intent)
        viz_action = next(a for a in result["actions"] if a["kind"] == "visualize")
        assert viz_action["chart_type"] is None


    def test_multi_chart_prompt_with_clustered_bar_and_pi_chart(self):
        intent = {
            "original_prompt": "generate the clustered bar graph for house ownership and the pi chart for male and female",
            "actions": [],
        }
        result = enrich_intent_with_visualization(intent)
        viz_actions = [a for a in result["actions"] if a["kind"] == "visualize"]

        assert len(viz_actions) == 2
        assert viz_actions[0]["chart_type"] == "bar"
        assert viz_actions[0]["bar_mode"] == "grouped"
        assert viz_actions[0]["description"] == "house ownership"
        assert viz_actions[1]["chart_type"] == "pie"
        assert viz_actions[1]["description"] == "male and female"


# ---------------------------------------------------------------------------
# Tests for should_produce_visualization
# ---------------------------------------------------------------------------


class TestShouldProduceVisualization:
    """Tests for the boolean utility function."""

    def test_returns_true_for_trigger(self):
        assert should_produce_visualization("show me a chart") is True

    def test_returns_false_for_no_trigger(self):
        assert should_produce_visualization("calculate revenue") is False

    def test_returns_false_for_empty(self):
        assert should_produce_visualization("") is False

    def test_returns_false_for_none(self):
        assert should_produce_visualization(None) is False


# ---------------------------------------------------------------------------
# Tests for Plan Compiler producing viz DAG nodes from VisualizeIntent
# ---------------------------------------------------------------------------


class TestPlanCompilerVisualizationIntegration:
    """Tests that Plan Compiler creates calc + viz DAG nodes from VisualizeIntent."""

    def test_compile_viz_plan_with_trigger_produces_nodes(self):
        """Req 2.1: VisualizeIntent → visualization DAG nodes."""
        nodes = compile_visualization_plan(
            prompt="Show me a bar chart of revenue",
            calc_step_ids=["calc_1"],
        )
        assert len(nodes) == 1
        assert nodes[0].kind == "visualization"
        assert nodes[0].depends_on == "calc_1"
        assert nodes[0].status == "pending"

    def test_compile_viz_plan_without_trigger_empty(self):
        """Req 13.1, 13.2: No trigger → empty list (no viz nodes)."""
        nodes = compile_visualization_plan(
            prompt="Calculate the total revenue for Q1",
            calc_step_ids=["calc_1"],
        )
        assert nodes == []

    def test_compile_viz_plan_multiple_calc_steps(self):
        """Req 5.2: Multiple calc steps → one viz node per calc step."""
        nodes = compile_visualization_plan(
            prompt="Visualize the results",
            calc_step_ids=["calc_1", "calc_2"],
        )
        assert len(nodes) == 2
        assert nodes[0].depends_on == "calc_1"
        assert nodes[1].depends_on == "calc_2"

    def test_compile_viz_plan_preserves_chart_type_hint(self):
        """Chart type hint from trigger phrase is propagated to node."""
        nodes = compile_visualization_plan(
            prompt="Show a pie chart of expenses",
            calc_step_ids=["calc_step_1"],
        )
        assert len(nodes) == 1
        assert nodes[0].chart_type == "pie"

    def test_compile_viz_plan_no_calc_steps_returns_empty(self):
        """No calc steps available → empty list (cannot create viz nodes)."""
        nodes = compile_visualization_plan(
            prompt="Show me a chart",
            calc_step_ids=[],
        )
        assert nodes == []


# ---------------------------------------------------------------------------
# Tests for Canonical Intent Compiler VisualizeIntent handling
# ---------------------------------------------------------------------------


class TestCanonicalCompilerVisualization:
    """Tests that the canonical compiler handles VisualizeIntent properly."""

    def test_compile_canonical_intent_with_visualize_action(self):
        """VisualizeIntent action triggers visualization in compile pipeline."""
        from finflow_agent.planning.compiler import compile_canonical_intent

        intent = CanonicalIntent(
            schema_version="2.0",
            resolution_status="resolved",
            original_prompt="Show me a bar chart of revenue",
            actions=[
                VisualizeIntent(kind="visualize", chart_type="bar", fields=[]),
            ],
            output_format="xlsx",
        )

        # The compiler should NOT raise ValueError for VisualizeIntent now.
        # It should produce a plan with visualization step.
        with patch("finflow_agent.planning.compiler.get_enable_visualization", return_value=True):
            plan = compile_canonical_intent(
                intent,
                resolved_file_path="/tmp/test.csv",
                file_type="csv",
                output_dir="/tmp/output",
                artifact_prefix="test",
            )

        # Plan should contain a visualization step
        step_agents = [s.agent for s in plan.steps]
        assert "visualization_agent" in step_agents

    def test_compile_canonical_intent_without_visualize_no_viz_step(self):
        """No VisualizeIntent → no visualization step (Req 13.1)."""
        from finflow_agent.planning.compiler import compile_canonical_intent
        from finflow_agent.planning.canonical_intent import (
            CleanIntent,
            CleaningIntentOperation,
        )

        intent = CanonicalIntent(
            schema_version="2.0",
            resolution_status="resolved",
            original_prompt="Clean the data",
            actions=[
                CleanIntent(kind="clean", mode="safe_default", operations=[]),
            ],
            output_format="xlsx",
        )

        plan = compile_canonical_intent(
            intent,
            resolved_file_path="/tmp/test.csv",
            file_type="csv",
            output_dir="/tmp/output",
            artifact_prefix="test",
        )

        step_agents = [s.agent for s in plan.steps]
        assert "visualization_agent" not in step_agents

    def test_compile_canonical_intent_viz_disabled_raises(self):
        """VisualizeIntent with viz disabled → VisualizationDisabledError."""
        from finflow_agent.planning.compiler import (
            compile_canonical_intent,
            VisualizationDisabledError,
        )

        intent = CanonicalIntent(
            schema_version="2.0",
            resolution_status="resolved",
            original_prompt="Show me a chart",
            actions=[
                VisualizeIntent(kind="visualize", chart_type=None, fields=[]),
            ],
            output_format="xlsx",
        )

        with patch("finflow_agent.planning.compiler.get_enable_visualization", return_value=False):
            with pytest.raises(VisualizationDisabledError):
                compile_canonical_intent(
                    intent,
                    resolved_file_path="/tmp/test.csv",
                    file_type="csv",
                    output_dir="/tmp/output",
                    artifact_prefix="test",
                )

    def test_compile_canonical_intent_viz_step_depends_on_last_step(self):
        """Visualization step depends on the preceding step in the plan."""
        from finflow_agent.planning.compiler import compile_canonical_intent

        intent = CanonicalIntent(
            schema_version="2.0",
            resolution_status="resolved",
            original_prompt="Show chart",
            actions=[
                VisualizeIntent(kind="visualize", chart_type="line", fields=[]),
            ],
            output_format="xlsx",
        )

        with patch("finflow_agent.planning.compiler.get_enable_visualization", return_value=True):
            plan = compile_canonical_intent(
                intent,
                resolved_file_path="/tmp/test.csv",
                file_type="csv",
                output_dir="/tmp/output",
                artifact_prefix="test",
            )

        viz_step = next(s for s in plan.steps if s.agent == "visualization_agent")
        # viz step depends on the step before it (ingest in this minimal case)
        assert viz_step.depends_on == ["ingest"]
